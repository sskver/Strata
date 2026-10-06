// src/prefill/moe_fp16tc_down.cu - see include/strata/prefill/moe_fp16tc_down.hpp.
//
// The weight decoders are moe_fused_iq.cu's: a 32-value unit of a GGUF block becomes 32 int8 codes (Q2_0: code - 1
// through a byte table; IQ4_NL: kvalues_iq4nl[nibble] through table16) and one scale, w = d * code, so the FP16 weight
// is the dequantized llama.cpp value rounded once.  The activations are the q8_1 values MMQ reads, converted the way
// moe_fp16tc.cu does (the exact signed code times the block's scale, in FP16).  The products are FP16 tensor-core
// wmma 16x16x16 with FP32 accumulation.
//
// Work item = (expert z, row tile y of 192 rows, column block cx of 128 weight rows).  The grid is 20 column blocks x
// NC classes (NC = SMs / 20: 4 on a V100, one block per SM).  The group's (expert, row tile) tiles, heaviest first,
// are dealt to the NC classes in snake order, so every class carries about the same number of 16-row fragments (the
// launch ends with its slowest block); block (q, cx) walks the tiles of class q at column block cx, its list rotated
// by cx so the blocks of a class do not all reach their epilogues at once.
//
// Block structure: 4 producer warps load the raw bytes of the next step into registers (one step ahead), decode the
// weights and convert the activations into one of two shared-memory stages; 8 consumer warps (RW row groups x 8/RW
// column groups) run the MMA on the other stage.  The halves hand stages over with named barriers (FULL 1/2, EMPTY
// 3/4); the pipeline runs on across the block's items, so the producers fill the next item's first stage while the
// consumers write the previous item out.  The epilogue stages each warp's accumulators through the stage it just
// consumed (the consumers first agree on barrier 5 that every warp is past its MMA) and writes whole 128-byte lines.
#include "strata/prefill/moe_fp16tc_down.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include "ggml.h"

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <type_traits>

namespace strata::prefill::fp16tc_down {
namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "prefill fp16tc down: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

constexpr int T_Q2_0 = GGML_TYPE_Q2_0, T_IQ4_NL = GGML_TYPE_IQ4_NL;
static_assert(sizeof(block_q2_0) == 18 && sizeof(block_iq4_nl) == 18, "the block layouts this file decodes");

constexpr int N = 2560, FF = 640;                      // output rows (n_embd) and K (n_ff)
constexpr int BN = 128, KS = 64, ACTB = 144, NCB = N / BN;   // weight rows per item, K per stage, q8_1 block bytes
constexpr int NK = FF / KS;                            // stages per item
constexpr int MCAP = 192, NPW = 4;                     // rows per item, producer warps
constexpr int NT = 256 + 32 * NPW, PT = 32 * NPW;      // threads per block, producer threads
constexpr int AP = 72, WP = 72;                        // smem leading dims (halves) of the activation / weight tiles
constexpr int ASZ = MCAP * AP, SSZ = ASZ + BN * WP;    // halves per stage
constexpr int SMEM = 2 * SSZ * 2;                      // bytes: two stages (90 KB)
constexpr int MAXN = 64, LMAX = 256;                   // experts per launch, items per block
struct DBlobs { const uint8_t* p[MAXN]; };   // expert z's down matrix starts at p[z]
constexpr int MIN_ROWS = 64;                           // below this group max_rows the caller's path is as fast (see down())
__host__ __device__ constexpr int row_bytes(int t) { return t == T_Q2_0 ? FF / 64 * 18 : FF / 32 * 18; }

__device__ __forceinline__ uint32_t ld16(const uint8_t* p) { return *(const uint16_t*) p; }
__device__ __forceinline__ uint32_t ld32(const uint8_t* p) { return ld16(p) | (ld16(p + 2) << 16); }
__device__ __forceinline__ float half_at(uint32_t w) { return __half2float(__ushort_as_half((unsigned short) w)); }
__device__ __forceinline__ __half2 h2_from_u32(uint32_t u) { __half2 h; __builtin_memcpy(&h, &u, 4); return h; }
__device__ __forceinline__ uint32_t u32_from_h2(__half2 h) { uint32_t u; __builtin_memcpy(&u, &h, 4); return u; }
// two biased bytes (code ^ 0x80 each, selected by `sel`) -> two halves of (code * sc): the bytes ride in the mantissa of
// 1024.0h, the 1152 offset leaves the signed code exactly, and sc is the scale
__device__ __forceinline__ uint32_t cvt2(uint32_t v, int sel, __half2 sc) {
    const __half2 c1152 = h2_from_u32(0x64806480u);
    return u32_from_h2(__hmul2(__hsub2(h2_from_u32(__byte_perm(v, 0x64646464u, sel)), c1152), sc));
}
__device__ __forceinline__ void bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void bar_arrive(int id, int n) { asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(n) : "memory"); }
// 8 nibbles of q4 through a 16-entry int8 table (4 words): the low nibbles' values in lo, the high ones' in hi
__device__ __forceinline__ void table16(uint32_t q4, const uint32_t (&t)[4], uint32_t& lo, uint32_t& hi) {
    uint32_t tmp[2];
    const uint32_t sel = 0x32103210u | ((q4 & 0x88888888u) >> 1);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const uint32_t sh = 16 * i;
        const uint32_t l = __byte_perm(t[0], t[1], q4 >> sh), h = __byte_perm(t[2], t[3], q4 >> sh);
        tmp[i] = __byte_perm(l, h, sel >> sh);
    }
    lo = __byte_perm(tmp[0], tmp[1], 0x6420);
    hi = __byte_perm(tmp[0], tmp[1], 0x7531);
}

// A 32-value unit of a weight row: `load` reads its raw bytes (value offset kv of the row), `convert` makes the 32
// int8 codes in natural order and the scale.
template <int WT> struct Fmt;
template <> struct Fmt<T_Q2_0> {   // 64-value blocks: fp16 d + 16 bytes of 2-bit codes (4 a byte, low first), w = (q - 1) d
    static constexpr int NW = 3;
    __device__ static void load(const uint8_t* row, int kv, uint32_t (&w)[NW]) {
        const uint8_t* bp = row + (kv >> 6) * 18;
        const int ib = (kv >> 5) & 1;
        w[0] = ld32(bp + 2 + 8 * ib); w[1] = ld32(bp + 6 + 8 * ib); w[2] = ld16(bp);
    }
    __device__ static void convert(const uint32_t (&w)[NW], const uint32_t (&)[4], uint32_t (&q)[8], float& s) {
#pragma unroll
        for (int h = 0; h < 4; ++h) {   // code - 1 via the byte table {-1, 0, 1, 2}
            const uint32_t c = (w[h >> 1] >> (16 * (h & 1))) & 0xFFFF;
            const uint32_t qe = __byte_perm(0x020100FFu, 0x020100FFu, c & 0x7777);
            const uint32_t qo = __byte_perm(0x020100FFu, 0x020100FFu, (c >> 2) & 0x7777);
            q[2 * h] = __byte_perm(qe, qo, 0x5140);
            q[2 * h + 1] = __byte_perm(qe, qo, 0x7362);
        }
        s = half_at(w[2]);
    }
};
template <> struct Fmt<T_IQ4_NL> {   // 32-value blocks: fp16 d + 16 nibble bytes (j low, j + 16 high), w = d kvalues[q]
    static constexpr int NW = 5;
    __device__ static void load(const uint8_t* row, int kv, uint32_t (&w)[NW]) {
        const uint8_t* bp = row + (kv >> 5) * 18;
#pragma unroll
        for (int k = 0; k < 4; ++k) w[k] = ld32(bp + 2 + 4 * k);
        w[4] = ld16(bp);
    }
    __device__ static void convert(const uint32_t (&w)[NW], const uint32_t (&kv)[4], uint32_t (&q)[8], float& s) {
#pragma unroll
        for (int k = 0; k < 4; ++k) table16(w[k], kv, q[k], q[4 + k]);
        s = half_at(w[4]);
    }
};

struct ARaw { uint4 q; float d; };
struct Item { int z, row0, local, cx; };

// This block's work list (thread 0, once): the (expert, row tile) tiles of class q = blockIdx.x / 20, heaviest first,
// rotated by cx = blockIdx.x % 20.  Full tiles all weigh the same and come first; the partial tiles follow by
// decreasing row count; tile i goes to class i % nc on even passes and nc - 1 - i % nc on odd ones (snake order).
// Kept out of line so its locals do not share the kernel's register budget.
__device__ __noinline__ void build_items(const int* s_b, int n, uint32_t* s_list, int* s_n) {
    const int nc = gridDim.x / NCB, q = blockIdx.x / NCB, cx = blockIdx.x % NCB;
    uint32_t part[MAXN];
    int prow[MAXN], np = 0, i = 0, c = 0;
    auto deal = [&](uint32_t tile) {
        const int r = i / nc, p = i % nc;
        if (((r & 1) ? nc - 1 - p : p) == q && c < LMAX) s_list[c++] = tile;
        ++i;
    };
    for (int z = 0; z < n; ++z) {
        const int rows = s_b[z + 1] - s_b[z];
        for (int y = 0; y < rows / MCAP; ++y) deal((uint32_t) z << 16 | (uint32_t) y);
        if (rows % MCAP) {   // insertion by decreasing row count
            int j = np++;
            while (j > 0 && prow[j - 1] < rows % MCAP) { part[j] = part[j - 1]; prow[j] = prow[j - 1]; --j; }
            part[j] = (uint32_t) z << 16 | (uint32_t) (rows / MCAP);
            prow[j] = rows % MCAP;
        }
    }
    for (int j = 0; j < np; ++j) deal(part[j]);
    if (c > 1) {   // rotate left by cx % c: three reversals
        const int rot = cx % c;
        auto rev = [&](int a, int b) { for (--b; a < b; ++a, --b) { const uint32_t t = s_list[a]; s_list[a] = s_list[b]; s_list[b] = t; } };
        rev(0, rot); rev(rot, c); rev(0, c);
    }
    *s_n = c;
}

// The consumer warps are RW row groups x (8 / RW) column groups: a warp owns the 16-row fragments rg, rg + RW, ...
// (MF of them) and SPAN / 16 column fragments.  RW 2 (6 x 2 fragments) is the gate/up kernel's layout.
template <int WT, int RW, int PF>
__global__ void __launch_bounds__(NT, 1)
down_kernel(const DBlobs dw, int n, int yt, const int32_t* __restrict__ bounds,
            const void* __restrict__ act, int64_t act_rows, float* __restrict__ dst, int64_t ld_dst, int64_t dst_row_base) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    using namespace nvcuda;
    using F = Fmt<WT>;
    constexpr int RB = row_bytes(WT);
    constexpr int CG = 8 / RW, SPAN = BN / CG, NCF = SPAN / 16, MF = MCAP / (16 * RW);
    constexpr int SLD = SPAN + 4;                      // epilogue staging leading dim (floats)
    constexpr int WU = (BN * 2 + PT - 1) / PT;         // weight units (32 values) per producer thread and step
    constexpr int AT = (MCAP * 4 + PT - 1) / PT;       // activation tasks (16 values) per producer thread and step
    static_assert(8 * 16 * SLD * 4 <= SSZ * 2, "the epilogue staging fits in one stage");
    extern __shared__ __align__(16) uint8_t smem[];
    __half* const bufs = (__half*) smem;
    __shared__ int s_b[MAXN + 1];
    __shared__ uint32_t s_list[LMAX];
    __shared__ int s_n;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;

    for (int i = tid; i <= n; i += NT) s_b[i] = bounds[i];
    __syncthreads();
    if (tid == 0) build_items(s_b, n, s_list, &s_n);
    __syncthreads();
    const int total = s_n;
    if (total == 0) return;
    const int nsteps = total * NK;
    const int cx = blockIdx.x % NCB;
    auto item_at = [&](int t) {
        const uint32_t e = s_list[t];
        Item it;
        it.z = (int) (e >> 16);
        it.row0 = s_b[it.z] + (int) (e & 0xFFFF) * MCAP;
        it.local = min(s_b[it.z + 1] - it.row0, MCAP);
        it.cx = cx;
        return it;
    };

    if (warp >= 8) {   // ---------------- producers: load the next step's bytes, decode / convert into the stage
        const int pt = tid - 256;
        uint32_t kv[4] = {0, 0, 0, 0};
        if constexpr (WT == T_IQ4_NL) {
#pragma unroll
            for (int k = 0; k < 16; ++k) kv[k >> 2] |= (uint32_t) (uint8_t) kvalues_iq4nl[k] << (8 * (k & 3));
        }
        uint32_t wr[PF][WU][F::NW];   // PF steps of raw bytes in flight
        ARaw ar[PF][AT];
        auto load = [&](auto pc, const Item& t, int k0) {
            constexpr int P = decltype(pc)::value;
            const uint8_t* wz = dw.p[t.z] + (size_t) (t.cx * BN) * RB;
#pragma unroll
            for (int u = 0; u < WU; ++u) {
                const int unit = pt + u * PT, row = unit >> 1, j = unit & 1;
                if (unit >= BN * 2) continue;
                F::load(wz + (size_t) row * RB, k0 + 32 * j, wr[P][u]);
            }
            const int kb = k0 >> 7;
#pragma unroll
            for (int j = 0; j < AT; ++j) {
                const int task = pt + j * PT, m = task >> 2, off = (k0 & 127) + (task & 3) * 16;
                if (task < MCAP * 4 && m < t.local) {
                    const uint8_t* base = (const uint8_t*) act + ((size_t) kb * (size_t) act_rows + (size_t) (t.row0 + m)) * ACTB;
                    ar[P][j].d = ((const float*) base)[off >> 5];
                    ar[P][j].q = *reinterpret_cast<const uint4*>(base + 16 + off);
                }
            }
        };
        auto store = [&](auto pc, const Item& t, int sb) {
            constexpr int P = decltype(pc)::value;
            __half* a_s = bufs + sb * SSZ;
            __half* w_s = a_s + ASZ;
#pragma unroll
            for (int u = 0; u < WU; ++u) {
                const int unit = pt + u * PT, row = unit >> 1, j = unit & 1;
                if (unit >= BN * 2) continue;
                uint32_t q[8];
                float sc;
                F::convert(wr[P][u], kv, q, sc);
                const __half2 s2 = __float2half2_rn(sc);
                uint4* o = reinterpret_cast<uint4*>(w_s + (size_t) row * WP + j * 32);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const uint32_t v0 = q[2 * i] ^ 0x80808080u, v1 = q[2 * i + 1] ^ 0x80808080u;
                    o[i] = make_uint4(cvt2(v0, 0x4140, s2), cvt2(v0, 0x4342, s2), cvt2(v1, 0x4140, s2), cvt2(v1, 0x4342, s2));
                }
            }
#pragma unroll
            for (int j = 0; j < AT; ++j) {
                const int task = pt + j * PT, m = task >> 2;
                if (task < MCAP * 4 && m < t.local) {
                    uint4* o = reinterpret_cast<uint4*>(a_s + (size_t) m * AP + (task & 3) * 16);
                    const __half2 d2 = __float2half2_rn(ar[P][j].d);
                    const uint32_t v0 = ar[P][j].q.x ^ 0x80808080u, v1 = ar[P][j].q.y ^ 0x80808080u, v2 = ar[P][j].q.z ^ 0x80808080u,
                                   v3 = ar[P][j].q.w ^ 0x80808080u;
                    o[0] = make_uint4(cvt2(v0, 0x4140, d2), cvt2(v0, 0x4342, d2), cvt2(v1, 0x4140, d2), cvt2(v1, 0x4342, d2));
                    o[1] = make_uint4(cvt2(v2, 0x4140, d2), cvt2(v2, 0x4342, d2), cvt2(v3, 0x4140, d2), cvt2(v3, 0x4342, d2));
                }
            }
        };
        // step s (item cur, k) sits in register set s % PF; once it is stored, step s + PF is loaded into that set
        Item cur = item_at(0), nxt = total > 1 ? item_at(1) : cur;
        bool more = total > 1;
        auto step = [&](auto pc, int k, int s) {
            const int b = s & 1;
            if (s >= 2) bar_sync(3 + b, NT);   // EMPTY(b): the consumers are done with stage b
            store(pc, cur, b);
            bar_arrive(1 + b, NT);             // FULL(b)
            if (k + PF < NK) load(pc, cur, (k + PF) * KS);
            else if (more) load(pc, nxt, (k + PF - NK) * KS);
        };
        load(std::integral_constant<int, 0>{}, cur, 0);
        if constexpr (PF == 2) load(std::integral_constant<int, 1>{}, cur, KS);
        static_assert(NK % PF == 0, "the register sets alternate within an item");
        int s = 0;
        for (int t = 0; t < total; ++t) {
            for (int k = 0; k < NK; k += PF, s += PF) {
                step(std::integral_constant<int, 0>{}, k, s);
                if constexpr (PF == 2) step(std::integral_constant<int, 1>{}, k + 1, s + 1);
            }
            cur = nxt;
            more = t + 2 < total;
            if (more) nxt = item_at(t + 2);
        }
        return;
    }
    // ---------------- consumers
    const int rg = warp / CG, n_base = (warp % CG) * SPAN;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[MF][NCF];
    auto mma_n = [&](const __half* a_s, const __half* w_s, auto na_c) {
        constexpr int NA = decltype(na_c)::value;
#pragma unroll
        for (int k16 = 0; k16 < KS / 16; ++k16) {
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> bf[NCF];
#pragma unroll
            for (int c = 0; c < NCF; ++c) wmma::load_matrix_sync(bf[c], w_s + (size_t) (n_base + 16 * c) * WP + k16 * 16, WP);
#pragma unroll
            for (int i = 0; i < NA; ++i) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> af;
                wmma::load_matrix_sync(af, a_s + (size_t) ((rg + RW * i) * 16) * AP + k16 * 16, AP);
#pragma unroll
                for (int c = 0; c < NCF; ++c) wmma::mma_sync(acc[i][c], af, bf[c], acc[i][c]);
            }
        }
    };
    int s = 0;
    for (int t = 0; t < total; ++t) {
        const Item it = item_at(t);
        const int nfr = (it.local + 15) >> 4;                       // 16-row fragments holding rows
        const int nact = min(MF, (nfr - rg + RW - 1) / RW);          // this warp's active fragments
#pragma unroll
        for (int i = 0; i < MF; ++i)
#pragma unroll
            for (int c = 0; c < NCF; ++c) wmma::fill_fragment(acc[i][c], 0.0f);
        for (int k = 0; k < NK; ++k, ++s) {
            const int b = s & 1;
            bar_sync(1 + b, NT);   // FULL(b)
            const __half* a_s = bufs + b * SSZ;
            const __half* w_s = a_s + ASZ;
            switch (nact) {   // no per-fragment guards in the MMA body: the count is a compile-time constant per case
                case 6: if constexpr (MF >= 6) mma_n(a_s, w_s, std::integral_constant<int, 6>{}); break;
                case 5: if constexpr (MF >= 5) mma_n(a_s, w_s, std::integral_constant<int, 5>{}); break;
                case 4: if constexpr (MF >= 4) mma_n(a_s, w_s, std::integral_constant<int, 4>{}); break;
                case 3: if constexpr (MF >= 3) mma_n(a_s, w_s, std::integral_constant<int, 3>{}); break;
                case 2: mma_n(a_s, w_s, std::integral_constant<int, 2>{}); break;
                case 1: mma_n(a_s, w_s, std::integral_constant<int, 1>{}); break;
                default: break;
            }
            if (k + 1 < NK) {
                if (s + 2 < nsteps) bar_arrive(3 + b, NT);   // EMPTY(b)
                continue;
            }
            // the item's last step: every consumer past its MMA, then stage b holds the accumulators on their way out
            bar_sync(5, 256);
            float* stg = reinterpret_cast<float*>(bufs + b * SSZ) + warp * 16 * SLD;
            float* drow = dst + (dst_row_base + it.row0) * ld_dst + it.cx * BN + n_base;
#pragma unroll
            for (int i = 0; i < MF; ++i) {
                const int f = rg + RW * i;
                if (f < nfr) {
#pragma unroll
                    for (int c = 0; c < NCF; ++c) wmma::store_matrix_sync(stg + 16 * c, acc[i][c], SLD, wmma::mem_row_major);
                    __syncwarp();
#pragma unroll
                    for (int h = 0; h < SPAN / 8; ++h) {   // a store instruction writes 512 / (4 SPAN) whole rows
                        const int idx = lane + 32 * h, r = idx / (SPAN / 4), c4 = idx % (SPAN / 4), m = f * 16 + r;
                        if (m < it.local)
                            *reinterpret_cast<float4*>(drow + (int64_t) m * ld_dst + 4 * c4) =
                                *reinterpret_cast<const float4*>(stg + r * SLD + 4 * c4);
                    }
                    __syncwarp();
                }
            }
            if (s + 2 < nsteps) bar_arrive(3 + b, NT);   // EMPTY(b), after the staging reads
        }
    }
#endif
}

struct DevInfo { bool attr[2][2] = {}; int classes = 0; };

template <int WT, int RW, int PF>
void launch(const DBlobs& dw, int n, int yt, int classes, const int32_t* bounds, const void* hq, int64_t hq_rows,
            float* dst, int64_t ld_dst, int64_t dst_row_base, cudaStream_t s) {
    down_kernel<WT, RW, PF><<<NCB * classes, NT, SMEM, s>>>(dw, n, yt, bounds, hq, hq_rows, dst, ld_dst,
                                                        dst_row_base);
    ck(cudaGetLastError(), "down_kernel");
}

// Per device: the opt-in to more than 48 KB of dynamic shared memory, and the class count (SMs / 20).
int prepare(int wt) {
    static std::mutex mu;
    static DevInfo info[16];
    int dev = 0;
    ck(cudaGetDevice(&dev), "cudaGetDevice");
    if (dev < 0 || dev >= 16) return 0;
    std::lock_guard<std::mutex> lk(mu);
    DevInfo& d = info[dev];
    if (d.classes == 0) {
        int sms = 0;
        ck(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev), "SM count");
        d.classes = sms / NCB > 0 ? sms / NCB : 1;
    }
    const int ti = wt == T_Q2_0 ? 0 : 1;
    if (!d.attr[ti][0]) {
        if (wt == T_Q2_0) ck(cudaFuncSetAttribute(down_kernel<T_Q2_0, 2, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM), "smem attribute");
        else ck(cudaFuncSetAttribute(down_kernel<T_IQ4_NL, 2, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM), "smem attribute");
        d.attr[ti][0] = true;
    }
    return d.classes;
}

bool type_ok(int t) { return t == T_Q2_0 || t == T_IQ4_NL; }

}  // namespace

bool built() { return true; }

bool available() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return false;
    int major = 0, minor = 0;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess) return false;
    if (cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess) return false;
    return major == 7 && minor == 0;   // Volta only: other architectures keep their MMQ dispatch
}

bool supported(int d_type, int64_t n_embd, int64_t n_ff) { return type_ok(d_type) && n_embd == N && n_ff == FF; }

static bool down_run(int d_type, const DBlobs& dw, int n_experts, int64_t max_rows, const int32_t* bounds, const void* hq,
                     int64_t hq_rows, float* dst, int64_t ld_dst, int64_t dst_row_base, void* stream) {
    if (!type_ok(d_type) || n_experts <= 0 || n_experts > MAXN || hq_rows <= 0) return false;
    if (ld_dst < N || ld_dst % 4 != 0) return false;
    for (int i = 0; i < n_experts; ++i)
        if (dw.p[i] == nullptr || (uintptr_t) dw.p[i] % 2 != 0) return false;   // 16-bit weight loads
    if (((uintptr_t) hq | (uintptr_t) dst) % 16 != 0) return false;   // 16-byte activation loads / output stores
    if (max_rows <= 0) return true;                                    // nothing to compute
    // Light groups (under 64 rows an expert, e.g. a 2048-token chunk) leave the MMA idle and the producers
    // latency-bound: there fp16tc::down (Q2_0) measured 10% faster and MMQ (IQ4_NL) as fast, so hand them back.
    if (max_rows < MIN_ROWS) return false;
    const int64_t yt = (max_rows + MCAP - 1) / MCAP;
    if (yt > 0xFFFF) return false;
    const int classes = prepare(d_type);
    if (classes <= 0 || (int64_t) n_experts * yt > (int64_t) classes * LMAX) return false;   // the per-block item list
    const cudaStream_t s = (cudaStream_t) stream;
    if (d_type == T_Q2_0) launch<T_Q2_0, 2, 1>(dw, n_experts, (int) yt, classes, bounds, hq, hq_rows, dst, ld_dst, dst_row_base, s);
    else launch<T_IQ4_NL, 2, 1>(dw, n_experts, (int) yt, classes, bounds, hq, hq_rows, dst, ld_dst, dst_row_base, s);
    return true;
}

bool down(int d_type, const void* w, size_t expert_bytes, int n_experts, int64_t max_rows, const int32_t* bounds,
          const void* hq, int64_t hq_rows, float* dst, int64_t ld_dst, int64_t dst_row_base, void* stream) {
    if (!type_ok(d_type) || n_experts <= 0 || n_experts > MAXN || !w) return false;
    if (expert_bytes < (size_t) N * row_bytes(d_type) || expert_bytes % 2 != 0) return false;
    DBlobs dw;
    for (int i = 0; i < n_experts; ++i) dw.p[i] = (const uint8_t*) w + (size_t) i * expert_bytes;
    return down_run(d_type, dw, n_experts, max_rows, bounds, hq, hq_rows, dst, ld_dst, dst_row_base, stream);
}

bool down_blobs(int d_type, const uint8_t* const* down_ptrs, int n_experts, int64_t max_rows, const int32_t* bounds,
                const void* hq, int64_t hq_rows, float* dst, int64_t ld_dst, int64_t dst_row_base, void* stream) {
    if (!type_ok(d_type) || !down_ptrs || n_experts <= 0 || n_experts > MAXN) return false;
    DBlobs dw;
    for (int i = 0; i < n_experts; ++i) dw.p[i] = down_ptrs[i];
    return down_run(d_type, dw, n_experts, max_rows, bounds, hq, hq_rows, dst, ld_dst, dst_row_base, stream);
}

}  // namespace strata::prefill::fp16tc_down
