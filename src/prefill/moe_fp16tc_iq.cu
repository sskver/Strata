// src/prefill/moe_fp16tc_iq.cu - see include/strata/prefill/moe_fp16tc_iq.hpp.
//
// The decoders (load_unit / convert) are moe_fused_iq.cu's: a 32-value sub-block of a GGUF i-quant block becomes 32
// int8 codes (the codebook entries with their signs applied) and a scale per 16 or 32 values, w = scale * code, so the
// FP16 weight is the dequantized llama.cpp value.  The activations are the q8_1 values MMQ reads, rounded the same
// way moe_fp16tc.cu does (the exact signed code times the block's scale, in FP16).  The products are FP16 tensor-core
// (wmma 16x16x16) with FP32 accumulation.
//
// Block structure: one thread block = 192 routed rows (MCAP) of one expert x 128 weight rows (BN).  4 producer warps
// load and decode the weights (2 sub-blocks of a row per thread pair) and convert the activations into one of two
// shared-memory stages; 8 consumer warps (2 row-halves x 4 column-quarters) run the MMA on the other stage.  The two
// halves hand stages over with named barriers (FULL 1/2, EMPTY 3/4), so decode and MMA overlap; with one block per SM
// (about 92 KB of shared memory) this is what lets the tensor cores and the decoders work at the same time.  The
// weights of an expert are decoded once per 192 rows instead of once per 64.
#include "strata/prefill/moe_fp16tc_iq.hpp"

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

namespace strata::prefill::fp16tc_iq {
namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "prefill fp16tc iq: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

constexpr int T_IQ2_XXS = GGML_TYPE_IQ2_XXS, T_IQ2_XS = GGML_TYPE_IQ2_XS, T_IQ2_S = GGML_TYPE_IQ2_S,
              T_IQ3_XXS = GGML_TYPE_IQ3_XXS, T_IQ3_S = GGML_TYPE_IQ3_S;
static_assert(sizeof(block_iq2_xxs) == 66 && sizeof(block_iq2_xs) == 74 && sizeof(block_iq2_s) == 82 &&
              sizeof(block_iq3_xxs) == 98 && sizeof(block_iq3_s) == 110, "the block layouts this file decodes");

constexpr int N = 2560, FF = 640, GU_ROWS = 2 * FF;   // K, n_ff, and the gate+up rows
constexpr int BN = 128, KS = 64, ACTB = 144;            // weight rows per block, K per stage, a q8_1 block's bytes
constexpr int MCAP = 192, NPW = 4;                      // rows per block, producer warps
constexpr int NT = 256 + 32 * NPW, PT = 32 * NPW;       // threads per block, producer threads
constexpr int AP = 72, WP = 72;                         // smem leading dims (halves) of the activation / weight tiles
constexpr int SMEM_STAGES = 2 * (MCAP * AP + BN * WP) * 2;   // bytes: two stages of (192 x 72 + 128 x 72) halves

__host__ __device__ constexpr int block_bytes(int t) {
    return t == T_IQ2_XXS ? 66 : t == T_IQ2_XS ? 74 : t == T_IQ2_S ? 82 : t == T_IQ3_XXS ? 98 : 110;
}
__host__ __device__ constexpr bool per16(int t) { return t == T_IQ2_XS || t == T_IQ2_S; }
__host__ __device__ constexpr int grid_bytes(int t) {
    return t == T_IQ2_XXS ? 256 * 8 : t == T_IQ2_XS ? 512 * 8 : t == T_IQ2_S ? 1024 * 8 : t == T_IQ3_XXS ? 256 * 4 : 512 * 4;
}
// the codebook fits next to the two stages in the 96 KB a block may use (IQ2_S's 8 KB does not: it is read from global)
template <int WT> constexpr bool grid_in_smem() { return grid_bytes(WT) + SMEM_STAGES <= 96 * 1024; }

__device__ __forceinline__ uint32_t ld16(const uint8_t* p) { return *(const uint16_t*) p; }
__device__ __forceinline__ uint32_t ld32(const uint8_t* p) { return ld16(p) | (ld16(p + 2) << 16); }
__device__ __forceinline__ float half_at(uint32_t w) { return __half2float(__ushort_as_half((unsigned short) w)); }
// llama.cpp's sign unpacking: 7 bits of signs, the 8th their parity (bit 7 of v may be anything)
__device__ __forceinline__ uint32_t unpack_ksigns(uint32_t v) {
    v &= 0xFF;
    const uint32_t p = __popc(v) & 1;
    return (v ^ p << 7) * 0x01010101u;
}
// 8 bytes of a codebook entry with the sign byte `s` (broadcast) applied: bits 0-3 to .x, 4-7 to .y
__device__ __forceinline__ void signed8(uint32_t gx, uint32_t gy, uint32_t s, uint32_t& qx, uint32_t& qy) {
    const uint32_t m0 = __vcmpne4(s & 0x08040201u, 0), m1 = __vcmpne4(s & 0x80402010u, 0);
    qx = __vsub4(gx ^ m0, m0);
    qy = __vsub4(gy ^ m1, m1);
}

// Raw bytes of sub-block `ib` (0..7) of the 256-value super-block at `bp`.
template <int T> __device__ __forceinline__ void load_unit(const uint8_t* bp, int ib, uint32_t (&w)[5]) {
    if constexpr (T == T_IQ2_XXS) {
        w[0] = ld32(bp + 2 + 8 * ib); w[1] = ld32(bp + 6 + 8 * ib); w[2] = ld16(bp);
    } else if constexpr (T == T_IQ2_XS) {
        w[0] = ld32(bp + 2 + 8 * ib); w[1] = ld32(bp + 6 + 8 * ib); w[2] = ld16(bp) | ((uint32_t) bp[66 + ib] << 16);
    } else if constexpr (T == T_IQ2_S) {
        w[0] = ld32(bp + 2 + 4 * ib); w[1] = ld32(bp + 34 + 4 * ib);
        w[2] = ld16(bp) | ((uint32_t) bp[66 + ib] << 16) | ((uint32_t) bp[74 + ib] << 24);
    } else if constexpr (T == T_IQ3_XXS) {
        w[0] = ld32(bp + 2 + 8 * ib); w[1] = ld32(bp + 6 + 8 * ib); w[2] = ld32(bp + 66 + 4 * ib); w[3] = ld16(bp);
    } else {   // IQ3_S
        w[0] = ld32(bp + 2 + 8 * ib); w[1] = ld32(bp + 6 + 8 * ib); w[2] = ld32(bp + 74 + 4 * ib);
        w[3] = ld16(bp) | ((uint32_t) bp[66 + ib] << 16) | ((uint32_t) ((bp[106 + ib / 2] >> (4 * (ib & 1))) & 15) << 24);
    }
}
// The sub-block as 32 int8 (q[0..7], natural order) and its scales (s0: values 0-15, s1: 16-31).
template <int T>
__device__ __forceinline__ void convert(const uint32_t (&w)[5], const uint8_t* grid, uint32_t (&q)[8], float& s0, float& s1) {
    if constexpr (T == T_IQ2_XXS) {
        const uint2* g = (const uint2*) grid;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint2 e = g[(w[0] >> (8 * l)) & 255];
            signed8(e.x, e.y, unpack_ksigns(w[1] >> (7 * l)), q[2 * l], q[2 * l + 1]);
        }
        s0 = s1 = half_at(w[2]) * (float) ((w[1] >> 27) | 1) * 0.125f;
    } else if constexpr (T == T_IQ2_XS) {
        const uint2* g = (const uint2*) grid;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t c = (w[l >> 1] >> (16 * (l & 1))) & 0xFFFF;
            const uint2 e = g[c & 511];
            signed8(e.x, e.y, unpack_ksigns(c >> 9), q[2 * l], q[2 * l + 1]);
        }
        const float d = half_at(w[2]);
        const uint32_t sc = w[2] >> 16;
        s0 = d * (float) (2 * (sc & 15) + 1) * 0.125f;
        s1 = d * (float) (2 * ((sc >> 4) & 15) + 1) * 0.125f;
    } else if constexpr (T == T_IQ2_S) {
        const uint2* g = (const uint2*) grid;
        const uint32_t qh = (w[2] >> 16) & 255;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint2 e = g[((w[0] >> (8 * l)) & 255) | ((qh << (8 - 2 * l)) & 0x300)];
            signed8(e.x, e.y, ((w[1] >> (8 * l)) & 255) * 0x01010101u, q[2 * l], q[2 * l + 1]);
        }
        const float d = half_at(w[2]);
        const uint32_t sc = w[2] >> 24;
        s0 = d * (float) (2 * (sc & 15) + 1) * 0.125f;
        s1 = d * (float) (2 * (sc >> 4) + 1) * 0.125f;
    } else if constexpr (T == T_IQ3_XXS) {
        const uint32_t* g = (const uint32_t*) grid;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t i0 = (w[l >> 1] >> (16 * (l & 1))) & 255, i1 = (w[l >> 1] >> (16 * (l & 1) + 8)) & 255;
            signed8(g[i0], g[i1], unpack_ksigns(w[2] >> (7 * l)), q[2 * l], q[2 * l + 1]);
        }
        s0 = s1 = half_at(w[3]) * (float) (2 * (w[2] >> 28) + 1) * 0.25f;
    } else {   // IQ3_S
        const uint32_t* g = (const uint32_t*) grid;
        const uint32_t qh = (w[3] >> 16) & 255;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t i0 = (w[l >> 1] >> (16 * (l & 1))) & 255, i1 = (w[l >> 1] >> (16 * (l & 1) + 8)) & 255;
            signed8(g[i0 | ((qh << (8 - 2 * l)) & 256)], g[i1 | ((qh << (7 - 2 * l)) & 256)],
                    ((w[2] >> (8 * l)) & 255) * 0x01010101u, q[2 * l], q[2 * l + 1]);
        }
        s0 = s1 = half_at(w[3]) * (float) (1 + 2 * (w[3] >> 24));
    }
}
template <int T> __device__ __forceinline__ const void* grid_src() {
    if constexpr (T == T_IQ2_XXS) return iq2xxs_grid;
    else if constexpr (T == T_IQ2_XS) return iq2xs_grid;
    else if constexpr (T == T_IQ2_S) return iq2s_grid;
    else if constexpr (T == T_IQ3_XXS) return iq3xxs_grid;
    else return iq3s_grid;
}

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

struct ARaw { uint4 q; float d; };

// Expert z = blockIdx.z owns the activation rows [bounds[z], bounds[z+1]); blockIdx.y takes 192 of them; blockIdx.x 128
// weight rows.  The whole block leaves together when its row range is empty.
template <int WT, int GS>
__global__ void __launch_bounds__(NT, 1)
gu_kernel(const uint8_t* __restrict__ wbase, size_t expert_bytes, const int32_t* __restrict__ bounds,
          const void* __restrict__ act, int64_t act_rows, float* __restrict__ dst, int64_t ld_dst) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    using namespace nvcuda;
    constexpr int BS = block_bytes(WT), ROWB = (N / 256) * BS;
    constexpr int MF = MCAP / 32;                          // m fragments per consumer warp (2 warp rows)
    constexpr int GB = GS ? grid_bytes(WT) : 0;
    constexpr int ASZ = MCAP * AP, WSZ = BN * WP, SSZ = ASZ + WSZ;
    constexpr int NK = N / KS;
    extern __shared__ __align__(16) uint8_t smem[];
    const int z = blockIdx.z;
    const int lo = bounds[z], hi = bounds[z + 1];
    const int row0 = lo + (int) blockIdx.y * MCAP;
    if (row0 >= hi) return;
    const int local = (hi - row0 < MCAP) ? (hi - row0) : MCAP;
    const uint8_t* const wb = wbase + (size_t) z * expert_bytes;
    const int out_base = (int) blockIdx.x * BN;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const uint8_t* grid;
    if constexpr (GS) {
        const uint32_t* gs = (const uint32_t*) grid_src<WT>();
        for (int i = tid; i < grid_bytes(WT) / 4; i += NT) ((uint32_t*) smem)[i] = gs[i];
        grid = smem;
        __syncthreads();
    } else {
        grid = (const uint8_t*) grid_src<WT>();
    }
    __half* bufs = (__half*) (smem + GB);

    if (warp >= 8) {   // ---------------- producers: load + decode weights, convert activations into the stage
        const int pt = tid - 256;
        constexpr int WU = (256 + PT - 1) / PT;       // weight units (32 values) per producer thread
        constexpr int AT = (MCAP * 4 + PT - 1) / PT;  // activation tasks (16 values) per producer thread
        uint32_t wr[WU][5];
        ARaw ar[AT];
        auto load = [&](int k0) {   // raw bytes into registers, one stage ahead
#pragma unroll
            for (int u = 0; u < WU; ++u) {
                const int unit = pt + u * PT, row = unit >> 1, dj = unit & 1;
                if (unit >= 256) continue;
                load_unit<WT>(wb + (size_t) (out_base + row) * ROWB + (size_t) (k0 >> 8) * BS, ((k0 & 255) >> 5) + dj, wr[u]);
            }
            const int kb = k0 >> 7;
#pragma unroll
            for (int j = 0; j < AT; ++j) {
                const int task = pt + j * PT, m = task >> 2, off = (k0 & 127) + (task & 3) * 16;
                if (m < local) {
                    const uint8_t* base = (const uint8_t*) act + ((size_t) kb * (size_t) act_rows + (size_t) (row0 + m)) * ACTB;
                    ar[j].d = ((const float*) base)[off >> 5];
                    ar[j].q = *reinterpret_cast<const uint4*>(base + 16 + off);
                }
            }
        };
        auto store = [&](int s) {   // decode into stage s
            __half* a_s = bufs + s * SSZ;
            __half* w_s = a_s + ASZ;
#pragma unroll
            for (int u = 0; u < WU; ++u) {
                const int unit = pt + u * PT, row = unit >> 1, dj = unit & 1;
                if (unit >= 256) continue;
                uint32_t q[8];
                float s0, s1;
                convert<WT>(wr[u], grid, q, s0, s1);
                const __half2 sa = __float2half2_rn(s0), sb = __float2half2_rn(per16(WT) ? s1 : s0);
                uint4* o = reinterpret_cast<uint4*>(w_s + (size_t) row * WP + dj * 32);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const uint32_t v0 = q[2 * i] ^ 0x80808080u, v1 = q[2 * i + 1] ^ 0x80808080u;
                    const __half2 sc = i < 2 ? sa : sb;
                    o[i] = make_uint4(cvt2(v0, 0x4140, sc), cvt2(v0, 0x4342, sc), cvt2(v1, 0x4140, sc), cvt2(v1, 0x4342, sc));
                }
            }
#pragma unroll
            for (int j = 0; j < AT; ++j) {
                const int task = pt + j * PT, m = task >> 2;
                if (m < local) {
                    uint4* o = reinterpret_cast<uint4*>(a_s + (size_t) m * AP + (task & 3) * 16);
                    const __half2 d2 = __float2half2_rn(ar[j].d);
                    const uint32_t v0 = ar[j].q.x ^ 0x80808080u, v1 = ar[j].q.y ^ 0x80808080u, v2 = ar[j].q.z ^ 0x80808080u,
                                   v3 = ar[j].q.w ^ 0x80808080u;
                    o[0] = make_uint4(cvt2(v0, 0x4140, d2), cvt2(v0, 0x4342, d2), cvt2(v1, 0x4140, d2), cvt2(v1, 0x4342, d2));
                    o[1] = make_uint4(cvt2(v2, 0x4140, d2), cvt2(v2, 0x4342, d2), cvt2(v3, 0x4140, d2), cvt2(v3, 0x4342, d2));
                }
            }
        };
        load(0);
        for (int k = 0; k < NK; ++k) {
            const int b = k & 1;
            if (k >= 2) bar_sync(3 + b, NT);   // EMPTY(b): the consumers finished the MMA that read stage b
            store(b);
            bar_arrive(1 + b, NT);             // FULL(b)
            if (k + 1 < NK) load((k + 1) * KS);
        }
        return;
    }
    // ---------------- consumers: 8 warps = 2 row-halves x 4 column-quarters; a row-half owns every second 16-row fragment
    const int rh = warp >> 2, cq = warp & 3, n_base = cq * 32;
    const int nfr = (local + 15) >> 4;                       // 16-row fragments holding rows
    const int nact = min(MF, (nfr - rh + 1) >> 1);           // this warp's active fragments
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[MF][2];
#pragma unroll
    for (int i = 0; i < MF; ++i)
#pragma unroll
        for (int c = 0; c < 2; ++c) wmma::fill_fragment(acc[i][c], 0.0f);
    auto mma_n = [&](const __half* a_s, const __half* w_s, auto na_c) {
        constexpr int NA = decltype(na_c)::value;
#pragma unroll
        for (int k16 = 0; k16 < KS / 16; ++k16) {
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> bf[2];
#pragma unroll
            for (int c = 0; c < 2; ++c) wmma::load_matrix_sync(bf[c], w_s + (size_t) (n_base + 16 * c) * WP + k16 * 16, WP);
#pragma unroll
            for (int i = 0; i < NA; ++i) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> af;
                wmma::load_matrix_sync(af, a_s + (size_t) ((rh + 2 * i) * 16) * AP + k16 * 16, AP);
                wmma::mma_sync(acc[i][0], af, bf[0], acc[i][0]);
                wmma::mma_sync(acc[i][1], af, bf[1], acc[i][1]);
            }
        }
    };
    for (int k = 0; k < NK; ++k) {
        const int b = k & 1;
        bar_sync(1 + b, NT);   // FULL(b)
        const __half* a_s = bufs + b * SSZ;
        const __half* w_s = a_s + ASZ;
        switch (nact) {   // no per-fragment guards in the MMA body: the count is a compile-time constant per case
            case 6: mma_n(a_s, w_s, std::integral_constant<int, 6>{}); break;
            case 5: mma_n(a_s, w_s, std::integral_constant<int, 5>{}); break;
            case 4: mma_n(a_s, w_s, std::integral_constant<int, 4>{}); break;
            case 3: mma_n(a_s, w_s, std::integral_constant<int, 3>{}); break;
            case 2: mma_n(a_s, w_s, std::integral_constant<int, 2>{}); break;
            case 1: mma_n(a_s, w_s, std::integral_constant<int, 1>{}); break;
            default: break;
        }
        if (k + 2 < NK) bar_arrive(3 + b, NT);   // EMPTY(b)
    }
    bar_sync(5, 256);   // all consumers are done reading the stage buffers: they become the epilogue staging
    float* stg = reinterpret_cast<float*>(bufs) + warp * 16 * 20;
#pragma unroll
    for (int i = 0; i < MF; ++i) {
        const int f = rh + 2 * i;
        if (f < nfr) {
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                wmma::store_matrix_sync(stg, acc[i][c], 20, wmma::mem_row_major);
                __syncwarp();
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int idx = lane + 32 * h, r = idx >> 2, c4 = idx & 3, m = f * 16 + r;
                    if (m < local)
                        *reinterpret_cast<float4*>(dst + (int64_t) (row0 + m) * ld_dst + out_base + n_base + 16 * c + 4 * c4) =
                            *reinterpret_cast<const float4*>(stg + r * 20 + 4 * c4);
                }
                __syncwarp();
            }
        }
    }
#endif
}

template <int WT>
void launch(const void* w, size_t expert_bytes, int n, int64_t max_rows, const int32_t* bounds, const void* xq,
            int64_t xq_rows, float* dst, cudaStream_t s) {
    constexpr int GS = grid_in_smem<WT>();
    constexpr int smem = (GS ? grid_bytes(WT) : 0) + SMEM_STAGES;
    {   // the per-device opt-in to more than 48 KB of dynamic shared memory
        static std::mutex mu;
        static bool done[16] = {};
        int dev = 0;
        ck(cudaGetDevice(&dev), "cudaGetDevice");
        std::lock_guard<std::mutex> lk(mu);
        if (dev >= 0 && dev < 16 && !done[dev]) {
            ck(cudaFuncSetAttribute(gu_kernel<WT, GS>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem), "smem attribute");
            done[dev] = true;
        }
    }
    const dim3 grid(GU_ROWS / BN, (unsigned) ((max_rows + MCAP - 1) / MCAP), (unsigned) n);
    gu_kernel<WT, GS><<<grid, NT, smem, s>>>((const uint8_t*) w, expert_bytes, bounds, xq, xq_rows, dst, GU_ROWS);
    ck(cudaGetLastError(), "gu_kernel");
}

bool type_ok(int t) {
    return t == T_IQ2_XXS || t == T_IQ2_XS || t == T_IQ2_S || t == T_IQ3_XXS || t == T_IQ3_S;
}

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

bool supported(int gu_type, int64_t n_embd, int64_t n_ff) { return type_ok(gu_type) && n_embd == N && n_ff == FF; }

bool gu(int gu_type, const void* w, size_t expert_bytes, int n_experts, int64_t max_rows, const int32_t* bounds,
        const void* xq, int64_t xq_rows, float* dst, int64_t ld_dst, void* stream) {
    if (!type_ok(gu_type) || ld_dst != GU_ROWS || n_experts <= 0 || n_experts > 65535 || xq_rows <= 0) return false;
    if (((uintptr_t) xq | (uintptr_t) dst) % 16 != 0) return false;   // the kernel's 16-byte activation loads / output stores
    if (max_rows <= 0) return true;                                    // nothing to compute
    if (max_rows > (int64_t) 65535 * MCAP) return false;               // grid.y limit
    const cudaStream_t s = (cudaStream_t) stream;
    switch (gu_type) {
        case T_IQ2_XXS: launch<T_IQ2_XXS>(w, expert_bytes, n_experts, max_rows, bounds, xq, xq_rows, dst, s); break;
        case T_IQ2_XS: launch<T_IQ2_XS>(w, expert_bytes, n_experts, max_rows, bounds, xq, xq_rows, dst, s); break;
        case T_IQ2_S: launch<T_IQ2_S>(w, expert_bytes, n_experts, max_rows, bounds, xq, xq_rows, dst, s); break;
        case T_IQ3_XXS: launch<T_IQ3_XXS>(w, expert_bytes, n_experts, max_rows, bounds, xq, xq_rows, dst, s); break;
        default: launch<T_IQ3_S>(w, expert_bytes, n_experts, max_rows, bounds, xq, xq_rows, dst, s); break;
    }
    return true;
}

}  // namespace strata::prefill::fp16tc_iq
