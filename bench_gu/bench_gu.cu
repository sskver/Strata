// bench_gu: MMQ (dp4a) vs a Volta FP16-HMMA gate/up kernel with fused i-quant decode, engine-scale routing.
// The decoders (load_unit/convert) are moe_fused_iq.cu's; the kernel body is moe_fp16tc.cu's with the Q2_0 weight stage
// replaced.  usage: bench_gu TYPE [tokens] [experts] [reps]   TYPE = iq2_xxs|iq2_xs|iq2_s|iq3_xxs|iq3_s
#include "strata/prefill/moe_mmq.hpp"
#include "ggml.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace mmq = strata::prefill::mmq;
constexpr int N = 2560, FF = 640, K = 10, GROUP = 16, GU_ROWS = 2 * FF;
constexpr int T_IQ2_XXS = GGML_TYPE_IQ2_XXS, T_IQ2_XS = GGML_TYPE_IQ2_XS, T_IQ2_S = GGML_TYPE_IQ2_S,
              T_IQ3_XXS = GGML_TYPE_IQ3_XXS, T_IQ3_S = GGML_TYPE_IQ3_S;

__host__ __device__ constexpr int block_bytes(int t) {
    return t == T_IQ2_XXS ? 66 : t == T_IQ2_XS ? 74 : t == T_IQ2_S ? 82 : t == T_IQ3_XXS ? 98 : 110;
}
__host__ __device__ constexpr bool per16(int t) { return t == T_IQ2_XS || t == T_IQ2_S; }
__host__ __device__ constexpr int grid_bytes(int t) {
    return t == T_IQ2_XXS ? 256 * 8 : t == T_IQ2_XS ? 512 * 8 : t == T_IQ2_S ? 1024 * 8 : t == T_IQ3_XXS ? 256 * 4 : 512 * 4;
}

__device__ __forceinline__ uint32_t ld16(const uint8_t* p) { return *(const uint16_t*) p; }
__device__ __forceinline__ uint32_t ld32(const uint8_t* p) { return ld16(p) | (ld16(p + 2) << 16); }
__device__ __forceinline__ float half_at(uint32_t w) { return __half2float(__ushort_as_half((unsigned short) w)); }
__device__ __forceinline__ uint32_t unpack_ksigns(uint32_t v) {
    v &= 0xFF;
    const uint32_t p = __popc(v) & 1;
    return (v ^ p << 7) * 0x01010101u;
}
__device__ __forceinline__ void signed8(uint32_t gx, uint32_t gy, uint32_t s, uint32_t& qx, uint32_t& qy) {
    const uint32_t m0 = __vcmpne4(s & 0x08040201u, 0), m1 = __vcmpne4(s & 0x80402010u, 0);
    qx = __vsub4(gx ^ m0, m0);
    qy = __vsub4(gy ^ m1, m1);
}
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

// ---- the kernel (moe_fp16tc.cu's geometry)
constexpr int BM = 64, BN = 128, KS = 64, TS = 256, AWMMA = 80, WWMMA = 72, ACTB = 144;
__device__ __forceinline__ __half2 h2_from_u32(uint32_t u) { __half2 h; __builtin_memcpy(&h, &u, 4); return h; }

struct Batch { int n = 0, max_rows = 0; const uint8_t* blob[16] = {}; };

__device__ __forceinline__ void dequant_act(int row0, int local, int k0, int tid, const void* act, int64_t act_rows, __half* a_s) {
    const int m = tid >> 2, part = tid & 3;
    __half* o = a_s + (size_t) m * AWMMA + part * 16;
    if (m >= local) {
#pragma unroll
        for (int j = 0; j < 16; ++j) o[j] = __float2half(0.0f);
    } else {
        const int64_t row = (int64_t) row0 + m;
        const int64_t kb = k0 / 128;
        const int off = (k0 % 128) + part * 16;
        const uint8_t* base = (const uint8_t*) act + ((size_t) kb * (size_t) act_rows + (size_t) row) * ACTB;
        const float d = ((const float*) base)[off >> 5];
        const uint32_t* qs = reinterpret_cast<const uint32_t*>(base + 16 + off);
        const __half2 d2 = __float2half2_rn(d);
        const __half2 c1152 = h2_from_u32(0x64806480u);
        __half2* o2 = reinterpret_cast<__half2*>(o);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint32_t v = qs[k] ^ 0x80808080u;
            o2[2 * k] = __hmul2(__hsub2(h2_from_u32(__byte_perm(v, 0x64646464u, 0x4140)), c1152), d2);
            o2[2 * k + 1] = __hmul2(__hsub2(h2_from_u32(__byte_perm(v, 0x64646464u, 0x4342)), c1152), d2);
        }
    }
}

template <int WT>
__global__ void __launch_bounds__(TS, 1)
gu_kernel(const Batch b, const int32_t* __restrict__ bounds, const void* __restrict__ act, int64_t act_rows,
          float* __restrict__ dst, int64_t ld_dst) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    using namespace nvcuda;
    constexpr int BS = block_bytes(WT), ROWB = (N / 256) * BS;
    const int z = blockIdx.z;
    const int lo = bounds[z], hi = bounds[z + 1];
    const int row0 = lo + (int) blockIdx.y * BM;
    if (row0 >= hi) return;
    const int local = (hi - row0 < BM) ? (hi - row0) : BM;
    const int out_base = (int) blockIdx.x * BN;
    const int tid = threadIdx.x, warp = tid >> 5;
    const int rh = warp >> 2, cq = warp & 3;
    const int m_base = rh * 32, n_base = cq * 32;

    __shared__ __align__(32) union {
        struct { __half a[BM * AWMMA]; __half w[BN * WWMMA]; } ab;
        float c[BM * BN];
    } sm;
    extern __shared__ __align__(16) uint8_t sgrid[];
    {
        const uint32_t* gs = (const uint32_t*) grid_src<WT>();
        for (int i = tid; i < grid_bytes(WT) / 4; i += TS) ((uint32_t*) sgrid)[i] = gs[i];
    }
    const uint8_t* wrow = b.blob[z] + (size_t) (out_base + (tid >> 1)) * ROWB;
    const int dj = tid & 1;
    const __half2 c1152 = h2_from_u32(0x64806480u);

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
        for (int c = 0; c < 2; ++c) wmma::fill_fragment(acc[r][c], 0.0f);

    for (int k0 = 0; k0 < N; k0 += KS) {
        __syncthreads();
        {   // this thread decodes sub-block (k0 % 256) / 32 + dj of its weight row into FP16
            const uint8_t* bp = wrow + (size_t) (k0 >> 8) * BS;
            uint32_t w[5], q[8];
            float s0, s1;
            load_unit<WT>(bp, ((k0 & 255) >> 5) + dj, w);
            convert<WT>(w, sgrid, q, s0, s1);
            const __half2 sa = __float2half2_rn(s0), sb = __float2half2_rn(per16(WT) ? s1 : s0);
            __half2* o = reinterpret_cast<__half2*>(sm.ab.w + (size_t) (tid >> 1) * WWMMA + dj * 32);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const uint32_t v = q[i] ^ 0x80808080u;
                const __half2 sc = i < 4 ? sa : sb;
                o[2 * i] = __hmul2(__hsub2(h2_from_u32(__byte_perm(v, 0x64646464u, 0x4140)), c1152), sc);
                o[2 * i + 1] = __hmul2(__hsub2(h2_from_u32(__byte_perm(v, 0x64646464u, 0x4342)), c1152), sc);
            }
        }
        dequant_act(row0, local, k0, tid, act, act_rows, sm.ab.a);
        __syncthreads();
#pragma unroll
        for (int k16 = 0; k16 < KS / 16; ++k16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> af[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> bf[2];
#pragma unroll
            for (int r = 0; r < 2; ++r)
                wmma::load_matrix_sync(af[r], &sm.ab.a[(size_t) (m_base + 16 * r) * AWMMA + k16 * 16], AWMMA);
#pragma unroll
            for (int c = 0; c < 2; ++c)
                wmma::load_matrix_sync(bf[c], &sm.ab.w[(size_t) (n_base + 16 * c) * WWMMA + k16 * 16], WWMMA);
#pragma unroll
            for (int r = 0; r < 2; ++r)
#pragma unroll
                for (int c = 0; c < 2; ++c) wmma::mma_sync(acc[r][c], af[r], bf[c], acc[r][c]);
        }
    }
    __syncthreads();
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
        for (int c = 0; c < 2; ++c)
            wmma::store_matrix_sync(&sm.c[(size_t) (m_base + 16 * r) * BN + n_base + 16 * c], acc[r][c], BN, wmma::mem_row_major);
    __syncthreads();
    for (int i = tid; i < BM * BN; i += TS) {
        const int grow = row0 + (i >> 7);
        if (grow < hi) dst[(int64_t) grow * ld_dst + out_base + (i & 127)] = sm.c[i];
    }
#endif
}

template <int WT>
void launch(const Batch& b, const int32_t* bounds, const void* act, int64_t act_rows, float* dst, cudaStream_t s) {
    if (b.n <= 0 || b.max_rows <= 0) return;
    const dim3 grid(GU_ROWS / BN, (b.max_rows + BM - 1) / BM, b.n);
    gu_kernel<WT><<<grid, TS, grid_bytes(WT), s>>>(b, bounds, act, act_rows, dst, GU_ROWS);
}
void launch_t(int t, const Batch& b, const int32_t* bounds, const void* act, int64_t act_rows, float* dst, cudaStream_t s) {
    switch (t) {
        case T_IQ2_XXS: launch<T_IQ2_XXS>(b, bounds, act, act_rows, dst, s); break;
        case T_IQ2_XS: launch<T_IQ2_XS>(b, bounds, act, act_rows, dst, s); break;
        case T_IQ2_S: launch<T_IQ2_S>(b, bounds, act, act_rows, dst, s); break;
        case T_IQ3_XXS: launch<T_IQ3_XXS>(b, bounds, act, act_rows, dst, s); break;
        default: launch<T_IQ3_S>(b, bounds, act, act_rows, dst, s); break;
    }
}

void ck(cudaError_t e, const char* w) { if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(2); } }
struct Dev { void* p = nullptr; explicit Dev(size_t n) { ck(cudaMalloc(&p, n), "malloc"); } ~Dev() { cudaFree(p); } template <typename T> T* as() const { return (T*) p; } };

int main(int argc, char** argv) {
    const std::string name = argc > 1 ? argv[1] : "iq3_s";
    const int T = argc > 2 ? std::atoi(argv[2]) : 8192, E = argc > 3 ? std::atoi(argv[3]) : 512, reps = argc > 4 ? std::atoi(argv[4]) : 5;
    const int ty = name == "iq2_xxs" ? T_IQ2_XXS : name == "iq2_xs" ? T_IQ2_XS : name == "iq2_s" ? T_IQ2_S : name == "iq3_xxs" ? T_IQ3_XXS : T_IQ3_S;
    const size_t ROWB = (size_t) (N / 256) * block_bytes(ty), EXB = (size_t) GU_ROWS * ROWB;
    if (!mmq::fits(ty, GU_ROWS)) { std::printf("MMQ does not fit %s\n", name.c_str()); return 77; }
    cudaStream_t s; ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
    std::mt19937 rng(7);
    // routing
    std::vector<int32_t> ids((size_t) T * K);
    std::uniform_int_distribution<int> any(0, E - 1);
    for (int t = 0; t < T; ++t) for (int k = 0; k < K; ++k) {
        int e; bool dup;
        do { e = any(rng); dup = false; for (int j = 0; j < k; ++j) dup |= ids[(size_t) t * K + j] == e; } while (dup);
        ids[(size_t) t * K + k] = e;
    }
    const int64_t rows = (int64_t) T * K;
    std::vector<int32_t> cnt((size_t) E, 0), off((size_t) E + 1, 0), fill((size_t) E, 0), tok((size_t) rows);
    for (int32_t e : ids) ++cnt[(size_t) e];
    for (int e = 0; e < E; ++e) off[(size_t) e + 1] = off[(size_t) e] + cnt[(size_t) e];
    for (int t = 0; t < T; ++t) for (int k = 0; k < K; ++k) { const int e = ids[(size_t) t * K + k]; tok[(size_t) (off[(size_t) e] + fill[(size_t) e]++)] = t; }
    std::vector<int32_t> order; for (int e = 0; e < E; ++e) if (cnt[(size_t) e] > 0) order.push_back(e);
    const size_t n = order.size(), ng = (n + GROUP - 1) / GROUP;
    std::vector<int32_t> bh(n + 1, 0);   // absolute sorted-row bounds in `order`'s sequence
    for (size_t j = 0; j < n; ++j) bh[j + 1] = bh[j] + cnt[(size_t) order[j]];
    std::printf("%s T=%d E=%d: rows %lld, %zu experts, %zu groups, row bytes %zu\n", name.c_str(), T, E, (long long) rows, n, ng, ROWB);
    // weights: random but valid (any bits are valid for these formats; the fp16 scale d is the first 2 bytes of a block)
    Dev wts(n * EXB);
    {
        std::vector<uint8_t> h(EXB);
        std::uniform_int_distribution<int> byte(0, 255);
        std::uniform_real_distribution<float> sc(0.004f, 0.03f);
        for (size_t j = 0; j < n; ++j) {
            for (auto& v : h) v = (uint8_t) byte(rng);
            for (size_t o = 0; o + block_bytes(ty) <= EXB; o += block_bytes(ty)) { const ggml_fp16_t d = ggml_fp32_to_fp16(sc(rng)); std::memcpy(&h[o], &d, 2); }
            ck(cudaMemcpy(wts.as<uint8_t>() + j * EXB, h.data(), EXB, cudaMemcpyHostToDevice), "w");
        }
    }
    // activations: T tokens, gathered by sorted row into one q8_1
    std::vector<float> xh((size_t) T * N); { std::normal_distribution<float> nd(0.f, 1.f); for (float& v : xh) v = nd(rng) * 0.5f; }
    Dev dx(xh.size() * 4), dtok((size_t) rows * 4), ident((size_t) rows * 4), dbounds((n + 1) * 4);
    ck(cudaMemcpy(dx.p, xh.data(), xh.size() * 4, cudaMemcpyHostToDevice), "x");
    ck(cudaMemcpy(dtok.p, tok.data(), (size_t) rows * 4, cudaMemcpyHostToDevice), "tok");
    ck(cudaMemcpy(dbounds.p, bh.data(), (n + 1) * 4, cudaMemcpyHostToDevice), "bounds");
    mmq::iota(ident.as<int32_t>(), rows, s);
    Dev xq(mmq::q8_bytes(rows, N));
    mmq::quantize(dx.as<float>(), dtok.as<int32_t>(), xq.p, ty, N, N, rows, s);
    Dev dm_mmq((size_t) rows * GU_ROWS * 4), dm_tc((size_t) rows * GU_ROWS * 4);
    ck(cudaMemset(dm_mmq.p, 0, (size_t) rows * GU_ROWS * 4), "z"); ck(cudaMemset(dm_tc.p, 0, (size_t) rows * GU_ROWS * 4), "z");
    ck(cudaStreamSynchronize(s), "prep");
    std::vector<int64_t> maxv(ng); std::vector<int> ngxv(ng);
    for (size_t g = 0; g < ng; ++g) { const size_t j0 = g * GROUP, j1 = std::min(n, j0 + GROUP); ngxv[g] = (int) (j1 - j0); int64_t mx = 0; for (size_t j = j0; j < j1; ++j) mx = std::max<int64_t>(mx, cnt[(size_t) order[j]]); maxv[g] = mx; }
    mmq::Context ctx;
    auto run_mmq = [&]() {
        for (size_t g = 0; g < ng; ++g) {
            mmq::Product p; p.w = wts.as<uint8_t>() + g * GROUP * EXB; p.type = ty; p.w_rows = GU_ROWS; p.w_cols = N; p.expert_bytes = EXB;
            p.n = ngxv[g]; p.xq = xq.p; p.bounds = dbounds.as<int32_t>() + g * GROUP; p.ids = ident.as<int32_t>();
            p.total_rows = rows; p.max_rows = maxv[g]; p.dst = dm_mmq.as<float>(); p.ld_dst = GU_ROWS;
            ctx.run(p, s);
        }
    };
    auto run_tc = [&]() {
        for (size_t g = 0; g < ng; ++g) {
            Batch b; b.n = ngxv[g]; b.max_rows = (int) maxv[g];
            for (int i = 0; i < ngxv[g]; ++i) b.blob[i] = wts.as<uint8_t>() + (g * GROUP + (size_t) i) * EXB;
            launch_t(ty, b, dbounds.as<int32_t>() + g * GROUP, xq.p, rows, dm_tc.as<float>(), s);
        }
        ck(cudaGetLastError(), "launch");
    };
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    auto timeit = [&](const char* nm, auto&& fn) {
        fn(); ck(cudaStreamSynchronize(s), "warm");
        float best = 1e30f, sum = 0;
        for (int r = 0; r < reps; ++r) { cudaEventRecord(e0, s); fn(); cudaEventRecord(e1, s); ck(cudaEventSynchronize(e1), "t"); float ms = 0; cudaEventElapsedTime(&ms, e0, e1); best = std::min(best, ms); sum += ms; }
        const double flop = (double) rows * GU_ROWS * N * 2.0;
        std::printf("%-10s best %8.3f ms  mean %8.3f ms  -> %6.2f TOPS\n", nm, best, sum / reps, flop / (best * 1e-3) / 1e12);
        return best;
    };
    const float a = timeit("MMQ gu", run_mmq);
    const float b = timeit("fp16 HMMA", run_tc);
    std::printf("speedup HMMA vs MMQ: %.2fx\n", a / b);
    for (int part = 0; part < 2; ++part) {
        const size_t r0 = part == 0 ? 0 : (size_t) rows / 2, pr = std::min<size_t>(4096, (size_t) rows - r0);
        std::vector<float> ym(pr * GU_ROWS), yt(pr * GU_ROWS);
        ck(cudaMemcpy(ym.data(), dm_mmq.as<float>() + r0 * GU_ROWS, pr * GU_ROWS * 4, cudaMemcpyDeviceToHost), "dl");
        ck(cudaMemcpy(yt.data(), dm_tc.as<float>() + r0 * GU_ROWS, pr * GU_ROWS * 4, cudaMemcpyDeviceToHost), "dl");
        double d2 = 0, r2 = 0; for (size_t i = 0; i < pr * GU_ROWS; ++i) { const double d = (double) ym[i] - yt[i]; d2 += d * d; r2 += (double) ym[i] * ym[i]; }
        std::printf("parity rows %zu..%zu: rel RMS diff %.3e\n", r0, r0 + pr, std::sqrt(d2 / r2));
    }
    return 0;
}
