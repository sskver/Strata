// bench_gu_v2: experiments on the Volta FP16-HMMA gate/up kernel (copy of bench_gu.cu; the prototype is variant 0).
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

#include <type_traits>
#include <cublas_v2.h>
// ================= v2 experiments =================
// FL ablation bits: 1 = no weight load/decode (constant stored), 2 = no act load/convert (constant stored),
// 4 = no MMA, 8 = no output store (kept alive behind a runtime-false branch)
constexpr int FL_NODEC = 1, FL_NOACT = 2, FL_NOMMA = 4, FL_NOEPI = 8, FL_HY = 16;   // HY: consumers convert act (kernel C)
constexpr int BM = 64, BN = 128, KS = 64, TS = 256, ACTB = 144;
__device__ __forceinline__ __half2 h2_from_u32(uint32_t u) { __half2 h; __builtin_memcpy(&h, &u, 4); return h; }
__device__ __forceinline__ uint32_t u32_from_h2(__half2 h) { uint32_t u; __builtin_memcpy(&u, &h, 4); return u; }
__device__ __forceinline__ uint32_t cvt2(uint32_t v, int sel, __half2 sc) {   // two biased bytes -> two halves * sc
    const __half2 c1152 = h2_from_u32(0x64806480u);
    return u32_from_h2(__hmul2(__hsub2(h2_from_u32(__byte_perm(v, 0x64646464u, sel)), c1152), sc));
}

struct Batch { int n = 0, max_rows = 0; const uint8_t* blob[16] = {}; };

// ---------------- kernel A: the prototype, with ablation flags and smem leading dims ----------------
template <int AP>
__device__ __forceinline__ void dequant_act(int row0, int local, int k0, int tid, const void* act, int64_t act_rows, __half* a_s, bool noact) {
    const int m = tid >> 2, part = tid & 3;
    __half* o = a_s + (size_t) m * AP + part * 16;
    if (m >= local || noact) {
#pragma unroll
        for (int j = 0; j < 16; ++j) o[j] = __float2half(noact ? 0.01f : 0.0f);
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

template <int WT, int FL, int AP, int WP>
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
        struct { __half a[BM * AP]; __half w[BN * WP]; } ab;
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
        {
            uint32_t q[8];
            float s0, s1;
            if constexpr (FL & FL_NODEC) {
#pragma unroll
                for (int i = 0; i < 8; ++i) q[i] = 0x01010101u * (uint32_t) (i + 1);
                s0 = s1 = 0.01f;
            } else {
                const uint8_t* bp = wrow + (size_t) (k0 >> 8) * BS;
                uint32_t w[5];
                load_unit<WT>(bp, ((k0 & 255) >> 5) + dj, w);
                convert<WT>(w, sgrid, q, s0, s1);
            }
            const __half2 sa = __float2half2_rn(s0), sb = __float2half2_rn(per16(WT) ? s1 : s0);
            __half2* o = reinterpret_cast<__half2*>(sm.ab.w + (size_t) (tid >> 1) * WP + dj * 32);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const uint32_t v = q[i] ^ 0x80808080u;
                const __half2 sc = i < 4 ? sa : sb;
                o[2 * i] = __hmul2(__hsub2(h2_from_u32(__byte_perm(v, 0x64646464u, 0x4140)), c1152), sc);
                o[2 * i + 1] = __hmul2(__hsub2(h2_from_u32(__byte_perm(v, 0x64646464u, 0x4342)), c1152), sc);
            }
        }
        dequant_act<AP>(row0, local, k0, tid, act, act_rows, sm.ab.a, (FL & FL_NOACT) != 0);
        __syncthreads();
        if constexpr (!(FL & FL_NOMMA)) {
#pragma unroll
            for (int k16 = 0; k16 < KS / 16; ++k16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> af[2];
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> bf[2];
#pragma unroll
                for (int r = 0; r < 2; ++r)
                    wmma::load_matrix_sync(af[r], &sm.ab.a[(size_t) (m_base + 16 * r) * AP + k16 * 16], AP);
#pragma unroll
                for (int c = 0; c < 2; ++c)
                    wmma::load_matrix_sync(bf[c], &sm.ab.w[(size_t) (n_base + 16 * c) * WP + k16 * 16], WP);
#pragma unroll
                for (int r = 0; r < 2; ++r)
#pragma unroll
                    for (int c = 0; c < 2; ++c) wmma::mma_sync(acc[r][c], af[r], bf[c], acc[r][c]);
            }
        } else {
            // keep the smem tiles alive: one cheap read per thread
            if (sm.ab.a[tid] == __float2half(-77.f) && sm.ab.w[tid] == __float2half(-77.f)) acc[0][0].x[0] += 1.f;
        }
    }
    __syncthreads();
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
        for (int c = 0; c < 2; ++c)
            wmma::store_matrix_sync(&sm.c[(size_t) (m_base + 16 * r) * BN + n_base + 16 * c], acc[r][c], BN, wmma::mem_row_major);
    __syncthreads();
    if ((FL & FL_NOEPI) && ld_dst > 0) {
        if (sm.c[tid] == -77.f) dst[tid] = 1.f;
        return;
    }
    for (int i = tid; i < BM * BN; i += TS) {
        const int grow = row0 + (i >> 7);
        if (grow < hi) dst[(int64_t) grow * ld_dst + out_base + (i & 127)] = sm.c[i];
    }
#endif
}

// ---------------- kernel B: whole-expert tile (up to MCAP rows), weights decoded once per expert ----------------
struct Item { int e, row0, n, pad; };

template <int WT> struct WRaw { uint32_t w[5]; };
struct ARaw { uint4 q; float d; };

// MCAP: max rows per block (multiple of 32); ST: smem stages (1 or 2); GS: grid in smem; AP/WP: smem leading dims
template <int WT, int MCAP, int ST, int GS, int AP, int WP, int FL, int DISP = 0>
__global__ void __launch_bounds__(TS, 1)
gu2_kernel(const uint8_t* const* __restrict__ blobs, const Item* __restrict__ items, const void* __restrict__ act,
           int64_t act_rows, float* __restrict__ dst, int64_t ld_dst) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    using namespace nvcuda;
    constexpr int BS = block_bytes(WT), ROWB = (N / 256) * BS;
    constexpr int MF = MCAP / 32;                       // m fragments per warp (2 warp rows)
    constexpr int AT = (MCAP * 4 + TS - 1) / TS;        // act tasks (16 values) per thread
    constexpr int GB = GS ? grid_bytes(WT) : 0;
    constexpr int ASZ = MCAP * AP, WSZ = BN * WP, SSZ = ASZ + WSZ;   // halves
    extern __shared__ __align__(16) uint8_t smem[];
    const Item it = items[blockIdx.y];
    const int local = it.n, row0 = it.row0;
    const int out_base = (int) blockIdx.x * BN;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int rh = warp >> 2, cq = warp & 3, n_base = cq * 32;
    const uint8_t* grid;
    if constexpr (GS) {
        const uint32_t* gs = (const uint32_t*) grid_src<WT>();
        for (int i = tid; i < grid_bytes(WT) / 4; i += TS) ((uint32_t*) smem)[i] = gs[i];
        grid = smem;
    } else {
        grid = (const uint8_t*) grid_src<WT>();
    }
    __half* bufs = (__half*) (smem + GB);
    const uint8_t* wrow = blobs[it.e] + (size_t) (out_base + (tid >> 1)) * ROWB;
    const int dj = tid & 1;
    const int nfr = (local + 15) >> 4;

    WRaw<WT> wr;
    ARaw ar[AT];
    auto load = [&](int k0) {
        if constexpr (!(FL & FL_NODEC)) load_unit<WT>(wrow + (size_t) (k0 >> 8) * BS, ((k0 & 255) >> 5) + dj, wr.w);
        if constexpr (!(FL & FL_NOACT)) {
            const int kb = k0 >> 7;
#pragma unroll
            for (int j = 0; j < AT; ++j) {
                const int task = tid + j * TS, m = task >> 2, off = (k0 & 127) + (task & 3) * 16;
                if (m < local) {
                    const uint8_t* base = (const uint8_t*) act + ((size_t) kb * (size_t) act_rows + (size_t) (row0 + m)) * ACTB;
                    ar[j].d = ((const float*) base)[off >> 5];
                    ar[j].q = *reinterpret_cast<const uint4*>(base + 16 + off);
                }
            }
        }
    };
    auto store = [&](int s) {
        __half* a_s = bufs + s * SSZ;
        __half* w_s = a_s + ASZ;
        {
            uint32_t q[8];
            float s0, s1;
            if constexpr (FL & FL_NODEC) {
#pragma unroll
                for (int i = 0; i < 8; ++i) q[i] = 0x01010101u * (uint32_t) (i + 1);
                s0 = s1 = 0.01f;
            } else {
                convert<WT>(wr.w, grid, q, s0, s1);
            }
            const __half2 sa = __float2half2_rn(s0), sb = __float2half2_rn(per16(WT) ? s1 : s0);
            uint4* o = reinterpret_cast<uint4*>(w_s + (size_t) (tid >> 1) * WP + dj * 32);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const uint32_t v0 = q[2 * i] ^ 0x80808080u, v1 = q[2 * i + 1] ^ 0x80808080u;
                const __half2 sc = i < 2 ? sa : sb;
                o[i] = make_uint4(cvt2(v0, 0x4140, sc), cvt2(v0, 0x4342, sc), cvt2(v1, 0x4140, sc), cvt2(v1, 0x4342, sc));
            }
        }
#pragma unroll
        for (int j = 0; j < AT; ++j) {
            const int task = tid + j * TS, m = task >> 2;
            if (m < local) {
                uint4* o = reinterpret_cast<uint4*>(a_s + (size_t) m * AP + (task & 3) * 16);
                if constexpr (FL & FL_NOACT) {
                    o[0] = o[1] = make_uint4(0x211f211fu, 0x211f211fu, 0x211f211fu, 0x211f211fu);
                } else {
                    const __half2 d2 = __float2half2_rn(ar[j].d);
                    const uint32_t v0 = ar[j].q.x ^ 0x80808080u, v1 = ar[j].q.y ^ 0x80808080u, v2 = ar[j].q.z ^ 0x80808080u,
                                   v3 = ar[j].q.w ^ 0x80808080u;
                    o[0] = make_uint4(cvt2(v0, 0x4140, d2), cvt2(v0, 0x4342, d2), cvt2(v1, 0x4140, d2), cvt2(v1, 0x4342, d2));
                    o[1] = make_uint4(cvt2(v2, 0x4140, d2), cvt2(v2, 0x4342, d2), cvt2(v3, 0x4140, d2), cvt2(v3, 0x4342, d2));
                }
            }
        }
    };

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[MF][2];
#pragma unroll
    for (int i = 0; i < MF; ++i)
#pragma unroll
        for (int c = 0; c < 2; ++c) wmma::fill_fragment(acc[i][c], 0.0f);

    const int nact = min(MF, (nfr - rh + 1) >> 1);   // this warp's active m fragments
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
    auto mma = [&](int s) {
        const __half* a_s = bufs + s * SSZ;
        const __half* w_s = a_s + ASZ;
        if constexpr (DISP && !(FL & FL_NOMMA)) {
            static_assert(MF <= 8, "MF");
            switch (nact) {
                case 8: if constexpr (MF >= 8) mma_n(a_s, w_s, std::integral_constant<int, (MF >= 8 ? 8 : 1)>{}); break;
                case 7: if constexpr (MF >= 7) mma_n(a_s, w_s, std::integral_constant<int, (MF >= 7 ? 7 : 1)>{}); break;
                case 6: if constexpr (MF >= 6) mma_n(a_s, w_s, std::integral_constant<int, (MF >= 6 ? 6 : 1)>{}); break;
                case 5: if constexpr (MF >= 5) mma_n(a_s, w_s, std::integral_constant<int, (MF >= 5 ? 5 : 1)>{}); break;
                case 4: if constexpr (MF >= 4) mma_n(a_s, w_s, std::integral_constant<int, (MF >= 4 ? 4 : 1)>{}); break;
                case 3: if constexpr (MF >= 3) mma_n(a_s, w_s, std::integral_constant<int, (MF >= 3 ? 3 : 1)>{}); break;
                case 2: mma_n(a_s, w_s, std::integral_constant<int, 2>{}); break;
                case 1: mma_n(a_s, w_s, std::integral_constant<int, 1>{}); break;
                default: break;
            }
        } else if constexpr (!(FL & FL_NOMMA)) {
#pragma unroll
            for (int k16 = 0; k16 < KS / 16; ++k16) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> bf[2];
#pragma unroll
                for (int c = 0; c < 2; ++c) wmma::load_matrix_sync(bf[c], w_s + (size_t) (n_base + 16 * c) * WP + k16 * 16, WP);
#pragma unroll
                for (int i = 0; i < MF; ++i) {
                    const int f = rh + 2 * i;
                    if (f < nfr) {
                        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> af;
                        wmma::load_matrix_sync(af, a_s + (size_t) (f * 16) * AP + k16 * 16, AP);
                        wmma::mma_sync(acc[i][0], af, bf[0], acc[i][0]);
                        wmma::mma_sync(acc[i][1], af, bf[1], acc[i][1]);
                    }
                }
            }
        } else {
            if (a_s[tid] == __float2half(-77.f) && w_s[tid] == __float2half(-77.f)) acc[0][0].x[0] += 1.f;
        }
    };

    if constexpr (GS) __syncthreads();
    constexpr int NK = N / KS;
    if constexpr (ST == 2) {
        load(0);
        store(0);
        __syncthreads();
        for (int k = 0; k < NK; ++k) {
            if (k + 1 < NK) load((k + 1) * KS);
            mma(k & 1);
            if (k + 1 < NK) store((k + 1) & 1);
            __syncthreads();
        }
    } else {
        for (int k = 0; k < NK; ++k) {
            load(k * KS);
            store(0);
            __syncthreads();
            mma(0);
            __syncthreads();
        }
    }
    // epilogue: per-warp 16x16 staging in the (now free) stage buffers
    float* stg = reinterpret_cast<float*>(bufs) + warp * 16 * 20;
    if ((FL & FL_NOEPI) && ld_dst > 0) {
        float sum = 0.f;
#pragma unroll
        for (int i = 0; i < MF; ++i)
#pragma unroll
            for (int c = 0; c < 2; ++c)
#pragma unroll
                for (int t = 0; t < acc[i][c].num_elements; ++t) sum += acc[i][c].x[t];
        if (sum == -77.f) dst[tid] = 1.f;
        return;
    }
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

// ---------------- kernel C: warp-specialized. 8 consumer warps (MMA, B's layout) + NPW producer warps (load+decode),
// 2 smem stages, named barriers FULL(1,2) / EMPTY(3,4), consumer-only barrier 5 ----------------
__device__ __forceinline__ void bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void bar_arrive(int id, int n) { asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(n) : "memory"); }

template <int WT, int MCAP, int NPW, int GS, int FL>
__global__ void __launch_bounds__(256 + 32 * NPW, 1)
gu3_kernel(const uint8_t* const* __restrict__ blobs, const Item* __restrict__ items, const void* __restrict__ act,
           int64_t act_rows, float* __restrict__ dst, int64_t ld_dst) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    using namespace nvcuda;
    constexpr int AP = 72, WP = 72;
    constexpr int NT = 256 + 32 * NPW, PT = 32 * NPW;
    constexpr int BS = block_bytes(WT), ROWB = (N / 256) * BS;
    constexpr int MF = MCAP / 32;
    constexpr int GB = GS ? grid_bytes(WT) : 0;
    constexpr int ASZ = MCAP * AP, WSZ = BN * WP, SSZ = ASZ + WSZ;
    constexpr int NK = N / KS;
    extern __shared__ __align__(16) uint8_t smem[];
    const Item it = items[blockIdx.y];
    const int local = it.n, row0 = it.row0;
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

    if (warp >= 8) {   // ---------------- producers
        const int pt = tid - 256;
        constexpr int WU = (256 + PT - 1) / PT;                // weight units (32 values) per producer thread
        constexpr int AT = (MCAP * 4 + PT - 1) / PT; // act tasks (16 values) per producer thread
        uint32_t wr[WU][5];
        ARaw ar[AT];
        auto load = [&](int k0) {
            if constexpr (!(FL & FL_NODEC)) {
#pragma unroll
                for (int u = 0; u < WU; ++u) {
                    const int unit = pt + u * PT, row = unit >> 1, dj = unit & 1; if (unit >= 256) continue;
                    load_unit<WT>(blobs[it.e] + (size_t) (out_base + row) * ROWB + (size_t) (k0 >> 8) * BS, ((k0 & 255) >> 5) + dj, wr[u]);
                }
            }
            if constexpr (!(FL & (FL_NOACT | FL_HY))) {
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
            }
        };
        auto store = [&](int s) {
            __half* a_s = bufs + s * SSZ;
            __half* w_s = a_s + ASZ;
#pragma unroll
            for (int u = 0; u < WU; ++u) {
                const int unit = pt + u * PT, row = unit >> 1, dj = unit & 1; if (unit >= 256) continue;
                uint32_t q[8];
                float s0, s1;
                if constexpr (FL & FL_NODEC) {
#pragma unroll
                    for (int i = 0; i < 8; ++i) q[i] = 0x01010101u * (uint32_t) (i + 1);
                    s0 = s1 = 0.01f;
                } else {
                    convert<WT>(wr[u], grid, q, s0, s1);
                }
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
            for (int j = 0; j < ((FL & FL_HY) ? 0 : AT); ++j) {
                const int task = pt + j * PT, m = task >> 2;
                if (m < local) {
                    uint4* o = reinterpret_cast<uint4*>(a_s + (size_t) m * AP + (task & 3) * 16);
                    if constexpr (FL & FL_NOACT) {
                        o[0] = o[1] = make_uint4(0x211f211fu, 0x211f211fu, 0x211f211fu, 0x211f211fu);
                    } else {
                        const __half2 d2 = __float2half2_rn(ar[j].d);
                        const uint32_t v0 = ar[j].q.x ^ 0x80808080u, v1 = ar[j].q.y ^ 0x80808080u, v2 = ar[j].q.z ^ 0x80808080u,
                                       v3 = ar[j].q.w ^ 0x80808080u;
                        o[0] = make_uint4(cvt2(v0, 0x4140, d2), cvt2(v0, 0x4342, d2), cvt2(v1, 0x4140, d2), cvt2(v1, 0x4342, d2));
                        o[1] = make_uint4(cvt2(v2, 0x4140, d2), cvt2(v2, 0x4342, d2), cvt2(v3, 0x4140, d2), cvt2(v3, 0x4342, d2));
                    }
                }
            }
        };
        load(0);
        for (int k = 0; k < NK; ++k) {
            const int b = k & 1;
            if (k >= 2) bar_sync(3 + b, NT);
            store(b);
            bar_arrive(1 + b, NT);
            if (k + 1 < NK) load((k + 1) * KS);
        }
        return;
    }
    // ---------------- consumers
    const int rh = warp >> 2, cq = warp & 3, n_base = cq * 32;
    const int nfr = (local + 15) >> 4;
    const int nact = min(MF, (nfr - rh + 1) >> 1);
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
    constexpr int CAT = (FL & FL_HY) ? (MCAP * 4 + 255) / 256 : 1;
    ARaw car[CAT];
    auto cact_load = [&](int k0) {
        const int kb = k0 >> 7;
#pragma unroll
        for (int j = 0; j < CAT; ++j) {
            const int task = tid + j * 256, m = task >> 2, off = (k0 & 127) + (task & 3) * 16;
            if (m < local) {
                const uint8_t* base = (const uint8_t*) act + ((size_t) kb * (size_t) act_rows + (size_t) (row0 + m)) * ACTB;
                car[j].d = ((const float*) base)[off >> 5];
                car[j].q = *reinterpret_cast<const uint4*>(base + 16 + off);
            }
        }
    };
    auto cact_store = [&](int s) {
        __half* a_s = bufs + s * SSZ;
#pragma unroll
        for (int j = 0; j < CAT; ++j) {
            const int task = tid + j * 256, m = task >> 2;
            if (m < local) {
                uint4* o = reinterpret_cast<uint4*>(a_s + (size_t) m * AP + (task & 3) * 16);
                const __half2 d2 = __float2half2_rn(car[j].d);
                const uint32_t v0 = car[j].q.x ^ 0x80808080u, v1 = car[j].q.y ^ 0x80808080u, v2 = car[j].q.z ^ 0x80808080u,
                               v3 = car[j].q.w ^ 0x80808080u;
                o[0] = make_uint4(cvt2(v0, 0x4140, d2), cvt2(v0, 0x4342, d2), cvt2(v1, 0x4140, d2), cvt2(v1, 0x4342, d2));
                o[1] = make_uint4(cvt2(v2, 0x4140, d2), cvt2(v2, 0x4342, d2), cvt2(v3, 0x4140, d2), cvt2(v3, 0x4342, d2));
            }
        }
    };
    if constexpr (FL & FL_HY) { cact_load(0); cact_store(0); }
    for (int k = 0; k < NK; ++k) {
        const int b = k & 1;
        if constexpr (FL & FL_HY) if (k + 1 < NK) cact_load((k + 1) * KS);
        bar_sync(1 + b, NT);
        const __half* a_s = bufs + b * SSZ;
        const __half* w_s = a_s + ASZ;
        if constexpr (FL & FL_HY) {
            // all consumers passed FULL(k), so all finished MMA(k-1): stage (k+1)&1 is free for act
            if constexpr (FL & FL_NOMMA) { if (a_s[tid] == __float2half(-77.f)) acc[0][0].x[0] += 1.f; }
            else switch (nact) {
                case 6: if constexpr (MF >= 6) mma_n(a_s, w_s, std::integral_constant<int, (MF >= 6 ? 6 : 1)>{}); break;
                case 5: if constexpr (MF >= 5) mma_n(a_s, w_s, std::integral_constant<int, (MF >= 5 ? 5 : 1)>{}); break;
                case 4: if constexpr (MF >= 4) mma_n(a_s, w_s, std::integral_constant<int, (MF >= 4 ? 4 : 1)>{}); break;
                case 3: if constexpr (MF >= 3) mma_n(a_s, w_s, std::integral_constant<int, (MF >= 3 ? 3 : 1)>{}); break;
                case 2: mma_n(a_s, w_s, std::integral_constant<int, 2>{}); break;
                case 1: mma_n(a_s, w_s, std::integral_constant<int, 1>{}); break;
                default: break;
            }
            if (k + 1 < NK) cact_store((k + 1) & 1);
        } else if constexpr (!(FL & FL_NOMMA)) {
            switch (nact) {
                case 8: if constexpr (MF >= 8) mma_n(a_s, w_s, std::integral_constant<int, (MF >= 8 ? 8 : 1)>{}); break;
                case 7: if constexpr (MF >= 7) mma_n(a_s, w_s, std::integral_constant<int, (MF >= 7 ? 7 : 1)>{}); break;
                case 6: if constexpr (MF >= 6) mma_n(a_s, w_s, std::integral_constant<int, (MF >= 6 ? 6 : 1)>{}); break;
                case 5: if constexpr (MF >= 5) mma_n(a_s, w_s, std::integral_constant<int, (MF >= 5 ? 5 : 1)>{}); break;
                case 4: if constexpr (MF >= 4) mma_n(a_s, w_s, std::integral_constant<int, (MF >= 4 ? 4 : 1)>{}); break;
                case 3: if constexpr (MF >= 3) mma_n(a_s, w_s, std::integral_constant<int, (MF >= 3 ? 3 : 1)>{}); break;
                case 2: mma_n(a_s, w_s, std::integral_constant<int, 2>{}); break;
                case 1: mma_n(a_s, w_s, std::integral_constant<int, 1>{}); break;
                default: break;
            }
        } else {
            if (a_s[tid] == __float2half(-77.f) && w_s[tid] == __float2half(-77.f)) acc[0][0].x[0] += 1.f;
        }
        if (k + 2 < NK) bar_arrive(3 + b, NT);
    }
    bar_sync(5, 256);   // all consumers done reading the stage buffers
    float* stg = reinterpret_cast<float*>(bufs) + warp * 16 * 20;
    if ((FL & FL_NOEPI) && ld_dst > 0) {
        float sum = 0.f;
#pragma unroll
        for (int i = 0; i < MF; ++i)
#pragma unroll
            for (int c = 0; c < 2; ++c)
#pragma unroll
                for (int t = 0; t < acc[i][c].num_elements; ++t) sum += acc[i][c].x[t];
        if (sum == -77.f) dst[tid] = 1.f;
        return;
    }
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

// ---------------- kernel D: C + coalesced raw super-block staging. Producers load the 128 rows x one 256-value
// super-block (128*BS bytes) with coalesced u16 loads into registers one super-block ahead, park it in smem, and decode
// from smem. KS = 32, NS stages of (192 x 40 act + 128 x 40 w) halves. Barriers: FULL 1..NS, EMPTY NS+1..2NS,
// consumers 2NS+1, producers 2NS+2.
template <int WT, int NS, int FL, int NPW = 4>
__global__ void __launch_bounds__(256 + 32 * NPW, 1)
gu4_kernel(const uint8_t* const* __restrict__ blobs, const Item* __restrict__ items, const void* __restrict__ act,
           int64_t act_rows, float* __restrict__ dst, int64_t ld_dst) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    using namespace nvcuda;
    constexpr int MCAP = 192, KD = 32, AP = 40, WP = 40;
    constexpr int NT = 256 + 32 * NPW, PT = 32 * NPW;
    constexpr int BS = block_bytes(WT), ROWB = (N / 256) * BS;
    constexpr int MF = MCAP / 32;
    constexpr int GBR = grid_bytes(WT), RAWB = BN * BS;            // raw super-block slab bytes
    constexpr int ASZ = MCAP * AP, WSZ = BN * WP, SSZ = ASZ + WSZ;
    constexpr int NK = N / KD, SPB = 256 / KD;                      // steps, steps per super-block
    constexpr int RU = (RAWB / 2 + PT - 1) / PT;                    // u16 raw loads per producer thread
    constexpr int BAR_C = 2 * NS + 1, BAR_P = 2 * NS + 2;
    extern __shared__ __align__(16) uint8_t smem[];
    const Item it = items[blockIdx.y];
    const int local = it.n, row0 = it.row0;
    const int out_base = (int) blockIdx.x * BN;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    {
        const uint32_t* gs = (const uint32_t*) grid_src<WT>();
        for (int i = tid; i < GBR / 4; i += NT) ((uint32_t*) smem)[i] = gs[i];
    }
    const uint8_t* grid = smem;
    uint8_t* raw = smem + GBR;
    __half* bufs = (__half*) (smem + ((GBR + RAWB + 15) & ~15));
    __syncthreads();

    if (warp >= 8) {   // ---------------- producers
        const int pt = tid - 256;
        constexpr int AT = (MCAP * 2 + PT - 1) / PT;   // act tasks (16 values) per thread per step (2 per row)
        const uint8_t* wbase = blobs[it.e] + (size_t) out_base * ROWB;
        uint16_t rr[RU];
        ARaw ar[AT];
        auto raw_load = [&](int kb) {
#pragma unroll
            for (int j = 0; j < RU; ++j) {
                const int i = pt + j * PT;
                if (i < RAWB / 2) {
                    const int row = i / (BS / 2), w = i - row * (BS / 2);
                    rr[j] = *reinterpret_cast<const uint16_t*>(wbase + (size_t) row * ROWB + (size_t) kb * BS + 2 * w);
                }
            }
        };
        auto raw_store = [&]() {
#pragma unroll
            for (int j = 0; j < RU; ++j) {
                const int i = pt + j * PT;
                if (i < RAWB / 2) reinterpret_cast<uint16_t*>(raw)[i] = rr[j];
            }
        };
        auto act_load = [&](int k0) {
            if constexpr (!(FL & FL_NOACT)) {
                const int kb = k0 >> 7;
#pragma unroll
                for (int j = 0; j < AT; ++j) {
                    const int task = pt + j * PT, m = task >> 1, off = (k0 & 127) + (task & 1) * 16;
                    if (m < local) {
                        const uint8_t* base = (const uint8_t*) act + ((size_t) kb * (size_t) act_rows + (size_t) (row0 + m)) * ACTB;
                        ar[j].d = ((const float*) base)[off >> 5];
                        ar[j].q = *reinterpret_cast<const uint4*>(base + 16 + off);
                    }
                }
            }
        };
        auto store = [&](int k, int s) {
            __half* a_s = bufs + s * SSZ;
            __half* w_s = a_s + ASZ;
#pragma unroll
            for (int u = 0; u < BN / PT; ++u) {
                const int row = pt + u * PT;
                uint32_t q[8];
                float s0, s1;
                if constexpr (FL & FL_NODEC) {
#pragma unroll
                    for (int i = 0; i < 8; ++i) q[i] = 0x01010101u * (uint32_t) (i + 1);
                    s0 = s1 = 0.01f;
                } else {
                    uint32_t w[5];
                    load_unit<WT>(raw + row * BS, k % SPB, w);
                    convert<WT>(w, grid, q, s0, s1);
                }
                const __half2 sa = __float2half2_rn(s0), sb = __float2half2_rn(per16(WT) ? s1 : s0);
                uint4* o = reinterpret_cast<uint4*>(w_s + (size_t) row * WP);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const uint32_t v0 = q[2 * i] ^ 0x80808080u, v1 = q[2 * i + 1] ^ 0x80808080u;
                    const __half2 sc = i < 2 ? sa : sb;
                    o[i] = make_uint4(cvt2(v0, 0x4140, sc), cvt2(v0, 0x4342, sc), cvt2(v1, 0x4140, sc), cvt2(v1, 0x4342, sc));
                }
            }
#pragma unroll
            for (int j = 0; j < AT; ++j) {
                const int task = pt + j * PT, m = task >> 1;
                if (m < local) {
                    uint4* o = reinterpret_cast<uint4*>(a_s + (size_t) m * AP + (task & 1) * 16);
                    if constexpr (FL & FL_NOACT) {
                        o[0] = o[1] = make_uint4(0x211f211fu, 0x211f211fu, 0x211f211fu, 0x211f211fu);
                    } else {
                        const __half2 d2 = __float2half2_rn(ar[j].d);
                        const uint32_t v0 = ar[j].q.x ^ 0x80808080u, v1 = ar[j].q.y ^ 0x80808080u, v2 = ar[j].q.z ^ 0x80808080u,
                                       v3 = ar[j].q.w ^ 0x80808080u;
                        o[0] = make_uint4(cvt2(v0, 0x4140, d2), cvt2(v0, 0x4342, d2), cvt2(v1, 0x4140, d2), cvt2(v1, 0x4342, d2));
                        o[1] = make_uint4(cvt2(v2, 0x4140, d2), cvt2(v2, 0x4342, d2), cvt2(v3, 0x4140, d2), cvt2(v3, 0x4342, d2));
                    }
                }
            }
        };
        if constexpr (!(FL & FL_NODEC)) { raw_load(0); raw_store(); }
        bar_sync(BAR_P, PT);
        act_load(0);
        for (int k = 0; k < NK; ++k) {
            const int kb = k / SPB, b = k % NS;
            if constexpr (!(FL & FL_NODEC)) if (k % SPB == 0 && kb + 1 < N / 256) raw_load(kb + 1);
            if (k >= NS) bar_sync(NS + 1 + b, NT);
            store(k, b);
            bar_arrive(1 + b, NT);
            if (k + 1 < NK) act_load((k + 1) * KD);
            if constexpr (!(FL & FL_NODEC)) {
                if (k % SPB == SPB - 1 && kb + 1 < N / 256) {
                    bar_sync(BAR_P, PT);
                    raw_store();
                    bar_sync(BAR_P, PT);
                }
            }
        }
        return;
    }
    // ---------------- consumers
    const int rh = warp >> 2, cq = warp & 3, n_base = cq * 32;
    const int nfr = (local + 15) >> 4;
    const int nact = min(MF, (nfr - rh + 1) >> 1);
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[MF][2];
#pragma unroll
    for (int i = 0; i < MF; ++i)
#pragma unroll
        for (int c = 0; c < 2; ++c) wmma::fill_fragment(acc[i][c], 0.0f);
    auto mma_n = [&](const __half* a_s, const __half* w_s, auto na_c) {
        constexpr int NA = decltype(na_c)::value;
#pragma unroll
        for (int k16 = 0; k16 < KD / 16; ++k16) {
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
        const int b = k % NS;
        bar_sync(1 + b, NT);
        const __half* a_s = bufs + b * SSZ;
        const __half* w_s = a_s + ASZ;
        if constexpr (!(FL & FL_NOMMA)) {
            switch (nact) {
                case 6: mma_n(a_s, w_s, std::integral_constant<int, 6>{}); break;
                case 5: mma_n(a_s, w_s, std::integral_constant<int, 5>{}); break;
                case 4: mma_n(a_s, w_s, std::integral_constant<int, 4>{}); break;
                case 3: mma_n(a_s, w_s, std::integral_constant<int, 3>{}); break;
                case 2: mma_n(a_s, w_s, std::integral_constant<int, 2>{}); break;
                case 1: mma_n(a_s, w_s, std::integral_constant<int, 1>{}); break;
                default: break;
            }
        } else {
            if (a_s[tid] == __float2half(-77.f) && w_s[tid] == __float2half(-77.f)) acc[0][0].x[0] += 1.f;
        }
        if (k + NS < NK) bar_arrive(NS + 1 + b, NT);
    }
    bar_sync(BAR_C, 256);
    float* stg = reinterpret_cast<float*>(bufs) + warp * 16 * 20;
    if ((FL & FL_NOEPI) && ld_dst > 0) {
        float sum = 0.f;
#pragma unroll
        for (int i = 0; i < MF; ++i)
#pragma unroll
            for (int c = 0; c < 2; ++c)
#pragma unroll
                for (int t = 0; t < acc[i][c].num_elements; ++t) sum += acc[i][c].x[t];
        if (sum == -77.f) dst[tid] = 1.f;
        return;
    }
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

void ck(cudaError_t e, const char* w) { if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(2); } }
struct Dev { void* p = nullptr; explicit Dev(size_t n) { ck(cudaMalloc(&p, n), "malloc"); } ~Dev() { cudaFree(p); } template <typename T> T* as() const { return (T*) p; } };

struct Env {
    size_t ng = 0;
    std::vector<int> ngx; std::vector<int64_t> maxv;
    const uint8_t* wts = nullptr; size_t exb = 0;
    const int32_t* bounds = nullptr;
    const void* xq = nullptr; int64_t rows = 0;
    float* dst = nullptr;
    const uint8_t* const* blobs = nullptr;
    // items per MCAP: device array, count, per-group offsets
    struct Items { Item* d = nullptr; int n = 0; std::vector<int> goff; };
    Items items[5];   // index by MCAP/64-? see mcap_idx
    cudaStream_t s;
    cublasHandle_t cb = nullptr; const __half* a16 = nullptr; const __half* b16 = nullptr;
};
constexpr int mcap_idx(int m) { return m == 64 ? 0 : m == 128 ? 1 : m == 160 ? 2 : m == 192 ? 3 : 4; }

template <int WT, int FL, int AP, int WP>
void runA(Env& e) {
    for (size_t g = 0; g < e.ng; ++g) {
        Batch b; b.n = e.ngx[g]; b.max_rows = (int) e.maxv[g];
        for (int i = 0; i < b.n; ++i) b.blob[i] = e.wts + (g * GROUP + (size_t) i) * e.exb;
        const dim3 grid(GU_ROWS / BN, (b.max_rows + BM - 1) / BM, b.n);
        gu_kernel<WT, FL, AP, WP><<<grid, TS, grid_bytes(WT), e.s>>>(b, e.bounds + g * GROUP, e.xq, e.rows, e.dst, GU_ROWS);
    }
}
template <int WT, int NS, int FL, int NPW = 4>
void runD(Env& e) {
    constexpr int smem = ((grid_bytes(WT) + BN * block_bytes(WT) + 15) & ~15) + NS * (192 * 40 + BN * 40) * 2;
    static_assert(smem <= 96 * 1024, "smem");
    static bool init = false;
    if (!init) { ck(cudaFuncSetAttribute(gu4_kernel<WT, NS, FL, NPW>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem), "attr"); init = true; }
    auto& it = e.items[mcap_idx(192)];
    gu4_kernel<WT, NS, FL, NPW><<<dim3(GU_ROWS / BN, it.n), 256 + 32 * NPW, smem, e.s>>>(e.blobs, it.d, e.xq, e.rows, e.dst, GU_ROWS);
}
template <int WT, int MCAP, int NPW, int FL>
void runC(Env& e) {
    constexpr int ss = 2 * (MCAP * 72 + BN * 72) * 2;
    constexpr int GS = grid_bytes(WT) + ss <= 96 * 1024;
    constexpr int smem = (GS ? grid_bytes(WT) : 0) + ss;
    static bool init = false;
    if (!init) { ck(cudaFuncSetAttribute(gu3_kernel<WT, MCAP, NPW, GS, FL>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem), "attr"); init = true; }
    auto& it = e.items[mcap_idx(MCAP)];
    gu3_kernel<WT, MCAP, NPW, GS, FL><<<dim3(GU_ROWS / BN, it.n), 256 + 32 * NPW, smem, e.s>>>(e.blobs, it.d, e.xq, e.rows, e.dst, GU_ROWS);
}
template <int WT, int MCAP, int ST, int GS, int AP, int WP, int FL, int DISP = 0>
void runB(Env& e, bool per_group) {
    constexpr int smem = (GS ? grid_bytes(WT) : 0) + ST * (MCAP * AP + BN * WP) * 2;
    static_assert(smem <= 96 * 1024, "smem");
    static bool init = false;
    if (!init) { ck(cudaFuncSetAttribute(gu2_kernel<WT, MCAP, ST, GS, AP, WP, FL, DISP>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem), "attr"); init = true; }
    auto& it = e.items[mcap_idx(MCAP)];
    if (!per_group) {
        gu2_kernel<WT, MCAP, ST, GS, AP, WP, FL, DISP><<<dim3(GU_ROWS / BN, it.n), TS, smem, e.s>>>(e.blobs, it.d, e.xq, e.rows, e.dst, GU_ROWS);
    } else {
        for (size_t g = 0; g < e.ng; ++g) {
            const int a = it.goff[g], b = it.goff[g + 1];
            gu2_kernel<WT, MCAP, ST, GS, AP, WP, FL, DISP><<<dim3(GU_ROWS / BN, b - a), TS, smem, e.s>>>(e.blobs, it.d + a, e.xq, e.rows, e.dst, GU_ROWS);
        }
    }
}

struct Variant { const char* name; bool exact; };
// exact = produces the real result (parity checked); ablations are timing-only
static const Variant VARS[] = {
    {"A proto", true},                 // 0
    {"A -dec", false},                 // 1
    {"A -act", false},                 // 2
    {"A -mma", false},                 // 3
    {"A -epi", false},                 // 4
    {"A -dec-act", false},             // 5
    {"A pad72/72", true},              // 6
    {"A pad80/80", true},              // 7
    {"A pad64/64", true},              // 8
    {"B m192 st1 grp", true},          // 9
    {"B m192 st1 all", true},          // 10
    {"B m192 st2 all g0", true},       // 11  grid in global (L1)
    {"B m160 st2 all", true},          // 12
    {"B m128 st2 all", true},          // 13
    {"B m64 st2 all", true},           // 14
    {"B m192 st2 -dec", false},        // 15
    {"B m192 st2 -act", false},        // 16
    {"B m192 st2 -mma", false},        // 17
    {"B m192 st2 -epi", false},        // 18
    {"B m192 st2 pad80/72 g0", true},  // 19
    {"B m160 st2 pad80/80", true},     // 20
    {"B m192 st1 all g1", true},       // 21
    {"B m192 st2 all g1*", true},      // 22  grid in smem when it fits
    {"B m256 st1 all", true},          // 23
    {"B m192 st1 mma-only", false},    // 24
    {"B m192 st1 -dec-act", false},    // 25
    {"B m192 st1 -mma", false},        // 26
    {"B m192 st1 -dec", false},        // 27
    {"B m192 st1 -act", false},        // 28
    {"B m192 st1 -epi", false},        // 29
    {"B m192 st1 disp", true},         // 30  unguarded count-dispatched MMA body
    {"B m192 st1 disp mma-only", false}, // 31
    {"B m192 st2 disp g1*", true},     // 32
    {"B m256 st1 disp", true},         // 33
    {"B m192 st1 disp -mma", false},   // 34
    {"C m192 np4", true},              // 35  warp-specialized
    {"C m192 np4 mma-only", false},    // 36
    {"C m192 np4 -mma", false},        // 37
    {"C m192 np2", true},              // 38
    {"C m192 np4 -dec", false},        // 39
    {"C m192 np4 -act", false},        // 40
    {"D ns3 (raw smem, KS32)", true},  // 41
    {"D ns2", true},                   // 42
    {"D ns3 mma-only", false},         // 43
    {"D ns3 -mma", false},             // 44
    {"D ns3 -dec", false},             // 45
    {"D ns3 -act", false},             // 46
    {"C np4 -mma-act (w only)", false},  // 47
    {"C np4 -mma-dec (act only)", false},// 48
    {"C np4 -mma-dec-act", false},     // 49
    {"C np6", true},                   // 50
    {"C np8", true},                   // 51
    {"C np4 HY (consumers do act)", true}, // 52
    {"C np4 HY -mma", false},          // 53
    {"C np4 HY -dec", false},          // 54
    {"cuBLAS dense fp16 same flops", false}, // 55
};
constexpr int NV = sizeof(VARS) / sizeof(VARS[0]);

template <int WT> void run_variant(int v, Env& e) {
    switch (v) {
        case 0: runA<WT, 0, 80, 72>(e); break;
        case 1: runA<WT, FL_NODEC, 80, 72>(e); break;
        case 2: runA<WT, FL_NOACT, 80, 72>(e); break;
        case 3: runA<WT, FL_NOMMA, 80, 72>(e); break;
        case 4: runA<WT, FL_NOEPI, 80, 72>(e); break;
        case 5: runA<WT, FL_NODEC | FL_NOACT, 80, 72>(e); break;
        case 6: runA<WT, 0, 72, 72>(e); break;
        case 7: runA<WT, 0, 80, 80>(e); break;
        case 8: runA<WT, 0, 64, 64>(e); break;
        case 9: runB<WT, 192, 1, 1, 72, 72, 0>(e, true); break;
        case 10: runB<WT, 192, 1, 1, 72, 72, 0>(e, false); break;
        case 11: runB<WT, 192, 2, 0, 72, 72, 0>(e, false); break;
        case 12: runB<WT, 160, 2, 1, 72, 72, 0>(e, false); break;
        case 13: runB<WT, 128, 2, 1, 72, 72, 0>(e, false); break;
        case 14: runB<WT, 64, 2, 1, 72, 72, 0>(e, false); break;
        case 15: runB<WT, 192, 2, 0, 72, 72, FL_NODEC>(e, false); break;
        case 16: runB<WT, 192, 2, 0, 72, 72, FL_NOACT>(e, false); break;
        case 17: runB<WT, 192, 2, 0, 72, 72, FL_NOMMA>(e, false); break;
        case 18: runB<WT, 192, 2, 0, 72, 72, FL_NOEPI>(e, false); break;
        case 19: runB<WT, 192, 2, 0, 80, 72, 0>(e, false); break;
        case 20: runB<WT, 160, 2, 0, 80, 80, 0>(e, false); break;
        case 21: runB<WT, 192, 1, 1, 72, 72, 0>(e, false); break;
        case 22: runB<WT, 192, 2, (grid_bytes(WT) + 2 * (192 * 72 + BN * 72) * 2 <= 96 * 1024), 72, 72, 0>(e, false); break;
        case 23: runB<WT, 256, 1, 1, 72, 72, 0>(e, false); break;
        case 24: runB<WT, 192, 1, 1, 72, 72, FL_NODEC | FL_NOACT | FL_NOEPI>(e, false); break;
        case 25: runB<WT, 192, 1, 1, 72, 72, FL_NODEC | FL_NOACT>(e, false); break;
        case 26: runB<WT, 192, 1, 1, 72, 72, FL_NOMMA>(e, false); break;
        case 27: runB<WT, 192, 1, 1, 72, 72, FL_NODEC>(e, false); break;
        case 28: runB<WT, 192, 1, 1, 72, 72, FL_NOACT>(e, false); break;
        case 29: runB<WT, 192, 1, 1, 72, 72, FL_NOEPI>(e, false); break;
        case 30: runB<WT, 192, 1, 1, 72, 72, 0, 1>(e, false); break;
        case 31: runB<WT, 192, 1, 1, 72, 72, FL_NODEC | FL_NOACT | FL_NOEPI, 1>(e, false); break;
        case 32: runB<WT, 192, 2, (grid_bytes(WT) + 2 * (192 * 72 + BN * 72) * 2 <= 96 * 1024), 72, 72, 0, 1>(e, false); break;
        case 33: runB<WT, 256, 1, 1, 72, 72, 0, 1>(e, false); break;
        case 34: runB<WT, 192, 1, 1, 72, 72, FL_NOMMA, 1>(e, false); break;
        case 35: runC<WT, 192, 4, 0>(e); break;
        case 36: runC<WT, 192, 4, FL_NODEC | FL_NOACT | FL_NOEPI>(e); break;
        case 37: runC<WT, 192, 4, FL_NOMMA>(e); break;
        case 38: runC<WT, 192, 2, 0>(e); break;
        case 39: runC<WT, 192, 4, FL_NODEC>(e); break;
        case 40: runC<WT, 192, 4, FL_NOACT>(e); break;
        case 41: runD<WT, 3, 0>(e); break;
        case 42: runD<WT, 2, 0>(e); break;
        case 43: runD<WT, 3, FL_NODEC | FL_NOACT | FL_NOEPI>(e); break;
        case 44: runD<WT, 3, FL_NOMMA>(e); break;
        case 45: runD<WT, 3, FL_NODEC>(e); break;
        case 46: runD<WT, 3, FL_NOACT>(e); break;
        case 47: runC<WT, 192, 4, FL_NOMMA | FL_NOACT>(e); break;
        case 48: runC<WT, 192, 4, FL_NOMMA | FL_NODEC>(e); break;
        case 49: runC<WT, 192, 4, FL_NOMMA | FL_NODEC | FL_NOACT>(e); break;
        case 50: runC<WT, 192, 6, 0>(e); break;
        case 51: runC<WT, 192, 8, 0>(e); break;
        case 52: runC<WT, 192, 4, FL_HY>(e); break;
        case 53: runC<WT, 192, 4, FL_HY | FL_NOMMA>(e); break;
        case 54: runC<WT, 192, 4, FL_HY | FL_NODEC>(e); break;
        case 55: { const float al = 1.f, be = 0.f; cublasSetStream(e.cb, e.s); if (cublasGemmEx(e.cb, CUBLAS_OP_T, CUBLAS_OP_N, GU_ROWS, (int) e.rows, N, &al, e.b16, CUDA_R_16F, N, e.a16, CUDA_R_16F, N, &be, e.dst, CUDA_R_32F, GU_ROWS, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) != CUBLAS_STATUS_SUCCESS) { std::fprintf(stderr, "cublas fail\n"); std::exit(3); } break; }
        default: break;
    }
}
void run_t(int t, int v, Env& e) {
    switch (t) {
        case T_IQ2_XXS: run_variant<T_IQ2_XXS>(v, e); break;
        case T_IQ2_XS: run_variant<T_IQ2_XS>(v, e); break;
        case T_IQ2_S: run_variant<T_IQ2_S>(v, e); break;
        case T_IQ3_XXS: run_variant<T_IQ3_XXS>(v, e); break;
        default: run_variant<T_IQ3_S>(v, e); break;
    }
}

int main(int argc, char** argv) {
    const std::string name = argc > 1 ? argv[1] : "iq3_s";
    const int T = argc > 2 ? std::atoi(argv[2]) : 8192, E = argc > 3 ? std::atoi(argv[3]) : 512, reps = argc > 4 ? std::atoi(argv[4]) : 5;
    std::vector<int> sel;
    if (argc > 5) { std::string l = argv[5]; size_t p = 0; while (p < l.size()) { size_t q = l.find(',', p); if (q == std::string::npos) q = l.size(); sel.push_back(std::atoi(l.substr(p, q - p).c_str())); p = q + 1; } }
    else for (int v = 0; v < NV; ++v) sel.push_back(v);
    const int ty = name == "iq2_xxs" ? T_IQ2_XXS : name == "iq2_xs" ? T_IQ2_XS : name == "iq2_s" ? T_IQ2_S : name == "iq3_xxs" ? T_IQ3_XXS : T_IQ3_S;
    const size_t ROWB = (size_t) (N / 256) * block_bytes(ty), EXB = (size_t) GU_ROWS * ROWB;
    if (!mmq::fits(ty, GU_ROWS)) { std::printf("MMQ does not fit %s\n", name.c_str()); return 77; }
    cudaStream_t s; ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
    std::mt19937 rng(7);
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
    std::vector<int32_t> bh(n + 1, 0);
    for (size_t j = 0; j < n; ++j) bh[j + 1] = bh[j] + cnt[(size_t) order[j]];
    int mx_all = 0, mn_all = 1 << 30; for (size_t j = 0; j < n; ++j) { mx_all = std::max(mx_all, cnt[(size_t) order[j]]); mn_all = std::min(mn_all, cnt[(size_t) order[j]]); }
    std::printf("%s T=%d E=%d: rows %lld, %zu experts (rows/expert %d..%d), %zu groups, row bytes %zu\n", name.c_str(), T, E, (long long) rows, n, mn_all, mx_all, ng, ROWB);
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
    std::vector<float> xh((size_t) T * N); { std::normal_distribution<float> nd(0.f, 1.f); for (float& v : xh) v = nd(rng) * 0.5f; }
    Dev dx(xh.size() * 4), dtok((size_t) rows * 4), ident((size_t) rows * 4), dbounds((n + 1) * 4);
    ck(cudaMemcpy(dx.p, xh.data(), xh.size() * 4, cudaMemcpyHostToDevice), "x");
    ck(cudaMemcpy(dtok.p, tok.data(), (size_t) rows * 4, cudaMemcpyHostToDevice), "tok");
    ck(cudaMemcpy(dbounds.p, bh.data(), (n + 1) * 4, cudaMemcpyHostToDevice), "bounds");
    mmq::iota(ident.as<int32_t>(), rows, s);
    Dev xq(mmq::q8_bytes(rows, N));
    mmq::quantize(dx.as<float>(), dtok.as<int32_t>(), xq.p, ty, N, N, rows, s);
    Dev dm_mmq((size_t) rows * GU_ROWS * 4), dm_tc((size_t) rows * GU_ROWS * 4);
    ck(cudaMemset(dm_mmq.p, 0, (size_t) rows * GU_ROWS * 4), "z");
    ck(cudaStreamSynchronize(s), "prep");
    std::vector<int64_t> maxv(ng); std::vector<int> ngxv(ng);
    for (size_t g = 0; g < ng; ++g) { const size_t j0 = g * GROUP, j1 = std::min(n, j0 + GROUP); ngxv[g] = (int) (j1 - j0); int64_t mx = 0; for (size_t j = j0; j < j1; ++j) mx = std::max<int64_t>(mx, cnt[(size_t) order[j]]); maxv[g] = mx; }

    Env env; env.ng = ng; env.ngx = ngxv; env.maxv = maxv; env.wts = wts.as<uint8_t>(); env.exb = EXB; env.bounds = dbounds.as<int32_t>();
    env.xq = xq.p; env.rows = rows; env.dst = dm_tc.as<float>(); env.s = s;
    std::vector<const uint8_t*> bl(n); for (size_t j = 0; j < n; ++j) bl[j] = wts.as<uint8_t>() + j * EXB;
    Dev dbl(n * sizeof(void*)); ck(cudaMemcpy(dbl.p, bl.data(), n * sizeof(void*), cudaMemcpyHostToDevice), "bl");
    env.blobs = dbl.as<const uint8_t*>();
    std::vector<std::unique_ptr<Dev>> keep;
    const int mcaps[5] = {64, 128, 160, 192, 256};
    for (int mi = 0; mi < 5; ++mi) {
        const int mc = mcaps[mi];
        std::vector<Item> it; std::vector<int> goff(1, 0);
        for (size_t j = 0; j < n; ++j) {
            const int c = bh[j + 1] - bh[j];
            const int nch = (c + mc - 1) / mc;
            const int chunk = (((c + nch - 1) / nch) + 15) / 16 * 16;
            for (int o = 0; o < c; o += chunk) it.push_back(Item{(int) j, bh[j] + o, std::min(chunk, c - o), 0});
            if ((j + 1) % GROUP == 0 || j + 1 == n) goff.push_back((int) it.size());
        }
        keep.emplace_back(new Dev(it.size() * sizeof(Item)));
        ck(cudaMemcpy(keep.back()->p, it.data(), it.size() * sizeof(Item), cudaMemcpyHostToDevice), "items");
        env.items[mi].d = keep.back()->as<Item>(); env.items[mi].n = (int) it.size(); env.items[mi].goff = goff;
        std::printf("MCAP %d: %zu items (%.1f%% padded MMA rows to 16)\n", mc, it.size(),
                    [&] { long long p = 0; for (auto& x : it) p += (x.n + 15) / 16 * 16; return 100.0 * (p - rows) / rows; }());
    }

    std::unique_ptr<Dev> a16, b16;
    if (std::find(sel.begin(), sel.end(), 55) != sel.end()) {
        a16.reset(new Dev((size_t) rows * N * 2)); b16.reset(new Dev((size_t) GU_ROWS * N * 2));
        ck(cudaMemset(a16->p, 0x11, (size_t) rows * N * 2), "a16"); ck(cudaMemset(b16->p, 0x11, (size_t) GU_ROWS * N * 2), "b16");
        cublasCreate(&env.cb); env.a16 = a16->as<__half>(); env.b16 = b16->as<__half>();
    }
    mmq::Context ctx;
    auto run_mmq = [&]() {
        for (size_t g = 0; g < ng; ++g) {
            mmq::Product p; p.w = wts.as<uint8_t>() + g * GROUP * EXB; p.type = ty; p.w_rows = GU_ROWS; p.w_cols = N; p.expert_bytes = EXB;
            p.n = ngxv[g]; p.xq = xq.p; p.bounds = dbounds.as<int32_t>() + g * GROUP; p.ids = ident.as<int32_t>();
            p.total_rows = rows; p.max_rows = maxv[g]; p.dst = dm_mmq.as<float>(); p.ld_dst = GU_ROWS;
            ctx.run(p, s);
        }
    };
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    const double flop = (double) rows * GU_ROWS * N * 2.0;
    auto timeit = [&](auto&& fn) {
        fn(); ck(cudaStreamSynchronize(s), "warm"); ck(cudaGetLastError(), "launch");
        float best = 1e30f;
        for (int r = 0; r < reps; ++r) { cudaEventRecord(e0, s); fn(); cudaEventRecord(e1, s); ck(cudaEventSynchronize(e1), "t"); float ms = 0; cudaEventElapsedTime(&ms, e0, e1); best = std::min(best, ms); }
        return best;
    };
    const float tm = timeit(run_mmq);
    std::vector<float> ym((size_t) rows * GU_ROWS), yt((size_t) rows * GU_ROWS);
    ck(cudaMemcpy(ym.data(), dm_mmq.p, ym.size() * 4, cudaMemcpyDeviceToHost), "dl");
    std::printf("%-26s %8.3f ms %6.2f TOPS\n", "MMQ", tm, flop / (tm * 1e-3) / 1e12);
    float tproto = 0;
    for (int v : sel) {
        if (v < 0 || v >= NV) continue;
        ck(cudaMemset(dm_tc.p, 0xFF, ym.size() * 4), "nan");   // NaN sentinel: any row not written shows up
        run_t(ty, v, env);
        ck(cudaStreamSynchronize(s), "run"); ck(cudaGetLastError(), "launch");
        char par[160] = "";
        if (VARS[v].exact) {
            ck(cudaMemcpy(yt.data(), dm_tc.p, yt.size() * 4, cudaMemcpyDeviceToHost), "dl");
            double d2 = 0, r2 = 0, s0d = 0, s0r = 0, s1d = 0, s1r = 0; size_t bad = 0, nz = 0;
            for (size_t i = 0; i < yt.size(); ++i) {
                if (!std::isfinite(yt[i])) { ++bad; continue; }
                nz += yt[i] != 0.f;
                const double d = (double) ym[i] - yt[i], dd = d * d, rr = (double) ym[i] * ym[i];
                d2 += dd; r2 += rr;
                const size_t r = i / GU_ROWS;
                if (r < 4096) { s0d += dd; s0r += rr; }
                if (r >= (size_t) rows / 2 && r < (size_t) rows / 2 + 4096) { s1d += dd; s1r += rr; }
            }
            std::snprintf(par, sizeof par, "parity all %.3e  s0 %.3e  s1 %.3e  nonfinite %zu  nonzero %.4f", std::sqrt(d2 / r2),
                          std::sqrt(s0d / s0r), std::sqrt(s1d / s1r), bad, (double) nz / yt.size());
        }
        const float t = timeit([&] { run_t(ty, v, env); });
        if (v == 0) tproto = t;
        std::printf("%2d %-23s %8.3f ms %6.2f TOPS  vsMMQ %.2fx  vsProto %.2fx  %s\n", v, VARS[v].name, t, flop / (t * 1e-3) / 1e12, tm / t,
                    tproto > 0 ? tproto / t : 0.0, par);
        std::fflush(stdout);
    }
    return 0;
}
