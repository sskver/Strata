// src/kernels/cuda/dense_pk.cu - persistent warp-per-row decode GEMVs for the dense projections (IQ4_XS, IQ3_S, Q5_K,
// Q6_K, IQ4_NL), BITWISE equal to the kernels they replace (native_mmvq.cu native_mmvq_multi_kernel / the ncols == 1
// kernels, iq_kernels.cu mmvq_multi_kernel<21>), plus one-launch fused GEMVs over 2-3 matrices that share an activation.
// See include/strata/kernels/dense_pk.hpp.  Developed and timed in bench/gemv_opus/gemv_opus.cu.
//
// Why the arithmetic is the old one: each row's float operations are the old kernel's.  Every lane of the old kernel's 4
// virtual warps (1 for IQ3_S) keeps its own sequential partial over the same blocks in the same order, the partials are
// added in the old order (vw0 + vw1 + vw2 + vw3), then the same xor butterfly; integer work (dp4a chains, scales, sign
// and codebook decode) is exact by construction.  What changes is who computes it and where operands come from: one warp
// per row, persistent blocks, the Q8_1 activation staged once per block in shared memory, the IQ3_S grid in shared memory.
//
// FTZ: this file is built with --use_fast_math (flush-to-zero, like native_mmvq.cu) while iq_kernels.cu (the old IQ3_S
// kernel) is not.  The difference cannot show: every float in these dot products is (fp16 x fp16) x integer, or a sum or
// rounding of such values, so it is an integer multiple of 2^-48 (fp16 values are multiples of 2^-24), and a nonzero
// one is >= 2^-48, far above the fp32 subnormal range: no input, product or sum is ever subnormal, so .ftz never fires.
// The fp16 -> fp32 conversions are the same non-flushing cvt.f32.f16 in both modes.
#include "strata/kernels/dense_pk.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/dp4a.hpp"

#include <cuda_runtime.h>

#if defined(STRATA_HIP_GFX906)
// gfx906 (wave64): not ported; every call takes the old kernels.
namespace strata::kernels {
bool dense_pk_mmvq(int, const void*, const void*, float*, int, int, int, void*) { return false; }
bool dense_pk_enabled() { return false; }
void dense_pk_set_enabled(bool) {}
bool dense_pk_fuse_enabled() { return false; }
void dense_pk_set_fuse_enabled(bool) {}
bool dense_pk_fused_supported(const MmvqMat*, int, int, int) { return false; }
void native_mmvq_fused(const MmvqMat* m, int count, const void* x_q8_1, int n_in, int ncols, void* stream) {
    for (int i = 0; i < count; ++i) native_mmvq(m[i].type, m[i].weights, x_q8_1, m[i].y, n_in, m[i].n_out, ncols, stream);
}
}  // namespace strata::kernels
#else
#include <cuda_fp16.h>

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>

#define IQ3S_FAST_SIGNS 1

namespace strata::kernels {
namespace {

constexpr int WARP = 32;
struct Q81 { half2 ds; int8_t qs[32]; };
static_assert(sizeof(Q81) == 36);

__device__ __align__(4) int8_t g_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffff, x, o, WARP);
    return x;
}
__device__ __forceinline__ int ld_b2(const uint8_t* p) {   // load_int_b2: two 16-bit loads, little-endian combine
    const uint16_t* x = reinterpret_cast<const uint16_t*>(p);
    int v = x[0] << 0;
    v |= x[1] << 16;
    return v;
}
// iq4_table_lookup with the table already in registers (same byte_perm network)
__device__ __forceinline__ int2 lookup_reg(int q4, uint32_t t0, uint32_t t1, uint32_t t2, uint32_t t3) {
    uint32_t tmp[2];
    const uint32_t sel = 0x32103210 | ((q4 & 0x88888888) >> 1);
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t shift = 16 * i;
        const uint32_t low = __byte_perm(t0, t1, q4 >> shift);
        const uint32_t high = __byte_perm(t2, t3, q4 >> shift);
        tmp[i] = __byte_perm(low, high, sel >> shift);
    }
    return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420), __byte_perm(tmp[0], tmp[1], 0x7531));
}
struct Tab { uint32_t t0, t1, t2, t3; };
__device__ __forceinline__ Tab iq4_tab() {
    const uint32_t* t = reinterpret_cast<const uint32_t*>(g_iq4nl);
    return Tab{t[0], t[1], t[2], t[3]};
}

// Staging: the activation is NC columns of nb32 Q8_1 blocks, contiguous (column j at x + j * nb32).  It is read as a flat array
// of 32-bit words with 16-byte loads (9 words per block: ds, qs[0..7]) and scattered into the format's layout by `put`.
template<class Put>
__device__ __forceinline__ void stage_flat(const Q81* __restrict__ x, int nblk, int tid, int nthr, Put put) {
    const int nwords = nblk * 9;
    const int4* x4 = reinterpret_cast<const int4*>(x);
    const int n4 = nwords / 4;
    for (int i = tid; i < n4; i += nthr) {
        const int4 v = x4[i];
        const int w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int word = 4 * i + k, B = word / 9, p = word - 9 * B;
            put(B, p, w[k]);
        }
    }
    for (int word = 4 * n4 + tid; word < nwords; word += nthr) {
        const int v = reinterpret_cast<const int*>(x)[word];
        const int B = word / 9, p = word - 9 * B;
        put(B, p, v);
    }
}
__device__ __forceinline__ float ds_lo(int w) { return __half2float(__ushort_as_half((unsigned short) (w & 0xffff))); }

// ============================================================================ formats
// kbx(vw, lane, it): the weight block the old kernel's thread (vw, lane) handles in its it-th iteration.
// iters(bpr): iterations of the old kernel's thread 0 (the one with the most).

// ---- IQ4_XS (native_mmvq.cu IQ4XSTraits: T = 8, BPI = 16, kqs = 4 * (tid % 8))
struct FIQ4XS {
    static constexpr int NVW = 4, DIV = 256, BLK = 136, ALIGN = 8;
    static constexpr int CH = 1;
    struct Raw { uint2 h; int2 a, b; };
    struct Ctx { Tab t; const int4* lo; const int4* hi; const float* d; int nb32; };
    __host__ __device__ static constexpr int kbx(int vw, int lane, int it) { return 4 * vw + (lane >> 3) + 16 * it; }
    __host__ __device__ static constexpr int iters(int bpr) { return (bpr + 15) / 16; }
    static size_t smem(int nc, int n_in) { return (size_t) nc * (n_in / 32) * 36; }
    __device__ static void load(Raw& r, const uint8_t* __restrict__ rowp, int kbx, int lane) {
        const uint8_t* b = rowp + (size_t) kbx * BLK;
        r.h = *reinterpret_cast<const uint2*>(b);
        const int2* q = reinterpret_cast<const int2*>(b + 8) + 2 * (lane & 7);
        r.a = q[0];
        r.b = q[1];
    }
    template<int NC>
    __device__ static Ctx stage(const Q81* __restrict__ x, unsigned char* sm, int nb32, int tid, int nthr) {
        const int nblk = NC * nb32;
        int4* lo = reinterpret_cast<int4*>(sm);
        int4* hi = lo + nblk;
        float* d = reinterpret_cast<float*>(hi + nblk);
        stage_flat(x, nblk, tid, nthr, [&](int B, int p, int v) {
            if (p == 0) d[B] = ds_lo(v);
            else if (p <= 4) reinterpret_cast<int*>(lo + B)[p - 1] = v;
            else reinterpret_cast<int*>(hi + B)[p - 5] = v;
        });
        return Ctx{iq4_tab(), lo, hi, d, nb32};
    }
    template<int NC>
    __device__ static void compute(const Raw& r, float (&P)[NC], const Ctx& c, int kbx, int lane, int nb32, bool valid) {
        const int kq = lane & 7, iqs = 4 * kq;
        const int2 v0 = lookup_reg(r.a.x, c.t.t0, c.t.t1, c.t.t2, c.t.t3), v1 = lookup_reg(r.a.y, c.t.t0, c.t.t1, c.t.t2, c.t.t3);
        const int2 v2 = lookup_reg(r.b.x, c.t.t0, c.t.t1, c.t.t2, c.t.t3), v3 = lookup_reg(r.b.y, c.t.t0, c.t.t1, c.t.t2, c.t.t3);
        const uint32_t sh = r.h.x >> 16;
        const uint32_t sl = (r.h.y >> (8 * (iqs / 8))) & 0xff;
        const int ls = int((sl >> (iqs & 0x04)) & 0x0f) | int(((sh >> (iqs / 2)) & 0x03) << 4);
        const float dw = __half2float(__ushort_as_half((unsigned short) (r.h.x & 0xffff)));
        const int b32 = kbx * 8 + kq;
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            const int idx = j * c.nb32 + b32;
            const int4 ul = c.lo[idx], uh = c.hi[idx];
            int sumi = 0;
            sumi = STRATA_DP4A(v0.x, ul.x, sumi); sumi = STRATA_DP4A(v0.y, uh.x, sumi);
            sumi = STRATA_DP4A(v1.x, ul.y, sumi); sumi = STRATA_DP4A(v1.y, uh.y, sumi);
            sumi = STRATA_DP4A(v2.x, ul.z, sumi); sumi = STRATA_DP4A(v2.y, uh.z, sumi);
            sumi = STRATA_DP4A(v3.x, ul.w, sumi); sumi = STRATA_DP4A(v3.y, uh.w, sumi);
            sumi *= ls - 32;
            const float d = dw * c.d[idx];
            { const float t_ = P[j] + d * sumi; P[j] = valid ? t_ : P[j]; }
        }
    }
};

// ---- Q5_K (Q5KTraits: T = 16, BPI = 8, kqs = 2 * (tid % 16)); block 176 B = 16-B aligned: dm + scales in one 16-byte load
__device__ __forceinline__ float q5_q8_dot_impl(const int* __restrict__ vl, const int* __restrict__ vh, const int* __restrict__ u,
                                                const uint8_t* __restrict__ sc, const uint8_t* __restrict__ m, const half2& dm5,
                                                const float* __restrict__ d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int vl0i = (vl[0] >> (4 * i)) & 0x0f0f0f0f;
        const int vl1i = (vl[1] >> (4 * i)) & 0x0f0f0f0f;
        const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;
        const int v0i = vl0i | vh0i;
        const int v1i = vl1i | vh1i;
        const int dot1 = STRATA_DP4A(v0i, u[2 * i], STRATA_DP4A(v1i, u[2 * i + 1], 0));
        const int dot2 = STRATA_DP4A(0x01010101, u[2 * i], STRATA_DP4A(0x01010101, u[2 * i + 1], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }
    const float2 dm5f = __half22float2(dm5);
    return dm5f.x * sumf_d - dm5f.y * sumf_m;
}
struct FQ5K {
    static constexpr int NVW = 4, DIV = 256, BLK = 176, ALIGN = 16;
    static constexpr int CH = 1;
    struct Raw { uint4 h; int ql0, ql1, qh0, qh1; };
    struct Ctx { const int2* pr; const float* d; int nb32; };   // pr[(j*2 + (b&1)) * nb32/2 + b/2][w] = (q[w], q[w+4])
    __host__ __device__ static constexpr int kbx(int vw, int lane, int it) { return 2 * vw + (lane >> 4) + 8 * it; }
    __host__ __device__ static constexpr int iters(int bpr) { return (bpr + 7) / 8; }
    static size_t smem(int nc, int n_in) { return (size_t) nc * (n_in / 32) * 36; }
    __device__ static void load(Raw& r, const uint8_t* __restrict__ rowp, int kbx, int lane) {
        const uint8_t* b = rowp + (size_t) kbx * BLK;
        const int kq = lane & 15, g = kq >> 2, w = kq & 3;
        r.h = *reinterpret_cast<const uint4*>(b);
        r.ql0 = *reinterpret_cast<const int*>(b + 48 + 32 * g + 4 * w);
        r.ql1 = *reinterpret_cast<const int*>(b + 48 + 32 * g + 4 * w + 16);
        r.qh0 = *reinterpret_cast<const int*>(b + 16 + 4 * w);
        r.qh1 = *reinterpret_cast<const int*>(b + 16 + 4 * w + 16);
    }
    template<int NC>
    __device__ static Ctx stage(const Q81* __restrict__ x, unsigned char* sm, int nb32, int tid, int nthr) {
        const int nblk = NC * nb32, half = nb32 / 2;
        int* pr = reinterpret_cast<int*>(sm);
        float* d = reinterpret_cast<float*>(sm + (size_t) nblk * 32);
        stage_flat(x, nblk, tid, nthr, [&](int B, int p, int v) {
            if (p == 0) { d[B] = ds_lo(v); return; }
            const int j = B / nb32, b = B - j * nb32, q = p - 1;
            pr[(((j * 2 + (b & 1)) * half + (b >> 1)) * 4 + (q & 3)) * 2 + (q >> 2)] = v;
        });
        return Ctx{reinterpret_cast<const int2*>(pr), d, nb32};
    }
    template<int NC>
    __device__ static void compute(const Raw& r, float (&P)[NC], const Ctx& c, int kbx, int lane, int nb32, bool valid) {
        const int kq = lane & 15, g = kq >> 2, w = kq & 3;
        const int bq8_offset = 2 * g;
        const int vl[2] = {r.ql0, r.ql1};
        const int vh[2] = {r.qh0 >> bq8_offset, r.qh1 >> bq8_offset};
        const int j2 = g, jm = j2 & 1;
        const uint32_t s0 = jm ? (r.h.y >> 16) : (r.h.y & 0xffff);
        const uint32_t s2 = jm ? (r.h.z >> 16) : (r.h.z & 0xffff);
        const uint32_t s4 = jm ? (r.h.w >> 16) : (r.h.w & 0xffff);
        const uint32_t hi = uint32_t(-int32_t(j2 >= 2));
        uint16_t aux[2];
        aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
        aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
        const uint8_t* sc = reinterpret_cast<const uint8_t*>(aux);
        half2 dm;
        memcpy(&dm, &r.h.x, 4);
        const int half = c.nb32 / 2;
        const int bh = kbx * 4 + g;   // (kbx * 8 + bq8_offset) / 2
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            const int2 p0 = c.pr[((j * 2 + 0) * half + bh) * 4 + w];
            const int2 p1 = c.pr[((j * 2 + 1) * half + bh) * 4 + w];
            const float2 dd = *reinterpret_cast<const float2*>(c.d + j * c.nb32 + 2 * bh);
            const int u[4] = {p0.x, p0.y, p1.x, p1.y};
            const float d8[2] = {dd.x, dd.y};
            { const float t_ = P[j] + q5_q8_dot_impl(vl, vh, u, sc, sc + 2, dm, d8); P[j] = valid ? t_ : P[j]; }
        }
    }
};

// ---- Q6_K (Q6KTraits: T = 32, BPI = 4, kqs = tid % 32); block 210 B, 2-byte aligned
__device__ __forceinline__ float q6_q8_dot_impl(int vl, int vh, const int* __restrict__ u, const int8_t* __restrict__ scales, float d,
                                                const float* __restrict__ d8) {
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int sc = scales[4 * i];
        const int vil = (vl >> (4 * i)) & 0x0f0f0f0f;
        const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
        const int vi = __vsubss4(vil | vih, 0x20202020);
        sumf += d8[i] * (STRATA_DP4A(vi, u[i], 0) * sc);
    }
    return d * sumf;
}
struct FQ6K {
    static constexpr int NVW = 4, DIV = 256, BLK = 210, ALIGN = 2;
    static constexpr int CH = 1;
    struct Raw { int vl, vh, sc0, sc4; float d; };
    struct Ctx { const int2* pr; const float2* d; };   // per (j, kbx): 4 pair slots x 8 words: (blk p, blk p+2), p in {0,1,4,5}
    __host__ __device__ static constexpr int kbx(int vw, int lane, int it) { return vw + 4 * it; }
    __host__ __device__ static constexpr int iters(int bpr) { return (bpr + 3) / 4; }
    static size_t smem(int nc, int n_in) { return (size_t) nc * (n_in / 32) * 36; }
    __device__ static void load(Raw& r, const uint8_t* __restrict__ rowp, int kbx, int lane) {
        const uint8_t* b = rowp + (size_t) kbx * BLK;
        const int iqs = lane;
        const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
        r.vl = ld_b2(b + 4 * iqs);
        r.vh = ld_b2(b + 128 + 4 * (8 * (iqs / 16) + iqs % 8));
        r.sc0 = (int8_t) b[192 + scale_offset];
        r.sc4 = (int8_t) b[192 + scale_offset + 4];
        r.d = __half2float(*reinterpret_cast<const half*>(b + 208));
    }
    template<int NC>
    __device__ static Ctx stage(const Q81* __restrict__ x, unsigned char* sm, int nb32, int tid, int nthr) {
        const int nblk = NC * nb32;
        int* pr = reinterpret_cast<int*>(sm);
        float* d = reinterpret_cast<float*>(sm + (size_t) nblk * 32);
        stage_flat(x, nblk, tid, nthr, [&](int B, int p, int v) {
            const int j = B / nb32, b = B - j * nb32, kb = b >> 3, rr = b & 7;
            const int pp = (rr & 2) ? rr - 2 : rr, slot = (rr & 2) ? 1 : 0;   // pair base p and which member
            const int pq = (pp & 1) | ((pp >> 2) << 1);
            const int base = (j * (nb32 >> 3) + kb) * 4 + pq;
            if (p == 0) d[base * 2 + slot] = ds_lo(v);
            else pr[(base * 8 + (p - 1)) * 2 + slot] = v;
        });
        return Ctx{reinterpret_cast<const int2*>(pr), reinterpret_cast<const float2*>(d)};
    }
    template<int NC>
    __device__ static void compute(const Raw& r, float (&P)[NC], const Ctx& c, int kbx, int lane, int nb32, bool valid) {
        const int iqs = lane;
        const int bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
        const int vh_shift = 2 * ((iqs % 16) / 8);
        const int vh = r.vh >> vh_shift;
        const int pq = (bq8_offset & 1) | ((bq8_offset >> 2) << 1);
        int vi[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int vil = (r.vl >> (4 * i)) & 0x0f0f0f0f;
            const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
            vi[i] = ((vil | vih | (int) 0x80808080) - 0x20202020) ^ (int) 0x80808080;   // == __vsubss4(vil | vih, 0x20202020): bytes are 0..63, no saturation
        }
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            const int base = (j * (nb32 >> 3) + kbx) * 4 + pq;
            const int2 uu = c.pr[base * 8 + (iqs % 8)];
            const float2 dd = c.d[base];
            const int u[2] = {uu.x, uu.y};
            const float d8[2] = {dd.x, dd.y};
            float sumf = 0.0f;
            sumf += d8[0] * (STRATA_DP4A(vi[0], u[0], 0) * r.sc0);
            sumf += d8[1] * (STRATA_DP4A(vi[1], u[1], 0) * r.sc4);
            { const float t_ = P[j] + r.d * sumf; P[j] = valid ? t_ : P[j]; }
        }
    }
};

// ---- IQ4_NL (SmallTraits<IQ4NLBlock, 4>: DIV = 32, T = 2, BPI = 64, kqs = 2 * (tid % 2)); block 18 B, 2-byte aligned
struct FIQ4NL {
    static constexpr int NVW = 4, DIV = 32, BLK = 18, ALIGN = 2;
    static constexpr int CH = 1;
    struct Raw { int q0, q1; float d; };
    struct Ctx { Tab t; const int4* q; const float* d; int nb32; };   // q[B*2 + h] = (q[2h], q[2h+1], q[2h+4], q[2h+5])
    __host__ __device__ static constexpr int kbx(int vw, int lane, int it) { return 16 * vw + (lane >> 1) + 64 * it; }
    __host__ __device__ static constexpr int iters(int bpr) { return (bpr + 63) / 64; }
    static size_t smem(int nc, int n_in) { return (size_t) nc * (n_in / 32) * 36; }
    __device__ static void load(Raw& r, const uint8_t* __restrict__ rowp, int kbx, int lane) {
        const uint8_t* b = rowp + (size_t) kbx * BLK;
        const int iqs = 2 * (lane & 1);
        r.q0 = ld_b2(b + 2 + 4 * iqs);
        r.q1 = ld_b2(b + 2 + 4 * (iqs + 1));
        r.d = __half2float(*reinterpret_cast<const half*>(b));
    }
    template<int NC>
    __device__ static Ctx stage(const Q81* __restrict__ x, unsigned char* sm, int nb32, int tid, int nthr) {
        const int nblk = NC * nb32;
        int* q = reinterpret_cast<int*>(sm);
        float* d = reinterpret_cast<float*>(sm + (size_t) nblk * 32);
        stage_flat(x, nblk, tid, nthr, [&](int B, int p, int v) {
            if (p == 0) { d[B] = ds_lo(v); return; }
            const int k = p - 1, hi = k >> 2, k4 = k & 3, h = k4 >> 1, e = k4 & 1;   // int k -> half h, slot e + 2*hi
            q[(B * 2 + h) * 4 + e + 2 * hi] = v;
        });
        return Ctx{iq4_tab(), reinterpret_cast<const int4*>(q), d, nb32};
    }
    template<int NC>
    __device__ static void compute(const Raw& r, float (&P)[NC], const Ctx& c, int kbx, int lane, int nb32, bool valid) {
        const int h = lane & 1;
        const int2 v0 = lookup_reg(r.q0, c.t.t0, c.t.t1, c.t.t2, c.t.t3), v1 = lookup_reg(r.q1, c.t.t0, c.t.t1, c.t.t2, c.t.t3);
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            const int B = j * c.nb32 + kbx;
            const int4 u = c.q[B * 2 + h];
            int sumi = 0;
            sumi = STRATA_DP4A(v0.x, u.x, sumi);
            sumi = STRATA_DP4A(v0.y, u.z, sumi);
            sumi = STRATA_DP4A(v1.x, u.y, sumi);
            sumi = STRATA_DP4A(v1.y, u.w, sumi);
            const float d = r.d * c.d[B];
            { const float t_ = P[j] + d * sumi; P[j] = valid ? t_ : P[j]; }
        }
    }
};

// ---- IQ3_S (iq_kernels.cu mmvq_multi_kernel<21>: one warp per row, call k = lane + 32 i, kbx = k / 8, iqs = 2 * (k % 8))
struct FIQ3S {
    static constexpr int NVW = 1, DIV = 256, BLK = 110, ALIGN = 2;
    static constexpr int CH = 4;
    struct Raw { int q0, q1, sg; uint8_t qh, scl; float d; };
    struct Ctx { const uint32_t* grid; const int4* lo; const int4* hi; const float* d; int nb32; };
    __host__ __device__ static constexpr int kbx(int, int lane, int it) { return (lane >> 3) + 4 * it; }
    __host__ __device__ static constexpr int iters(int bpr) { return (bpr * 8 + 31) / 32; }
    static size_t smem(int nc, int n_in) { return (size_t) nc * (n_in / 32) * 36 + 2048; }
    __device__ static void load(Raw& r, const uint8_t* __restrict__ rowp, int kbx, int lane) {
        const uint8_t* b = rowp + (size_t) kbx * BLK;
        const int sb = lane & 7;   // iqs = 2 * sb
        r.q0 = ld_b2(b + 2 + 8 * sb);
        r.q1 = ld_b2(b + 2 + 8 * sb + 4);
        r.qh = b[66 + sb];
        r.sg = ld_b2(b + 74 + 4 * sb);
        r.scl = b[106 + sb / 2];
        r.d = __half2float(*reinterpret_cast<const half*>(b));
    }
    template<int NC>
    __device__ static Ctx stage(const Q81* __restrict__ x, unsigned char* sm, int nb32, int tid, int nthr) {
        uint32_t* grid = reinterpret_cast<uint32_t*>(sm);
        for (int i = tid; i < 512; i += nthr) grid[i] = iq3s_grid[i];
        const int nblk = NC * nb32;
        int4* lo = reinterpret_cast<int4*>(sm + 2048);
        int4* hi = lo + nblk;
        float* d = reinterpret_cast<float*>(hi + nblk);
        stage_flat(x, nblk, tid, nthr, [&](int B, int p, int v) {
            if (p == 0) d[B] = ds_lo(v);
            else if (p <= 4) reinterpret_cast<int*>(lo + B)[p - 1] = v;
            else reinterpret_cast<int*>(hi + B)[p - 5] = v;
        });
        return Ctx{grid, lo, hi, d, nb32};
    }
    template<int NC>
    __device__ static void compute(const Raw& r, float (&P)[NC], const Ctx& c, int kbx, int lane, int nb32, bool valid) {
        const int sb = lane & 7, iqs = 2 * sb;
        const int2 qs_packed = make_int2(r.q0, r.q1);
        const uint8_t* qs = reinterpret_cast<const uint8_t*>(&qs_packed);
        const int qh = r.qh;
        const uint8_t* signs_packed_8 = reinterpret_cast<const uint8_t*>(&r.sg);
        int g[8];
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int2 grid_pos = make_int2(c.grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)], c.grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);
#if IQ3S_FAST_SIGNS
            // == the vcmpne4/vsub4 pair: sign bit k of the nibble -> byte k mask 0xff; grid bytes are odd 1..15, so (g ^ m) + (m & 1)
            // never carries out of a byte and equals the per-byte g - m of __vsub4(g ^ m, m)
            const uint32_t sp = signs_packed_8[l0 / 2];
            const uint32_t m0 = (((sp & 0x0f) * 0x00204081u) & 0x01010101u) * 0xffu;
            const uint32_t m1 = (((sp >> 4) * 0x00204081u) & 0x01010101u) * 0xffu;
            g[l0 + 0] = (int) (((uint32_t) grid_pos.x ^ m0) + (m0 & 0x01010101u));
            g[l0 + 1] = (int) (((uint32_t) grid_pos.y ^ m1) + (m1 & 0x01010101u));
#else
            const int signs0 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x03) << 7) | ((signs_packed_8[l0 / 2] & 0x0C) << 21), 0x00000000);
            const int signs1 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x30) << 3) | ((signs_packed_8[l0 / 2] & 0xC0) << 17), 0x00000000);
            g[l0 + 0] = __vsub4(grid_pos.x ^ signs0, signs0);
            g[l0 + 1] = __vsub4(grid_pos.y ^ signs1, signs1);
#endif
        }
        const int ls = 1 + 2 * ((r.scl >> ((iqs << 1) & 0x04)) & 0x0F);
        const int b32 = kbx * 8 + sb;
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            const int idx = j * c.nb32 + b32;
            const int4 ul = c.lo[idx], uh = c.hi[idx];
            const int u[8] = {ul.x, ul.y, ul.z, ul.w, uh.x, uh.y, uh.z, uh.w};
            int sumi = 0;
#pragma unroll
            for (int k = 0; k < 8; ++k) sumi = STRATA_DP4A(g[k], u[k], sumi);
            sumi *= ls;
            const float d = r.d * c.d[idx];
            { const float t_ = P[j] + d * sumi; P[j] = valid ? t_ : P[j]; }
        }
    }
};

// ---- Q6_K reading the plain (lo/hi/d) activation layout, so it can share a fused launch with IQ4_XS / IQ3_S
struct FQ6Kp : FQ6K {
    struct Ctx { const int4* lo; const int4* hi; const float* d; int nb32; };
    template<int NC>
    __device__ static void compute(const Raw& r, float (&P)[NC], const Ctx& c, int kbx, int lane, int nb32, bool valid) {
        const int iqs = lane;
        const int bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
        const int vh_shift = 2 * ((iqs % 16) / 8);
        const int vh = r.vh >> vh_shift;
        int vi[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int vil = (r.vl >> (4 * i)) & 0x0f0f0f0f;
            const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
            vi[i] = ((vil | vih | (int) 0x80808080) - 0x20202020) ^ (int) 0x80808080;
        }
        const int wq = iqs % 8;
        const int* base = reinterpret_cast<const int*>(wq < 4 ? c.lo : c.hi) + (wq & 3);
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            const int b0 = j * c.nb32 + kbx * 8 + bq8_offset;
            const int u[2] = {base[4 * b0], base[4 * (b0 + 2)]};
            const float d8[2] = {c.d[b0], c.d[b0 + 2]};
            float sumf = 0.0f;
            sumf += d8[0] * (STRATA_DP4A(vi[0], u[0], 0) * r.sc0);
            sumf += d8[1] * (STRATA_DP4A(vi[1], u[1], 0) * r.sc4);
            { const float t_ = P[j] + r.d * sumf; P[j] = valid ? t_ : P[j]; }
        }
    }
};
template<class F> __device__ __forceinline__ typename F::Ctx ctx_from_plain(const int4* lo, const int4* hi, const float* d, int nb32, const uint32_t* grid) {
    if constexpr (std::is_same_v<F, FIQ4XS>) return typename F::Ctx{iq4_tab(), lo, hi, d, nb32};
    else if constexpr (std::is_same_v<F, FIQ3S>) return typename F::Ctx{grid, lo, hi, d, nb32};
    else return typename F::Ctx{lo, hi, d, nb32};
}

// ============================================================================ the persistent kernel
// Warp gw takes rows gw, gw + nw, ...  A row is NCH chunks of CH old-kernel iterations; BPR (weight blocks per row) is a
// compile-time constant, so after unrolling every (iteration, virtual warp) slot is statically all-valid, never-valid or
// lane-dependent (then the load is clamped to a real block and the add is a select: no branch, the slots interleave).
// With PF the next chunk's raw weight loads (the next row's first chunk at the end of a row) are issued before the
// current chunk is decoded and multiplied.
template<class F, int BPR>
struct Shape {
    static constexpr int NI = F::iters(BPR), CH = F::CH < NI ? F::CH : NI, NCH = (NI + CH - 1) / CH, NB32 = BPR * F::DIV / 32;
};
template<class F, int BPR>
__device__ __forceinline__ void load_chunk(typename F::Raw (&R)[F::CH][F::NVW], const uint8_t* __restrict__ rowp, int ch, int lane) {
    using S = Shape<F, BPR>;
#pragma unroll
    for (int c = 0; c < S::CH; ++c)
#pragma unroll
        for (int vw = 0; vw < F::NVW; ++vw) {
            const int it = ch * S::CH + c;
            if (it < S::NI && F::kbx(vw, 0, it) < BPR) {
                const int kb = min(F::kbx(vw, lane, it), BPR - 1);
                F::load(R[c][vw], rowp, kb, lane);
            }
        }
}
template<class F, int BPR, int NC>
__device__ __forceinline__ void compute_chunk(const typename F::Raw (&R)[F::CH][F::NVW], float (&P)[F::NVW][NC], const typename F::Ctx& cx, int ch,
                                              int lane) {
    using S = Shape<F, BPR>;
#pragma unroll
    for (int c = 0; c < S::CH; ++c)
#pragma unroll
        for (int vw = 0; vw < F::NVW; ++vw) {
            const int it = ch * S::CH + c;
            if (it < S::NI && F::kbx(vw, 0, it) < BPR) {
                const int kr = F::kbx(vw, lane, it);
                F::template compute<NC>(R[c][vw], P[vw], cx, min(kr, BPR - 1), lane, S::NB32, kr < BPR);
            }
        }
}
template<class F, int NC, int WPB, int MINB, int BPR, bool PF>
__global__ void __launch_bounds__(WPB * 32, MINB) pk_kernel(const uint8_t* __restrict__ w, const Q81* __restrict__ x, float* __restrict__ y,
                                                            int n_out) {
    using S = Shape<F, BPR>;
    extern __shared__ __align__(16) unsigned char smem[];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    constexpr size_t row_bytes = (size_t) BPR * F::BLK;
    const int nw = gridDim.x * WPB;
    int row = blockIdx.x * WPB + wid;
    typename F::Raw R[F::CH][F::NVW], N[F::CH][F::NVW];
    if (row < n_out) load_chunk<F, BPR>(R, w + (size_t) row * row_bytes, 0, lane);
    const typename F::Ctx cx = F::template stage<NC>(x, smem, S::NB32, threadIdx.x, blockDim.x);
    __syncthreads();
    while (row < n_out) {
        const int nrow = row + nw;
        float P[F::NVW][NC];
#pragma unroll
        for (int vw = 0; vw < F::NVW; ++vw)
#pragma unroll
            for (int j = 0; j < NC; ++j) P[vw][j] = 0.0f;
#pragma unroll
        for (int ch = 0; ch < S::NCH; ++ch) {
            if (PF) {
                if (ch + 1 < S::NCH) load_chunk<F, BPR>(N, w + (size_t) row * row_bytes, ch + 1, lane);
                else if (nrow < n_out) load_chunk<F, BPR>(N, w + (size_t) nrow * row_bytes, 0, lane);
            }
            compute_chunk<F, BPR, NC>(R, P, cx, ch, lane);
            if (PF) {
#pragma unroll
                for (int c = 0; c < F::CH; ++c)
#pragma unroll
                    for (int vw = 0; vw < F::NVW; ++vw) R[c][vw] = N[c][vw];
            } else {
                if (ch + 1 < S::NCH) load_chunk<F, BPR>(R, w + (size_t) row * row_bytes, ch + 1, lane);
                else if (nrow < n_out) load_chunk<F, BPR>(R, w + (size_t) nrow * row_bytes, 0, lane);
            }
        }
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            float t = P[0][j];
#pragma unroll
            for (int vw = 1; vw < F::NVW; ++vw) t += P[vw][j];
            t = warp_sum(t);
            if (lane == j) y[(size_t) j * n_out + row] = t;
        }
        row = nrow;
    }
}


// ============================================================================ two matrices, one activation, one launch
// Rows [0, nA) are matrix A's, [nA, nA + nB) matrix B's; a warp takes rows gw, gw + nw, ... of the joint range.  Each row is
// computed exactly as by pk_kernel (same per-format arithmetic), so each output is bitwise the old kernel's.  A and B must share
// the activation layout: IQ4_XS + IQ3_S (the plain lo/hi/d layout; IQ3_S's ctx also carries the staged grid) or IQ4_XS + IQ4_XS.
template<class F, int BPR, int NC>
__device__ __forceinline__ void row_simple(const uint8_t* __restrict__ w, int row, int n_out, float* __restrict__ y, const typename F::Ctx& cx,
                                           int lane) {
    using S = Shape<F, BPR>;
    constexpr size_t row_bytes = (size_t) BPR * F::BLK;
    typename F::Raw R[F::CH][F::NVW];
    float P[F::NVW][NC];
#pragma unroll
    for (int vw = 0; vw < F::NVW; ++vw)
#pragma unroll
        for (int j = 0; j < NC; ++j) P[vw][j] = 0.0f;
#pragma unroll
    for (int ch = 0; ch < S::NCH; ++ch) {
        load_chunk<F, BPR>(R, w + (size_t) row * row_bytes, ch, lane);
        compute_chunk<F, BPR, NC>(R, P, cx, ch, lane);
    }
#pragma unroll
    for (int j = 0; j < NC; ++j) {
        float t = P[0][j];
#pragma unroll
        for (int vw = 1; vw < F::NVW; ++vw) t += P[vw][j];
        t = warp_sum(t);
        if (lane == j) y[(size_t) j * n_out + row] = t;
    }
}
template<class FA, class FB, class FC, int NC, int WPB, int BPR>
__global__ void __launch_bounds__(WPB * 32, (WPB <= 4 ? 4 : (WPB <= 8 ? 2 : 1)))
fused3_kernel(const uint8_t* __restrict__ wA, int nA, float* __restrict__ yA, const uint8_t* __restrict__ wB, int nB, float* __restrict__ yB,
              const uint8_t* __restrict__ wC, int nC, float* __restrict__ yC, const Q81* __restrict__ x) {
    extern __shared__ __align__(16) unsigned char smem[];
    constexpr bool GRID = std::is_same_v<FA, FIQ3S> || std::is_same_v<FB, FIQ3S> || std::is_same_v<FC, FIQ3S>;
    constexpr int NB32 = BPR * 8;
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    // the plain layout (+ the IQ3_S grid when a matrix needs it), staged once for all three matrices
    FIQ3S::Ctx p;
    if constexpr (GRID) p = FIQ3S::stage<NC>(x, smem, NB32, threadIdx.x, blockDim.x);
    else {
        const FIQ4XS::Ctx q = FIQ4XS::stage<NC>(x, smem, NB32, threadIdx.x, blockDim.x);
        p = FIQ3S::Ctx{nullptr, q.lo, q.hi, q.d, q.nb32};
    }
    const typename FA::Ctx ca = ctx_from_plain<FA>(p.lo, p.hi, p.d, p.nb32, p.grid);
    const typename FB::Ctx cb = ctx_from_plain<FB>(p.lo, p.hi, p.d, p.nb32, p.grid);
    const typename FC::Ctx cc = ctx_from_plain<FC>(p.lo, p.hi, p.d, p.nb32, p.grid);
    __syncthreads();
    const int nw = gridDim.x * WPB, n = nA + nB + nC;
    for (int r = blockIdx.x * WPB + wid; r < n; r += nw) {
        if (r < nA) row_simple<FA, BPR, NC>(wA, r, nA, yA, ca, lane);
        else if (r < nA + nB) row_simple<FB, BPR, NC>(wB, r - nA, nB, yB, cb, lane);
        else row_simple<FC, BPR, NC>(wC, r - nA - nB, nC, yC, cc, lane);
    }
}

// ============================================================================ host side
// Launch shape: 8 warps per block, 2 blocks per SM (16 resident warps per SM: more blocks than fit in registers make a
// second wave).  No dynamic shared memory above 48 KB is ever needed (the largest is 6 columns of n_in = 6144: 41.5 KB),
// so there is no cudaFuncSetAttribute and no per-launch host state: every launch is legal inside stream capture.
constexpr int PK_WPB = 8, PK_BPS = 2, PK_MAX_NCOLS = 6;
constexpr size_t PK_SMEM_MAX = 48 * 1024;

bool env_flag(const char* name) {   // unset or anything but "0": on
    const char* v = std::getenv(name);
    return v == nullptr || std::strcmp(v, "0") != 0;
}
bool g_pk_on = env_flag("STRATA_PK_MMVQ");
bool g_pk_fuse_on = env_flag("STRATA_PK_FUSE");

int sm_count() {
    static int cache[64] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) return 80;
    if (cache[dev] == 0) {
        int n = 0;
        cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
        cache[dev] = n > 0 ? n : 80;
    }
    return cache[dev];
}
bool aligned(const void* p, std::uintptr_t a) { return p != nullptr && reinterpret_cast<std::uintptr_t>(p) % a == 0; }
void announce() {   // once per process, so a log shows which path ran
    static bool done = false;
    if (done) return;
    done = true;
    std::fprintf(stderr, "strata dense_pk: STRATA_PK_MMVQ %s, STRATA_PK_FUSE %s\n", g_pk_on ? "on" : "off", g_pk_fuse_on ? "on" : "off");
}
bool weights_ok(int type, const void* w) {
    if (aligned(w, type == 13 ? 16 : type == 23 ? 8 : 2)) return true;
    static bool warned[64] = {};
    if (type >= 0 && type < 64 && !warned[type]) {
        warned[type] = true;
        std::fprintf(stderr, "strata dense_pk: type %d weights at %p are not aligned for the wide loads: old kernel\n", type, w);
    }
    return false;
}

// Where the new kernel is faster than the old one (bench/gemv_opus table_graph.txt) and instantiated.  Everything else
// keeps the old kernel: small matrices at ncols >= 2 (they gain only inside a fused launch), ssm_out-like n_in = 6144
// IQ4_XS at one column, Q6_K with few rows.
bool shape_ok(int type, int n_in, int n_out, int ncols) {
    switch (type) {
        case 23: return (n_in == 2560 && (n_out >= 2048 || ncols == 1)) || (n_in == 6144 && ncols >= 2);
        case 21: return n_in == 2560;
        case 20: return n_in == 640;
        case 14: return n_in == 6144 || (n_in == 2560 && n_out >= 2048);
        case 13: return n_in == 2560 && n_out >= 2048;
        default: return false;
    }
}

template<class F, int NC, int BPR>
void launch_one(const void* w, const void* x, float* y, int n_in, int n_out, cudaStream_t s) {
    const size_t sm = F::smem(NC, n_in);
    const int blocks = std::max(1, std::min((n_out + PK_WPB - 1) / PK_WPB, sm_count() * PK_BPS));
    pk_kernel<F, NC, PK_WPB, 2, BPR, true><<<blocks, PK_WPB * 32, sm, s>>>(static_cast<const uint8_t*>(w), static_cast<const Q81*>(x), y, n_out);
}
template<class F, int BPR>
void launch_nc(const void* w, const void* x, float* y, int n_in, int n_out, int ncols, cudaStream_t s) {
    switch (ncols) {
        case 1: launch_one<F, 1, BPR>(w, x, y, n_in, n_out, s); break;
        case 2: launch_one<F, 2, BPR>(w, x, y, n_in, n_out, s); break;
        case 3: launch_one<F, 3, BPR>(w, x, y, n_in, n_out, s); break;
        case 4: launch_one<F, 4, BPR>(w, x, y, n_in, n_out, s); break;
        case 5: launch_one<F, 5, BPR>(w, x, y, n_in, n_out, s); break;
        default: launch_one<F, 6, BPR>(w, x, y, n_in, n_out, s); break;
    }
}

template<class FA, class FB, class FC, int NC>
void launch_fused_nc(const MmvqMat* m, int count, const void* x, cudaStream_t s) {
    const int nA = m[0].n_out, nB = m[1].n_out, nC = count > 2 ? m[2].n_out : 0;
    const size_t sm = FIQ3S::smem(NC, 2560);
    const int blocks = std::max(1, std::min((nA + nB + nC + PK_WPB - 1) / PK_WPB, sm_count() * PK_BPS));
    fused3_kernel<FA, FB, FC, NC, PK_WPB, 10><<<blocks, PK_WPB * 32, sm, s>>>(
        static_cast<const uint8_t*>(m[0].weights), nA, m[0].y, static_cast<const uint8_t*>(m[1].weights), nB, m[1].y,
        count > 2 ? static_cast<const uint8_t*>(m[2].weights) : nullptr, nC, count > 2 ? m[2].y : nullptr, static_cast<const Q81*>(x));
}
template<class FA, class FB, class FC>
void launch_fused(const MmvqMat* m, int count, const void* x, int ncols, cudaStream_t s) {
    switch (ncols) {
        case 1: launch_fused_nc<FA, FB, FC, 1>(m, count, x, s); break;
        case 2: launch_fused_nc<FA, FB, FC, 2>(m, count, x, s); break;
        case 3: launch_fused_nc<FA, FB, FC, 3>(m, count, x, s); break;
        case 4: launch_fused_nc<FA, FB, FC, 4>(m, count, x, s); break;
        case 5: launch_fused_nc<FA, FB, FC, 5>(m, count, x, s); break;
        default: launch_fused_nc<FA, FB, FC, 6>(m, count, x, s); break;
    }
}

}  // namespace

bool dense_pk_enabled() { return g_pk_on; }
void dense_pk_set_enabled(bool on) { g_pk_on = on; }
bool dense_pk_fuse_enabled() { return g_pk_fuse_on; }
void dense_pk_set_fuse_enabled(bool on) { g_pk_fuse_on = on; }

bool dense_pk_mmvq(int type, const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream) {
    announce();
    if (!g_pk_on || ncols < 1 || ncols > PK_MAX_NCOLS || n_out <= 0 || !stream || !y) return false;
    if (ncols > 1 && !native_mmvq_multi_exact()) return false;   // the upstream (non-exact) layout is a different sum
    if (!shape_ok(type, n_in, n_out, ncols) || !aligned(x_q8_1, 16) || !weights_ok(type, weights)) return false;
    const auto s = static_cast<cudaStream_t>(stream);
    switch (type) {
        case 23: if (n_in == 2560) launch_nc<FIQ4XS, 10>(weights, x_q8_1, y, n_in, n_out, ncols, s);
                 else launch_nc<FIQ4XS, 24>(weights, x_q8_1, y, n_in, n_out, ncols, s);
                 break;
        case 21: launch_nc<FIQ3S, 10>(weights, x_q8_1, y, n_in, n_out, ncols, s); break;
        case 20: launch_nc<FIQ4NL, 20>(weights, x_q8_1, y, n_in, n_out, ncols, s); break;
        case 14: if (n_in == 2560) launch_nc<FQ6K, 10>(weights, x_q8_1, y, n_in, n_out, ncols, s);
                 else launch_nc<FQ6K, 24>(weights, x_q8_1, y, n_in, n_out, ncols, s);
                 break;
        case 13: launch_nc<FQ5K, 10>(weights, x_q8_1, y, n_in, n_out, ncols, s); break;
        default: return false;
    }
    return true;
}

bool dense_pk_fused_supported(const MmvqMat* m, int count, int n_in, int ncols) {
    announce();
    if (!g_pk_on || !g_pk_fuse_on || count < 2 || count > 3 || n_in != 2560 || ncols < 1 || ncols > PK_MAX_NCOLS) return false;
    if (ncols > 1 && !native_mmvq_multi_exact()) return false;
    for (int i = 0; i < count; ++i)
        if (m[i].n_out <= 0 || !m[i].y || !weights_ok(m[i].type, m[i].weights)) return false;
    if (count == 2) return m[0].type == 23 && (m[1].type == 21 || m[1].type == 23);
    return m[0].type == 23 && m[1].type == 23 && m[2].type == 14;
}

void native_mmvq_fused(const MmvqMat* m, int count, const void* x_q8_1, int n_in, int ncols, void* stream) {
    if (aligned(x_q8_1, 16) && stream && dense_pk_fused_supported(m, count, n_in, ncols)) {
        const auto s = static_cast<cudaStream_t>(stream);
        if (count == 3) launch_fused<FIQ4XS, FIQ4XS, FQ6Kp>(m, count, x_q8_1, ncols, s);
        else if (m[1].type == 21) launch_fused<FIQ4XS, FIQ3S, FIQ4XS>(m, count, x_q8_1, ncols, s);
        else launch_fused<FIQ4XS, FIQ4XS, FIQ4XS>(m, count, x_q8_1, ncols, s);
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) throw std::runtime_error(std::string("native_mmvq_fused: ") + cudaGetErrorString(e));
        return;
    }
    for (int i = 0; i < count; ++i) native_mmvq(m[i].type, m[i].weights, x_q8_1, m[i].y, n_in, m[i].n_out, ncols, stream);
}

}  // namespace strata::kernels
#endif  // STRATA_HIP_GFX906
