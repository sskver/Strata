// src/kernels/dense_pk_parity.cpp - dense_pk.cu against the kernels it replaces, bitwise, at the engine's real shapes.
//
//     build/dense_pk_parity [--quick]
//
// For every (format, n_in, n_out) the decode window launches and every width T = 1..6: the OLD path (native_mmvq with
// dense_pk switched off: native_mmvq.cu / iq_kernels.cu exactly as before) against the NEW one (dense_pk_mmvq called
// directly, which must accept the call where the dispatcher would route it); then each fused group (native_mmvq_fused)
// against the old per-matrix calls.  Data as mmvq_multi_parity: random weight bytes with every fp16 block scale rewritten
// to a normal fp16 (random bytes there give inf/NaN, which compare equal whatever produced them), activations quantized
// by native_quantize_q8_1 from normal floats.  A second pass uses TINY data (weight scales near the fp16 subnormal
// range, activations ~1e-6 so the Q8_1 scales are fp16 subnormals): it is where flush-to-zero would show if it could
// (dense_pk.cu is built with --use_fast_math, iq_kernels.cu is not).  Every output must be finite and bitwise equal.
// NEGATIVE CONTROL: the new kernel against the old one with multi_exact OFF (llama.cpp's generic layout, which groups the
// sums differently) at T = 5, 6 must find differences, or the comparison has no power.
#include "strata/kernels/dense_pk.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace K = strata::kernels;

namespace {

int blk_bytes(int t) { return t == 23 ? 136 : t == 13 ? 176 : t == 14 ? 210 : t == 20 ? 18 : 110; }
int blk_elems(int t) { return t == 20 ? 32 : 256; }
std::vector<int> scale_at(int t) { return t == 13 ? std::vector<int>{0, 2} : t == 14 ? std::vector<int>{208} : std::vector<int>{0}; }

uint16_t scale_half(std::mt19937& rng, bool tiny) {   // normal fp16, +-[2^-10, 2^-5) or (tiny) +-[2^-14, 2^-12)
    const uint32_t r = rng();
    const uint32_t e = tiny ? 1u + (r >> 10) % 2u : 5u + (r >> 10) % 5u;
    return (uint16_t) (((r >> 31) << 15) | (e << 10) | (r & 0x3ffu));
}

struct Mat { int type, n_in, n_out; void* w = nullptr; };

void make(Mat& m, unsigned seed, bool tiny) {
    const size_t nblk = (size_t) m.n_out * (m.n_in / blk_elems(m.type)), wb = nblk * blk_bytes(m.type);
    std::mt19937 rng(seed);
    std::vector<uint8_t> h(wb);
    for (auto& b : h) b = (uint8_t) rng();
    for (size_t k = 0; k < nblk; ++k)
        for (int at : scale_at(m.type)) { const uint16_t v = scale_half(rng, tiny); std::memcpy(&h[k * blk_bytes(m.type) + at], &v, 2); }
    if (!m.w) cudaMalloc(&m.w, wb);
    cudaMemcpy(m.w, h.data(), wb, cudaMemcpyHostToDevice);
}
void* make_x(int n_in, int T, unsigned seed, bool tiny, cudaStream_t s) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, tiny ? 1e-6f : 1.f);
    std::vector<float> x((size_t) n_in * T);
    for (auto& v : x) v = nd(rng);
    float* dx;
    void* xq;
    cudaMalloc(&dx, x.size() * 4);
    cudaMalloc(&xq, K::native_q8_1_bytes(n_in, T));
    cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
    K::native_quantize_q8_1(dx, xq, n_in, T, s);
    cudaStreamSynchronize(s);
    cudaFree(dx);
    return xq;
}
// differing outputs; non-finite outputs count as differing
long long diff(const float* a, const float* b, size_t n) {
    std::vector<uint32_t> ha(n), hb(n);
    cudaMemcpy(ha.data(), a, n * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(hb.data(), b, n * 4, cudaMemcpyDeviceToHost);
    long long d = 0;
    for (size_t i = 0; i < n; ++i) {
        float f;
        std::memcpy(&f, &ha[i], 4);
        d += ha[i] != hb[i] || !std::isfinite(f);
    }
    return d;
}

}  // namespace

int main(int argc, char** argv) {
    const bool quick = argc > 1 && !std::strcmp(argv[1], "--quick");
    cudaStream_t s;
    cudaStreamCreate(&s);
    std::vector<Mat> mats = {{23, 2560, 10240}, {23, 2560, 12288}, {23, 2560, 640}, {23, 2560, 512}, {23, 6144, 2560},
                             {21, 2560, 6144},  {21, 2560, 640},   {20, 640, 2560},  {14, 6144, 2560}, {14, 2560, 512}};
    if (!quick) mats.push_back({13, 2560, 248320});   // the output head, 437 MB
    long long bad = 0, compared = 0, control = 0;
    int routed = 0, kept = 0;
    for (int pass = 0; pass < 2; ++pass) {
        const bool tiny = pass == 1;
        std::printf("\n=== %s data ===\n%-8s %6s %7s %3s %10s %8s\n", tiny ? "TINY (fp16-subnormal scales)" : "normal", "type", "n_in", "n_out", "T",
                    "outputs", "differ");
        for (Mat& m : mats) {
            make(m, 77u + (unsigned) m.type * 31u + (unsigned) m.n_out + (tiny ? 5u : 0u), tiny);
            for (int T = 1; T <= 6; ++T) {
                void* xq = make_x(m.n_in, T, 1000u + T + (unsigned) m.n_in, tiny, s);
                float *yo, *yn;
                const size_t n = (size_t) m.n_out * T;
                cudaMalloc(&yo, n * 4);
                cudaMalloc(&yn, n * 4);
                cudaMemset(yn, 0xff, n * 4);
                K::dense_pk_set_enabled(false);
                K::native_mmvq(m.type, m.w, xq, yo, m.n_in, m.n_out, T, s);
                K::dense_pk_set_enabled(true);
                const bool took = K::dense_pk_mmvq(m.type, m.w, xq, yn, m.n_in, m.n_out, T, s);
                if (cudaStreamSynchronize(s) != cudaSuccess) { std::printf("CUDA error\n"); return 1; }
                if (!took) {
                    ++kept;
                    std::printf("%-8d %6d %7d %3d %10s %8s  old kernel kept (not faster here)\n", m.type, m.n_in, m.n_out, T, "-", "-");
                } else {
                    ++routed;
                    const long long d = diff(yo, yn, n);
                    bad += d;
                    compared += (long long) n;
                    long long c = -1;
                    if (!tiny && T >= 5) {   // negative control: the generic (non-exact) layout
                        K::native_mmvq_set_multi_exact(false);
                        K::dense_pk_set_enabled(false);
                        K::native_mmvq(m.type, m.w, xq, yo, m.n_in, m.n_out, T, s);
                        cudaStreamSynchronize(s);
                        K::native_mmvq_set_multi_exact(true);
                        K::dense_pk_set_enabled(true);
                        c = diff(yo, yn, n);
                        control += c;
                    }
                    std::printf("%-8d %6d %7d %3d %10zu %8lld  %s", m.type, m.n_in, m.n_out, T, n, d, d ? "FAIL" : "ok");
                    if (c >= 0) std::printf("   control (non-exact layout) differs in %lld", c);
                    std::printf("\n");
                }
                cudaFree(yo); cudaFree(yn); cudaFree(xq);
            }
        }
        // fused groups vs the old per-matrix calls
        struct G { const char* name; std::vector<int> idx; };
        const G groups[] = {{"gdn qkv+gate", {0, 5}}, {"shexp gate+up", {2, 6}}, {"qsa q+k+v", {1, 3, 9}}};
        for (const G& g : groups) {
            for (int T = 1; T <= 6; ++T) {
                void* xq = make_x(2560, T, 2000u + T, tiny, s);
                std::vector<K::MmvqMat> fm;
                std::vector<float*> yo, yn;
                for (int i : g.idx) {
                    float *a, *b;
                    cudaMalloc(&a, (size_t) mats[i].n_out * T * 4);
                    cudaMalloc(&b, (size_t) mats[i].n_out * T * 4);
                    cudaMemset(b, 0xff, (size_t) mats[i].n_out * T * 4);
                    yo.push_back(a); yn.push_back(b);
                    K::dense_pk_set_enabled(false);
                    K::native_mmvq(mats[i].type, mats[i].w, xq, a, 2560, mats[i].n_out, T, s);
                    K::dense_pk_set_enabled(true);
                    fm.push_back({mats[i].type, mats[i].w, b, mats[i].n_out});
                }
                const bool sup = K::dense_pk_fused_supported(fm.data(), (int) fm.size(), 2560, T);
                K::native_mmvq_fused(fm.data(), (int) fm.size(), xq, 2560, T, s);
                cudaStreamSynchronize(s);
                long long d = 0, n = 0;
                for (size_t k = 0; k < g.idx.size(); ++k) {
                    d += diff(yo[k], yn[k], (size_t) mats[g.idx[k]].n_out * T);
                    n += (long long) mats[g.idx[k]].n_out * T;
                }
                bad += d;
                compared += n;
                if (!sup) ++bad;   // the engine's groups must take the fused launch
                std::printf("fused %-14s T=%d %10lld outputs, %lld differ  %s%s\n", g.name, T, n, d, d ? "FAIL" : "ok", sup ? "" : "  NOT FUSED");
                for (auto p : yo) cudaFree(p);
                for (auto p : yn) cudaFree(p);
                cudaFree(xq);
            }
        }
    }
    for (Mat& m : mats) cudaFree(m.w);
    std::printf("\n%lld outputs compared (%d single-matrix cases routed to dense_pk, %d kept on the old kernel), %lld differ; "
                "negative control found %lld differences\n", compared, routed, kept, bad, control);
    if (bad || control == 0) {
        std::printf("dense_pk_parity: FAILED%s\n", control == 0 ? " (negative control has no power)" : "");
        return 1;
    }
    std::printf("dense_pk_parity: ok\n");
    return 0;
}
