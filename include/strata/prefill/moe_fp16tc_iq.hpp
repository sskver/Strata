// include/strata/prefill/moe_fp16tc_iq.hpp - sm_70 (Volta) FP16 tensor-core gate/up for the native i-quant experts.
//
// Volta's MMQ path runs the i-quant gate/up products on CUDA-core dp4a (about 23 TOPS).  This module decodes the same
// GGUF blocks (IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS / IQ3_S, with moe_fused_iq.cu's decoders) into FP16 and runs the
// products on the FP16 tensor cores with FP32 accumulation.  One thread block takes up to 192 routed rows of one
// expert, so the weights are decoded once per expert instead of once per 64-row tile, and a block's producer warps
// (load + decode) run concurrently with its consumer warps (MMA).
//
// The activations are the q8_1 values MMQ reads (the same layout); the weights are the dequantized llama.cpp values
// (code * scale, rounded to FP16); the products accumulate in FP32.  The result is not bit-identical to MMQ (which
// accumulates int8 products exactly and applies the scales in FP32); measured against MMQ the relative RMS difference
// is 4.2e-4 on every type, below the 1e-2 that the shared q8_1 activation rounding contributes against an FP32
// reference.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::prefill::fp16tc_iq {

#ifdef STRATA_PREFILL_FP16TC

/// The kernels are in this build.
bool built();
/// The current device is compute capability 7.0 (Volta) and the kernels are built.
bool available();
/// A gate/up matrix of this ggml type, n_embd (K) and n_ff (rows / 2) runs on the kernel: one of the five i-quants,
/// n_embd 2560 and n_ff 640 (the geometry the kernel is compiled for).
bool supported(int gu_type, int64_t n_embd, int64_t n_ff);

/// Gate/up of `n_experts` experts into `dst` ([rows][2 n_ff], gate at [0, n_ff), up at [n_ff, 2 n_ff)).  Expert i's
/// weights start at `w + i * expert_bytes` ([2 n_ff rows][K] in the GGUF blocks of `gu_type`, gate rows then up
/// rows); it reads rows [bounds[i], bounds[i+1]) of `xq` (the layer's q8_1 in MMQ's layout, `xq_rows` rows, absolute
/// sorted rows) and writes the same rows of `dst`.  `max_rows` is the largest row count of the experts (the launch
/// grid).  Returns false, launching nothing, when the arguments are outside what the kernel handles (unsupported
/// type, a pointer that is not 16-byte aligned, ld_dst != 2 n_ff): the caller then uses MMQ.
bool gu(int gu_type, const void* w, size_t expert_bytes, int n_experts, int64_t max_rows, const int32_t* bounds,
        const void* xq, int64_t xq_rows, float* dst, int64_t ld_dst, void* stream);
/// gu() with the SwiGLU in its epilogue: writes H = silu(gate) * up into `dst` ([rows][n_ff], ld_dst == n_ff) instead
/// of the gate/up product, with mmq::swiglu's formula (non-interleaved), so the result is bit-identical to gu() then
/// mmq::swiglu(interleaved = false).  The same weights, `bounds` (absolute rows, the same rows of `dst`), activations
/// and `max_rows` as gu().  Returns false, launching nothing, under gu()'s conditions (ld_dst != n_ff here): the caller
/// then runs gu() (or MMQ) and mmq::swiglu.
bool gu_swiglu(int gu_type, const void* w, size_t expert_bytes, int n_experts, int64_t max_rows, const int32_t* bounds,
               const void* xq, int64_t xq_rows, float* dst, int64_t ld_dst, void* stream);
/// gu_swiglu() that also writes H's q8_1 into `hq`, byte-identical to mmq::quantize(H + r0 * n_ff, nullptr, hq,
/// hq_type, n_ff, n_ff, rows) of the launch's rows (r0 = bounds[0], rows = bounds[n_experts] - r0: hq is group-local,
/// its row 0 is the launch's first row, and the 512-padded tail blocks are written too).  `dst` may be null (no FP32
/// H).  Returns false, launching nothing, under gu_swiglu()'s conditions, or when hq is null or not 16-byte aligned,
/// or hq_type's MMQ q8_1 layout is not D4: the caller then quantizes H itself.
bool gu_swiglu_q8_1(int gu_type, const void* w, size_t expert_bytes, int n_experts, int64_t max_rows,
                    const int32_t* bounds, const void* xq, int64_t xq_rows, float* dst, int64_t ld_dst, int hq_type,
                    void* hq, void* stream);

#else  // !STRATA_PREFILL_FP16TC - no-ops so the caller needs no #ifdef.

inline bool built() { return false; }
inline bool available() { return false; }
inline bool supported(int, int64_t, int64_t) { return false; }
inline bool gu(int, const void*, size_t, int, int64_t, const int32_t*, const void*, int64_t, float*, int64_t, void*) {
    return false;
}
inline bool gu_swiglu(int, const void*, size_t, int, int64_t, const int32_t*, const void*, int64_t, float*, int64_t, void*) {
    return false;
}
inline bool gu_swiglu_q8_1(int, const void*, size_t, int, int64_t, const int32_t*, const void*, int64_t, float*, int64_t, int,
                           void*, void*) {
    return false;
}

#endif

}  // namespace strata::prefill::fp16tc_iq
