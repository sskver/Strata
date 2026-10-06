// include/strata/prefill/moe_fp16tc_down.hpp - sm_70 (Volta) FP16 tensor-core down projection for the routed experts.
//
// The prompt path's down product (per expert [n_embd = 2560 rows][n_ff = 640] times the group's SwiGLU outputs in
// q8_1) ran on CUDA-core dp4a MMQ for IQ4_NL and on fp16tc::down (64-row tiles, decode and MMA serialized) for Q2_0.
// This module decodes the GGUF blocks (Q2_0: (code - 1) * d; IQ4_NL: kvalues_iq4nl[nibble] * d) into FP16 and runs
// the products on the FP16 tensor cores with FP32 accumulation.  One thread block owns up to 192 routed rows of one
// expert x 128 weight rows per work item, so the weights are decoded once per 192 rows; a block's producer warps
// (load + decode + activation conversion) run concurrently with its consumer warps (MMA), and a block walks several
// items (persistent, one block per SM) so the pipeline does not drain between them.
//
// The activations are the q8_1 values MMQ reads (the same D4 layout, group-local rows); the products accumulate in
// FP32.  Not bit-identical to MMQ (which accumulates the int8 products exactly and scales in FP32): measured against
// MMQ the relative RMS difference is 2.9e-4 (Q2_0) and 3.6e-4 (IQ4_NL), well below what the shared q8_1 activation
// rounding contributes against an FP32 reference.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::prefill::fp16tc_down {

#ifdef STRATA_PREFILL_FP16TC

/// The kernels are in this build.
bool built();
/// The current device is compute capability 7.0 (Volta) and the kernels are built.
bool available();
/// A down matrix of this ggml type, n_embd (rows) and n_ff (K) runs on the kernel: Q2_0 (42) or IQ4_NL (20),
/// n_embd 2560 and n_ff 640 (the geometry the kernel is compiled for).
bool supported(int d_type, int64_t n_embd, int64_t n_ff);

/// Down of `n_experts` experts (one group) into `dst`.  Expert i's weights start at `w + i * expert_bytes`
/// ([n_embd rows][n_ff] in the GGUF blocks of `d_type`); it reads rows [bounds[i], bounds[i+1]) of `hq` (the group's
/// q8_1 in MMQ's layout, `hq_rows` rows, group-local: row 0 is the group's first row; `bounds` on the device, n+1
/// entries, group-relative) and writes dst rows dst_row_base + bounds[i] .. (ld_dst floats apart).  `max_rows` is the
/// largest row count of the experts.  Returns false, launching nothing, when the arguments are outside what the kernel
/// handles (unsupported type, more than 64 experts, a pointer that is not 16-byte aligned, ld_dst not a multiple of 4
/// or below n_embd, a row count beyond the per-block work list) and for light groups (max_rows under 64, where
/// fp16tc::down and MMQ measured as fast or faster): the caller then uses its existing path.
bool down(int d_type, const void* w, size_t expert_bytes, int n_experts, int64_t max_rows, const int32_t* bounds,
          const void* hq, int64_t hq_rows, float* dst, int64_t ld_dst, int64_t dst_row_base, void* stream);
/// down() on the experts' own down matrices, without gathering them first: expert i's matrix ([n_embd rows][n_ff] GGUF
/// blocks) starts at `down_ptrs[i]` (2-byte aligned; at most 64 experts).  The matrices must stay valid until the launch
/// finishes.  The same conditions and false returns as down() (nothing is launched when it returns false).
bool down_blobs(int d_type, const uint8_t* const* down_ptrs, int n_experts, int64_t max_rows, const int32_t* bounds,
                const void* hq, int64_t hq_rows, float* dst, int64_t ld_dst, int64_t dst_row_base, void* stream);

#else  // !STRATA_PREFILL_FP16TC - no-ops so the caller needs no #ifdef.

inline bool built() { return false; }
inline bool available() { return false; }
inline bool supported(int, int64_t, int64_t) { return false; }
inline bool down(int, const void*, size_t, int, int64_t, const int32_t*, const void*, int64_t, float*, int64_t, int64_t,
                 void*) {
    return false;
}
inline bool down_blobs(int, const uint8_t* const*, int, int64_t, const int32_t*, const void*, int64_t, float*, int64_t, int64_t,
                       void*) {
    return false;
}

#endif

}  // namespace strata::prefill::fp16tc_down
