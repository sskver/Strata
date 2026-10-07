// include/strata/kernels/dense_pk.hpp - persistent decode GEMVs for the dense projections (src/kernels/cuda/dense_pk.cu).
//
// native_mmvq() routes IQ4_XS, IQ3_S, Q5_K, Q6_K and IQ4_NL calls here when the shape is one of the instantiated,
// measured-faster ones (n_in 2560 / 6144 / 640, ncols 1..6, see shape_ok in the .cu), the pointers are aligned for the
// wide loads (activation 16 B, IQ4_XS weights 8 B, Q5_K 16 B) and, for ncols > 1, native_mmvq_multi_exact() is on.
// The result is BITWISE the old kernel's (every column also bitwise a single-column call); anything else falls back.
//
// STRATA_PK_MMVQ=0 turns all of it off (read once at load); STRATA_PK_FUSE=0 turns off only the fused launches
// (native_mmvq_fused then makes one native_mmvq call per matrix, which may still take the single-matrix kernels).
// Set the switches before graph capture: captured graphs keep the kernels they captured.
#pragma once

namespace strata::kernels {

// Launches the new kernel and returns true, or returns false (nothing enqueued) when the call is not eligible.
bool dense_pk_mmvq(int ggml_type, const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream);

bool dense_pk_enabled();
void dense_pk_set_enabled(bool on);
bool dense_pk_fuse_enabled();
void dense_pk_set_fuse_enabled(bool on);

// One weight matrix of a fused call: y (n_out x ncols, column-major as native_mmvq) = W (ggml_type, n_out x n_in) * x.
struct MmvqMat {
    int type;
    const void* weights;
    float* y;
    int n_out;
};
// 2 or 3 matrices that read the SAME Q8_1 activation (n_in, ncols): one launch when the combination is supported
// (IQ4_XS + IQ3_S, IQ4_XS + IQ4_XS, IQ4_XS + IQ4_XS + Q6_K at n_in 2560), else one native_mmvq per matrix.  Each output
// is bitwise what native_mmvq gives for that matrix alone.
bool dense_pk_fused_supported(const MmvqMat* mats, int count, int n_in, int ncols);
void native_mmvq_fused(const MmvqMat* mats, int count, const void* x_q8_1, int n_in, int ncols, void* stream);

}  // namespace strata::kernels
