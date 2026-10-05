#!/usr/bin/env bash
set -e
F=$HOME/strata-v100; L=$F/build/_deps/strata_llamacpp-src; CU=/nix/store/arma8qsncv5fy7rkfg73a9iwka4fh44m-cuda-merged-12.9
cd $HOME/strata/prof/bench_gu
nvcc -forward-unknown-to-host-compiler -DSTRATA_PREFILL_FP16TC=1 -I$L/ggml/include -I$L/ggml/src -I$L/ggml/src/ggml-cuda -I$F/include \
  -O3 -DNDEBUG -std=c++20 --generate-code=arch=compute_70,code=[compute_70,sm_70] --extended-lambda -use_fast_math -Xptxas -v -x cu -c bench_gu_v2.cu -o bench_gu_v2.o 2> ptxas_v2.log || { grep -v "^ptxas info" ptxas_v2.log | head -40; exit 1; }
g++ -O3 bench_gu_v2.o -o bench_gu_v2 -L/nix/store/m757fr2qv0vwl4xa83hwhzjldcvckvk4-cuda12.9-cuda_nvcc-12.9.86/nvvm/lib $F/build/libstrata_mmq.a \
  $F/build/ggml/src/libggml-base.a $CU/lib/libcudart.so $CU/lib/libcublas.so -lm -ldl $CU/lib/libcublasLt.so $CU/lib/libculibos.a -lcudadevrt -lcudart_static -lrt -lpthread -ldl
echo BUILD_OK
