#!/usr/bin/env bash
# Build and run the S4 WMMA microkernel on a GPU box (Colab T4 or the Track-B VM).
#
#   ./run.sh                         # sm_75 (T4), fp16, M=N=2048 K=64, 50 timed iters
#   ./run.sh 2048 2048 128           # K=128 (needs the >48 KB dynamic-smem opt-in; fits T4's 64 KB)
#   ARCH=sm_80 ./run.sh --bf16       # bf16 fragments exist only for sm_80+
#   NVCC_EXTRA="-DUSE_CUBLAS -lcublas" ./run.sh   # add the cuBLAS S-GEMM reference point
#
# -Xptxas -v prints registers / spill stores+loads per kernel: paste that block into
# docs/STAGES.md for S4 (zero spills is the gate). -lineinfo keeps ncu source correlation.
set -euo pipefail
cd "$(dirname "$0")"
ARCH="${ARCH:-sm_75}"
NVCC="${NVCC:-nvcc}"
NVCC_EXTRA="${NVCC_EXTRA:-}"
mkdir -p build  # build/ is gitignored repo-wide: the binary, ptxas log and any --dump land here

echo "== build: $NVCC -O3 -std=c++17 -lineinfo -arch=$ARCH -Xptxas -v $NVCC_EXTRA wmma_gemm.cu"
# shellcheck disable=SC2086
"$NVCC" -O3 -std=c++17 -lineinfo -arch="$ARCH" -Xptxas -v $NVCC_EXTRA wmma_gemm.cu \
  -o build/wmma_gemm 2>&1 | tee "build/ptxas_$ARCH.txt"

echo "== run: build/wmma_gemm $*"
build/wmma_gemm "$@"
