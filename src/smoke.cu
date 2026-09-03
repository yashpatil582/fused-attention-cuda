// S0 smoke kernel: o[i] = v[i] * scale, fp32, grid-stride.
//
// WHY it exists: before any real kernel is written, the build -> run -> CSV -> git push
// round trip on Colab has to be proven end to end (cmake preset, -Xptxas -v capture, ctest,
// compute-sanitizer, fa_bench CSV row, gpu_run.sh push). It is also the kernel the sanitizer
// gate runs under `fa_bench --smoke` in every stage. Name `fa_smoke` is fixed by CONTRACT §2
// (ncu --kernel-name regex:fa_smoke), so it lives at global scope, not in a namespace.
#include <cuda_runtime.h>

#include <cstddef>

#include "fa/cuda_check.cuh"

__global__ void fa_smoke(const float* __restrict__ v, float* __restrict__ o, float scale,
                         std::size_t n) {
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += stride) {
    o[i] = v[i] * scale;
  }
}

namespace fa {
namespace smoke {

// Host launcher. Declared (not via a header) at the top of bench/bench_main.cu.
cudaError_t forward(const float* v, float* o, float scale, std::size_t n, cudaStream_t stream) {
  if (n == 0) return cudaSuccess;
  constexpr int kThreads = 256;
  // Enough blocks to cover a T4 (40 SMs x 16 blocks of 256) several times over; grid-stride
  // handles any n.
  const std::size_t want = (n + kThreads - 1) / kThreads;
  const int blocks = static_cast<int>(want < 4096 ? want : 4096);
  fa_smoke<<<blocks, kThreads, 0, stream>>>(v, o, scale, n);
  FA_CUDA_TRY_LAUNCH();
  return cudaSuccess;
}

}  // namespace smoke
}  // namespace fa
