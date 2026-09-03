// CUDA error-handling macros.
//
// Two flavours on purpose:
//   CUDA_CHECK / CUDA_CHECK_LAST  abort with file:line — for the bench harness, tests and
//                                 tools, where a CUDA error means the process is worthless.
//   FA_CUDA_TRY                   return the error to the caller — for fa::forward and the
//                                 stage launchers, whose contract is "never aborts; callers
//                                 decide" (the PyTorch op turns it into a Python exception).
// Every launch is followed by CUDA_CHECK_LAST()/FA_CUDA_TRY_LAUNCH() because a bad launch
// configuration (too much dynamic smem, too many threads) is only reported by
// cudaGetLastError(), not by the <<<>>> expression itself.
#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(x)                                                                    \
  do {                                                                                   \
    cudaError_t fa_err__ = (x);                                                          \
    if (fa_err__ != cudaSuccess) {                                                       \
      std::fprintf(stderr, "CUDA error at %s:%d: %s (%d) in `%s`\n", __FILE__, __LINE__, \
                   cudaGetErrorString(fa_err__), static_cast<int>(fa_err__), #x);        \
      std::fflush(stderr);                                                               \
      std::abort();                                                                      \
    }                                                                                    \
  } while (0)

// After a kernel launch: catches launch-configuration errors and sticky errors from earlier
// asynchronous work. Does not synchronize.
#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())

// Non-aborting variant: evaluates `x` and returns its cudaError_t from the enclosing
// function when it is not cudaSuccess. The enclosing function must return cudaError_t.
#define FA_CUDA_TRY(x)             \
  do {                             \
    cudaError_t fa_err__ = (x);    \
    if (fa_err__ != cudaSuccess) { \
      return fa_err__;             \
    }                              \
  } while (0)

// Non-aborting launch check, to be placed right after a <<<>>> launch in a stage launcher.
#define FA_CUDA_TRY_LAUNCH() FA_CUDA_TRY(cudaGetLastError())
