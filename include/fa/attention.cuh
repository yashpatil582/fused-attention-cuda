// fused-attention-cuda — public kernel API.
//
// FROZEN CONTRACT (S0, 2026-09-02): every stage implements this interface so the
// benchmark harness, the PyTorch op and the tests never change between stages.
// Adding fields is allowed only at the end of structs; renaming is not.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace fa {

// Element type of q/k/v/o. Scores, running max/sum and the O accumulator are
// ALWAYS fp32 inside every kernel regardless of Dtype (fp16 overflow safety).
enum class Dtype : int { F32 = 0, F16 = 1, BF16 = 2 };

// One value per optimization stage. Numbers match the README/STAGES.md ladder.
enum class Impl : int {
  Naive = 1,  // S1: three kernels (QK^T, row softmax, PV), fp32, global memory only
  Tiled = 2,  // S2: shared-memory tiled GEMMs + coalesced loads, S/P still materialized
  Fused = 3,  // S3: single kernel, online softmax, never materializes the NxN scores
  Wmma = 4,   // S4: fused + tensor cores (nvcuda::wmma), fp16/bf16 in, fp32 accumulate
  Tuned = 5,  // S5: S4 + Nsight-guided tuning (padding/swizzle, shuffles, launch bounds, prefetch)
};

struct Params {
  int B;        // batch size
  int H;        // number of query heads
  int Hkv;      // number of key/value heads; H % Hkv == 0 (Hkv == H unless GQA/MQA)
  int N;        // sequence length; Nq == Nkv in-window; N % 32 == 0 enforced by S1..S5
  int D;        // head dimension: 64 or 128
  float scale;  // softmax scale applied to QK^T; PyTorch default is 1/sqrt(D)
  bool causal;  // top-left aligned causal mask (matches torch SDPA is_causal for Nq == Nkv)
};

// Memory layout of q, k, v, o: [B, H, N, D] contiguous, D fastest (k/v use Hkv heads).
// lse: [B, H, N] fp32 logsumexp per query row = m*scale + log(l); may be nullptr for
//      Naive/Tiled; Fused/Wmma/Tuned always write it (needed later by split-KV/backward).
// workspace: device buffer of at least workspace_bytes(); nullptr allowed when that is 0.
//
// Launch is asynchronous on `stream`. Returns cudaSuccess, or the first CUDA error
// (cudaErrorNotSupported when (impl, dtype, D, arch) is not implemented — see
// impl_supports()). Never aborts; callers decide.
cudaError_t forward(Impl impl, Dtype dtype, const void* q, const void* k, const void* v,
                    void* o, float* lse, void* workspace, const Params& p,
                    cudaStream_t stream);

// Bytes of scratch memory `forward` needs. Naive/Tiled: fp32 S and P, i.e.
// 2 * B * H * N * N * sizeof(float). Fused/Wmma/Tuned: 0.
std::size_t workspace_bytes(Impl impl, Dtype dtype, const Params& p);

// True when this build can run (impl, dtype, D) on a device of the given compute
// capability (major*10+minor, e.g. 75 for T4). Wmma/Tuned need F16 (sm_75+) or
// BF16 (sm_80+); Naive/Tiled/Fused support F32 and F16.
bool impl_supports(Impl impl, Dtype dtype, int D, int sm);

// Tile-size overrides for the tile-invariance tests (0 = kernel default). Reads
// FA_FORCE_BR / FA_FORCE_BC from the environment when `from_env` is true.
struct DebugOverrides {
  int block_m = 0;
  int block_n = 0;
};
void set_debug_overrides(const DebugOverrides& o);
DebugOverrides get_debug_overrides(bool from_env = true);

const char* impl_name(Impl impl);    // "naive", "tiled", "fused", "wmma", "tuned"
const char* dtype_name(Dtype dtype);  // "fp32", "fp16", "bf16"
std::size_t dtype_size(Dtype dtype);  // 4, 2, 2

}  // namespace fa
