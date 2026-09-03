// Implements every function of include/fa/attention.cuh. The only file that knows which
// stage kernels exist in this build.
//
// WHY a registry: the harness, the PyTorch op and the tests are written once against the
// frozen API; each stage lands by (1) adding its .cu to sources.txt and (2) replacing ONE
// nullptr pair in kStages below. Nothing else in the repo changes between stages, which is
// what makes the S1..S5 ladder comparable.
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "fa/attention.cuh"
#include "fa/cuda_check.cuh"

namespace fa {

// ---------------------------------------------------------------------------------------
// Stage ABI. Every src/stageN_*.cu defines these two functions in its own namespace with
// EXACTLY these signatures (a mismatch is a link error, which is the point):
//   forward():  validated inputs, launches on `stream`, returns cudaGetLastError()-style
//               status, never synchronises. `dbg` carries FA_FORCE_BR/BC (0 = default),
//               `sm` is major*10+minor of the current device (75 on T4).
//   supports(): the stage's own (dtype, D, sm) gate; it is ANDed with contract_allows().
// ---------------------------------------------------------------------------------------
using StageForwardFn = cudaError_t (*)(Dtype dtype, const void* q, const void* k, const void* v,
                                       void* o, float* lse, void* workspace, const Params& p,
                                       const DebugOverrides& dbg, int sm, cudaStream_t stream);
using StageSupportsFn = bool (*)(Dtype dtype, int D, int sm);

#define FA_DECLARE_STAGE(ns)                                                                   \
  namespace ns {                                                                               \
  cudaError_t forward(Dtype dtype, const void* q, const void* k, const void* v, void* o,       \
                      float* lse, void* workspace, const Params& p, const DebugOverrides& dbg, \
                      int sm, cudaStream_t stream);                                            \
  bool supports(Dtype dtype, int D, int sm);                                                   \
  }
FA_DECLARE_STAGE(stage1)  // src/stage1_naive.cu   fa_s1_qk / fa_s1_softmax / fa_s1_pv
FA_DECLARE_STAGE(stage2)  // src/stage2_tiled.cu   fa_s2_qk_tiled / fa_s2_softmax / fa_s2_pv_tiled
FA_DECLARE_STAGE(stage3)  // src/stage3_fused.cu   fa_s3_fused
FA_DECLARE_STAGE(stage4)  // src/stage4_wmma.cu    fa_s4_wmma
FA_DECLARE_STAGE(stage5)  // src/stage5_tuned.cu   fa_s5_tuned
#undef FA_DECLARE_STAGE

struct StageEntry {
  Impl impl;
  StageForwardFn forward;    // nullptr = not implemented in this build
  StageSupportsFn supports;  // nullptr = never supported
};

// ======================= STAGE REGISTRY (one line per stage) =============================
// Implemented: S1 (Impl::Naive), S2 (Impl::Tiled), S3 (Impl::Fused). Unregistered stages (nullptr pairs) make
// forward() return cudaErrorNotSupported and impl_supports() false. When stage N lands, replace
// its pair with {Impl::X, stageN::forward, stageN::supports} and add src/stageN_*.cu to
// sources.txt.
static const StageEntry kStages[] = {
    {Impl::Naive, stage1::forward, stage1::supports},  // S1: src/stage1_naive.cu
    {Impl::Tiled, stage2::forward, stage2::supports},  // S2: src/stage2_tiled.cu
    {Impl::Fused, stage3::forward, stage3::supports},  // S3: src/stage3_fused.cu
    {Impl::Wmma, nullptr, nullptr},   // S4 -> {Impl::Wmma,  stage4::forward, stage4::supports}
    {Impl::Tuned, nullptr, nullptr},  // S5 -> {Impl::Tuned, stage5::forward, stage5::supports}
};
// ========================================================================================

namespace {

const StageEntry* find_stage(Impl impl) {
  for (const StageEntry& e : kStages) {
    if (e.impl == impl) return &e;
  }
  return nullptr;
}

// Contract-level gate (attention.cuh comments): D in {64,128}; sm_75+; Naive/Tiled/Fused take
// fp32/fp16; Wmma/Tuned take fp16 (sm_75+) or bf16 (sm_80+: no bf16 tensor cores on Turing).
bool contract_allows(Impl impl, Dtype dtype, int D, int sm) {
  if (D != 64 && D != 128) return false;
  if (sm < 75) return false;
  switch (impl) {
    case Impl::Naive:
    case Impl::Tiled:
    case Impl::Fused:
      return dtype == Dtype::F32 || dtype == Dtype::F16;
    case Impl::Wmma:
    case Impl::Tuned:
      return dtype == Dtype::F16 || (dtype == Dtype::BF16 && sm >= 80);
  }
  return false;
}

bool verbose() {
  static const bool v = std::getenv("FA_VERBOSE") != nullptr;
  return v;
}

// Returns nullptr when valid, else a reason (printed only under FA_VERBOSE=1 because the
// API contract is "never aborts, never spams; callers decide").
const char* validate(Dtype dtype, const void* q, const void* k, const void* v, const void* o,
                     const float* lse, const Params& p, Impl impl) {
  if (p.B <= 0 || p.H <= 0 || p.Hkv <= 0 || p.N <= 0 || p.D <= 0) return "B,H,Hkv,N,D must be > 0";
  if (p.D != 64 && p.D != 128) return "D must be 64 or 128";
  if (p.N % 32 != 0) return "N % 32 != 0 (ragged N is a documented in-window limitation)";
  if (p.H % p.Hkv != 0) return "H % Hkv != 0";
  if (q == nullptr || k == nullptr || v == nullptr || o == nullptr)
    return "q/k/v/o must be non-null";
  if (dtype_size(dtype) == 0) return "unknown dtype";
  // Fused/Wmma/Tuned always write lse (needed by split-KV and backward later); Naive/Tiled may
  // skip it, so nullptr is only legal for those.
  if (lse == nullptr && impl != Impl::Naive && impl != Impl::Tiled) return "lse must be non-null";
  return nullptr;
}

cudaError_t current_sm(int* sm) {
  int dev = 0, major = 0, minor = 0;
  FA_CUDA_TRY(cudaGetDevice(&dev));
  FA_CUDA_TRY(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));
  FA_CUDA_TRY(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev));
  *sm = major * 10 + minor;
  return cudaSuccess;
}

// Parses a positive integer from the environment; 0 when unset, empty or malformed
// (malformed values are reported once so a typo in FA_FORCE_BC cannot silently test nothing).
int env_positive_int(const char* name) {
  const char* s = std::getenv(name);
  if (s == nullptr || *s == '\0') return 0;
  char* end = nullptr;
  const long v = std::strtol(s, &end, 10);
  if (end == s || *end != '\0' || v <= 0 || v > (1 << 20)) {
    std::fprintf(stderr, "fa: ignoring %s='%s' (expected a positive integer)\n", name, s);
    return 0;
  }
  return static_cast<int>(v);
}

DebugOverrides g_overrides;  // set_debug_overrides() state; env values layer on top

}  // namespace

cudaError_t forward(Impl impl, Dtype dtype, const void* q, const void* k, const void* v, void* o,
                    float* lse, void* workspace, const Params& p, cudaStream_t stream) {
  if (const char* why = validate(dtype, q, k, v, o, lse, p, impl)) {
    if (verbose()) std::fprintf(stderr, "fa::forward: invalid params: %s\n", why);
    return cudaErrorInvalidValue;
  }
  int sm = 0;
  FA_CUDA_TRY(current_sm(&sm));
  const StageEntry* e = find_stage(impl);
  if (e == nullptr || e->forward == nullptr || !impl_supports(impl, dtype, p.D, sm)) {
    if (verbose()) {
      std::fprintf(stderr, "fa::forward: %s/%s/D=%d not supported on sm_%d in this build\n",
                   impl_name(impl), dtype_name(dtype), p.D, sm);
    }
    return cudaErrorNotSupported;
  }
  if (workspace_bytes(impl, dtype, p) > 0 && workspace == nullptr) {
    if (verbose()) std::fprintf(stderr, "fa::forward: workspace required but nullptr\n");
    return cudaErrorInvalidValue;
  }
  return e->forward(dtype, q, k, v, o, lse, workspace, p, get_debug_overrides(true), sm, stream);
}

std::size_t workspace_bytes(Impl impl, Dtype /*dtype*/, const Params& p) {
  // Materialising stages keep fp32 S and P regardless of the input dtype (contract).
  if (impl != Impl::Naive && impl != Impl::Tiled) return 0;
  const std::size_t n = static_cast<std::size_t>(p.N);
  return 2u * static_cast<std::size_t>(p.B) * static_cast<std::size_t>(p.H) * n * n * sizeof(float);
}

bool impl_supports(Impl impl, Dtype dtype, int D, int sm) {
  if (!contract_allows(impl, dtype, D, sm)) return false;
  const StageEntry* e = find_stage(impl);
  if (e == nullptr || e->supports == nullptr) return false;
  return e->supports(dtype, D, sm);
}

void set_debug_overrides(const DebugOverrides& o) {
  g_overrides = o;
}

DebugOverrides get_debug_overrides(bool from_env) {
  DebugOverrides r = g_overrides;
  if (from_env) {
    // Environment wins so the tile-invariance tests can steer a process without code changes.
    if (const int br = env_positive_int("FA_FORCE_BR")) r.block_m = br;
    if (const int bc = env_positive_int("FA_FORCE_BC")) r.block_n = bc;
  }
  return r;
}

const char* impl_name(Impl impl) {
  switch (impl) {
    case Impl::Naive:
      return "naive";
    case Impl::Tiled:
      return "tiled";
    case Impl::Fused:
      return "fused";
    case Impl::Wmma:
      return "wmma";
    case Impl::Tuned:
      return "tuned";
  }
  return "unknown";
}

const char* dtype_name(Dtype dtype) {
  switch (dtype) {
    case Dtype::F32:
      return "fp32";
    case Dtype::F16:
      return "fp16";
    case Dtype::BF16:
      return "bf16";
  }
  return "unknown";
}

std::size_t dtype_size(Dtype dtype) {
  switch (dtype) {
    case Dtype::F32:
      return 4;
    case Dtype::F16:
      return 2;
    case Dtype::BF16:
      return 2;
  }
  return 0;
}

}  // namespace fa
