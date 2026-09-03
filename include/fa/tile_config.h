// Per-architecture capacity caps and the default tile shapes for every (ARCH, D).
//
// Plain C++17 header: no CUDA syntax, so host .cpp code and tools can include it. When
// <cuda_runtime.h> is available (any .cu, or a host TU built against the toolkit) the
// cudaDeviceProp overload of arch_of() is compiled in as well.
//
// Design (judge_plan §4.1, tile budgets derived, not measured):
//   * ARCH values equal __CUDA_ARCH__ (750, 800, 860, 890, 900) so device code can pick
//     TileCfg<__CUDA_ARCH__, D> and host code TileCfg<arch_of(prop), D> with one template.
//   * Shared-memory caps are the documented per-block opt-in maxima
//     (cudaFuncAttributeMaxDynamicSharedMemorySize):
//       Turing sm_75  64 KB   (Turing tuning guide)
//       A100   sm_80 163 KB   (Ampere tuning guide)
//       GA10x  sm_86  99 KB   (Ampere tuning guide, GA10x section)
//       Ada    sm_89  99 KB   (Ada tuning guide)
//       Hopper sm_90 227 KB   (Hopper tuning guide)
//     Anything above 48 KB needs `extern __shared__` + cudaFuncSetAttribute with the return
//     value checked; kNeedsOptIn* tells the launcher whether that call is required.
//   * Row padding: fp32 tiles +4 floats, fp16 tiles +8 halves. Both are +16 B per row. Any
//     row stride that is a multiple of 128 B (the repo's D=64/128 rows are 256 B / 512 B
//     unpadded) maps column c of every row to the same bank, so a warp walking one column
//     conflicts 32-way. +16 B shifts consecutive rows by 4 banks: conflict-free for 16-B-chunk
//     walks (float4 / ldmatrix-style, 8 lanes per phase cover 32 banks) and 4-way for scalar
//     column walks (bank = (4r + c) mod 32 hits 8 banks x 4 rows). +1 element would be the
//     scalar-walk fix but is rejected because WMMA load/store_matrix_sync and float4 need the
//     leading dimension to be a multiple of 16 B. S tiles are always fp32 (+4). The residual
//     conflict count is measured (`l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`),
//     never argued from this comment.
//   * Occupancy implied by the tiles is DERIVED below (kBlocksPerSmBySmem*, kWarpsPerSmBySmem*)
//     so docs quote a constant, not a typed number: on T4 the S3 d=64 tile (30 KB) admits 2
//     blocks/SM = 8 warps = 25% of the 32-warp ceiling; d=128 and both S4 tiles admit 1.
#pragma once

#include <cstddef>
#include <cstdint>

#if __has_include(<cuda_runtime.h>)
#include <cuda_runtime.h>
#define FA_HAVE_CUDA_RUNTIME 1
#endif

#if defined(__CUDACC__)
#define FA_HD __host__ __device__
#else
#define FA_HD
#endif

namespace fa {

// Architecture buckets (== __CUDA_ARCH__ of the SASS being compiled).
constexpr int kArchTuring = 750;
constexpr int kArchA100 = 800;
constexpr int kArchGA10x = 860;
constexpr int kArchAda = 890;
constexpr int kArchHopper = 900;

// The 48 KB static / no-opt-in ceiling shared by every architecture in scope.
constexpr std::size_t kSmemNoOptInBytes = 48u * 1024u;

template <int ARCH>
struct ArchCaps;  // only the five buckets below exist on purpose: unknown arch = compile error

template <>
struct ArchCaps<750> {
  static constexpr std::size_t kMaxDynSmemBytes = 65536;
  static constexpr int kMaxThreadsPerSm = 1024;
  static constexpr int kWarpsPerSm = 32;
};
template <>
struct ArchCaps<800> {
  static constexpr std::size_t kMaxDynSmemBytes = 166912;
  static constexpr int kMaxThreadsPerSm = 2048;
  static constexpr int kWarpsPerSm = 64;
};
template <>
struct ArchCaps<860> {
  static constexpr std::size_t kMaxDynSmemBytes = 101376;
  static constexpr int kMaxThreadsPerSm = 1536;
  static constexpr int kWarpsPerSm = 48;
};
template <>
struct ArchCaps<890> {
  static constexpr std::size_t kMaxDynSmemBytes = 101376;
  static constexpr int kMaxThreadsPerSm = 1536;
  static constexpr int kWarpsPerSm = 48;
};
template <>
struct ArchCaps<900> {
  static constexpr std::size_t kMaxDynSmemBytes = 232448;
  static constexpr int kMaxThreadsPerSm = 2048;
  static constexpr int kWarpsPerSm = 64;
};

// Runtime mirrors of ArchCaps for host dispatch (the arch is only known at runtime there).
FA_HD constexpr std::size_t max_dyn_smem_bytes(int arch) {
  return arch >= 900   ? ArchCaps<900>::kMaxDynSmemBytes
         : arch >= 890 ? ArchCaps<890>::kMaxDynSmemBytes
         : arch >= 860 ? ArchCaps<860>::kMaxDynSmemBytes
         : arch >= 800 ? ArchCaps<800>::kMaxDynSmemBytes
                       : ArchCaps<750>::kMaxDynSmemBytes;
}
FA_HD constexpr int max_threads_per_sm(int arch) {
  return arch >= 900   ? ArchCaps<900>::kMaxThreadsPerSm
         : arch >= 890 ? ArchCaps<890>::kMaxThreadsPerSm
         : arch >= 860 ? ArchCaps<860>::kMaxThreadsPerSm
         : arch >= 800 ? ArchCaps<800>::kMaxThreadsPerSm
                       : ArchCaps<750>::kMaxThreadsPerSm;
}
FA_HD constexpr int warps_per_sm(int arch) {
  return max_threads_per_sm(arch) / 32;
}

// Compute capability -> bucket. Unknown minors fall to the nearest LOWER bucket of the same
// generation (conservative: a smaller smem cap never breaks a launch). Anything below 7.5 is
// reported as 750 so the caller gets the smallest caps; impl_supports() rejects sm < 75
// separately, this function never does.
FA_HD constexpr int arch_of(int major, int minor) {
  return major >= 9                 ? kArchHopper
         : major == 8 && minor >= 9 ? kArchAda
         : major == 8 && minor >= 6 ? kArchGA10x
         : major == 8               ? kArchA100
                                    : kArchTuring;
}
// sm as used by fa::impl_supports (major*10+minor, e.g. 75) -> bucket.
FA_HD constexpr int arch_of_sm(int sm) {
  return arch_of(sm / 10, sm % 10);
}
#if defined(FA_HAVE_CUDA_RUNTIME)
inline int arch_of(const cudaDeviceProp& p) {
  return arch_of(p.major, p.minor);
}
#endif

// Bytes of shared memory for one fused-attention block: Q[br][d+pad] + K[bc][d+pad] +
// V[bc][d+pad] in the input dtype plus S[br][bc+s_pad] in fp32. Runtime-callable so the
// FA_FORCE_BR/BC overrides can be budget-checked before launch with the same formula.
FA_HD constexpr std::size_t fused_smem_bytes(int br, int bc, int d, int elem_bytes, int pad_elems,
                                             int s_pad_floats) {
  return static_cast<std::size_t>(br + 2 * bc) * static_cast<std::size_t>(d + pad_elems) *
             static_cast<std::size_t>(elem_bytes) +
         static_cast<std::size_t>(br) * static_cast<std::size_t>(bc + s_pad_floats) * sizeof(float);
}

// Default tiles. F32 rows are the S3 fused kernel (fp32 in, fp32 acc); F16 rows are the S4/S5
// WMMA kernels (fp16/bf16 in, fp32 acc). Bc shrinks with D so the tile stays under the T4 cap
// with head-room for the fp32 S tile; Br stays fixed so the grid shape is D-independent.
// kThreads is a default a kernel may override; the smem budget is the binding constraint.
template <int ARCH, int D>
struct TileCfg {
  static_assert(D == 64 || D == 128, "head dim must be 64 or 128 (CONTRACT §1)");

  static constexpr int kPadF32 = 4;  // +16 B per fp32 row
  static constexpr int kPadF16 = 8;  // +16 B per fp16 row, keeps WMMA ldm % 8 == 0
  static constexpr int kSPad = 4;    // S tile is fp32 in every stage

  static constexpr int kBrF32 = 32;
  static constexpr int kBcF32 = (D == 64) ? 32 : 16;
  static constexpr int kThreadsF32 = 128;  // 4 warps

  static constexpr int kBrF16 = 64;  // 4 warps x one 16-row WMMA fragment each
  static constexpr int kBcF16 = (D == 64) ? 64 : 32;
  static constexpr int kThreadsF16 = 128;

  static constexpr std::size_t kSmemF32 = fused_smem_bytes(kBrF32, kBcF32, D, 4, kPadF32, kSPad);
  static constexpr std::size_t kSmemF16 = fused_smem_bytes(kBrF16, kBcF16, D, 2, kPadF16, kSPad);

  static constexpr bool kNeedsOptInF32 = kSmemF32 > kSmemNoOptInBytes;
  static constexpr bool kNeedsOptInF16 = kSmemF16 > kSmemNoOptInBytes;

  static_assert(kSmemF32 <= ArchCaps<ARCH>::kMaxDynSmemBytes,
                "fp32 fused tile exceeds this architecture's shared-memory cap");
  static_assert(kSmemF16 <= ArchCaps<ARCH>::kMaxDynSmemBytes,
                "fp16 fused tile exceeds this architecture's shared-memory cap");

  // Blocks per SM limited by shared memory alone (the per-SM capacity is the per-block opt-in
  // maximum plus ~1 KB of driver reserve on every arch in scope, so this is exact for the
  // integer quotient), and the warps that implies. Registers and the warp ceiling can only
  // lower it; the achieved value is sm__warps_active.avg.pct_of_peak_sustained_active.
  static constexpr int kBlocksPerSmBySmemF32 =
      static_cast<int>(ArchCaps<ARCH>::kMaxDynSmemBytes / kSmemF32);
  static constexpr int kBlocksPerSmBySmemF16 =
      static_cast<int>(ArchCaps<ARCH>::kMaxDynSmemBytes / kSmemF16);
  static constexpr int kWarpsPerSmBySmemF32 = kBlocksPerSmBySmemF32 * (kThreadsF32 / 32);
  static constexpr int kWarpsPerSmBySmemF16 = kBlocksPerSmBySmemF16 * (kThreadsF16 / 32);
  static_assert(kBrF32 % 32 == 0 && kBcF32 % 16 == 0 && kBrF16 % 16 == 0 && kBcF16 % 16 == 0,
                "tiles must be multiples of the warp / WMMA fragment size");
};

namespace detail {
// Force every (ARCH, D) instantiation at header-parse time so a bad budget fails the build of
// any TU that includes this header, not just the one that happens to use the offending pair.
constexpr bool all_tile_budgets_ok() {
  return TileCfg<750, 64>::kSmemF32 > 0 && TileCfg<750, 128>::kSmemF32 > 0 &&
         TileCfg<800, 64>::kSmemF32 > 0 && TileCfg<800, 128>::kSmemF32 > 0 &&
         TileCfg<860, 64>::kSmemF32 > 0 && TileCfg<860, 128>::kSmemF32 > 0 &&
         TileCfg<890, 64>::kSmemF32 > 0 && TileCfg<890, 128>::kSmemF32 > 0 &&
         TileCfg<900, 64>::kSmemF32 > 0 && TileCfg<900, 128>::kSmemF32 > 0;
}
static_assert(all_tile_budgets_ok(), "tile budget check");
// Derived numbers quoted in docs (judge_plan §4.1 rounds them): keep them honest here.
static_assert(TileCfg<750, 64>::kSmemF32 == 30720, "S3 d=64: 3x8.5 KB tiles + 4.5 KB S");
static_assert(TileCfg<750, 128>::kSmemF32 == 36352, "S3 d=128 (Bc=16): ~35.5 KB");
static_assert(TileCfg<750, 64>::kSmemF16 == 45056, "S4 d=64: 3x9216 B + 17408 B S = 44 KB");
static_assert(TileCfg<750, 128>::kSmemF16 == 44032, "S4 d=128 (Bc=32): 43 KB");
// Occupancy numbers quoted in docs/INTERVIEW.md section 6 (T4, 32-warp ceiling).
static_assert(TileCfg<750, 64>::kWarpsPerSmBySmemF32 == 8, "S3 d=64 on T4: 2 blocks, 8 warps");
static_assert(TileCfg<750, 128>::kWarpsPerSmBySmemF32 == 4, "S3 d=128 on T4: 1 block, 4 warps");
static_assert(TileCfg<750, 64>::kWarpsPerSmBySmemF16 == 4, "S4 d=64 on T4: 1 block, 4 warps");
static_assert(TileCfg<750, 128>::kWarpsPerSmBySmemF16 == 4, "S4 d=128 on T4: 1 block, 4 warps");
}  // namespace detail

}  // namespace fa
