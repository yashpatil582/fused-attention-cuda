// S5: the S4 WMMA kernel (fa_s4_wmma) with two Nsight-guided tuning knobs, each a template
// parameter so every variant is its own SASS entry with its own ptxas line: the BLOCK HEIGHT
// (WARPS, 16 query rows per warp) and a REGISTER-STAGED K/V PREFETCH (PREFETCH). Same online
// softmax (Milakov & Gimelshein 2018), same FlashAttention-2 loop order, same deferred
// normalisation, same guards, same barriers and same fragment layout as S4 — PREFETCH=false with
// S4's WARPS is S4 to the bit; the knobs only move WHEN loads are issued and HOW MANY warps share
// one shared-memory carve. Everything measured by the metrics below, kept only by the rule at the
// end of this header.
//
// KNOB 1 — WARPS (block height BR = 16*WARPS rows, 32*WARPS threads). WHY: S4's T4 profiles show
// 1 block per SM at the default tiles (45,056 B of the 64 KB per-SM shared memory for D=64/BC=64;
// 43,008 B for D=128/BC=32), i.e. 4 resident warps = 12.5% of Turing's 32-warp ceiling (2 warps
// = 6.25% at D=128). Shared memory binds before registers do, so a second block never fits and
// the ONLY occupancy lever left on a 64 KB/SM part is more warps inside the one block: WARPS=8
// gives 8 resident warps (BR=128, needs BC <= 32 to fit: 62,464 B). More warps per SM also means
// the scheduler has other warps to issue while one waits at the two block barriers per tile, and
// a taller block halves the K/V global traffic per query row (each K/V tile is loaded once per
// block: dram__bytes.sum scales with N/BR). Smaller WARPS goes the other way (BR=32/BC=16 at D=64
// is 17,920 B -> 3 blocks = 6 warps/SM, but 2x the K/V traffic) and is instantiated for the
// tile-invariance tests and the sweep, not as a candidate default. Metrics:
//   sm__warps_active.avg.pct_of_peak_sustained_active     4/32 -> 8/32 expected for WARPS=8
//   smsp__warp_issue_stalled_barrier_per_warp_active.pct  the barrier share; may rise per warp
//                                                         (8 warps meet at each barrier) while
//                                                         issue slots per SM rise — measured
//   dram__bytes.sum                                       K/V re-reads: N/BR tile loads per head
// Registers per thread do not depend on WARPS (every warp does the same 16-row work), so the
// register-bound block count is unchanged and __launch_bounds__(32*WARPS) forces no cap below
// 255 (65,536 / 256 threads = 256).
//
// KNOB 2 — PREFETCH (register-staged double buffering). WHY: S4 loads K/V tile t synchronously —
// LDG.128 -> STS.128 -> __syncthreads — so at the top of every tile the whole block sits on DRAM
// latency with nothing to issue (stall reason long_scoreboard, i.e. waiting for an L1TEX/global
// result). The Ampere answer is cp.async into a second smem buffer, but cp.async is sm_80+ with
// no Turing fallback ("Requires sm_80 or higher", PTX ISA cp.async, quoted in
// .research/tech-tensor-core-programming-constraints-by-n.json: "On T4 you must hand-roll double
// buffering with LDG->registers->STS->__syncthreads, which costs you registers"). That is what
// PREFETCH does: right after tile t's RAW barrier every thread issues the 16-B global loads of
// tile t+1 (the same cooperative thread->vector mapping as the direct load, rows >= N zero-filled
// exactly like the direct path) into a fixed-size uint4 array, runs tile t's S GEMM / softmax /
// P*V while those loads are in flight, and writes the registers to Ks/Vs only after tile t's WAR
// barrier. Global latency overlaps compute; the smem footprint is unchanged (no second buffer,
// which the 64 KB budget could not afford at the default tiles). Cost: 2*BC*D/threads halves of
// registers per thread held across the whole tile body — 64 halves = 32 registers at
// D=64/BC=64/128 threads, 128 halves = 64 registers at D=128/BC=32/64 threads (table below).
// Metrics:
//   smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct  the target; expected down
//   dram__throughput.avg.pct_of_peak_sustained_elapsed            up (loads in flight during MMA)
//   launch__registers_per_thread                                  the price, from ptxas -v below
// The loads are issued in program order before the MMA section; ptxas normally keeps LDGs early
// (it schedules for latency), but under register pressure it may sink them toward the STS and
// silently undo the overlap — the long_scoreboard metric, not this comment, decides.
//
// BARRIERS with PREFETCH (still exactly two __syncthreads per K/V tile, S4's count):
//   RAW  top of tile t: Ks/Vs hold tile t (direct load on t=0 / the previous iteration's register
//        stores), and on t=0 it fences the Q fragment loads so Qs may become P.
//   WAR  bottom of tile t: every warp finished its P*V reads of Vs (and its S GEMM reads of Ks).
//   The register -> Ks/Vs stores sit strictly between the WAR barrier of tile t and the RAW barrier
//   of tile t+1; no lane reads Ks/Vs in that window (the P*V GEMM ended before the WAR barrier and
//   the next S GEMM starts after the RAW barrier). After the last tile nothing is stored and the
//   WAR barrier fences the O staging as in S4. P aliases Qs, never Ks/Vs, so the prefetch stores
//   cannot touch the P strip regardless of timing.
//
// WHY SHARED-MEMORY ROUND TRIPS (unchanged from S4): WMMA fragment layouts are opaque — "the way
// matrix elements map to fragment internal storage is unspecified" (CUDA C++ Programming Guide,
// Warp Matrix Functions) and "the mapping between individual matrix parameters to their
// fragments is opaque, so your program should not make assumptions about it" (NVIDIA blog,
// Programming Tensor Cores in CUDA 9; both quoted in
// .research/tech-tensor-core-programming-constraints-by-n.json). Element-wise work on a fragment
// is legal, but everything attention needs between the two GEMMs is position-dependent: the row
// max/sum, the causal/partial-tile mask, and feeding P back as the A operand of P*V. The only
// portable route is store_matrix_sync into smem in a KNOWN row-major layout, do the row-wise work
// there, load_matrix_sync back — what experiments/wmma_gemm/wmma_gemm.cu de-risked. Per K/V tile:
// S out (fp32), P in (half), and — because O <- alpha*O + P*V is row-wise too — O out, rescale,
// O in (skipped when no row of the warp needs it, step 4). FlashAttention-2 avoids all of this
// with raw mma.sync, whose PTX-documented lane<->element map keeps S and P in registers; that is
// out of scope here (PLAN.md W6) and is the honest ceiling of a WMMA kernel.
//
// THREAD MAPPING: warp w owns query rows 16w..16w+15 of the block for EVERYTHING — the S strip
// (16 x BC), the softmax statistics, the P strip and the O strip — the FA2 "each warp owns whole
// rows" partition, so the online-softmax bookkeeping never crosses warps and the only block-wide
// dependencies are the cooperative K/V tile loads. Inside the softmax, lane owns row 16w + lane/2
// and one half of its columns ([(lane&1)*BC/2, +BC/2)); the two lanes of a row combine with ONE
// __shfl_xor_sync(.., 1) for the max and one for the sum, and both hold identical m, l, alpha.
// Q fragments (D/16 matrix_a, row_major) are loaded ONCE per block and stay in registers for the
// whole K/V loop; the O accumulator is D/16 fp32 accumulator fragments, also in registers.
//
// SHARED MEMORY (dynamic, one carve per block, 128-B aligned base; halves for T tiles):
//   Qs [BR][D+8]   T     Q tile; DEAD after the Q fragment loads -> P [BR][BC+8] aliases it.
//   Ks [BC][D+8]   T     K tile, read as matrix_b COL_MAJOR: element (k, n) of Kt sits at
//                        ptr[n*ldm + k] = Ks[n][k], i.e. where row-major K already is.
//   Vs [BC][D+8]   T     V tile, matrix_b row_major (K x N = keys x d).
//   Ss [BR][max(BC,D)+4] fp32 per-warp strips: S for the softmax, then O for the rescale round
//                        trip and the epilogue (max(BC, D): the O strip is D wide).
// P aliases Qs (not Ks): the P strip needs BR*(BC+8) halves <= BR*(D+8) = Qs because BC <= D for
// every tile in scope (static_assert); Qs is dead once every warp holds its Q fragments — the
// first tile's RAW barrier publishes that — so P needs no extra barrier, and Ks/Vs stay free for
// the prefetch stores. +8 halves / +4 floats (16 B) of row padding: load/store_matrix_sync need
// ldm to be a multiple of 8 halves / 4 floats and every fragment pointer 256-bit aligned (+1 is
// illegal); the same 16 B shifts consecutive rows by 4 banks (tile_config.h; measured by
// l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum, never argued from here).
// FRAGMENT POINTER ALIGNMENT (256-bit): every fragment pointer starts at a row that is a multiple
// of 16 (16*144 B, 16*272 B, 16*528 B are multiples of 32 B) and a column that is a multiple of
// 16 elements from a 128-B-aligned region base (region offsets static_asserted).
//
// INSTANTIATIONS (the T4 64 KB per-block cap decides; sm_80+ could fit more but the primary GPU
// is the T4 and one matrix keeps the sweep comparable). Bytes = BR*(D+8)*2 + 2*BC*(D+8)*2 +
// BR*(max(BC,D)+4)*4; "opt-in" = > 48 KB -> cudaFuncSetAttribute as in S3/S4; blocks/SM by smem
// on T4 (64 KB, also capped at 16 blocks and 32 warps); registers from `nvcc -O3 -std=c++17
// -lineinfo -arch=sm_75 -Xptxas -v --expt-relaxed-constexpr -Iinclude -c src/stage5_tuned.cu`
// (nvcc 12.8.93, fp16; 0 spill bytes in every instantiated entry; P = PREFETCH):
//   D   WARPS  BR  BC  smem B  opt-in  blk/SM  regs P=0  regs P=1  staging halves/thread (P=1)
//   64    1    16  16  11,264    no      5      120       124        64  (8 x uint32)
//   64    1    16  32  15,872    no      4      127       181       128 (16 x uint32)
//   64    1    16  64  25,088    no      2      164        --       256 (32 x uint32) (*)
//   64    2    32  16  17,920    no      3       93       111        32
//   64    2    32  32  22,528    no      2      124       147        64
//   64    2    32  64  31,744    no      2      151       215       128
//   64    4    64  16  31,232    no      2       92        96        16   (S4 D=64 BC=16: 92)
//   64    4    64  32  35,840    no      1      113       121        32   (S4 D=64 BC=32: 113)
//   64    4    64  64  45,056    no      1      151       168        64   (S4 D=64 default: 151)
//   64    8   128  16  57,856   yes      1       94       107         8   (128 vecs < 256 thr)
//   64    8   128  32  62,464   yes      1      108       115        16
//   64    8   128  64  71,680   -- NOT instantiated: 71,680 > 65,536 (T4 cap)
//  128    2    32  16  34,304    no      1      168       179        64   (S4 D=128 BC=16: 168)
//  128    2    32  32  43,008    no      1      168       254       128   (S4 D=128 default: 168)
//  128    4    64  16  59,904   yes      1      154       168        32
//  128    4    64  32  68,608   -- NOT instantiated: 68,608 > 65,536 (T4 cap)
//  128    x     x  64  -- NOT instantiated: S4's BC=64 already sits at 253 registers (0 spills)
//                        with 60,416 B = 1 block/SM; any staging on top of it spills.
// (*) D=64/WARPS=1/BC=64/P=1 is the one dropped variant: 256 staging halves = 128 registers on
//     top of the 164 the direct variant uses -> ptxas 255 registers, 56 B spill stores / 68 B
//     spill loads. kPrefetchOk<64, 1, 64> = false; that tile runs with FA_S5_PREFETCH=0 only.
// Every P=0 entry at S4's block height (WARPS 4 / 2) reproduces S4's register count exactly,
// which is the check that the PREFETCH=false path is S4's code. D=128/WARPS=2/BC=32/P=1 (the
// D=128 default with prefetch on) sits at 254 registers with 0 spills: legal, but at the ceiling
// ptxas may have sunk some of the staged loads toward their stores — the long_scoreboard metric
// decides whether that variant earns its keep. Registers never bind on T4 in this table: the
// count by registers, 65,536 / (32*WARPS*regs), is >= the smem count in every row, so shared
// memory stays the binding constraint, exactly the S4 situation the WARPS knob attacks.
// Other archs (same build, CI list): fp16 on sm_80/86/89/90 lands within +-15 registers of the
// sm_75 column with 0 spills (D=128/WARPS=2/BC=32/P=1 is 241 on sm_80); bf16 on sm_80+ runs a
// few registers higher than fp16 (254 vs 241 there) and has ONE spilling entry, bf16
// D=64/WARPS=4/BC=16/P=1 on sm_80/86/89: 96 registers with 16 B spill stores / 12 B spill loads
// (0 on sm_90, 0 for the fp16 twin). It stays instantiated: the 0-spill rule of this stage is
// the sm_75 column (the T4 never runs bf16), dropping it would make the bf16 tile-invariance
// test (BR 0 / BC 16 at N=64) fail on an sm_80+ box, and an 8-byte stack frame at 96 registers
// is a ptxas allocation artefact, not pressure. On sm_75 every bf16 entry is a 4-register trap
// stub.
//
// RUNTIME SELECTION (fa::stage5::forward):
//   FA_FORCE_BR (dbg.block_m)  BR = 16*WARPS: 16/32/64/128 -> WARPS 1/2/4/8 (16 exists for D=64
//                              only: test_tile_invariance asks for (BR, BC) = (16, 16) at N=64);
//                              0 -> S4's choice, 64 (4 warps) for D=64 and 32 (2 warps) for D=128.
//   FA_FORCE_BC (dbg.block_n)  as in S4: 16/32/64 at D=64, 16/32 at D=128; 0 -> the stage default
//                              (S4's 64 at D=64; at D=128 16 from N=2048 up, S4's 32 below)
//                              (TileCfg<kArchTuring, D>::kBcF16: 64 for D=64, 32 for D=128).
//   FA_S5_PREFETCH             unset or 1 -> register-staged prefetch on; 0 -> off (S4 path).
//                              Read once per process (getenv), like FA_VERBOSE.
//   Any (D, BR, BC, PREFETCH) that is not in the table -> cudaErrorInvalidValue, and under
//   FA_VERBOSE the valid set for that D is printed.
//
// SOFTMAX CONVENTION (identical to include/fa/softmax_math.cuh, S3, S4, python/fused_attn/
// reference.py tiled_online_softmax_reference — checked line by line):
//   S      = Q Kt                  RAW scores, fp32 accumulate; the scale is NOT applied here
//   m_new  = max(m, rowmax(S))     raw units
//   alpha  = exp2((m - m_new) * c) c = scale*log2(e), folded once per launch (rescale_alpha)
//   p      = exp2(s*c - m_new*c)   one FFMA + one EX2 per score (exp2_scaled), 0 if m_new == -inf
//   l      = alpha * l + rowsum(p) rowsum from the fp32 p, BEFORE rounding p to T (as FA2 does)
//   O      = alpha * O + P V       P rounded to T for the tensor-core GEMM; O stays fp32
//   m      = m_new                 AFTER alpha and p used the old / new value
//   o      = O * (1 / l)           one divide per row at the end;  lse = m*scale + ln(l)
// fp16 range (reality-check BUG 3): only the Q/K/V/P OPERANDS are half; S, m, l, alpha and O are
// fp32 fragments/registers. The scale is applied exactly once, inside c (BUG 4).
//
// CAUSAL / PARTIAL TILES: top-left aligned, key j visible to query i iff j <= i; the mask uses
// GLOBAL i and j with a strict `j > i` (BUG 5) on RAW scores, only for tiles that need it
// (block-uniform test). Tiles entirely above the diagonal are skipped by the loop bound
// n_tiles = ceil(min(i0+BR, N) / BC) (reference.py `j0 > i1 - 1: continue`). Rows past N of a
// partial Q tile and keys past N of a partial K/V tile are zero-filled in smem (direct AND
// prefetched path); the keys are also masked to -inf so p = 0 and 0 * 0 (never 0 * garbage);
// the rows are computed as finite garbage and never stored. With BR=128 and N=32 six of the
// eight warps own only such rows; they cost time, not correctness (the sweep's N >= 128).
//
// GUARDS (BUG 2): m == -inf (no unmasked key so far) -> alpha = 0 (l and O are still 0, exact),
// p = 0 for the whole tile; at the end finalize_inv_l / finalize_lse turn an empty row into
// o = 0, lse = +inf. Impossible for top-left causal with Nq == Nkv; kept so the invariant is local.
//
// KEEP RULE (docs/STAGES.md S5 entries, .research/PLAN.md S5a row): each S5 commit changes ONE
// thing and keeps it only if it wins >= 3% on the median at the canonical config with p10/p90
// shown next to it; a knob that loses stays instantiated (it is the "tried and rejected"
// sub-entry with its profile) but is not the default. Defaults: D=64 is S4's tile with PREFETCH
// on (no block height beat it in the A/B); D=128 is WARPS=4/BC=16 from N=2048 up and S4's
// WARPS=2/BC=32 below, both with PREFETCH on, picked by the T4 A/B and sweep quoted at
// kTallTileMinN128 — those ms figures are the only speed claims in this file and they point at
// the committed CSVs.
//
// LIMITS: gridDim.y = B*H <= 65,535; q/k/v/o 16-B aligned (rows are 128 B / 256 B, so every tile
// base and every 16-B vector inherits it); N % 32 == 0 (contract); bf16 needs sm_80+ — the
// kernel body is compiled for __CUDA_ARCH__ >= 800 only (mma.h has no bf16 fragments below
// that), an sm_75 pass gets a __trap() stub that supports() never lets anyone launch.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>

#include "fa/attention.cuh"
#include "fa/cuda_check.cuh"
#include "fa/softmax_math.cuh"
#include "fa/tile_config.h"

namespace fa {
namespace stage5 {

constexpr int kFrag = 16;                                 // WMMA m = n = k
constexpr int kPadT = TileCfg<kArchTuring, 64>::kPadF16;  // +8 halves = 16 B per T row
constexpr int kPadF = TileCfg<kArchTuring, 64>::kSPad;    // +4 floats = 16 B per fp32 row
constexpr int kVecElems = 8;                              // one 16-B vector = 8 halves/bf16
constexpr unsigned kFullMask = 0xffffffffu;

// Default tiles when FA_FORCE_BR / FA_FORCE_BC are 0 (both override everything below).
// D=64 keeps S4's 4 warps x BC=64: the T4 A/B (bench/results/t4/stage5_tuning.csv @ d835758,
// canonical B2_H8_N2048_D64, 585 MHz) found no taller or shorter block that beats it.
constexpr int kDefaultWarps64 = 4;
constexpr int kDefaultBc64 = TileCfg<kArchTuring, 64>::kBcF16;  // 64, S4's
// D=128 has two tiles, picked by N. The same A/B made WARPS=4 x BC=16 the winner at the canonical
// config (8.4526 vs 9.8815 ms for S4's WARPS=2 x BC=32 with prefetch, -14.5%), but the 48-config
// sweep at 15b83a5 (tall tile everywhere) against d835758 (S4's tile everywhere), both with
// prefetch, bench/results/t4/stage5.csv, shows it is a size trade-off:
//   N >= 2048: tall tile faster by 2.7-20.7% (N=4096: -20%; B8_H32_N2048 causal=0: -14.5%)
//   N <= 1024: tall tile slower by 2.5-46% (B1_H8_N128: 0.062 vs 0.045 ms): half the blocks
//              (16 < 40 SMs at B1_H8_N128), twice the K/V tiles = twice the barriers and S round
//              trips per query row, and too few tiles for the prefetch to amortise.
// The crossover sits between 1024 and 2048 (mixed at 1024: B8_H32 -6.8%, B1_H8 +4.8%), so the tall
// tile is the default from N = 2048 up and S4's tile below it.
constexpr int kTallTileMinN128 = 2048;
constexpr int kTallWarps128 = 4, kTallBc128 = 16;    // 59,904 B smem (opt-in), 168 regs, 0 spills
constexpr int kShortWarps128 = 2, kShortBc128 = 32;  // S4's tile: 43,008 B, 254 regs with prefetch
static_assert(TileCfg<kArchTuring, 64>::kBrF16 / kFrag == kDefaultWarps64,
              "S5 D=64 default WARPS drifted from tile_config.h");

// bf16 WMMA fragments exist only for compute capability >= 8.0: mma.h declares them under
// `!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800`, so the host pass and sm_80+ device passes
// see them and an sm_75 device pass does not. The kernel discards its body with `if constexpr`
// on that pass (wmma_gemm.cu's fa_wmma_bf16_kernel does the same with #if).
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)
constexpr bool kArchHasBf16Wmma = false;
#else
constexpr bool kArchHasBf16Wmma = true;
#endif

// Tile geometry and the shared-memory carve for one (D, WARPS, BC) instantiation (header table).
template <int D, int WARPS, int BC>
struct Cfg {
  static_assert(D == 64 || D == 128, "D must be 64 or 128 (CONTRACT 1)");
  static_assert(WARPS == 1 || WARPS == 2 || WARPS == 4 || WARPS == 8, "WARPS must be 1/2/4/8");
  static_assert(BC == 16 || BC == 32 || BC == 64, "BC must be 16, 32 or 64");

  static constexpr int kWarps = WARPS;
  static constexpr int kBr = WARPS * kFrag;  // query rows per block: one 16-row strip per warp
  static constexpr int kThreads = WARPS * 32;

  static constexpr int kLdT = D + kPadT;                  // Q/K/V row stride (T elements)
  static constexpr int kLdP = BC + kPadT;                 // P row stride (T elements)
  static constexpr int kLdS = (BC > D ? BC : D) + kPadF;  // fp32 staging strip row stride

  static constexpr std::size_t kQBytes = static_cast<std::size_t>(kBr) * kLdT * 2;
  static constexpr std::size_t kKvBytes = static_cast<std::size_t>(BC) * kLdT * 2;
  static constexpr std::size_t kSBytes = static_cast<std::size_t>(kBr) * kLdS * sizeof(float);
  static constexpr std::size_t kOffK = kQBytes;
  static constexpr std::size_t kOffV = kOffK + kKvBytes;
  static constexpr std::size_t kOffS = kOffV + kKvBytes;
  static constexpr std::size_t kBytes = kOffS + kSBytes;
  static constexpr bool kNeedsOptIn = kBytes > kSmemNoOptInBytes;

  // Budget: the per-arch opt-in caps of tile_config.h; Turing's 64 KB is the smallest in scope
  // and is what decides the instantiation table in the header.
  static_assert(kBytes <= ArchCaps<kArchTuring>::kMaxDynSmemBytes, "S5 tile exceeds the sm_75 cap");
  static_assert(kBytes <= ArchCaps<kArchA100>::kMaxDynSmemBytes, "S5 tile exceeds the sm_80 cap");
  static_assert(kBytes <= ArchCaps<kArchGA10x>::kMaxDynSmemBytes, "S5 tile exceeds the sm_86 cap");
  static_assert(kBytes <= ArchCaps<kArchAda>::kMaxDynSmemBytes, "S5 tile exceeds the sm_89 cap");
  static_assert(kBytes <= ArchCaps<kArchHopper>::kMaxDynSmemBytes, "S5 tile exceeds the sm_90 cap");
  // Alignment: every region starts 128-B aligned so fragment pointers (multiples of 16 rows and
  // 16 columns inside a region) are 256-bit aligned, and 16-B vector loads/stores stay aligned.
  static_assert(kOffK % 128 == 0 && kOffV % 128 == 0 && kOffS % 128 == 0,
                "smem regions must be 128-B aligned");
  static_assert(kLdT % 8 == 0 && kLdP % 8 == 0, "T ldm must be a multiple of 8 elements (16 B)");
  static_assert(kLdS % 4 == 0, "fp32 ldm must be a multiple of 4 floats (16 B)");
  // P aliases the dead Q tile: BR*(BC+8) <= BR*(D+8) because BC <= D for every tile in scope.
  static_assert(static_cast<std::size_t>(kBr) * kLdP * 2 <= kQBytes, "P strip must fit in Qs");
  static_assert(kLdS >= D + kPadF && kLdS >= BC + kPadF, "staging strip must hold S and O rows");
  // Cooperative loads: the Q tile is a whole number of 16-B vectors per thread (2*WARPS*D vectors
  // over 32*WARPS threads = D/16 each); the K/V tile may be SMALLER than one vector per thread
  // (BC=16, D=64, 256 threads: 128 vectors) and is then guarded per vector in TileCopy.
  static_assert(D % kVecElems == 0, "D must be a multiple of 8");
  static_assert((kBr * D / kVecElems) % kThreads == 0, "Q tile: vectors per thread");
  // Softmax lane mapping: each lane's half row of S / P / O is a whole number of 16-B vectors.
  static_assert((BC / 2) % kVecElems == 0, "half a P row must be whole 16-B vectors");
  static_assert((BC / 2) % 4 == 0 && (D / 2) % 4 == 0, "half rows must be whole float4s");
  static_assert(kBr % kFrag == 0 && BC % kFrag == 0 && D % kFrag == 0, "fragment multiples");
};

// The only place fp16 and bf16 differ: conversions and the packed 2-vector. bf16 conversions
// compile on every arch (bit arithmetic below sm_80); the WMMA gate is kArchHasBf16Wmma.
template <typename T>
struct Elem;

template <>
struct Elem<__half> {
  using Vec2 = __half2;
  __device__ static __forceinline__ __half2 pack2(float a, float b) {
    return __floats2half2_rn(a, b);  // .x = a at the lower address
  }
  __device__ static __forceinline__ float2 unpack2(__half2 v) {
    return __half22float2(v);
  }
};

template <>
struct Elem<__nv_bfloat16> {
  using Vec2 = __nv_bfloat162;
  __device__ static __forceinline__ __nv_bfloat162 pack2(float a, float b) {
    return __floats2bfloat162_rn(a, b);
  }
  __device__ static __forceinline__ float2 unpack2(__nv_bfloat162 v) {
    return __bfloat1622float2(v);
  }
};

// 8 floats -> one 16-B vector of T (round to nearest even), and back.
template <typename T>
__device__ __forceinline__ uint4 pack8(const float* f) {
  using Vec2 = typename Elem<T>::Vec2;
  Vec2 v0 = Elem<T>::pack2(f[0], f[1]);
  Vec2 v1 = Elem<T>::pack2(f[2], f[3]);
  Vec2 v2 = Elem<T>::pack2(f[4], f[5]);
  Vec2 v3 = Elem<T>::pack2(f[6], f[7]);
  uint4 u;
  u.x = *reinterpret_cast<const unsigned*>(&v0);
  u.y = *reinterpret_cast<const unsigned*>(&v1);
  u.z = *reinterpret_cast<const unsigned*>(&v2);
  u.w = *reinterpret_cast<const unsigned*>(&v3);
  return u;
}

// Cooperative, coalesced copy of a [ROWS][D] T tile (ROWS*D CONTIGUOUS elements in global memory:
// rows of one head are adjacent) into a padded [ROWS][D+8] smem tile, 16-B vectors, consecutive
// threads walking the contiguous D axis (reality-check BUG 7), split into the two halves the
// prefetch needs: fetch() global -> registers (rows >= rows_valid are never read and become
// zeros, so nothing in smem is ever uninitialised: 0 * V = 0, bitwise reproducible runs) and
// store() registers -> smem with the SAME thread -> vector mapping. load() = fetch + store is the
// direct path (S4's load_tile, instruction for instruction when kExact). When the tile has fewer
// vectors than threads (kExact false: K/V tile of BC=16 at 256 threads) the surplus threads skip
// both halves; the `vec < kVecs` test is compiled out otherwise.
template <typename T, int ROWS, int D, int THREADS>
struct TileCopy {
  static constexpr int kLd = D + kPadT;
  static constexpr int kVecs = ROWS * D / kVecElems;
  static constexpr int kIters = (kVecs + THREADS - 1) / THREADS;  // uint4 per thread
  static constexpr bool kExact = (kVecs % THREADS) == 0;
  static_assert(kIters >= 1, "tile must be at least one vector");

  __device__ static __forceinline__ void fetch(const T* __restrict__ g, int rows_valid,
                                               uint4 (&regs)[kIters]) {
    const int valid_elems = rows_valid * D;
#pragma unroll
    for (int it = 0; it < kIters; ++it) {
      const int vec = it * THREADS + static_cast<int>(threadIdx.x);
      const int e = vec * kVecElems;
      uint4 x = make_uint4(0u, 0u, 0u, 0u);
      if ((kExact || vec < kVecs) && e < valid_elems) x = *reinterpret_cast<const uint4*>(g + e);
      regs[it] = x;
    }
  }

  __device__ static __forceinline__ void store(const uint4 (&regs)[kIters], T* __restrict__ dst) {
#pragma unroll
    for (int it = 0; it < kIters; ++it) {
      const int vec = it * THREADS + static_cast<int>(threadIdx.x);
      if (kExact || vec < kVecs) {
        const int e = vec * kVecElems;
        const int r = e / D;
        const int d = e - r * D;
        *reinterpret_cast<uint4*>(dst + r * kLd + d) = regs[it];
      }
    }
  }

  // Direct path = S4's load_tile, instruction for instruction: one vector fetched and stored per
  // iteration (one uint4 live), NOT fetch-all-then-store-all — that ordering keeps every K+V
  // vector live at once and costs the PREFETCH=false variant registers S4 never paid (measured:
  // 168 -> 218 at D=128/BC=32 before this was split out). The prefetch stage is the opposite by
  // design: all of its loads are in flight together, and it pays for that in the table.
  __device__ static __forceinline__ void load(const T* __restrict__ g, int rows_valid,
                                              T* __restrict__ dst) {
    const int valid_elems = rows_valid * D;
#pragma unroll
    for (int it = 0; it < kIters; ++it) {
      const int vec = it * THREADS + static_cast<int>(threadIdx.x);
      if (kExact || vec < kVecs) {
        const int e = vec * kVecElems;
        const int r = e / D;
        const int d = e - r * D;
        uint4 x = make_uint4(0u, 0u, 0u, 0u);
        if (e < valid_elems) x = *reinterpret_cast<const uint4*>(g + e);
        *reinterpret_cast<uint4*>(dst + r * kLd + d) = x;
      }
    }
  }
};

// The register staging area of the prefetch: the next K and V tiles as uint4 arrays of a
// compile-time size, indexed only by unrolled constants so they live in registers, never local
// memory. With ENABLED=false both members are one dead uint4 and both methods are empty, so the
// PREFETCH=false kernel carries no trace of them (that variant must be S4 to the bit).
template <typename T, int BC, int D, int THREADS, bool ENABLED>
struct KvStage {
  using Copy = TileCopy<T, BC, D, THREADS>;
  static constexpr int kSlots = ENABLED ? Copy::kIters : 1;
  uint4 k[kSlots];
  uint4 v[kSlots];

  __device__ __forceinline__ void fetch([[maybe_unused]] const T* __restrict__ kg,
                                        [[maybe_unused]] const T* __restrict__ vg,
                                        [[maybe_unused]] int rows_valid) {
    if constexpr (ENABLED) {
      Copy::fetch(kg, rows_valid, k);
      Copy::fetch(vg, rows_valid, v);
    }
  }
  __device__ __forceinline__ void store([[maybe_unused]] T* __restrict__ Ks,
                                        [[maybe_unused]] T* __restrict__ Vs) const {
    if constexpr (ENABLED) {
      Copy::store(k, Ks);
      Copy::store(v, Vs);
    }
  }
};

// The block body (a __device__ template so the __global__ wrapper can discard it for bf16 on
// sm_75 without instantiating any bf16 fragment type there).
template <typename T, int D, int WARPS, int BC, bool PREFETCH>
__device__ __forceinline__ void attention_block(const T* __restrict__ q, const T* __restrict__ k,
                                                const T* __restrict__ v, T* __restrict__ o,
                                                float* __restrict__ lse, int N, int H, int Hkv,
                                                float scale, float scale_log2, bool causal) {
  using namespace nvcuda;
  using C = Cfg<D, WARPS, BC>;
  using FragQ = wmma::fragment<wmma::matrix_a, kFrag, kFrag, kFrag, T, wmma::row_major>;
  using FragKt = wmma::fragment<wmma::matrix_b, kFrag, kFrag, kFrag, T, wmma::col_major>;
  using FragP = wmma::fragment<wmma::matrix_a, kFrag, kFrag, kFrag, T, wmma::row_major>;
  using FragV = wmma::fragment<wmma::matrix_b, kFrag, kFrag, kFrag, T, wmma::row_major>;
  using FragAcc = wmma::fragment<wmma::accumulator, kFrag, kFrag, kFrag, float>;
  constexpr int kBr = C::kBr;
  constexpr int kThreads = C::kThreads;
  constexpr int kLdT = C::kLdT;
  constexpr int kLdP = C::kLdP;
  constexpr int kLdS = C::kLdS;
  constexpr int kDFrags = D / kFrag;   // Q fragments per warp == O fragments per warp
  constexpr int kNFrags = BC / kFrag;  // S column fragments == P k-steps
  constexpr int kHalfC = BC / 2;       // softmax columns per lane
  constexpr int kHalfD = D / 2;        // O columns per lane
  using QCopy = TileCopy<T, kBr, D, kThreads>;
  using KvCopy = TileCopy<T, BC, D, kThreads>;
  using Stage = KvStage<T, BC, D, kThreads, PREFETCH>;

  extern __shared__ __align__(128) unsigned char smem_raw[];
  T* Qs = reinterpret_cast<T*>(smem_raw);  // [BR][D+8], then P [BR][BC+8] (alias, see header)
  T* Ps = Qs;
  T* Ks = reinterpret_cast<T*>(smem_raw + C::kOffK);          // [BC][D+8]
  T* Vs = reinterpret_cast<T*>(smem_raw + C::kOffV);          // [BC][D+8]
  float* Ss = reinterpret_cast<float*>(smem_raw + C::kOffS);  // [BR][max(BC,D)+4]

  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int i0 = static_cast<int>(blockIdx.x) * kBr;  // first query row of this block
  const int bh = static_cast<int>(blockIdx.y);        // flattened (b, h)
  const int b = bh / H;
  const int h = bh - b * H;
  const int kvh = h / (H / Hkv);  // GQA: query head h reads kv head h / (H/Hkv) (reference.py)

  // 64-bit offsets: B*H*N*D elements can exceed 2^31 in the sweep's largest configs.
  const std::size_t q_off = (static_cast<std::size_t>(bh) * N + i0) * D;
  const std::size_t kv_off = static_cast<std::size_t>(b * Hkv + kvh) * N * D;
  const T* __restrict__ qg = q + q_off;   // this block's Q tile, [rows_q][D] contiguous
  T* __restrict__ og = o + q_off;         // and its O tile
  const T* __restrict__ kg = k + kv_off;  // the head's K, [N][D]
  const T* __restrict__ vg = v + kv_off;  // the head's V, [N][D]

  // ---- Q tile -> smem once, then this warp's 16-row strip -> matrix_a fragments, kept in
  // registers for every K/V tile. Rows past N (partial Q tile) are zero-filled.
  const int rows_q = min(kBr, N - i0);
  QCopy::load(qg, rows_q, Qs);
  __syncthreads();
  FragQ qf[kDFrags];
#pragma unroll
  for (int kk = 0; kk < kDFrags; ++kk)
    wmma::load_matrix_sync(qf[kk], Qs + (warp * kFrag) * kLdT + kk * kFrag, kLdT);
  // Qs is dead for this warp from here; the first tile's barrier makes that true block-wide.

  FragAcc oacc[kDFrags];
#pragma unroll
  for (int n = 0; n < kDFrags; ++n)
    wmma::fill_fragment(oacc[n], 0.0f);

  // Softmax lane mapping: row r of the block, columns [half*BC/2, +BC/2) (and [half*D/2, +D/2)
  // of the O strip). Both lanes of a row carry identical m, l.
  const int r = warp * kFrag + (lane >> 1);
  const int half = lane & 1;
  const int i = i0 + r;                          // GLOBAL query row of this lane
  const bool row_valid = i < N;                  // false only in a partial last Q tile
  float* const Sw = Ss + (warp * kFrag) * kLdS;  // this warp's staging strip
  float* const srow = Sw + (lane >> 1) * kLdS + half * kHalfC;  // lane's half row of S
  float* const orow = Sw + (lane >> 1) * kLdS + half * kHalfD;  // lane's half row of O
  T* const prow = Ps + r * kLdP + half * kHalfC;                // lane's half row of P
  const T* const pwarp = Ps + (warp * kFrag) * kLdP;            // this warp's P strip

  float m = fa::kNegInf;  // running max of RAW scores
  float l = 0.0f;         // running sum of exp2((s - m) * c)

  // Causal: keys j >= min(i0+BR, N) are invisible to every row of this block, so those tiles
  // are skipped BEFORE any load (reference.py: `if j0 > i1 - 1: continue`). n_tiles >= 1.
  const int kv_end = causal ? min(i0 + kBr, N) : N;
  const int n_tiles = (kv_end + BC - 1) / BC;

  // PREFETCH: tile 0 takes the direct route (there is nothing to overlap it with); the loop's
  // first RAW barrier publishes it. Tiles 1.. arrive through the register stage.
  if constexpr (PREFETCH) {
    KvCopy::load(kg, min(BC, N), Ks);
    KvCopy::load(vg, min(BC, N), Vs);
  }

  for (int t = 0; t < n_tiles; ++t) {
    const int j0 = t * BC;
    const int rows_kv = min(BC, N - j0);  // < BC only for a partial last tile (BC=64, N=96)

    // (1) cooperative K/V tile loads (direct path), then the RAW barrier: Ks/Vs hold tile t
    // (also the Qs -> Ps hand-over on t=0).
    if constexpr (!PREFETCH) {
      KvCopy::load(kg + static_cast<std::size_t>(j0) * D, rows_kv, Ks);
      KvCopy::load(vg + static_cast<std::size_t>(j0) * D, rows_kv, Vs);
    }
    __syncthreads();

    // (1') PREFETCH: issue tile t+1's global loads into registers NOW, before any of this tile's
    // tensor-core / softmax work, so their latency overlaps it. Nothing touches Ks/Vs here; the
    // registers are only written to smem after the WAR barrier at (6). Rows >= N zero-filled as
    // in the direct path. Block-uniform branch (t and n_tiles are the same for every thread).
    Stage stage;
    const bool has_next = PREFETCH && (t + 1 < n_tiles);
    if (has_next) {
      const int j1 = j0 + BC;
      stage.fetch(kg + static_cast<std::size_t>(j1) * D, vg + static_cast<std::size_t>(j1) * D,
                  min(BC, N - j1));
    }

    // (2) S strip = Q_strip * Kt on tensor cores, one 16x16 column fragment at a time (only one
    // fp32 accumulator live), Kt for free via matrix_b col_major (wmma_gemm.cu step 2), stored
    // into the warp's fp32 strip in a known row-major layout (the escape hatch).
#pragma unroll
    for (int nf = 0; nf < kNFrags; ++nf) {
      FragAcc sacc;
      wmma::fill_fragment(sacc, 0.0f);
#pragma unroll
      for (int kk = 0; kk < kDFrags; ++kk) {
        FragKt kf;
        wmma::load_matrix_sync(kf, Ks + (nf * kFrag) * kLdT + kk * kFrag, kLdT);
        wmma::mma_sync(sacc, qf[kk], kf, sacc);
      }
      wmma::store_matrix_sync(Sw + nf * kFrag, sacc, kLdS, wmma::mem_row_major);
    }
    __syncwarp();  // strip complete and visible to every lane of this warp

    // (3) online softmax on the strip, warp-local. Lane reads its BC/2 raw scores (float4).
    float s[kHalfC];
#pragma unroll
    for (int c = 0; c < kHalfC; c += 4) {
      const float4 x = *reinterpret_cast<const float4*>(srow + c);
      s[c + 0] = x.x;
      s[c + 1] = x.y;
      s[c + 2] = x.z;
      s[c + 3] = x.w;
    }
    // Mask on RAW scores, only for tiles that need it (block-uniform): a partial last tile
    // (keys >= N) or a tile straddling the causal diagonal for at least one row of the block.
    // Per element with GLOBAL indices: key jj is masked iff jj >= N or (causal and jj > i).
    if (rows_kv < BC || (causal && j0 + BC - 1 > i0)) {
#pragma unroll
      for (int c = 0; c < kHalfC; ++c) {
        const int jj = j0 + half * kHalfC + c;
        if (jj >= N || (causal && jj > i)) s[c] = fa::kNegInf;
      }
    }
    float tile_max = s[0];
#pragma unroll
    for (int c = 1; c < kHalfC; ++c)
      tile_max = fmaxf(tile_max, s[c]);
    tile_max = fmaxf(tile_max, __shfl_xor_sync(kFullMask, tile_max, 1));  // the row's other half
    const float m_new = fmaxf(m, tile_max);
    // alpha = exp2((m - m_new) * c), 0 when m == -inf. It rescales BOTH l and O (BUG 1).
    const float alpha = fa::rescale_alpha(m, m_new, scale_log2);
    // p = exp2(s*c - m_new*c), m_new*c hoisted; the m_new == -inf guard of p_from_score hoisted
    // out of the loop (every p is 0 then). Masked s = -inf -> exp2f(-inf) = 0 (c > 0).
    float psum = 0.0f;
    if (m_new != fa::kNegInf) {
      const float m_c = m_new * scale_log2;
#pragma unroll
      for (int c = 0; c < kHalfC; ++c) {
        const float p = fa::exp2_scaled(s[c], scale_log2, m_c);
        psum += p;  // row sum from the fp32 p, before rounding to T (FA2 softmax.h)
        s[c] = p;
      }
    } else {
#pragma unroll
      for (int c = 0; c < kHalfC; ++c)
        s[c] = 0.0f;
    }
    psum += __shfl_xor_sync(kFullMask, psum, 1);
    l = alpha * l + psum;
    // P (as T) into the dead Q tile, 16-B vectors of 8 packed pairs.
#pragma unroll
    for (int c = 0; c < kHalfC; c += kVecElems)
      *reinterpret_cast<uint4*>(prow + c) = pack8<T>(s + c);
    const float m_old = m;
    m = m_new;  // AFTER alpha and p used the old / new value

    // (4) O <- alpha * O, row-wise, so it must go through smem (opaque fragment layout).
    // Skipped when no row of the warp needs it (warp vote, exact): alpha == 1 is the identity
    // (the running max did not move — the common case once it stabilises) and m_old == -inf
    // means this warp's O rows are still exactly 0 (every earlier p was 0).
    const bool trivial = (alpha == 1.0f) || (m_old == fa::kNegInf);
    if (!__all_sync(kFullMask, trivial)) {
      __syncwarp();  // every lane has read its S values before O overwrites the strip
#pragma unroll
      for (int n = 0; n < kDFrags; ++n)
        wmma::store_matrix_sync(Sw + n * kFrag, oacc[n], kLdS, wmma::mem_row_major);
      __syncwarp();
#pragma unroll
      for (int c = 0; c < kHalfD; c += 4) {
        float4 x = *reinterpret_cast<const float4*>(orow + c);
        x.x *= alpha;
        x.y *= alpha;
        x.z *= alpha;
        x.w *= alpha;
        *reinterpret_cast<float4*>(orow + c) = x;
      }
      __syncwarp();
#pragma unroll
      for (int n = 0; n < kDFrags; ++n)
        wmma::load_matrix_sync(oacc[n], Sw + n * kFrag, kLdS, wmma::mem_row_major);
    }
    __syncwarp();  // P strip visible to load_matrix_sync (and O strip round trip complete)

    // (5) O += P * V on tensor cores: P re-enters as matrix_a row_major (ldm BC+8) from the warp's
    // strip, V as matrix_b row_major (K x N = keys x d, ldm D+8). One P fragment live at a time.
#pragma unroll
    for (int kk = 0; kk < kNFrags; ++kk) {
      FragP pf;
      wmma::load_matrix_sync(pf, pwarp + kk * kFrag, kLdP);
#pragma unroll
      for (int n = 0; n < kDFrags; ++n) {
        FragV vf;
        wmma::load_matrix_sync(vf, Vs + (kk * kFrag) * kLdT + n * kFrag, kLdT);
        wmma::mma_sync(oacc[n], pf, vf, oacc[n]);
      }
    }

    // (6) WAR barrier: every warp is done reading Ks (step 2) and Vs (step 5), so the next tile
    // may overwrite them — by the direct loads at the top of the next iteration, or by the
    // prefetch stores right here; after the last tile it fences the O staging below.
    __syncthreads();

    // (7) PREFETCH: registers -> Ks/Vs. Strictly after this tile's WAR barrier and before the
    // next tile's RAW barrier; no lane reads Ks/Vs in between. Empty when !PREFETCH or on the
    // last tile (has_next is block-uniform, so no thread stores while another could read).
    if (has_next) stage.store(Ks, Vs);
  }

  // ---- Epilogue: O fragments -> the warp's fp32 strip, each lane scales its half row by
  // 1/l (finalize_inv_l: 0 for an empty row -> o = 0), lse in natural-log units with the raw m
  // re-scaled (finalize_lse: +inf for an empty row), then the block stores O coalesced as T.
#pragma unroll
  for (int n = 0; n < kDFrags; ++n)
    wmma::store_matrix_sync(Sw + n * kFrag, oacc[n], kLdS, wmma::mem_row_major);
  __syncwarp();
  const float inv_l = fa::finalize_inv_l(l);
#pragma unroll
  for (int c = 0; c < kHalfD; c += 4) {
    float4 x = *reinterpret_cast<const float4*>(orow + c);
    x.x *= inv_l;
    x.y *= inv_l;
    x.z *= inv_l;
    x.w *= inv_l;
    *reinterpret_cast<float4*>(orow + c) = x;
  }
  if (half == 0 && row_valid)
    lse[static_cast<std::size_t>(bh) * N + i] = fa::finalize_lse(m, l, scale);
  __syncthreads();
  // Coalesced store: consecutive threads take consecutive 16-B vectors of the contiguous
  // [rows_q][D] O tile, exactly the inverse of the Q load. Rows >= N are never written.
  constexpr int kVecs = kBr * D / kVecElems;
#pragma unroll
  for (int it = 0; it < kVecs / kThreads; ++it) {
    const int e = (it * kThreads + tid) * kVecElems;
    const int rr = e / D;
    const int d = e - rr * D;
    if (rr < rows_q) {
      const float* src = Ss + rr * kLdS + d;
      float f[kVecElems];
      const float4 a = *reinterpret_cast<const float4*>(src);
      const float4 c4 = *reinterpret_cast<const float4*>(src + 4);
      f[0] = a.x, f[1] = a.y, f[2] = a.z, f[3] = a.w;
      f[4] = c4.x, f[5] = c4.y, f[6] = c4.z, f[7] = c4.w;
      *reinterpret_cast<uint4*>(og + e) = pack8<T>(f);
    }
  }
}

}  // namespace stage5
}  // namespace fa

// ---------------------------------------------------------------------------------------
// The kernel. Global scope and this exact name are fixed by CONTRACT 2 (ncu --kernel-name
// regex:fa_s5_tuned matches the mangled prefix). Template on (input type, head dim, warps per
// block, KV tile, prefetch). bf16 on an sm < 80 pass is a trap stub (no bf16 fragments exist
// there); supports() keeps it from ever being launched, and the launcher double-checks the
// loaded binary's version.
// ---------------------------------------------------------------------------------------
template <typename T, int D, int WARPS, int BC, bool PREFETCH>
__global__ void __launch_bounds__(fa::stage5::Cfg<D, WARPS, BC>::kThreads)
    fa_s5_tuned(const T* __restrict__ q, const T* __restrict__ k, const T* __restrict__ v,
                T* __restrict__ o, float* __restrict__ lse, int N, int H, int Hkv, float scale,
                float scale_log2, bool causal) {
  if constexpr (std::is_same_v<T, __nv_bfloat16> && !fa::stage5::kArchHasBf16Wmma) {
    __trap();
  } else {
    fa::stage5::attention_block<T, D, WARPS, BC, PREFETCH>(q, k, v, o, lse, N, H, Hkv, scale,
                                                           scale_log2, causal);
  }
}

// ---------------------------------------------------------------------------------------
// Host side: fa::stage5::forward / supports with the exact ABI of src/dispatch.cu.
// ---------------------------------------------------------------------------------------
namespace fa {
namespace stage5 {
namespace {

bool verbose() {
  static const bool v = std::getenv("FA_VERBOSE") != nullptr;
  return v;
}

// FA_S5_PREFETCH: unset or "1" -> on, "0" -> off. Read once per process, like FA_VERBOSE.
bool prefetch_enabled() {
  static const bool on = [] {
    const char* s = std::getenv("FA_S5_PREFETCH");
    if (s == nullptr || *s == '\0') return true;
    if (std::strcmp(s, "0") == 0) return false;
    if (std::strcmp(s, "1") != 0 && verbose()) {
      std::fprintf(stderr, "fa::stage5: FA_S5_PREFETCH='%s' is not 0 or 1; treating it as 1\n", s);
    }
    return true;
  }();
  return on;
}

bool tall_tile_128(int N) {
  return N >= kTallTileMinN128;
}

int default_bc(int D, int N) {
  // D=64 keeps S4's BC from tile_config.h (the static_assert keeps the two files from drifting);
  // D=128 picks by N (kTallTileMinN128). FA_FORCE_BC overrides both.
  static_assert(TileCfg<kArchTuring, 64>::kBcF16 == kDefaultBc64, "S5 D=64 default BC drifted");
  if (D == 64) return kDefaultBc64;
  return tall_tile_128(N) ? kTallBc128 : kShortBc128;
}

int default_warps(int D, int N) {
  if (D == 64) return kDefaultWarps64;
  return tall_tile_128(N) ? kTallWarps128 : kShortWarps128;
}

bool aligned16(const void* p) {
  return (reinterpret_cast<std::uintptr_t>(p) & 15u) == 0;
}

// Once per (kernel instantiation, device): opt in to > 48 KB of dynamic smem when the tile needs
// it, and refuse to launch a bf16 stub (a binary whose only code for this device is an sm < 80
// pass, e.g. PTX JIT-compiled from compute_75). Attributes are per (kernel, device); cached per
// device so the benchmark's back-to-back launches do not pay a runtime call each time.
template <typename T, int D, int WARPS, int BC, bool PREFETCH>
cudaError_t prepare_once() {
  using C = Cfg<D, WARPS, BC>;
  int dev = 0;
  FA_CUDA_TRY(cudaGetDevice(&dev));
  static std::atomic<std::uint64_t> ready{0};  // one bit per device id < 64
  if (dev < 64 && ((ready.load(std::memory_order_acquire) >> dev) & 1u)) return cudaSuccess;
  if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    cudaFuncAttributes attr;
    FA_CUDA_TRY(cudaFuncGetAttributes(&attr, fa_s5_tuned<T, D, WARPS, BC, PREFETCH>));
    if (attr.binaryVersion < 80) {
      if (verbose()) {
        std::fprintf(stderr,
                     "fa::stage5: bf16 kernel in this binary was built for sm_%d (< 80): trap "
                     "stub, rebuild with an sm_80+ architecture\n",
                     attr.binaryVersion);
      }
      return cudaErrorNotSupported;
    }
  }
  if constexpr (C::kNeedsOptIn) {
    // Opt in above 48 KB (Turing tuning guide: a block may address the full 64 KB only after
    // cudaFuncSetAttribute); the return value is checked because it fails on a smaller device.
    FA_CUDA_TRY(cudaFuncSetAttribute(fa_s5_tuned<T, D, WARPS, BC, PREFETCH>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     static_cast<int>(C::kBytes)));
  }
  if (dev < 64) ready.fetch_or(std::uint64_t{1} << dev, std::memory_order_acq_rel);
  return cudaSuccess;
}

template <typename T, int D, int WARPS, int BC, bool PREFETCH>
cudaError_t launch(const void* q, const void* k, const void* v, void* o, float* lse,
                   const Params& p, int sm, cudaStream_t stream) {
  using C = Cfg<D, WARPS, BC>;
  if constexpr (C::kNeedsOptIn) {
    const std::size_t cap = max_dyn_smem_bytes(arch_of_sm(sm));
    if (C::kBytes > cap) {  // unreachable for the archs in scope (static_assert), kept honest
      if (verbose()) {
        std::fprintf(stderr,
                     "fa::stage5: D=%d BR=%d BC=%d needs %zu B of shared memory > %zu B "
                     "per-block cap on sm_%d\n",
                     D, C::kBr, BC, C::kBytes, cap, sm);
      }
      return cudaErrorInvalidValue;
    }
  }
  FA_CUDA_TRY((prepare_once<T, D, WARPS, BC, PREFETCH>()));
  const dim3 grid(static_cast<unsigned>((p.N + C::kBr - 1) / C::kBr),
                  static_cast<unsigned>(p.B) * static_cast<unsigned>(p.H));
  fa_s5_tuned<T, D, WARPS, BC, PREFETCH><<<grid, C::kThreads, C::kBytes, stream>>>(
      static_cast<const T*>(q), static_cast<const T*>(k), static_cast<const T*>(v),
      static_cast<T*>(o), lse, p.N, p.H, p.Hkv, p.scale, scale_log2(p.scale), p.causal);
  FA_CUDA_TRY_LAUNCH();
  return cudaSuccess;
}

// ---- The instantiation table (header INSTANTIATIONS). One row per (D, WARPS, BC) that fits the
// T4 cap; each row yields the PREFETCH=false and PREFETCH=true kernels. A prefetch variant that
// spills on sm_75 is excluded here with kPrefetchOk (one does: D=64/WARPS=1/BC=64, header table).
using LaunchFn = cudaError_t (*)(const void*, const void*, const void*, void*, float*,
                                 const Params&, int, cudaStream_t);

struct Variant {
  int D;
  int warps;
  int bc;
  bool prefetch;
  LaunchFn fn;  // nullptr = this PREFETCH flavour is deliberately not instantiated
};

template <int D, int WARPS, int BC>
constexpr bool kPrefetchOk = true;
// D=64/WARPS=1/BC=64: 256 staging halves per thread (128 registers) on top of the 168 the
// direct variant already uses -> ptxas 255 registers + 56 B spill stores / 68 B spill loads on
// sm_75. Dropped (0 spills is the rule); FA_FORCE_BR=16 FA_FORCE_BC=64 needs FA_S5_PREFETCH=0.
template <>
constexpr bool kPrefetchOk<64, 1, 64> = false;

template <typename T, int D, int WARPS, int BC, bool PREFETCH>
LaunchFn launch_ptr() {
  if constexpr (PREFETCH && !kPrefetchOk<D, WARPS, BC>) {
    return nullptr;
  } else {
    return &launch<T, D, WARPS, BC, PREFETCH>;
  }
}

#define FA_S5_VARIANTS(X) \
  X(64, 1, 16)            \
  X(64, 1, 32)            \
  X(64, 1, 64)            \
  X(64, 2, 16)            \
  X(64, 2, 32)            \
  X(64, 2, 64)            \
  X(64, 4, 16)            \
  X(64, 4, 32)            \
  X(64, 4, 64)            \
  X(64, 8, 16)            \
  X(64, 8, 32)            \
  X(128, 2, 16)           \
  X(128, 2, 32)           \
  X(128, 4, 16)

template <typename T>
const Variant* variants(int* count) {
#define FA_S5_ROW(D_, W_, BC_)                                      \
  Variant{D_, W_, BC_, false, launch_ptr<T, D_, W_, BC_, false>()}, \
      Variant{D_, W_, BC_, true, launch_ptr<T, D_, W_, BC_, true>()},
  static const Variant table[] = {FA_S5_VARIANTS(FA_S5_ROW)};
#undef FA_S5_ROW
  *count = static_cast<int>(sizeof(table) / sizeof(table[0]));
  return table;
}
#undef FA_S5_VARIANTS

template <typename T>
LaunchFn find_variant(int D, int warps, int bc, bool prefetch) {
  int n = 0;
  const Variant* tab = variants<T>(&n);
  for (int idx = 0; idx < n; ++idx) {
    const Variant& e = tab[idx];
    if (e.D == D && e.warps == warps && e.bc == bc && e.prefetch == prefetch) return e.fn;
  }
  return nullptr;
}

template <typename T>
void print_valid_set(int D) {
  int n = 0;
  const Variant* tab = variants<T>(&n);
  std::fprintf(stderr,
               "fa::stage5: instantiated (FA_FORCE_BR, FA_FORCE_BC, FA_S5_PREFETCH) for D=%d:", D);
  for (int idx = 0; idx < n; ++idx) {
    const Variant& e = tab[idx];
    if (e.D == D && e.fn != nullptr) {
      std::fprintf(stderr, " (%d,%d,%d)", e.warps * kFrag, e.bc, e.prefetch ? 1 : 0);
    }
  }
  std::fprintf(stderr, "\n");
}

template <typename T>
cudaError_t launch_selected(int warps, int bc, bool prefetch, const void* q, const void* k,
                            const void* v, void* o, float* lse, const Params& p, int sm,
                            cudaStream_t stream) {
  const LaunchFn fn = find_variant<T>(p.D, warps, bc, prefetch);
  if (fn == nullptr) {
    if (verbose()) {
      std::fprintf(stderr,
                   "fa::stage5: D=%d FA_FORCE_BR=%d (WARPS=%d) FA_FORCE_BC=%d FA_S5_PREFETCH=%d is "
                   "not an instantiated variant\n",
                   p.D, warps * kFrag, warps, bc, prefetch ? 1 : 0);
      print_valid_set<T>(p.D);
    }
    return cudaErrorInvalidValue;
  }
  return fn(q, k, v, o, lse, p, sm, stream);
}

}  // namespace

bool supports(Dtype dtype, int D, int sm) {
  if (D != 64 && D != 128) return false;
  switch (dtype) {
    case Dtype::F32:
      return false;  // tensor-core stage: fp16/bf16 operands only (attention.cuh)
    case Dtype::F16:
      return sm >= 75;  // Turing tensor cores: fp16 in, fp32 accumulate
    case Dtype::BF16:
      return sm >= 80;  // bf16 WMMA fragments exist from compute capability 8.0
  }
  return false;
}

cudaError_t forward(Dtype dtype, const void* q, const void* k, const void* v, void* o, float* lse,
                    void* /*workspace: fused stages need none (workspace_bytes == 0)*/,
                    const Params& p, const DebugOverrides& dbg, int sm, cudaStream_t stream) {
  // dispatch.cu already validated shapes, pointers and N % 32; the checks below are the ones
  // specific to this kernel's launch geometry and numerics. Messages only under FA_VERBOSE
  // (API contract: never aborts, never spams).
  if (!supports(dtype, p.D, sm)) return cudaErrorNotSupported;
  if (lse == nullptr) {
    if (verbose()) std::fprintf(stderr, "fa::stage5: lse must be non-null (fused stages)\n");
    return cudaErrorInvalidValue;
  }
  if (p.N % 32 != 0 || p.H % p.Hkv != 0) {
    if (verbose()) std::fprintf(stderr, "fa::stage5: N %% 32 != 0 or H %% Hkv != 0\n");
    return cudaErrorInvalidValue;
  }
  if (static_cast<long long>(p.B) * p.H > 65535LL) {
    if (verbose()) {
      std::fprintf(stderr, "fa::stage5: B*H = %lld exceeds gridDim.y (65535)\n",
                   static_cast<long long>(p.B) * p.H);
    }
    return cudaErrorInvalidValue;
  }
  // Masking is applied to RAW scores before the fold into c = scale*log2e; a non-positive or
  // non-finite scale would turn the -inf sentinel into +inf/NaN (reference.py shares the
  // convention and the limitation).
  if (!(p.scale > 0.0f) || !std::isfinite(p.scale)) {
    if (verbose()) std::fprintf(stderr, "fa::stage5: scale must be finite and > 0\n");
    return cudaErrorInvalidValue;
  }
  // 16-B vector loads/stores; rows are D*2 = 128 B / 256 B so every tile base inherits it.
  if (!aligned16(q) || !aligned16(k) || !aligned16(v) || !aligned16(o)) {
    if (verbose()) std::fprintf(stderr, "fa::stage5: q/k/v/o must be 16-byte aligned\n");
    return cudaErrorInvalidValue;
  }
  // FA_FORCE_BR -> WARPS = BR / 16 (BR must be a positive multiple of 16); 0 -> the default
  // height for (D, N) (S4's at D=64; at D=128 4 warps from N=2048 up, S4's 2 below).
  int warps = default_warps(p.D, p.N);
  if (dbg.block_m != 0) {
    if (dbg.block_m % kFrag != 0) {
      if (verbose()) {
        std::fprintf(stderr,
                     "fa::stage5: FA_FORCE_BR=%d is not a multiple of %d (16 rows per warp)\n",
                     dbg.block_m, kFrag);
        print_valid_set<__half>(p.D);
      }
      return cudaErrorInvalidValue;
    }
    warps = dbg.block_m / kFrag;
  }
  const int bc = dbg.block_n != 0 ? dbg.block_n : default_bc(p.D, p.N);
  const bool prefetch = prefetch_enabled();
  switch (dtype) {
    case Dtype::F16:
      return launch_selected<__half>(warps, bc, prefetch, q, k, v, o, lse, p, sm, stream);
    case Dtype::BF16:
      return launch_selected<__nv_bfloat16>(warps, bc, prefetch, q, k, v, o, lse, p, sm, stream);
    case Dtype::F32:
      break;
  }
  return cudaErrorNotSupported;
}

}  // namespace stage5
}  // namespace fa
