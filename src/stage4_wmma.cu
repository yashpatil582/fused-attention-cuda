// S4: the fused kernel of S3 with both GEMMs on tensor cores — nvcuda::wmma 16x16x16, fp16 (bf16
// on sm_80+) operands, fp32 accumulate. Same online softmax (Milakov & Gimelshein 2018), same
// FlashAttention-2 loop order, same deferred normalisation, same guards as S3. Correctness first;
// S5 tunes it.
//
// WHAT THIS STAGE MOVES (the ncu metric): S3 does every multiply-add as one FFMA whose operands
// come from shared memory or registers (1 LDS.128 per 4 FFMA), so its tensor pipe is idle
// (sm__pipe_tensor_op_hmma_cycles_active == 0) and its top stalls are mio_throttle and barrier.
// One wmma::mma_sync performs a 16x16x16 tile (4,096 multiply-adds per warp) from three fragments
// already in registers; the A fragment feeds 16 output columns instead of one, so the operand
// fetches per FLOP drop by ~16x compared with S3's FFMA stream. Expected on T4:
//   sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active  0 (S3) -> clearly > 0
//   sm__inst_executed_pipe_tensor.sum                                       0 (S3) -> > 0
//   smsp__sass_thread_inst_executed_op_ffma_pred_on.sum                    down by the GEMM share
//     of S3's FFMAs (what remains is the softmax: BC exp2 + a few FMAs per row per K/V tile).
// Likely new top stalls: barrier (two block barriers per K/V tile at 1-2 blocks per SM, see
// OCCUPANCY) and mio_throttle (the shared-memory round trips below). Those are S5's targets.
//
// WHY SHARED-MEMORY ROUND TRIPS: WMMA fragment layouts are opaque — "the way matrix elements map
// to fragment internal storage is unspecified" (CUDA C++ Programming Guide, Warp Matrix Functions)
// and "the mapping between individual matrix parameters to their fragments is opaque, so your
// program should not make assumptions about it" (NVIDIA blog, Programming Tensor Cores in CUDA 9;
// both quoted in .research/tech-tensor-core-programming-constraints-by-n.json). Element-wise work
// on a fragment is legal, but everything attention needs between the two GEMMs is
// position-dependent: the row max/sum, the causal/partial-tile mask, and feeding P back as the A
// operand of P*V. The only portable route is store_matrix_sync into smem in a KNOWN row-major
// layout, do the row-wise work there, load_matrix_sync back — exactly what
// experiments/wmma_gemm/wmma_gemm.cu de-risked. Per K/V tile that is: S out (fp32), P in (half),
// and — because O <- alpha*O + P*V is row-wise too — O out, rescale, O in (skipped when no row of
// the warp needs it, see step 4). FlashAttention-2 avoids all of this with raw mma.sync, whose
// PTX-documented lane<->element map lets it keep S and P in registers; that is out of scope here
// (PLAN.md W6) and is the honest ceiling of a WMMA kernel.
//
// THREAD MAPPING: warp w owns query rows 16w..16w+15 of the block for EVERYTHING — the S strip
// (16 x BC), the softmax statistics, the P strip and the O strip — the FA2 "each warp owns whole
// rows" partition, so the online-softmax bookkeeping never crosses warps and the only block-wide
// dependencies are the cooperative K/V tile loads. Inside the softmax, lane owns row 16w + lane/2
// and one half of its columns ([(lane&1)*BC/2, +BC/2)); the two lanes of a row combine with ONE
// __shfl_xor_sync(.., 1) for the max and one for the sum, and both hold identical m, l, alpha.
//   BR = 64 (4 warps, 128 threads) for D = 64; BR = 32 (2 warps, 64 threads) for D = 128 — the
//   smem budget below forces the smaller tile at D = 128 (the fp32 staging strip is D wide).
// Q fragments (D/16 matrix_a, row_major) are loaded ONCE per block and stay in registers for the
// whole K/V loop (fp16: 16 halves = 8 registers per fragment, 32 at D = 64, 64 at D = 128; bf16
// fragments hold 8 elements = 4 registers, mma.h); the O accumulator is D/16 fp32 accumulator
// fragments (8 registers each: 32 / 64), also in registers across tiles.
//
// SHARED MEMORY (dynamic, one carve per block, 128-B aligned base; halves for T tiles):
//   Qs [BR][D+8]   T     Q tile; DEAD after the Q fragment loads -> P [BR][BC+8] aliases it.
//   Ks [BC][D+8]   T     K tile, read as matrix_b COL_MAJOR: element (k, n) of Kt sits at
//                        ptr[n*ldm + k] = Ks[n][k], i.e. where row-major K already is. Kt is
//                        free (wmma_gemm.cu step 2).
//   Vs [BC][D+8]   T     V tile, matrix_b row_major (K x N = keys x d).
//   Ss [BR][max(BC,D)+4] fp32 per-warp strips: S for the softmax, then O for the rescale round
//                        trip and the epilogue. ldm BC+4 would be too narrow for the D-wide O
//                        strip when D > BC, hence max(BC, D).
// WHY P aliases Qs and not Ks (wmma_gemm.cu put P in the dead K tile): the P strip needs
// BR*(BC+8) halves, which exceeds the Ks buffer BC*(D+8) for D=64/BC=16 (1,536 > 1,152) and
// D=64/BC=32 (2,560 > 2,304), while BC <= D makes BR*(BC+8) <= BR*(D+8) hold for every tile in
// scope (static_assert). Qs is dead once every warp holds its Q fragments — the first tile's RAW
// barrier publishes that — so P needs NO extra "all warps finished the S GEMM" barrier, and Ks
// stays free during the softmax + P*V phase (an S5 prefetch slot). Vs could alias Qs instead;
// it does not, because P uses it.
// WHY +8 halves / +4 floats (16 B) of row padding: load/store_matrix_sync require ldm to be a
// multiple of 8 halves / 4 floats (16 B) and every fragment pointer to be 256-bit aligned, so
// the pad must itself be a multiple of 8 halves (+1 is illegal). The same 16 B also shifts
// consecutive rows by 4 banks: an unpadded 128-B / 256-B row maps the same column of every row
// to one bank, a 144-B / 272-B stride spreads 8 consecutive rows over all 32 banks for 16-B
// chunk walks (tile_config.h). Expected from the bank arithmetic; S5a measures it with
// l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum.
// Bytes = BR*(D+8)*2 + 2*BC*(D+8)*2 + BR*(max(BC,D)+4)*4:
//   D=64,  BR=64: BC=16 31,232 | BC=32 35,840 | BC=64 45,056 (default)
//   D=128, BR=32: BC=16 34,304 | BC=32 43,008 (default) | BC=64 60,416 -> > 48 KB, dynamic opt-in
// Every value is static_asserted against the per-arch caps of include/fa/tile_config.h (the
// Turing 64 KB cap is the binding one); the launcher opts in above 48 KB exactly like S3.
// FRAGMENT POINTER ALIGNMENT (256-bit): row strides 144 B / 272 B (T) and 272 B / 528 B (fp32)
// are NOT multiples of 32 B, but every fragment pointer below starts at a row that is a multiple
// of 16 (16*144 = 2,304 = 72*32; 16*272 = 8,704 = 272*32; 16*528 = 8,448 = 264*32) and a column
// that is a multiple of 16 elements (32 B for T, 64 B for fp32) from a 128-B-aligned region
// base (all region offsets are multiples of 128 B, static_asserted). Same rule as the microkernel.
//
// OCCUPANCY (T4: 64 KB smem per SM, 64K registers, 32 warps, 16 blocks) — stated honestly:
//   D=64/BC=64 (45,056 B)  -> 1 block/SM = 4 warps = 12.5% of the 32-warp ceiling
//   D=64/BC=32 (35,840 B)  -> 1 block/SM = 4 warps;  D=64/BC=16 (31,232 B) -> 2 blocks = 8 warps
//   D=128/BC=32 (43,008 B) -> 1 block/SM = 2 warps = 6.25%; BC=16 -> 1 block; BC=64 -> 1 block
// Shared memory binds before registers do: ptxas -v (nvcc 12.8, sm_75, fp16, 0 spills in every
// instantiation) D=64: BC=16 92, BC=32 113, BC=64 151 registers -> 5 / 4 / 3 blocks by registers;
// D=128: BC=16 168, BC=32 168, BC=64 253 -> 6 / 6 / 4 blocks (64 threads). D=128/BC=64 sits at
// the 255 ceiling with 0 spills (Q + O fragments alone are 128 registers); it is the FA_FORCE_BC
// experiment, not a default. The tensor-core win must therefore come from per-warp throughput,
// not from latency hiding — which is why the round trips and barriers above show up as the new
// stalls and why S5 (prefetch, smaller staging, launch bounds) exists.
//
// SOFTMAX CONVENTION (identical to include/fa/softmax_math.cuh, S3, python/fused_attn/
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
// fp32 fragments/registers, so scores past 65,504 and sums past fp16 range are safe; p in [0, 1]
// always fits in half. The scale is applied exactly once, inside c (BUG 4).
//
// CAUSAL / PARTIAL TILES: top-left aligned, key j visible to query i iff j <= i; the mask uses
// GLOBAL i and j with a strict `j > i` (BUG 5) on RAW scores, only for tiles that need it
// (block-uniform test, as in S3). Tiles entirely above the diagonal are skipped by the loop bound
// n_tiles = ceil(min(i0+BR, N) / BC) (reference.py `j0 > i1 - 1: continue`). Rows past N of a
// partial Q tile and keys past N of a partial K/V tile are zero-filled in smem; the keys are
// also masked to -inf so p = 0 and 0 * 0 (never 0 * garbage = NaN); the rows are computed as
// finite garbage and never stored.
//
// GUARDS (BUG 2): m == -inf (no unmasked key so far) -> alpha = 0 (l and O are still 0, exact),
// p = 0 for the whole tile; at the end finalize_inv_l / finalize_lse turn an empty row into
// o = 0, lse = +inf. Impossible for top-left causal with Nq == Nkv; kept so the invariant is local.
//
// BARRIERS (BUG 8): exactly two __syncthreads per K/V tile — after the cooperative K/V load
// (RAW; on the first tile it also fences the Q fragment loads so Qs may become P) and after the
// P*V GEMM (WAR: the next tile's loads overwrite Ks/Vs; after the last tile it fences the O
// staging). Everything else is warp-private and ordered by __syncwarp: store_matrix_sync ->
// lane reads, lane writes -> load_matrix_sync.
//
// TILE DEFAULTS: BC = TileCfg<kArchTuring, D>::kBcF16 (64 for D=64, 32 for D=128). FA_FORCE_BC
// selects among {16, 32, 64} (all instantiated for both D); anything else is
// cudaErrorInvalidValue. FA_FORCE_BR is IGNORED (reported once under FA_VERBOSE): BR is fixed by
// the warp partition and the smem budget, and the tile-invariance test only needs BC to vary.
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
#include <type_traits>

#include "fa/attention.cuh"
#include "fa/cuda_check.cuh"
#include "fa/softmax_math.cuh"
#include "fa/tile_config.h"

namespace fa {
namespace stage4 {

constexpr int kFrag = 16;                                 // WMMA m = n = k
constexpr int kPadT = TileCfg<kArchTuring, 64>::kPadF16;  // +8 halves = 16 B per T row
constexpr int kPadF = TileCfg<kArchTuring, 64>::kSPad;    // +4 floats = 16 B per fp32 row
constexpr int kVecElems = 8;                              // one 16-B vector = 8 halves/bf16
constexpr unsigned kFullMask = 0xffffffffu;

// Instantiated BC values. Anything else from FA_FORCE_BC is rejected in forward().
constexpr int kBcMin = 16;
constexpr int kBcMid = 32;
constexpr int kBcMax = 64;

// bf16 WMMA fragments exist only for compute capability >= 8.0: mma.h declares them under
// `!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800`, so the host pass and sm_80+ device passes
// see them and an sm_75 device pass does not. The kernel discards its body with `if constexpr`
// on that pass (wmma_gemm.cu's fa_wmma_bf16_kernel does the same with #if).
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)
constexpr bool kArchHasBf16Wmma = false;
#else
constexpr bool kArchHasBf16Wmma = true;
#endif

// Tile geometry and the shared-memory carve for one (D, BC) instantiation (header comment table).
template <int D, int BC>
struct Cfg {
  static_assert(D == 64 || D == 128, "D must be 64 or 128 (CONTRACT 1)");
  static_assert(BC == kBcMin || BC == kBcMid || BC == kBcMax, "BC must be 16, 32 or 64");

  static constexpr int kBr = (D == 64) ? 64 : 32;  // query rows per block
  static constexpr int kWarps = kBr / kFrag;       // one 16-row strip per warp
  static constexpr int kThreads = kWarps * 32;

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

  // Budget: the per-arch opt-in caps of tile_config.h; Turing's 64 KB is the smallest in scope.
  static_assert(kBytes <= ArchCaps<kArchTuring>::kMaxDynSmemBytes, "S4 tile exceeds the sm_75 cap");
  static_assert(kBytes <= ArchCaps<kArchA100>::kMaxDynSmemBytes, "S4 tile exceeds the sm_80 cap");
  static_assert(kBytes <= ArchCaps<kArchGA10x>::kMaxDynSmemBytes, "S4 tile exceeds the sm_86 cap");
  static_assert(kBytes <= ArchCaps<kArchAda>::kMaxDynSmemBytes, "S4 tile exceeds the sm_89 cap");
  static_assert(kBytes <= ArchCaps<kArchHopper>::kMaxDynSmemBytes, "S4 tile exceeds the sm_90 cap");
  // Alignment: every region starts 128-B aligned so fragment pointers (multiples of 16 rows and
  // 16 columns inside a region) are 256-bit aligned, and 16-B vector loads/stores stay aligned.
  static_assert(kOffK % 128 == 0 && kOffV % 128 == 0 && kOffS % 128 == 0,
                "smem regions must be 128-B aligned");
  static_assert(kLdT % 8 == 0 && kLdP % 8 == 0, "T ldm must be a multiple of 8 elements (16 B)");
  static_assert(kLdS % 4 == 0, "fp32 ldm must be a multiple of 4 floats (16 B)");
  // P aliases the dead Q tile: BR*(BC+8) <= BR*(D+8) because BC <= D for every tile in scope.
  static_assert(static_cast<std::size_t>(kBr) * kLdP * 2 <= kQBytes, "P strip must fit in Qs");
  static_assert(kLdS >= D + kPadF && kLdS >= BC + kPadF, "staging strip must hold S and O rows");
  // Cooperative loads: whole number of 16-B vectors per thread, no vector straddles a row.
  static_assert(D % kVecElems == 0, "D must be a multiple of 8");
  static_assert((kBr * D / kVecElems) % kThreads == 0, "Q tile: vectors per thread");
  static_assert((BC * D / kVecElems) % kThreads == 0, "K/V tile: vectors per thread");
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
// threads walking the contiguous D axis (reality-check BUG 7). Rows >= rows_valid are never read
// and are written as zeros so nothing in smem is ever uninitialised (0 * V = 0, bitwise
// reproducible runs).
template <typename T, int ROWS, int D, int THREADS>
__device__ __forceinline__ void load_tile(const T* __restrict__ g, int rows_valid,
                                          T* __restrict__ dst) {
  constexpr int kLd = D + kPadT;
  constexpr int kVecs = ROWS * D / kVecElems;
  const int valid_elems = rows_valid * D;
#pragma unroll
  for (int it = 0; it < kVecs / THREADS; ++it) {
    const int e = (it * THREADS + static_cast<int>(threadIdx.x)) * kVecElems;
    const int r = e / D;
    const int d = e - r * D;
    uint4 x = make_uint4(0u, 0u, 0u, 0u);
    if (e < valid_elems) x = *reinterpret_cast<const uint4*>(g + e);
    *reinterpret_cast<uint4*>(dst + r * kLd + d) = x;
  }
}

// The block body (a __device__ template so the __global__ wrapper can discard it for bf16 on
// sm_75 without instantiating any bf16 fragment type there).
template <typename T, int D, int BC>
__device__ __forceinline__ void attention_block(const T* __restrict__ q, const T* __restrict__ k,
                                                const T* __restrict__ v, T* __restrict__ o,
                                                float* __restrict__ lse, int N, int H, int Hkv,
                                                float scale, float scale_log2, bool causal) {
  using namespace nvcuda;
  using C = Cfg<D, BC>;
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
  load_tile<T, kBr, D, kThreads>(qg, rows_q, Qs);
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

  for (int t = 0; t < n_tiles; ++t) {
    const int j0 = t * BC;
    const int rows_kv = min(BC, N - j0);  // < BC only for a partial last tile (BC=64, N=96)

    // (1) cooperative K/V tile loads, then the RAW barrier (also the Qs -> Ps hand-over on t=0).
    load_tile<T, BC, D, kThreads>(kg + static_cast<std::size_t>(j0) * D, rows_kv, Ks);
    load_tile<T, BC, D, kThreads>(vg + static_cast<std::size_t>(j0) * D, rows_kv, Vs);
    __syncthreads();

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

    // (6) WAR barrier: the next iteration's loads overwrite Ks/Vs (and P/S strips are reused);
    // after the last tile it fences the O staging below.
    __syncthreads();
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

}  // namespace stage4
}  // namespace fa

// ---------------------------------------------------------------------------------------
// The kernel. Global scope and this exact name are fixed by CONTRACT 2 (ncu --kernel-name
// regex:fa_s4_wmma matches the mangled prefix). Template on (input type, head dim, KV tile).
// bf16 on an sm < 80 pass is a trap stub (no bf16 fragments exist there); supports() keeps it
// from ever being launched, and the launcher double-checks the loaded binary's version.
// ---------------------------------------------------------------------------------------
template <typename T, int D, int BC>
__global__ void __launch_bounds__(fa::stage4::Cfg<D, BC>::kThreads)
    fa_s4_wmma(const T* __restrict__ q, const T* __restrict__ k, const T* __restrict__ v,
               T* __restrict__ o, float* __restrict__ lse, int N, int H, int Hkv, float scale,
               float scale_log2, bool causal) {
  if constexpr (std::is_same_v<T, __nv_bfloat16> && !fa::stage4::kArchHasBf16Wmma) {
    __trap();
  } else {
    fa::stage4::attention_block<T, D, BC>(q, k, v, o, lse, N, H, Hkv, scale, scale_log2, causal);
  }
}

// ---------------------------------------------------------------------------------------
// Host side: fa::stage4::forward / supports with the exact ABI of src/dispatch.cu.
// ---------------------------------------------------------------------------------------
namespace fa {
namespace stage4 {
namespace {

bool verbose() {
  static const bool v = std::getenv("FA_VERBOSE") != nullptr;
  return v;
}

int default_bc(int D) {
  // The plan's S4 defaults live in tile_config.h (64 for D=64, 32 for D=128); static_assert
  // keeps the two files from drifting apart.
  static_assert(TileCfg<kArchTuring, 64>::kBcF16 == kBcMax, "S4 D=64 default BC drifted");
  static_assert(TileCfg<kArchTuring, 128>::kBcF16 == kBcMid, "S4 D=128 default BC drifted");
  return D == 64 ? TileCfg<kArchTuring, 64>::kBcF16 : TileCfg<kArchTuring, 128>::kBcF16;
}

bool aligned16(const void* p) {
  return (reinterpret_cast<std::uintptr_t>(p) & 15u) == 0;
}

// Once per (kernel instantiation, device): opt in to > 48 KB of dynamic smem when the tile needs
// it, and refuse to launch a bf16 stub (a binary whose only code for this device is an sm < 80
// pass, e.g. PTX JIT-compiled from compute_75). Attributes are per (kernel, device); cached per
// device so the benchmark's back-to-back launches do not pay a runtime call each time.
template <typename T, int D, int BC>
cudaError_t prepare_once() {
  using C = Cfg<D, BC>;
  int dev = 0;
  FA_CUDA_TRY(cudaGetDevice(&dev));
  static std::atomic<std::uint64_t> ready{0};  // one bit per device id < 64
  if (dev < 64 && ((ready.load(std::memory_order_acquire) >> dev) & 1u)) return cudaSuccess;
  if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    cudaFuncAttributes attr;
    FA_CUDA_TRY(cudaFuncGetAttributes(&attr, fa_s4_wmma<T, D, BC>));
    if (attr.binaryVersion < 80) {
      if (verbose()) {
        std::fprintf(stderr,
                     "fa::stage4: bf16 kernel in this binary was built for sm_%d (< 80): trap "
                     "stub, rebuild with an sm_80+ architecture\n",
                     attr.binaryVersion);
      }
      return cudaErrorNotSupported;
    }
  }
  if constexpr (C::kNeedsOptIn) {
    // Opt in above 48 KB (Turing tuning guide: a block may address the full 64 KB only after
    // cudaFuncSetAttribute); the return value is checked because it fails on a smaller device.
    FA_CUDA_TRY(cudaFuncSetAttribute(fa_s4_wmma<T, D, BC>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     static_cast<int>(C::kBytes)));
  }
  if (dev < 64) ready.fetch_or(std::uint64_t{1} << dev, std::memory_order_acq_rel);
  return cudaSuccess;
}

template <typename T, int D, int BC>
cudaError_t launch(const void* q, const void* k, const void* v, void* o, float* lse,
                   const Params& p, int sm, cudaStream_t stream) {
  using C = Cfg<D, BC>;
  if constexpr (C::kNeedsOptIn) {
    const std::size_t cap = max_dyn_smem_bytes(arch_of_sm(sm));
    if (C::kBytes > cap) {  // unreachable for the archs in scope (static_assert), kept honest
      if (verbose()) {
        std::fprintf(stderr,
                     "fa::stage4: D=%d BC=%d needs %zu B of shared memory > %zu B per-block cap "
                     "on sm_%d (try FA_FORCE_BC=%d)\n",
                     D, BC, C::kBytes, cap, sm, kBcMin);
      }
      return cudaErrorInvalidValue;
    }
  }
  FA_CUDA_TRY((prepare_once<T, D, BC>()));
  const dim3 grid(static_cast<unsigned>((p.N + C::kBr - 1) / C::kBr),
                  static_cast<unsigned>(p.B) * static_cast<unsigned>(p.H));
  fa_s4_wmma<T, D, BC><<<grid, C::kThreads, C::kBytes, stream>>>(
      static_cast<const T*>(q), static_cast<const T*>(k), static_cast<const T*>(v),
      static_cast<T*>(o), lse, p.N, p.H, p.Hkv, p.scale, scale_log2(p.scale), p.causal);
  FA_CUDA_TRY_LAUNCH();
  return cudaSuccess;
}

template <typename T, int D>
cudaError_t launch_bc(int bc, const void* q, const void* k, const void* v, void* o, float* lse,
                      const Params& p, int sm, cudaStream_t stream) {
  switch (bc) {
    case kBcMin:
      return launch<T, D, kBcMin>(q, k, v, o, lse, p, sm, stream);
    case kBcMid:
      return launch<T, D, kBcMid>(q, k, v, o, lse, p, sm, stream);
    case kBcMax:
      return launch<T, D, kBcMax>(q, k, v, o, lse, p, sm, stream);
    default:
      return cudaErrorInvalidValue;  // unreachable: forward() validated bc
  }
}

template <typename T>
cudaError_t launch_d(int D, int bc, const void* q, const void* k, const void* v, void* o,
                     float* lse, const Params& p, int sm, cudaStream_t stream) {
  return D == 64 ? launch_bc<T, 64>(bc, q, k, v, o, lse, p, sm, stream)
                 : launch_bc<T, 128>(bc, q, k, v, o, lse, p, sm, stream);
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
    if (verbose()) std::fprintf(stderr, "fa::stage4: lse must be non-null (fused stages)\n");
    return cudaErrorInvalidValue;
  }
  if (p.N % 32 != 0 || p.H % p.Hkv != 0) {
    if (verbose()) std::fprintf(stderr, "fa::stage4: N %% 32 != 0 or H %% Hkv != 0\n");
    return cudaErrorInvalidValue;
  }
  if (static_cast<long long>(p.B) * p.H > 65535LL) {
    if (verbose()) {
      std::fprintf(stderr, "fa::stage4: B*H = %lld exceeds gridDim.y (65535)\n",
                   static_cast<long long>(p.B) * p.H);
    }
    return cudaErrorInvalidValue;
  }
  // Masking is applied to RAW scores before the fold into c = scale*log2e; a non-positive or
  // non-finite scale would turn the -inf sentinel into +inf/NaN (reference.py shares the
  // convention and the limitation).
  if (!(p.scale > 0.0f) || !std::isfinite(p.scale)) {
    if (verbose()) std::fprintf(stderr, "fa::stage4: scale must be finite and > 0\n");
    return cudaErrorInvalidValue;
  }
  // 16-B vector loads/stores; rows are D*2 = 128 B / 256 B so every tile base inherits it.
  if (!aligned16(q) || !aligned16(k) || !aligned16(v) || !aligned16(o)) {
    if (verbose()) std::fprintf(stderr, "fa::stage4: q/k/v/o must be 16-byte aligned\n");
    return cudaErrorInvalidValue;
  }
  if (dbg.block_m != 0 && verbose()) {
    static bool said = false;  // once per process; benign race, it is only a log line
    if (!said) {
      said = true;
      std::fprintf(stderr,
                   "fa::stage4: FA_FORCE_BR=%d ignored (BR is fixed at %d for D=64 and %d for "
                   "D=128 by the warp-per-16-rows partition; only FA_FORCE_BC varies the tiling)\n",
                   dbg.block_m, Cfg<64, kBcMax>::kBr, Cfg<128, kBcMid>::kBr);
    }
  }
  const int bc = dbg.block_n != 0 ? dbg.block_n : default_bc(p.D);
  if (bc != kBcMin && bc != kBcMid && bc != kBcMax) {
    if (verbose()) {
      std::fprintf(stderr,
                   "fa::stage4: FA_FORCE_BC=%d is not an instantiated tile ({%d, %d, %d})\n", bc,
                   kBcMin, kBcMid, kBcMax);
    }
    return cudaErrorInvalidValue;
  }
  switch (dtype) {
    case Dtype::F16:
      return launch_d<__half>(p.D, bc, q, k, v, o, lse, p, sm, stream);
    case Dtype::BF16:
      return launch_d<__nv_bfloat16>(p.D, bc, q, k, v, o, lse, p, sm, stream);
    case Dtype::F32:
      break;
  }
  return cudaErrorNotSupported;
}

}  // namespace stage4
}  // namespace fa
