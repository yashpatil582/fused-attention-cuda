// S2: shared-memory tiled QK^T and PV GEMMs with coalesced float4 loads and a shuffle-reduced
// row softmax. S and P are STILL materialised in the fp32 workspace (same layout as S1: S first,
// then P), so this stage changes exactly one thing versus S1: the memory access pattern.
//
// WHY: S1 read every operand straight from global memory (thread per (i, j), K rows walked
// column-wise). S2 keeps S1's three-kernel structure and the N x N intermediates so that the
// S1 -> S2 ncu delta isolates memory-access improvements and the S2 -> S3 delta isolates fusion
// (FA1 Theorem 2). Metrics expected to move (docs/PROFILING_PROTOCOL.md section 3; directions
// are the hypothesis, magnitudes are measured, never claimed here):
//   * smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct -> ~100: every global load
//     is one 16-byte float4 (one 8-byte load of 4 halves for fp16) with consecutive lanes walking
//     the contiguous D axis, so each 32-byte sector a warp touches is fully consumed;
//   * l1tex__t_sector_hit_rate.pct / lts__t_sector_hit_rate.pct up and dram__bytes.sum down for
//     the GEMMs: a Q/K (P/V) row is fetched once per 64-wide tile instead of once per output
//     element, i.e. 64x fewer operand elements requested from L1 per output than S1;
//   * l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum is RECORDED, not fixed: the
//     inner loop reads Ks column-wise on a row-major tile (analysis at fa_s2_qk_tiled), which is
//     S5a's "before" number; PAD = 0 here on purpose, S5a measures +4 (tile_config.h kPadF32);
//   * fa_s2_softmax: smsp__warp_issue_stalled_barrier_per_warp_active.pct and shared-load
//     instructions down versus a smem tree: 2 barriers per row instead of 2 x log2(256).
//
// Tile choice (ONE config, no sweep; judge_plan section 4.3, sweeps are W3): 64 x 64 output
// tile, 256 threads as a 16 x 16 thread grid, 4 x 4 register tile per thread (2D register
// blocking, cuda-samples matrixMul lineage): every smem value loaded into a register feeds 4
// FMAs, so the inner loop issues 8 LDS per 16 FFMA instead of 2 LDS per FFMA for a scalar tile.
// Reduction chunk 16 (D for QK^T, j for PV) keeps the static smem small (derivation at
// kSmemQkBytes / kSmemPvBytes below: 8 KB per block). T4 (sm_75, ArchCaps<750>): 1024
// threads/SM = 4 blocks of 256 = 32 warps = the full warp ceiling; 4 x 8 KB = 32 KB of the
// 64 KB smem, so smem never limits; the 65536-entry register file admits 4 blocks only at
// <= 64 registers/thread. Measured with -Xptxas -v for sm_75 (nvcc 12.8.93, 2026-09-02), the
// unroll factor of the 16-step inner loop being the only knob touched:
//   full unroll       qk 80 / pv 66 regs, 0 spills      -> 3 blocks/SM = 24 warps (75 %)
//   #pragma unroll 2  qk 61 / pv 61 (fp16 61 / 59), 0 spills -> 4 blocks/SM = 32 warps (100 %)
//   full unroll + __launch_bounds__(256, 4): 64 regs, 36 B (qk) / 4 B (pv) spilled -> rejected
// Shipped: unroll 2 (per unrolled pair of k-steps the sm_75 SASS is 8 LDS.64 + 32 independent
// FFMA — nvcc fuses Qs[row][kk], Qs[row][kk+1] into one 64-bit shared load; 8 branches/chunk).
// Achieved occupancy is sm__warps_active.avg.pct_of_peak_sustained_active, not this arithmetic.
// No __launch_bounds__ on purpose: that is the S5c variable (spill-free only).
//
// Numerics: fp32 accumulation everywhere (fp16 inputs are converted on load, outputs on store);
// S holds SCALED scores with -inf where masked, P = exp(S - rowmax) / rowsum, and
// lse = rowmax + log(rowsum) (+inf for an all-masked row), exactly python/fused_attn/
// reference.py::math_reference and src/reference/cpu_reference.cpp::attention_rows.
//
// DebugOverrides (FA_FORCE_BR/BC) are ignored: the tiles are compile-time constants and the
// tile-invariance tests target the online-softmax stages (3+). Registry line for dispatch.cu:
// {Impl::Tiled, stage2::forward, stage2::supports}.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <climits>
#include <cstddef>
#include <cstdint>

#include "fa/attention.cuh"
#include "fa/cuda_check.cuh"
#include "fa/softmax_math.cuh"

namespace {

constexpr int kBM = 64;  // output-tile rows (query rows i in both GEMMs)
constexpr int kBN = 64;  // output-tile cols (keys j for QK^T, head dims d for PV)
constexpr int kBK = 16;  // reduction chunk staged through smem per k-step
constexpr int kPad = 0;  // smem row padding in floats; S2 = 0 (S5a measures +4)
constexpr int kThreads = 256;
constexpr int kTM = 4;            // register tile rows per thread
constexpr int kTN = 4;            // register tile cols per thread
constexpr int kTX = kBN / kTN;    // 16 threads across the tile columns
constexpr int kTY = kBM / kTM;    // 16 threads down the tile rows
constexpr int kLdA = kBK + kPad;  // row stride of Qs / Ks / Ps  ([64][16+PAD])
constexpr int kLdV = kBN + kPad;  // row stride of Vs            ([16][64+PAD])

static_assert(kTX * kTY == kThreads, "16x16 threads must cover the 64x64 tile with 4x4 tiles");
static_assert(kBM * kBK == kThreads * 4, "one float4 per thread per 64x16 sub-tile load");
static_assert(kBK * kBN == kThreads * 4, "one float4 per thread per 16x64 sub-tile load");
static_assert(kPad % 4 == 0, "row stride must stay a multiple of 16 B for float4 smem access");
// Static shared memory per block, derived: each GEMM stages two 1024-float sub-tiles.
//   QK: Qs[64][16+PAD] + Ks[64][16+PAD] = 2 x 64 x 16 x 4 B = 8192 B at PAD = 0 (10240 at +4)
//   PV: Ps[64][16+PAD] + Vs[16][64+PAD] = (1024 + 1024) x 4 B = 8192 B at PAD = 0 (9472 at +4)
constexpr std::size_t kSmemQkBytes =
    static_cast<std::size_t>(kBM * kLdA + kBN * kLdA) * sizeof(float);
constexpr std::size_t kSmemPvBytes =
    static_cast<std::size_t>(kBM * kLdA + kBK * kLdV) * sizeof(float);
static_assert(kSmemQkBytes <= 48u * 1024u && kSmemPvBytes <= 48u * 1024u,
              "static smem must not need the dynamic opt-in");

// ---- 4-element vector load/store, always fp32 in registers ------------------------------
// fp32: one 16-byte LDG.128 / STG.128. fp16: one 8-byte LDG.64 of four halves converted with
// __half22float2 (round trip through fp32 keeps S / P / acc in fp32 per attention.cuh).
__device__ __forceinline__ float4 load4(const float* p) {
  return *reinterpret_cast<const float4*>(p);
}
__device__ __forceinline__ float4 load4(const __half* p) {
  const uint2 raw = *reinterpret_cast<const uint2*>(p);  // halves 0,1 in .x; 2,3 in .y
  const float2 lo = __half22float2(*reinterpret_cast<const __half2*>(&raw.x));
  const float2 hi = __half22float2(*reinterpret_cast<const __half2*>(&raw.y));
  return make_float4(lo.x, lo.y, hi.x, hi.y);
}
__device__ __forceinline__ void store4(float* p, float4 v) {
  *reinterpret_cast<float4*>(p) = v;
}
__device__ __forceinline__ void store4(__half* p, float4 v) {
  const __half2 lo = __floats2half2_rn(v.x, v.y);  // round-to-nearest, same as __float2half
  const __half2 hi = __floats2half2_rn(v.z, v.w);
  uint2 raw;
  raw.x = *reinterpret_cast<const unsigned int*>(&lo);
  raw.y = *reinterpret_cast<const unsigned int*>(&hi);
  *reinterpret_cast<uint2*>(p) = raw;
}

// ---- warp / block reductions (cuda-samples reduction_kernel.cu style) ---------------------
struct MaxOp {
  __device__ __forceinline__ float operator()(float a, float b) const {
    return fmaxf(a, b);
  }
};
struct SumOp {
  __device__ __forceinline__ float operator()(float a, float b) const {
    return a + b;
  }
};
// __shfl_down_sync tree: after 5 steps lane 0 holds the warp result. Every lane of the warp
// must reach the call (the softmax loops end before it, so the warp is converged).
template <typename Op>
__device__ __forceinline__ float warp_reduce(float v, Op op) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    v = op(v, __shfl_down_sync(0xffffffffu, v, off));
  return v;
}
// Block result = warp results exchanged once through `red[kWarps]`, then reduced by EVERY thread
// in the same fixed order (broadcast smem reads): all threads hold the identical value, no
// second barrier is needed, and the result is bitwise deterministic run to run.
template <int kWarps, typename Op>
__device__ __forceinline__ float block_reduce(float v, float* red, Op op) {
  v = warp_reduce(v, op);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
  __syncthreads();
  float r = red[0];
#pragma unroll
  for (int w = 1; w < kWarps; ++w)
    r = op(r, red[w]);
  return r;
}

}  // namespace

// =========================================================================================
// fa_s2_qk_tiled<T>: S[i][j] = scale * sum_d Q[i][d] * K[j][d] for one (b, h), -inf where
// causal masks (j > i, top-left). Grid (ceil(N/64) j-tiles, ceil(N/64) i-tiles, B*H).
//
// Thread mapping: tid = ty * 16 + tx; thread (tx, ty) owns rows i0 + 4*ty .. +3 and columns
// j0 + 4*tx .. +3 of the 64x64 tile in acc[4][4]. A warp is ty in {2w, 2w+1} x tx in 0..15,
// i.e. 8 tile rows by all 64 columns, so its epilogue writes 16 lanes x float4 = 256 contiguous
// bytes per row (coalesced STG.128).
// Loads per k-step (16 of D): the 64x16 Q and K sub-tiles are 1024 floats each = 256 float4s,
// one per thread: thread t fetches row t/4, elements (t%4)*4 .. +3. Lanes 4r..4r+3 read 64
// contiguous bytes of one row (two full 32-byte sectors), a warp reads 8 rows: nothing fetched
// is left unused, which is what smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct
// measures (S1's per-(i,j) K walk touched a fresh line per lane).
// Shared-memory pattern (the S5a "before"): the inner loop reads a[r] = Qs[4*ty+r][kk] and
// b[c] = Ks[4*tx+c][kk]. With row stride kLdA = 16 words, the 16 tx lanes of a half-warp read
// words (4*tx+c)*16 + kk whose bank is (16*c + kk) mod 32 for EVERY tx (64*tx = 0 mod 32): one
// bank, 16 distinct words -> 16-way serialisation per Ks read (the other half-warp reads the
// same words: broadcast). Qs reads are 2-way (two ty values, words 64 apart). Expected to show
// as l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum and mio_throttle/short_scoreboard
// stalls. Note for S5a: +4 padding alone gives 8-way on this scalar column walk (80*tx = 16*tx
// mod 32); the conflict-free fixes are a transposed Ks[16][64+PAD] read as float4 along j (the
// Vs layout below) or an XOR swizzle of the 16-byte chunk index.
// =========================================================================================
template <typename T>
__global__ void fa_s2_qk_tiled(const T* __restrict__ q, const T* __restrict__ k,
                               float* __restrict__ S, int N, int D, int H, int Hkv, float scale,
                               bool causal) {
  __shared__ __align__(16) float Qs[kBM][kLdA];  // [i][d] chunk
  __shared__ __align__(16) float Ks[kBN][kLdA];  // [j][d] chunk (row-major: the S5a "before")

  const int tid = static_cast<int>(threadIdx.x);
  const int tx = tid % kTX;  // 0..15: tile column group (4 columns each)
  const int ty = tid / kTX;  // 0..15: tile row group (4 rows each)
  // Grid: x = key tile, y = query tile, z = b * H + h.
  const int i0 = static_cast<int>(blockIdx.y) * kBM;
  const int j0 = static_cast<int>(blockIdx.x) * kBN;
  const int bh = static_cast<int>(blockIdx.z);
  const int b = bh / H, h = bh % H;
  const int kvh = h / (H / Hkv);  // GQA/MQA mapping (reference.py _expand_kv)

  // Head base pointers. q/o are [B, H, N, D]; k/v are [B, Hkv, N, D]; S is [B, H, N, N].
  const T* qb = q + static_cast<std::size_t>(bh) * N * D;
  const T* kb = k + (static_cast<std::size_t>(b) * Hkv + kvh) * N * D;
  float* Sb = S + static_cast<std::size_t>(bh) * N * N;

  const int ri = i0 + ty * kTM;  // first row of this thread's register tile
  const int cj = j0 + tx * kTN;  // first column of this thread's register tile
  // N % 32 == 0 but the tile is 64: the last tile may be half empty. Columns come in groups of 4
  // and N % 4 == 0, so a float4 column group is either entirely inside or entirely outside.
  const bool cols_ok = cj < N;

  if (causal && j0 > i0 + kBM - 1) {
    // Every (i, j) of this tile has j >= j0 > i0 + 63 >= i: all masked. S must still be fully
    // defined for the softmax, so write -inf and leave before any load or barrier (the
    // condition is block-uniform, so no thread is left waiting at a __syncthreads).
    if (cols_ok) {
      const float4 ninf = make_float4(fa::kNegInf, fa::kNegInf, fa::kNegInf, fa::kNegInf);
#pragma unroll
      for (int r = 0; r < kTM; ++r) {
        if (ri + r < N) store4(Sb + static_cast<std::size_t>(ri + r) * N + cj, ninf);
      }
    }
    return;
  }

  // Cooperative-load mapping: thread t stages row t/4, 4-element chunk (t%4)*4 of both sub-tiles.
  const int lr = tid / 4;        // 0..63: sub-tile row
  const int lc = (tid % 4) * 4;  // 0,4,8,12: first of the 4 elements along D
  const bool q_ok = i0 + lr < N;
  const bool k_ok = j0 + lr < N;
  // Rows past N are clamped to a valid row and then zeroed: no out-of-bounds read, no garbage.
  const T* qp = qb + static_cast<std::size_t>(q_ok ? i0 + lr : N - 1) * D + lc;
  const T* kp = kb + static_cast<std::size_t>(k_ok ? j0 + lr : N - 1) * D + lc;
  const float4 zero4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

  float acc[kTM][kTN] = {};  // fp32 partial dot products of this thread's 4x4 outputs

  // D in {64, 128}: 4 or 8 full chunks of 16 along the head dimension.
  for (int d0 = 0; d0 < D; d0 += kBK) {
    const float4 qv = q_ok ? load4(qp + d0) : zero4;  // Q[i0+lr][d0+lc .. +3]
    const float4 kv = k_ok ? load4(kp + d0) : zero4;  // K[j0+lr][d0+lc .. +3]
    *reinterpret_cast<float4*>(&Qs[lr][lc]) = qv;
    *reinterpret_cast<float4*>(&Ks[lr][lc]) = kv;
    // Barrier 1 (read-after-write): a thread's inner loop reads Qs/Ks rows written by OTHER
    // threads; every store above must be visible before any read below.
    __syncthreads();
#pragma unroll 2  // NOT full: measured register trade-off in the file header
    for (int kk = 0; kk < kBK; ++kk) {
      float a[kTM], bb[kTN];
#pragma unroll
      for (int r = 0; r < kTM; ++r)
        a[r] = Qs[ty * kTM + r][kk];  // 4 rows of Q, column kk
#pragma unroll
      for (int c = 0; c < kTN; ++c)
        bb[c] = Ks[tx * kTN + c][kk];  // 4 rows of K (= K^T cols)
#pragma unroll
      for (int r = 0; r < kTM; ++r) {
#pragma unroll
        for (int c = 0; c < kTN; ++c)
          acc[r][c] = fmaf(a[r], bb[c], acc[r][c]);
      }
    }
    // Barrier 2 (write-after-read): the next iteration overwrites Qs/Ks; without it a warp that
    // finished its FMAs early would clobber a row a slower warp is still reading.
    __syncthreads();
  }

  // Epilogue: scale and mask at store time; S = -inf for j > i (top-left causal). Each thread
  // stores 4 rows x one float4; cj is a multiple of 4 and N of 32, so the store is 16-B aligned.
  if (!cols_ok) return;
#pragma unroll
  for (int r = 0; r < kTM; ++r) {
    const int i = ri + r;
    if (i >= N) break;
    float4 out;
    out.x = (causal && cj + 0 > i) ? fa::kNegInf : acc[r][0] * scale;
    out.y = (causal && cj + 1 > i) ? fa::kNegInf : acc[r][1] * scale;
    out.z = (causal && cj + 2 > i) ? fa::kNegInf : acc[r][2] * scale;
    out.w = (causal && cj + 3 > i) ? fa::kNegInf : acc[r][3] * scale;
    store4(Sb + static_cast<std::size_t>(i) * N + cj, out);
  }
}

// =========================================================================================
// fa_s2_softmax<kThreads>: one block per row of S (grid = B*H*N). Three streaming passes over
// the row with consecutive threads on consecutive columns (a warp reads 128 contiguous bytes):
//   pass 1  m = max_j S[j]            (block_reduce: shuffle tree + one smem exchange)
//   pass 2  l = sum_j exp(S[j] - m)   (block_reduce)
//   pass 3  P[j] = exp(S[j] - m) / l  (recomputes the exp: one MUFU per element is cheaper than
//                                      a second write+read of the row through L2)
// then lse[row] = m + log(l) when requested. S is already scaled, so m is in scaled units and
// lse matches torch.logsumexp(scale * q k^T) (cpu_reference.cpp attention_rows, direct form).
// Guard: m == -inf (row with no visible key; impossible for top-left causal with Nq == Nkv but
// the invariant is kept local) -> P = 0 for the row and lse = +inf, never exp(-inf - -inf) = NaN.
// Rows with a visible key have l >= 1 (the max element contributes exp(0)), so 1/l is finite.
// =========================================================================================
template <int kThreadsSm>
__global__ void fa_s2_softmax(const float* __restrict__ S, float* __restrict__ P,
                              float* __restrict__ lse, int N) {
  constexpr int kWarps = kThreadsSm / 32;
  __shared__ float red_max[kWarps];
  __shared__ float red_sum[kWarps];  // separate array: no WAR hazard between the two reductions

  const std::size_t row = blockIdx.x;  // (b * H + h) * N + i
  const float* s = S + row * static_cast<std::size_t>(N);
  float* p = P + row * static_cast<std::size_t>(N);
  const int tid = static_cast<int>(threadIdx.x);

  float m = fa::kNegInf;
  for (int j = tid; j < N; j += kThreadsSm)
    m = fmaxf(m, s[j]);
  m = block_reduce<kWarps>(m, red_max, MaxOp{});

  if (m == fa::kNegInf) {  // block-uniform: every thread holds the same m
    for (int j = tid; j < N; j += kThreadsSm)
      p[j] = 0.0f;
    if (lse != nullptr && tid == 0) lse[row] = fa::kPosInf;
    return;
  }

  float l = 0.0f;
  for (int j = tid; j < N; j += kThreadsSm)
    l += expf(s[j] - m);  // masked: exp(-inf) = 0
  l = block_reduce<kWarps>(l, red_sum, SumOp{});

  const float inv_l = 1.0f / l;
  for (int j = tid; j < N; j += kThreadsSm)
    p[j] = expf(s[j] - m) * inv_l;
  if (lse != nullptr && tid == 0) lse[row] = m + logf(l);
}

// =========================================================================================
// fa_s2_pv_tiled<T>: O[i][d] = sum_j P[i][j] * V[j][d] for one (b, h), written as T.
// Grid (D/64 d-tiles, ceil(N/64) i-tiles, B*H): D = 64 is one tile column, D = 128 two.
//
// Thread mapping as in fa_s2_qk_tiled: thread (tx, ty) owns rows i0 + 4*ty .. +3 and head dims
// d0 + 4*tx .. +3. Reduction over j in chunks of 16 (N % 32 == 0: every chunk is full):
//   Ps[64][16+PAD]  thread t stages P[i0 + t/4][j0 + (t%4)*4 .. +3]   (float4, 4 lanes = 64 B)
//   Vs[16][64+PAD]  thread t stages V[j0 + t/16][d0 + (t%16)*4 .. +3] (16 lanes = 256 B fp32 /
//                                                                     128 B fp16 of one V row)
// Inner loop: a[r] = Ps[4*ty+r][kk] (column walk, 2-way like Qs), b[0..3] = Vs[kk][4*tx .. +3]
// as ONE float4 (LDS.128): the 16 tx lanes read 16 consecutive 16-byte chunks = 256 B, i.e.
// each 8-lane phase covers all 32 banks once: conflict-free. Smem footprint: kSmemPvBytes.
// Causal: masked P entries are exactly 0, so the full j loop is exact (skipping tiles past the
// diagonal is S6 block skipping and is deliberately absent here).
// =========================================================================================
template <typename T>
__global__ void fa_s2_pv_tiled(const float* __restrict__ P, const T* __restrict__ v,
                               T* __restrict__ o, int N, int D, int H, int Hkv) {
  __shared__ __align__(16) float Ps[kBM][kLdA];  // [i][j] chunk
  __shared__ __align__(16) float Vs[kBK][kLdV];  // [j][d] chunk

  const int tid = static_cast<int>(threadIdx.x);
  const int tx = tid % kTX;
  const int ty = tid / kTX;
  // Grid: x = head-dim tile, y = query tile, z = b * H + h.
  const int d0 = static_cast<int>(blockIdx.x) * kBN;
  const int i0 = static_cast<int>(blockIdx.y) * kBM;
  const int bh = static_cast<int>(blockIdx.z);
  const int b = bh / H, h = bh % H;
  const int kvh = h / (H / Hkv);

  const float* Pb = P + static_cast<std::size_t>(bh) * N * N;
  const T* vb = v + (static_cast<std::size_t>(b) * Hkv + kvh) * N * D;
  T* ob = o + static_cast<std::size_t>(bh) * N * D;

  // P sub-tile (64 x 16): row t/4, columns (t%4)*4 .. +3 of the current j chunk.
  const int pr = tid / 4;
  const int pc = (tid % 4) * 4;
  const bool p_ok = i0 + pr < N;  // last i-tile may be half empty (N % 64 == 32)
  const float* pp = Pb + static_cast<std::size_t>(p_ok ? i0 + pr : N - 1) * N + pc;
  // V sub-tile (16 x 64): row t/16 of the chunk, head dims d0 + (t%16)*4 .. +3 (always < D).
  const int vr = tid / 16;
  const int vc = (tid % 16) * 4;
  const T* vp = vb + static_cast<std::size_t>(vr) * D + d0 + vc;
  const float4 zero4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

  float acc[kTM][kTN] = {};

  for (int j0 = 0; j0 < N; j0 += kBK) {
    const float4 pv = p_ok ? load4(pp + j0) : zero4;                 // P[i0+pr][j0+pc .. +3]
    const float4 vv = load4(vp + static_cast<std::size_t>(j0) * D);  // V[j0+vr][d0+vc .. +3]
    *reinterpret_cast<float4*>(&Ps[pr][pc]) = pv;
    *reinterpret_cast<float4*>(&Vs[vr][vc]) = vv;
    __syncthreads();  // barrier 1: RAW, see fa_s2_qk_tiled
#pragma unroll 2      // NOT full: measured register trade-off in the file header
    for (int kk = 0; kk < kBK; ++kk) {
      float a[kTM];
#pragma unroll
      for (int r = 0; r < kTM; ++r)
        a[r] = Ps[ty * kTM + r][kk];  // P[4 rows][j = kk]
      // V[j = kk][4 head dims] as one LDS.128 (conflict-free, see the header).
      const float4 bv = *reinterpret_cast<const float4*>(&Vs[kk][tx * kTN]);
      const float bb[kTN] = {bv.x, bv.y, bv.z, bv.w};
#pragma unroll
      for (int r = 0; r < kTM; ++r) {
#pragma unroll
        for (int c = 0; c < kTN; ++c)
          acc[r][c] = fmaf(a[r], bb[c], acc[r][c]);
      }
    }
    __syncthreads();  // barrier 2: WAR, see fa_s2_qk_tiled
  }

  // Epilogue: O rows i0 + 4*ty + r (skip rows past N), dims d0 + 4*tx .. +3, as T. A warp's 16
  // tx lanes write 256 B (fp32) / 128 B (fp16) contiguous per row.
  const int ri = i0 + ty * kTM;
  const int cd = d0 + tx * kTN;
#pragma unroll
  for (int r = 0; r < kTM; ++r) {
    const int i = ri + r;
    if (i >= N) break;
    store4(ob + static_cast<std::size_t>(i) * D + cd,
           make_float4(acc[r][0], acc[r][1], acc[r][2], acc[r][3]));
  }
}

// =========================================================================================
// Stage launcher: the ABI declared in src/dispatch.cu (FA_DECLARE_STAGE(stage2)).
// =========================================================================================
namespace fa {
namespace stage2 {

namespace {

template <typename T>
cudaError_t launch(const void* q, const void* k, const void* v, void* o, float* lse,
                   void* workspace, const Params& p, cudaStream_t stream) {
  // Workspace layout (workspace_bytes = 2 * B*H*N*N floats): S first, then P; same as S1.
  const std::size_t nn = static_cast<std::size_t>(p.B) * p.H * p.N * p.N;
  float* S = static_cast<float*>(workspace);
  float* P = S + nn;
  const std::size_t bh = static_cast<std::size_t>(p.B) * p.H;
  const int tiles_n = (p.N + kBM - 1) / kBM;  // N % 32 == 0: a trailing half tile is possible
  // Launch-geometry limits (never hit in-window, checked so a bad shape fails loudly, not UB):
  // grid.z <= 65535 for the GEMMs, grid.x <= INT_MAX rows for the softmax.
  if (bh > 65535) return cudaErrorInvalidConfiguration;
  if (static_cast<std::size_t>(bh) * p.N > static_cast<std::size_t>(INT_MAX))
    return cudaErrorInvalidConfiguration;
  const dim3 block(kThreads);

  const dim3 grid_qk(tiles_n, tiles_n, bh);
  fa_s2_qk_tiled<T><<<grid_qk, block, 0, stream>>>(static_cast<const T*>(q),
                                                   static_cast<const T*>(k), S, p.N, p.D, p.H,
                                                   p.Hkv, p.scale, p.causal);
  FA_CUDA_TRY_LAUNCH();

  const dim3 grid_sm(static_cast<unsigned int>(bh) * static_cast<unsigned int>(p.N));
  fa_s2_softmax<kThreads><<<grid_sm, block, 0, stream>>>(S, P, lse, p.N);
  FA_CUDA_TRY_LAUNCH();

  const dim3 grid_pv(p.D / kBN, tiles_n, bh);  // D in {64, 128}: 1 or 2 tile columns
  fa_s2_pv_tiled<T><<<grid_pv, block, 0, stream>>>(P, static_cast<const T*>(v), static_cast<T*>(o),
                                                   p.N, p.D, p.H, p.Hkv);
  FA_CUDA_TRY_LAUNCH();
  return cudaSuccess;
}

}  // namespace

// S2 gate: fp32 and fp16 inputs, D in {64, 128}, any sm_75+ device (no tensor cores, no
// dynamic smem, so nothing is architecture-specific). dispatch.cu ANDs this with the contract.
bool supports(Dtype dtype, int D, int sm) {
  return sm >= 75 && (D == 64 || D == 128) && (dtype == Dtype::F32 || dtype == Dtype::F16);
}

// `dbg` (FA_FORCE_BR/BC) is ignored on purpose: S2's tiles are compile-time constants and the
// tile-invariance tests target stages 3+. `sm` only gates supports(). Never synchronises;
// returns the first launch error (dispatch.cu already validated p and the workspace).
cudaError_t forward(Dtype dtype, const void* q, const void* k, const void* v, void* o, float* lse,
                    void* workspace, const Params& p, const DebugOverrides& /*dbg*/, int sm,
                    cudaStream_t stream) {
  if (!supports(dtype, p.D, sm)) return cudaErrorNotSupported;
  if (workspace == nullptr) return cudaErrorInvalidValue;
  switch (dtype) {
    case Dtype::F32:
      return launch<float>(q, k, v, o, lse, workspace, p, stream);
    case Dtype::F16:
      return launch<__half>(q, k, v, o, lse, workspace, p, stream);
    case Dtype::BF16:
      break;  // contract: bf16 is Wmma/Tuned on sm_80+ only
  }
  return cudaErrorNotSupported;
}

}  // namespace stage2
}  // namespace fa
