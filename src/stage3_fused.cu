// S3: one fused attention kernel — online softmax (Milakov & Gimelshein 2018) with the
// FlashAttention-2 loop order and deferred normalisation. fp32 arithmetic, no tensor cores.
//
// WHAT THIS STAGE MOVES (the ncu metric): S1/S2 write the N x N score matrix S and the
// probability matrix P to HBM and read them back, so dram__bytes.sum is O(B*H*N^2). Here a
// thread block owns ONE tile of BR query rows for its whole life, streams K/V tiles of BC rows
// through shared memory, and keeps the softmax statistics (m, l) and the un-normalised output
// accumulator in registers. Nothing of size N x N ever leaves the SM, so the modelled minimum
// traffic is Q + K + V + O + LSE = O(B*H*N*D) bytes (FA1 Theorem 2, arXiv 2205.14135) — the
// bench's `gbps` column uses exactly that model (bench_main.cu bytes_moved()):
//   bytes = 2*B*H*N*D*es (Q, O) + 2*B*Hkv*N*D*es (K, V) + B*H*N*4 (LSE)
// K/V are re-read once per Q tile (N/BR times per head) but from L2, not DRAM: a head's K+V is
// 2*N*D*4 B = 1 MB at N=2048, D=64 fp32, well inside T4's 4 MB L2, and the N/BR blocks of one
// head are launched back to back (blockIdx.x is the Q tile, blockIdx.y the head). Expected on
// T4: dram__bytes.sum collapses to ~O(B*H*N*D), DRAM throughput % falls, and the top stall
// reasons become smsp__warp_issue_stalled_mio_throttle (every FMA below has a shared-memory
// operand: 1 LDS.128 per 4 FFMA) and _barrier (two __syncthreads per K/V tile with only 1-2
// blocks per SM, see the occupancy arithmetic below). Low sm__throughput here is the point:
// it is the recorded motivation for S4 (tensor cores fetch 16x fewer operands per FLOP) and
// S5 (prefetch, launch bounds, shuffles). Honest expectation (judge_plan 4.3): fp32 S3 may
// still lose to SDPA(EFFICIENT) at N=4096; the stage is about correctness and the traffic
// collapse, not peak speed.
//
// THREAD MAPPING: one thread per query row, blockDim = BR = 64 (two warps). The thread keeps
// its row's running max m (RAW score units, init -inf), running sum l (init 0), the fp32
// accumulator acc[D] and the current tile's BC raw scores s[BC] in registers; every reduction
// (row max, row sum) is therefore thread-local: no shuffles, no smem round trip, no barrier in
// the softmax — FA2's "each warp owns whole rows" argument taken to its extreme. The price is
// register pressure (acc[D] alone is 64/128 registers), which is why this mapping is only
// viable with BR threads per block and no tensor cores; S4 replaces it.
//
// SHARED-MEMORY LAYOUT (dynamic, one carve per block, all fp32 regardless of the input dtype):
//   Qs[BR][D+1]   the block's Q tile, PADDED by one float per row. A warp reads column d of
//                 32 DIFFERENT rows at once (thread t <-> row t), i.e. addresses t*(D+1)+d.
//                 With D % 32 == 0 an unpadded stride would put every row's column d in the
//                 same bank (32-way conflict); stride D+1 gives bank (t + d) mod 32: 32
//                 distinct banks. (+1 is legal here because Qs is only ever read with scalar
//                 LDS.32; tile_config.h's +4 pad is for float4/WMMA-aligned tiles, S4/S5.)
//   Ks[BC][D]     UNPADDED: all 64 threads read the SAME K row j (their own row's dot product
//   Vs[BC][D]     with it) and the same V row j, so a warp-wide LDS is a broadcast of one
//                 address — no conflict possible — and 16-B alignment lets it be an LDS.128.
// Bytes = 4*(BR*(D+1) + 2*BC*D):
//   D=64 : BC=16 24,832 | BC=32 33,024 (default) | BC=64 49,408
//   D=128: BC=16 49,408 (default)  | BC=32 65,792   | BC=64 not instantiated (see below)
// The static/no-opt-in ceiling is 48 KB = 49,152 B: both 49,408-B tiles miss it by the 256 B
// of Q padding, so the kernel takes `extern __shared__` and the launcher opts in with
// cudaFuncSetAttribute(MaxDynamicSharedMemorySize) whenever bytes > kSmemNoOptInBytes, up to
// the per-arch cap from include/fa/tile_config.h (T4: 64 KB, so D=64 admits BC in {16,32,64}
// and D=128 admits BC=16 only; sm_80+ admits D=128/BC=32 too; the launcher returns
// cudaErrorInvalidValue for anything above the cap of the running device).
//
// OCCUPANCY (T4: 64 KB smem per SM, 64K registers, 32 warps, 16 blocks):
//   blocks/SM by smem  = floor(65,536 / bytes): D=64/BC=32 -> 1 (2 warps, 6.25%);
//                        D=64/BC=16 -> 2 (4 warps, 12.5%); D=128/BC=16 -> 1 (2 warps).
//   blocks/SM by regs  = floor(65,536 / (64 * regs)); ptxas -v (nvcc 12.8, sm_75, 0 spills):
//                        D=64/BC=16 144-149, D=64/BC=32 152-162, D=64/BC=64 201-254,
//                        D=128/BC=16 191-196, D=128/BC=32 254-255 (ranges over fp32/fp16/
//                        bf16) -> 4-7 blocks. Registers never bind before shared memory does.
// So the Q tile (16.6 KB / 33 KB) is what caps occupancy; that is the S4/S5 trade (fp16
// tiles, Q fragments in registers). Registers above 128 at D=128 are inherent to
// thread-per-row (acc[128] alone; acc[D] + s[BC] are live together during the PV update),
// accepted because they are not the binding limit and there are 0 spills. The allocator uses
// the head-room up to 255 (__launch_bounds__(64) only, no min-blocks hint) to schedule the
// unrolled LDS.128 stream early — that is ILP, not pressure. D=128/BC=64 is the one shape
// where the live set (128 + 64 registers before temporaries) does spill (ptxas: 255 regs,
// 184-232 B stack, 232-412 B spill stores); it would also need 98,560 B of smem, above T4's
// cap, so it is not instantiated and FA_FORCE_BC=64 with D=128 returns cudaErrorInvalidValue.
//
// TILE DEFAULTS: BC = TileCfg<kArchTuring, D>::kBcF32 (32 for D=64, 16 for D=128). BR is
// fixed at 64 by the mapping (64 rows = 64 threads = 2 warps; 32 would halve the warps per
// block and double the K/V re-reads per head for the same smem/row). FA_FORCE_BC selects
// among the instantiated BC: {16, 32, 64} for D=64, {16, 32} for D=128; FA_FORCE_BR is
// IGNORED (reported once under FA_VERBOSE) — the tile-invariance test only needs BC to vary,
// which it does (block_n in {N/2, N/4} for N in {64, 128} = {16, 32, 64}, all instantiated).
// First tuning experiment for the bench, one env var: FA_FORCE_BC=16 at D=64 trades 2x the
// barriers for 2 blocks/SM (4 warps) — measure, do not guess.
//
// SOFTMAX CONVENTION (identical to include/fa/softmax_math.cuh, python/fused_attn/
// reference.py tiled_online_softmax_reference and src/reference/cpu_reference.cpp
// attention_tiled — checked line by line):
//   s      = q . k                     RAW dot product, the scale is NOT applied here
//   m_new  = max(m, max_j s_j)         raw units, so no scaling pass over the tile
//   alpha  = exp2((m - m_new) * c)     c = scale * log2(e), folded once per launch
//   p_j    = exp2(s_j * c - m_new * c) one FFMA + one EX2 per score (exp2_scaled)
//   l      = alpha * l + sum_j p_j
//   acc    = alpha * acc + sum_j p_j * v_j
//   m      = m_new                     updated AFTER alpha and p are computed
//   o      = acc * (1 / l)             ONE divide per row, after the last tile
//   lse    = m * scale + ln(l)         natural log, torch.logsumexp units
// WHY alpha is a multiply, never an inverse: m_new >= m so alpha <= 1; multiplying by it can
// neither overflow nor amplify error (flash-attention softmax.h softmax_rescale_o does the
// same; FA2's paper notation diag(...)^-1 is the un-normalised form, not the code). WHY the
// divide is deferred: FA1 normalised after every tile (a divide per output element per tile
// and an extra register per row); FA2 keeps acc un-normalised and divides once — the
// "deferred normalisation" — which is also why l == 0 must be guarded only once, at the end.
// The scale is applied exactly once, inside c (BUG 4 of the reality-check research: never
// on s and again in c); masking happens on raw scores BEFORE any scaling, which requires
// scale > 0 (validated in forward()).
//
// CAUSAL: top-left aligned, key j visible to query i iff j <= i (torch SDPA is_causal for
// Nq == Nkv); masked entries are s_j = -inf, masking uses `j > i` with GLOBAL i and j (the
// diagonal off-by-one is the reality-check's BUG 5). Tiles entirely above the diagonal are
// skipped by the loop bound (n_tiles = ceil(min(i0+BR, N) / BC)) exactly as reference.py's
// `j0 > i1 - 1: continue`; only tiles straddling the diagonal pay the element-wise mask.
// The last K/V tile can be partial when N % BC != 0 (N % 32 == 0 but BC = 64): rows past N
// are zero-filled in smem AND masked to -inf, so p = 0 and 0 * 0 (never 0 * garbage = NaN).
//
// GUARDS (reality-check BUG 2): a row that has seen no unmasked key has m == -inf; then
// rescale_alpha() returns 0 (acc and l are still 0, so exact), p is 0 for the whole tile
// (hoisted guard on m_new), and finalize_inv_l()/finalize_lse() turn the empty row into
// o = 0, lse = +inf. Impossible for top-left causal with Nq == Nkv (row i always sees key 0)
// but kept so the invariant is local, as softmax_math.cuh asks.
//
// BARRIERS (reality-check BUG 8): exactly two per K/V tile — one after the cooperative load
// (RAW: the tile must be complete before anyone reads it; on the first tile it also publishes
// the Q tile) and one after the PV update (WAR: no thread may start overwriting Ks/Vs with
// the next tile while a slower thread still reads the current one). The trailing barrier of
// the last tile also fences the reuse of Qs as the O staging buffer.
//
// LIMITS: gridDim.y = B*H <= 65,535 (B=8, H=32 -> 256 is far below); B*H beyond that returns
// cudaErrorInvalidValue. q/k/v/o must be 16-B aligned (any torch/cudaMalloc allocation; rows
// are 256 B / 512 B so tile bases inherit the alignment).
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "fa/attention.cuh"
#include "fa/cuda_check.cuh"
#include "fa/softmax_math.cuh"
#include "fa/tile_config.h"

namespace fa {
namespace stage3 {

constexpr int kBr = 64;       // query rows per block == threads per block (thread-per-row)
constexpr int kQChunk = 8;    // Q row elements held in registers per pass over the K tile
constexpr int kVecElems = 4;  // elements per cooperative-load vector (float4 / 4 halves)

// Instantiated BC values. Anything else from FA_FORCE_BC is rejected in forward().
constexpr int kBcMin = 16;
constexpr int kBcMid = 32;
constexpr int kBcMax = 64;

// Shared-memory carve for one block (see the layout table in the header comment).
template <int D, int BC>
struct Smem {
  static_assert(D == 64 || D == 128, "D must be 64 or 128 (CONTRACT 1)");
  static_assert(BC == kBcMin || BC == kBcMid || BC == kBcMax, "BC must be 16, 32 or 64");
  static constexpr int kQStride = D + 1;  // +1 float: conflict-free thread-per-row column reads
  static constexpr std::size_t kQFloats = static_cast<std::size_t>(kBr) * kQStride;
  static constexpr std::size_t kKvFloats = static_cast<std::size_t>(BC) * D;
  static constexpr std::size_t kBytes = (kQFloats + 2 * kKvFloats) * sizeof(float);
  // Ks/Vs are read with float4 (LDS.128): their base must stay 16-B aligned after the padded
  // Q tile. BR*(D+1)*4 = 64*65*4 = 16,640 and 64*129*4 = 33,024 are both multiples of 16.
  static_assert(kQFloats * sizeof(float) % 16 == 0, "K/V tiles must start 16-B aligned");
  static_assert(kKvFloats % (kBr * kVecElems) == 0, "K/V tile must be a whole number of loads");
  static_assert(D % kQChunk == 0 && D % 4 == 0, "D must be a multiple of the register chunk");
};

// Element conversion. Scores/m/l/acc are fp32 for every T (fp16 overflow safety, NUMERICS.md
// section 3): inputs are widened on the way into shared memory, outputs narrowed on the way
// out, and nothing else in the kernel depends on T.
template <typename T>
struct Elem;

template <>
struct Elem<float> {
  __device__ static __forceinline__ float to_float(float x) {
    return x;
  }
  __device__ static __forceinline__ float from_float(float x) {
    return x;
  }
  // 4 consecutive elements at a 16-B aligned address.
  __device__ static __forceinline__ float4 load4(const float* p) {
    return *reinterpret_cast<const float4*>(p);
  }
};

template <>
struct Elem<__half> {
  __device__ static __forceinline__ float to_float(__half x) {
    return __half2float(x);
  }
  __device__ static __forceinline__ __half from_float(float x) {
    return __float2half_rn(x);
  }
  // 4 halves = 8 B (one uint2): 32 lanes x 8 B = 256 B per warp request, fully coalesced.
  __device__ static __forceinline__ float4 load4(const __half* p) {
    const uint2 raw = *reinterpret_cast<const uint2*>(p);
    const __half2 h0 = *reinterpret_cast<const __half2*>(&raw.x);
    const __half2 h1 = *reinterpret_cast<const __half2*>(&raw.y);
    const float2 f0 = __half22float2(h0);
    const float2 f1 = __half22float2(h1);
    return make_float4(f0.x, f0.y, f1.x, f1.y);
  }
};

// bf16: no tensor cores are involved in S3, so the conversions compile for every arch (the
// cuda_bf16 intrinsics fall back to bit arithmetic below sm_80). supports() still gates it to
// sm_80+ to match the contract ("BF16 sm_80+"), and dispatch.cu's contract_allows() does not
// route BF16 to Fused at all, so this path is reachable only through a direct stage3 call.
template <>
struct Elem<__nv_bfloat16> {
  __device__ static __forceinline__ float to_float(__nv_bfloat16 x) {
    return __bfloat162float(x);
  }
  __device__ static __forceinline__ __nv_bfloat16 from_float(float x) {
    return __float2bfloat16_rn(x);
  }
  __device__ static __forceinline__ float4 load4(const __nv_bfloat16* p) {
    const uint2 raw = *reinterpret_cast<const uint2*>(p);
    const __nv_bfloat162 h0 = *reinterpret_cast<const __nv_bfloat162*>(&raw.x);
    const __nv_bfloat162 h1 = *reinterpret_cast<const __nv_bfloat162*>(&raw.y);
    const float2 f0 = __bfloat1622float2(h0);
    const float2 f1 = __bfloat1622float2(h1);
    return make_float4(f0.x, f0.y, f1.x, f1.y);
  }
};

// Cooperative, coalesced copy of one K or V tile (rows j0 .. j0+BC-1 of a [N][D] head slice)
// into an unpadded fp32 [BC][D] smem tile. The tile is BC*D CONTIGUOUS elements in global
// memory (rows of one head are adjacent), so this is a flat copy: consecutive threads take
// consecutive 4-element vectors (reality-check BUG 7: threadIdx.x walks the contiguous D axis),
// and the smem stores are consecutive float4s — conflict-free (8 lanes per phase cover all 32
// banks). Rows >= rows_valid (a partial last tile) are never read from global memory and are
// written as zeros so that 0 * Vs[j][d] is 0, never 0 * (stale smem) = NaN, and so runs are
// bitwise reproducible (nothing in smem is ever uninitialised).
template <typename T, int D, int BC>
__device__ __forceinline__ void load_kv_tile(const T* __restrict__ g, int rows_valid,
                                             float* __restrict__ dst) {
  constexpr int kVecs = BC * D / kVecElems;
  const int valid_elems = rows_valid * D;
#pragma unroll
  for (int it = 0; it < kVecs / kBr; ++it) {
    const int e = (it * kBr + static_cast<int>(threadIdx.x)) * kVecElems;  // D % 4 == 0: no
    float4 f = make_float4(0.0f, 0.0f, 0.0f, 0.0f);                        // row straddling
    if (e < valid_elems) f = Elem<T>::load4(g + e);
    *reinterpret_cast<float4*>(dst + e) = f;
  }
}

}  // namespace stage3
}  // namespace fa

// ---------------------------------------------------------------------------------------
// The kernel. Global scope and this exact name are fixed by CONTRACT 2 (ncu --kernel-name
// regex:fa_s3_fused matches the mangled prefix). Template on (input type, head dim, KV tile).
// ---------------------------------------------------------------------------------------
template <typename T, int D, int BC>
__global__ void __launch_bounds__(fa::stage3::kBr)
    fa_s3_fused(const T* __restrict__ q, const T* __restrict__ k, const T* __restrict__ v,
                T* __restrict__ o, float* __restrict__ lse, int N, int H, int Hkv, float scale,
                float scale_log2, bool causal) {
  using namespace fa::stage3;
  using SmemT = Smem<D, BC>;
  constexpr int kQStride = SmemT::kQStride;

  extern __shared__ __align__(16) float smem[];
  float* Qs = smem;                    // [BR][D+1]
  float* Ks = smem + SmemT::kQFloats;  // [BC][D]
  float* Vs = Ks + SmemT::kKvFloats;   // [BC][D]

  const int tid = static_cast<int>(threadIdx.x);
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

  // ---- Q tile -> padded smem (once per block). Scalar and coalesced: thread tid takes
  // elements tid, tid+64, ... so a warp reads 32 consecutive elements (one 128-B / 64-B
  // request) and writes 32 consecutive floats of one row (32 distinct banks; the +1 pad only
  // shifts where the next row starts). Rows past N (partial Q tile: N % 32 == 0 but BR = 64)
  // are zero-filled so their threads compute finite garbage they never store.
  const int rows_q = min(kBr, N - i0);
  for (int e = tid; e < kBr * D; e += kBr) {
    const int r = e / D;
    const int d = e - r * D;
    Qs[r * kQStride + d] = (r < rows_q) ? Elem<T>::to_float(qg[e]) : 0.0f;
  }
  // (published to the other warp by the first barrier inside the tile loop)

  const int i = i0 + tid;        // this thread's GLOBAL query row
  const bool row_valid = i < N;  // false only in a partial last Q tile

  // Per-row running state, registers only (FA2: never spilled to smem, no per-tile divide).
  float m = fa::kNegInf;  // running max of RAW scores
  float l = 0.0f;         // running sum of exp2((s - m) * c)
  float acc[D];           // un-normalised output row
#pragma unroll
  for (int d = 0; d < D; ++d)
    acc[d] = 0.0f;

  // Causal: keys j >= min(i0+BR, N) are invisible to every row of this block, so those tiles
  // are skipped BEFORE any load (reference.py: `if j0 > i1 - 1: continue`). Non-causal
  // streams all of K/V. n_tiles >= 1 always (i0 < N).
  const int kv_end = causal ? min(i0 + kBr, N) : N;
  const int n_tiles = (kv_end + BC - 1) / BC;

  for (int t = 0; t < n_tiles; ++t) {
    const int j0 = t * BC;
    const int rows_kv = min(BC, N - j0);  // < BC only for a partial last tile (BC=64, N=96)

    // (1) cooperative K/V tile load, then the RAW barrier.
    load_kv_tile<T, D, BC>(kg + static_cast<std::size_t>(j0) * D, rows_kv, Ks);
    load_kv_tile<T, D, BC>(vg + static_cast<std::size_t>(j0) * D, rows_kv, Vs);
    __syncthreads();

    // (2) s_j = q_i . k_j for the BC keys of the tile, RAW (scale folded into c later).
    // Loop order: an 8-wide chunk of the thread's Q row is copied into registers, then every
    // K row of the tile is dotted against it, so Qs is read D times per tile per thread
    // instead of BC*D times, and the K reads are warp-broadcast LDS.128 (all threads read the
    // same address). s[] is fully unrolled over j so it lives in registers, not local memory;
    // the d0 loop is deliberately NOT unrolled to keep the SASS near the instruction cache.
    float s[BC];
#pragma unroll
    for (int j = 0; j < BC; ++j)
      s[j] = 0.0f;
    const float* __restrict__ qrow = Qs + tid * kQStride;
#pragma unroll 1
    for (int d0 = 0; d0 < D; d0 += kQChunk) {
      float qc[kQChunk];
#pragma unroll
      for (int dd = 0; dd < kQChunk; ++dd)
        qc[dd] = qrow[d0 + dd];  // bank (tid + d) mod 32
#pragma unroll
      for (int j = 0; j < BC; ++j) {
        const float* __restrict__ krow = Ks + j * D + d0;
        const float4 k0 = *reinterpret_cast<const float4*>(krow);      // broadcast
        const float4 k1 = *reinterpret_cast<const float4*>(krow + 4);  // broadcast
        float sj = s[j];
        sj = fmaf(qc[0], k0.x, sj);
        sj = fmaf(qc[1], k0.y, sj);
        sj = fmaf(qc[2], k0.z, sj);
        sj = fmaf(qc[3], k0.w, sj);
        sj = fmaf(qc[4], k1.x, sj);
        sj = fmaf(qc[5], k1.y, sj);
        sj = fmaf(qc[6], k1.z, sj);
        sj = fmaf(qc[7], k1.w, sj);
        s[j] = sj;
      }
    }

    // (3) mask, on RAW scores, only for tiles that need it (block-uniform branch):
    //     a partial last tile (keys >= N) or a tile that straddles the causal diagonal for at
    //     least one row of the block (its last key j0+BC-1 exceeds the block's first row i0).
    //     Inside, the test is per element with the GLOBAL indices: key jj is masked iff
    //     jj >= N or (causal and jj > i) — strictly greater, so the diagonal stays visible.
    if (rows_kv < BC || (causal && j0 + BC - 1 > i0)) {
#pragma unroll
      for (int j = 0; j < BC; ++j) {
        const int jj = j0 + j;
        if (jj >= N || (causal && jj > i)) s[j] = fa::kNegInf;
      }
    }

    // (4) online-softmax update, exactly reference.py's recurrence.
    float tile_max = s[0];
#pragma unroll
    for (int j = 1; j < BC; ++j)
      tile_max = fmaxf(tile_max, s[j]);
    const float m_new = fmaxf(m, tile_max);
    // alpha = exp2((m - m_new) * c), 0 when m == -inf (first tile / all-masked prefix). It
    // rescales BOTH l and acc (forgetting acc is reality-check BUG 1) and is a multiply.
    const float alpha = fa::rescale_alpha(m, m_new, scale_log2);
    l *= alpha;
#pragma unroll
    for (int d = 0; d < D; ++d)
      acc[d] *= alpha;
    // p_j = exp2(s_j*c - m_new*c) with m_new*c hoisted (softmax_math.cuh exp2_scaled); the
    // m_new == -inf guard of p_from_score is hoisted out of the j loop: every p is 0 then.
    // Masked s_j = -inf gives fmaf(-inf, c, -m_c) = -inf and exp2f(-inf) = 0 (c > 0).
    if (m_new != fa::kNegInf) {
      const float m_c = m_new * scale_log2;
#pragma unroll
      for (int j = 0; j < BC; ++j) {
        const float p = fa::exp2_scaled(s[j], scale_log2, m_c);
        l += p;
        const float* __restrict__ vrow = Vs + j * D;
#pragma unroll
        for (int d0 = 0; d0 < D; d0 += 4) {
          const float4 vv = *reinterpret_cast<const float4*>(vrow + d0);  // broadcast
          acc[d0 + 0] = fmaf(p, vv.x, acc[d0 + 0]);
          acc[d0 + 1] = fmaf(p, vv.y, acc[d0 + 1]);
          acc[d0 + 2] = fmaf(p, vv.z, acc[d0 + 2]);
          acc[d0 + 3] = fmaf(p, vv.w, acc[d0 + 3]);
        }
      }
    }
    m = m_new;  // AFTER alpha and p used the old m / the new m respectively

    // (5) WAR barrier: the next iteration's loads overwrite Ks/Vs; after the last tile it also
    // fences the reuse of Qs below.
    __syncthreads();
  }

  // ---- Epilogue: deferred normalisation (one 1/l per row, l == 0 guarded -> o = 0), LSE in
  // natural-log units with the raw m re-scaled (finalize_lse: +inf for an empty row). O is
  // staged through Qs (no longer needed) so the global store is coalesced like the Q load;
  // the staging writes hit bank (tid + d) mod 32 thanks to the same +1 pad.
  const float inv_l = fa::finalize_inv_l(l);
  float* __restrict__ orow = Qs + tid * kQStride;
#pragma unroll
  for (int d = 0; d < D; ++d)
    orow[d] = acc[d] * inv_l;
  if (row_valid) lse[static_cast<std::size_t>(bh) * N + i] = fa::finalize_lse(m, l, scale);
  __syncthreads();
  for (int e = tid; e < kBr * D; e += kBr) {
    const int r = e / D;
    const int d = e - r * D;
    if (r < rows_q) og[e] = Elem<T>::from_float(Qs[r * kQStride + d]);
  }
}

// ---------------------------------------------------------------------------------------
// Host side: fa::stage3::forward / supports with the exact ABI of src/dispatch.cu.
// ---------------------------------------------------------------------------------------
namespace fa {
namespace stage3 {
namespace {

bool verbose() {
  static const bool v = std::getenv("FA_VERBOSE") != nullptr;
  return v;
}

int default_bc(int D) {
  // The plan's S3 defaults (tile_config.h): 32 for D=64, 16 for D=128. Both are in the
  // instantiated set; static_assert keeps the two files from drifting apart.
  static_assert(TileCfg<kArchTuring, 64>::kBcF32 == kBcMid, "S3 D=64 default BC drifted");
  static_assert(TileCfg<kArchTuring, 128>::kBcF32 == kBcMin, "S3 D=128 default BC drifted");
  return D == 64 ? TileCfg<kArchTuring, 64>::kBcF32 : TileCfg<kArchTuring, 128>::kBcF32;
}

bool aligned16(const void* p) {
  return (reinterpret_cast<std::uintptr_t>(p) & 15u) == 0;
}

// True for the (D, BC) pairs that are compiled. D=128/BC=64 is left out on purpose: its live
// register set (acc[128] + s[64]) spills under thread-per-row, and its 98,560 B of smem exceed
// the T4 cap anyway (header comment, OCCUPANCY).
constexpr bool instantiated(int D, int BC) {
  return !(D == 128 && BC == kBcMax);
}

template <typename T, int D, int BC>
cudaError_t launch_impl(const void* q, const void* k, const void* v, void* o, float* lse,
                        const Params& p, int sm, cudaStream_t stream) {
  constexpr std::size_t kBytes = Smem<D, BC>::kBytes;
  if (kBytes > kSmemNoOptInBytes) {
    const std::size_t cap = max_dyn_smem_bytes(arch_of_sm(sm));
    if (kBytes > cap) {
      if (verbose()) {
        std::fprintf(stderr,
                     "fa::stage3: D=%d BC=%d needs %zu B of shared memory > %zu B per-block cap "
                     "on sm_%d (try FA_FORCE_BC=%d)\n",
                     D, BC, kBytes, cap, sm, kBcMin);
      }
      return cudaErrorInvalidValue;
    }
    // Opt in above 48 KB (Turing tuning guide: a block may address the full 64 KB only after
    // cudaFuncSetAttribute). Attributes are per (kernel, device); cache per device so the
    // benchmark's back-to-back launches do not pay a runtime call each time.
    int dev = 0;
    FA_CUDA_TRY(cudaGetDevice(&dev));
    static std::atomic<std::uint64_t> opted_in{0};  // one bit per device id < 64
    const bool cached = dev < 64 && ((opted_in.load(std::memory_order_acquire) >> dev) & 1u);
    if (!cached) {
      FA_CUDA_TRY(cudaFuncSetAttribute(fa_s3_fused<T, D, BC>,
                                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       static_cast<int>(kBytes)));
      if (dev < 64) opted_in.fetch_or(std::uint64_t{1} << dev, std::memory_order_acq_rel);
    }
  }
  const dim3 grid(static_cast<unsigned>((p.N + kBr - 1) / kBr),
                  static_cast<unsigned>(p.B) * static_cast<unsigned>(p.H));
  fa_s3_fused<T, D, BC><<<grid, kBr, kBytes, stream>>>(
      static_cast<const T*>(q), static_cast<const T*>(k), static_cast<const T*>(v),
      static_cast<T*>(o), lse, p.N, p.H, p.Hkv, p.scale, scale_log2(p.scale), p.causal);
  FA_CUDA_TRY_LAUNCH();
  return cudaSuccess;
}

// Compile-time gate in front of launch_impl: the excluded (D, BC) pair never instantiates the
// kernel, so it costs no compile time and cannot spill; at run time it is a clean error.
template <typename T, int D, int BC>
cudaError_t launch(const void* q, const void* k, const void* v, void* o, float* lse,
                   const Params& p, int sm, cudaStream_t stream) {
  if constexpr (instantiated(D, BC)) {
    return launch_impl<T, D, BC>(q, k, v, o, lse, p, sm, stream);
  } else {
    (void)q, (void)k, (void)v, (void)o, (void)lse, (void)p, (void)sm, (void)stream;
    if (verbose()) {
      std::fprintf(stderr,
                   "fa::stage3: D=%d BC=%d is not instantiated (register spills, %d B smem); "
                   "use FA_FORCE_BC in {%d, %d}\n",
                   D, BC, 4 * (kBr * (D + 1) + 2 * BC * D), kBcMin, kBcMid);
    }
    return cudaErrorInvalidValue;
  }
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
    case Dtype::F16:
      return true;  // any sm this build was compiled for
    case Dtype::BF16:
      return sm >= 80;  // contract: bf16 is an sm_80+ dtype (dispatch.cu never routes it here)
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
    if (verbose()) std::fprintf(stderr, "fa::stage3: lse must be non-null (fused stages)\n");
    return cudaErrorInvalidValue;
  }
  if (p.N % 32 != 0 || p.H % p.Hkv != 0) {
    if (verbose()) std::fprintf(stderr, "fa::stage3: N %% 32 != 0 or H %% Hkv != 0\n");
    return cudaErrorInvalidValue;
  }
  if (static_cast<long long>(p.B) * p.H > 65535LL) {
    if (verbose()) {
      std::fprintf(stderr, "fa::stage3: B*H = %lld exceeds gridDim.y (65535)\n",
                   static_cast<long long>(p.B) * p.H);
    }
    return cudaErrorInvalidValue;
  }
  // Masking is applied to RAW scores before the fold into c = scale*log2e; a non-positive or
  // non-finite scale would turn the -inf sentinel into +inf/NaN (reference.py shares the
  // convention and the limitation).
  if (!(p.scale > 0.0f) || !std::isfinite(p.scale)) {
    if (verbose()) std::fprintf(stderr, "fa::stage3: scale must be finite and > 0\n");
    return cudaErrorInvalidValue;
  }
  if (!aligned16(q) || !aligned16(k) || !aligned16(v) || !aligned16(o)) {
    if (verbose()) std::fprintf(stderr, "fa::stage3: q/k/v/o must be 16-byte aligned\n");
    return cudaErrorInvalidValue;
  }
  if (dbg.block_m != 0 && verbose()) {
    static bool said = false;  // once per process; benign race, it is only a log line
    if (!said) {
      said = true;
      std::fprintf(stderr,
                   "fa::stage3: FA_FORCE_BR=%d ignored (BR is fixed at %d by the "
                   "thread-per-row mapping; only FA_FORCE_BC varies the tiling)\n",
                   dbg.block_m, kBr);
    }
  }
  const int bc = dbg.block_n != 0 ? dbg.block_n : default_bc(p.D);
  if (bc != kBcMin && bc != kBcMid && bc != kBcMax) {
    if (verbose()) {
      std::fprintf(stderr,
                   "fa::stage3: FA_FORCE_BC=%d is not an instantiated tile ({%d, %d, %d})\n", bc,
                   kBcMin, kBcMid, kBcMax);
    }
    return cudaErrorInvalidValue;
  }
  switch (dtype) {
    case Dtype::F32:
      return launch_d<float>(p.D, bc, q, k, v, o, lse, p, sm, stream);
    case Dtype::F16:
      return launch_d<__half>(p.D, bc, q, k, v, o, lse, p, sm, stream);
    case Dtype::BF16:
      return launch_d<__nv_bfloat16>(p.D, bc, q, k, v, o, lse, p, sm, stream);
  }
  return cudaErrorNotSupported;
}

}  // namespace stage3
}  // namespace fa
