// S1 — naive attention as THREE kernels, fp32 math, global memory only, S and P materialised.
//
// WHY this stage exists (README S1 row, judge_plan §4.3): it is the "before" of every later ncu
// comparison, so each kernel is written the way a first CUDA port would be and the metric that
// will indict it is named in its header. Nothing here is tuned on purpose:
//   fa_s1_qk       one thread per score, K rows read at stride D  -> uncoalesced loads (S2 fixes)
//   fa_s1_softmax  one block per row, three passes over N scores  -> N^2 fp32 P round-trips HBM
//   fa_s1_pv       one thread per output element, N-long dot      -> V re-read once per query
//                  row (N times per head), each P row read D/32 times (S3 deletes both)
//
// Math (python/fused_attn/reference.py math_reference, src/reference/cpu_reference.cpp
// attention_rows — the fp64 oracles the tests and the bench self-check compare against):
//   S[i,j] = scale * dot(Q[i,:], K[j,:]);  S[i,j] = -inf when causal and j > i (top-left mask)
//   m_i    = max_j S[i,j]
//   P[i,j] = exp(S[i,j] - m_i) / l_i,  l_i = sum_j exp(S[i,j] - m_i)
//   O[i,:] = sum_j P[i,j] * V[j,:]
//   lse_i  = m_i + log(l_i)   (S is already scaled, so this is logsumexp_j(scale*q_i.k_j), the
//                              natural-log torch.logsumexp quantity reference.py documents)
// Guards (include/fa/softmax_math.cuh conventions): a row whose max is -inf (every key masked —
// impossible for top-left causal with Nq == Nkv, kept so the invariant is local) writes P = 0,
// O = 0, lse = +inf instead of NaN.
//
// Precision: Q/K/V/O are T in {float, __half, __nv_bfloat16}; every score, max, sum and
// accumulator is fp32 (converted on load, converted back on the single O store), so fp16 inputs
// scaled by 1e2 (tests/test_kernels_gpu.py::test_half_precision_dynamic_range) cannot overflow.
//
// Layouts (CONTRACT §1): q/o [B,H,N,D], k/v [B,Hkv,N,D], all contiguous, D fastest; query head h
// reads kv head h / (H/Hkv). Workspace (CONTRACT: 2*B*H*N*N*sizeof(float)):
//   S = ws[0, B*H*N*N)            fp32 scaled+masked scores, row-major [B,H,N(i),N(j)]
//   P = ws[B*H*N*N, 2*B*H*N*N)    fp32 normalised probabilities, same layout
//
// DebugOverrides (FA_FORCE_BR/BC) are accepted and IGNORED: S1 has no tiles. The tile-invariance
// tests only run for stages >= 3, and the harness passes the same struct to every stage.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <climits>
#include <cstddef>
#include <cstdint>

#include "fa/attention.cuh"
#include "fa/cuda_check.cuh"
#include "fa/softmax_math.cuh"

namespace {

// Element <-> fp32 conversion, one overload per input dtype. The bf16 intrinsics have a software
// fallback below sm_80 (cuda_bf16.hpp), so the bf16 instantiation COMPILES for every arch; the
// host gate in stage1::supports() decides whether it may RUN (CONTRACT dtype gate: sm_80+).
__device__ __forceinline__ float to_f32(float x) {
  return x;
}
__device__ __forceinline__ float to_f32(__half x) {
  return __half2float(x);
}
__device__ __forceinline__ float to_f32(__nv_bfloat16 x) {
  return __bfloat162float(x);
}
template <typename T>
__device__ __forceinline__ T from_f32(float x);
template <>
__device__ __forceinline__ float from_f32<float>(float x) {
  return x;
}
template <>
__device__ __forceinline__ __half from_f32<__half>(float x) {
  return __float2half(x);  // round-to-nearest-even, same as torch's .half()
}
template <>
__device__ __forceinline__ __nv_bfloat16 from_f32<__nv_bfloat16>(float x) {
  return __float2bfloat16(x);
}

constexpr int kThreads = 256;  // every S1 kernel: 8 warps per block

// Block-wide reductions for fa_s1_softmax. Warp butterfly (__shfl_xor_sync) -> one slot per warp
// in `red` -> every thread folds the kWarps slots itself (no second broadcast round). Two
// barriers: the first publishes the slots, the second lets the caller reuse `red` right away.
// The fold order is fixed, so the result is bitwise reproducible run to run (the 50-run
// determinism test relies on that; no atomics anywhere in S1).
template <int kBlock>
__device__ __forceinline__ float block_max(float x, float* red) {
  constexpr int kWarps = kBlock / 32;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, off));
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (lane == 0) red[warp] = x;
  __syncthreads();
  float r = red[0];
#pragma unroll
  for (int w = 1; w < kWarps; ++w)
    r = fmaxf(r, red[w]);
  __syncthreads();
  return r;
}

template <int kBlock>
__device__ __forceinline__ float block_sum(float x, float* red) {
  constexpr int kWarps = kBlock / 32;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    x += __shfl_xor_sync(0xffffffffu, x, off);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (lane == 0) red[warp] = x;
  __syncthreads();
  float r = red[0];
#pragma unroll
  for (int w = 1; w < kWarps; ++w)
    r += red[w];
  __syncthreads();
  return r;
}

}  // namespace

// ============================================================================================
// fa_s1_qk<T, D>: S[b,h,i,j] = scale * dot(Q[b,h,i,:], K[b,hkv,j,:]), -inf above the diagonal.
//
// Thread mapping: grid (ceil(N*N / 256), B*H); thread t = blockIdx.x*256 + threadIdx.x is the
// linear score index WITHIN head blockIdx.y with j fastest: i = t / N, j = t % N. Because
// N % 32 == 0 (CONTRACT §1) a warp of 32 consecutive t never straddles a row, so i is
// warp-uniform: the Q[i,:] loads are a 32-lane broadcast of one address (one sector per
// request). The K loads are the deliberate sin of this stage: lane l reads K[j+l, d], i.e. 32
// addresses D*sizeof(T) bytes apart, so every warp-level K load request touches 32 different
// 32-byte sectors and uses only sizeof(T) bytes of each.
//
// Expected ncu symptom (docs/PROFILING_PROTOCOL.md): on the K loads only sizeof(T) of every
// 32-byte sector is used (4/32 = 12.5% fp32, 2/32 = 6% fp16), so this kernel's
// smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct sits far below fa_s1_pv's and
// below S2's float4 walk (the kernel-wide average also contains the Q loads, which are 32-lane
// broadcasts of one address, so the absolute figure is measured, not asserted). Sectors per
// request are 32 here versus 4 (fp32) / 2 (fp16) for a coalesced walk along d, i.e. 8x/16x the
// L1 sector traffic; the rest of each sector is re-fetched from L1 on the next d iteration, so
// l1tex__t_sector_hit_rate.pct is high. S2's smem tiles with float4 loads along the contiguous
// axis are the "after". Every K row is also re-read once per query row (N times per head) — the
// S3 argument, visible as lts__t_bytes.sum ∝ N^2*D while the K tensor is N*D.
//
// The S write itself is coalesced (adjacent j -> adjacent floats) and is the first N^2 fp32
// pass over HBM that dram__bytes.sum will show growing with N^2.
// ============================================================================================
template <typename T, int D>
__global__ void __launch_bounds__(kThreads)
    fa_s1_qk(const T* __restrict__ q, const T* __restrict__ k, float* __restrict__ S, int H,
             int Hkv, int N, int h_per_kv, float scale, bool causal) {
  static_assert(D == 64 || D == 128, "head dim must be 64 or 128 (CONTRACT §1)");
  const unsigned nn = static_cast<unsigned>(N) * static_cast<unsigned>(N);  // host checks N*N fits
  const unsigned t = blockIdx.x * blockDim.x + threadIdx.x;  // score index within this head
  if (t >= nn) return;
  const int i = static_cast<int>(t / static_cast<unsigned>(N));      // query row (warp-uniform)
  const int j = static_cast<int>(t - static_cast<unsigned>(i) * N);  // key column (lane-varying)
  const int bh = static_cast<int>(blockIdx.y);                       // b*H + h
  const int b = bh / H;
  const int h = bh - b * H;
  const int hkv = h / h_per_kv;  // GQA/MQA: query head h reads kv head h / (H/Hkv)

  // S[b,h,i,j]: row (bh*N + i) of length N, column j.
  float* s_out = S + (static_cast<std::size_t>(bh) * N + i) * N + j;
  if (causal && j > i) {  // top-left mask: key j is visible to query i iff j <= i
    *s_out = fa::kNegInf;
    return;
  }
  // Q[b,h,i,:] and K[b,hkv,j,:]: rows of D contiguous elements.
  const T* qi = q + (static_cast<std::size_t>(bh) * N + i) * D;
  const T* kj = k + ((static_cast<std::size_t>(b) * Hkv + hkv) * N + j) * D;
  float acc = 0.0f;
#pragma unroll 8
  for (int d = 0; d < D; ++d)
    acc = fmaf(to_f32(qi[d]), to_f32(kj[d]), acc);
  *s_out = acc * scale;  // scale applied AFTER the dot, as in reference.py (matmul * scale)
}

// ============================================================================================
// fa_s1_softmax<kBlock>: P[row, :] = softmax(S[row, :]) and optionally lse[row], one block of
// kBlock (256) threads per row (b,h,i) — grid = B*H*N blocks, blockIdx.x is the row and also the
// lse index because lse is [B,H,N] with the same (b,h,i) order as the S rows.
//
// Three passes over the row, each a strided loop j = threadIdx.x + k*256 (adjacent threads ->
// adjacent floats, so every pass is coalesced):
//   pass 1  m = max_j S[j]                       (block_max)
//   pass 2  P[j] = exp(S[j] - m), l = sum_j P[j] (un-normalised P written to global; block_sum)
//   pass 3  P[j] *= 1/l                          (re-read the thread's OWN pass-2 stores)
// Each thread rescales exactly the elements it stored in pass 2, so program order guarantees
// visibility; no fence or barrier is needed between the passes beyond the reduction barriers.
//
// Expected ncu symptom: this kernel does no math worth measuring; it is the N^2 HBM round-trip
// the S3 fused kernel deletes. Per row it reads S twice, writes P twice and reads it once
// (lts__t_bytes.sum ~ 5 * B*H*N^2 * 4 B at L2; at DRAM the pass-2 S re-read and pass-3 P
// re-read of an 8 KB row hit the 4 MB L2 and the two P writes merge before write-back, so
// dram__bytes.sum is expected between 2x and 5x B*H*N^2*4 B — the multiple is measured, the
// N^2 growth is the point: the FA1 "IO-complexity" argument as a counter). sm__throughput
// stays low; the roofline point sits at the far left.
// ============================================================================================
template <int kBlock>
__global__ void __launch_bounds__(kBlock)
    fa_s1_softmax(const float* __restrict__ S, float* __restrict__ P, float* __restrict__ lse,
                  int N) {
  static_assert(kBlock % 32 == 0 && kBlock <= 1024, "block must be whole warps");
  __shared__ float red[kBlock / 32];
  const std::size_t row = blockIdx.x;  // (b*H + h)*N + i
  const float* s = S + row * N;
  float* p = P + row * N;
  const int tid = static_cast<int>(threadIdx.x);

  // pass 1: row max (in scaled-score units, -inf for masked entries).
  float m = fa::kNegInf;
  for (int j = tid; j < N; j += kBlock)
    m = fmaxf(m, s[j]);
  m = block_max<kBlock>(m, red);

  // pass 2: un-normalised P and its row sum. expf(-inf - m) = 0 for masked j; the m == -inf
  // guard (row entirely masked) avoids (-inf) - (-inf) = NaN and writes P = 0.
  float l = 0.0f;
  const bool empty = (m == fa::kNegInf);
  for (int j = tid; j < N; j += kBlock) {
    const float pj = empty ? 0.0f : expf(s[j] - m);
    p[j] = pj;
    l += pj;
  }
  l = block_sum<kBlock>(l, red);

  // pass 3: normalise. finalize_inv_l returns 0 when l == 0 (empty row) so P stays 0, not NaN.
  const float inv_l = fa::finalize_inv_l(l);
  for (int j = tid; j < N; j += kBlock)
    p[j] *= inv_l;

  // lse = m + log(l): scores are already scaled, so scale = 1 here (fmaf(m, 1, log l) is exactly
  // m + log l) and the guard turns an empty row into +inf per softmax_math.cuh.
  if (lse != nullptr && tid == 0) lse[row] = fa::finalize_lse(m, l, 1.0f);
}

// ============================================================================================
// fa_s1_pv<T, D>: O[b,h,i,d] = sum_j P[b,h,i,j] * V[b,hkv,j,d].
//
// Thread mapping: grid (ceil(N*D / 256), B*H); thread t = blockIdx.x*256 + threadIdx.x is the
// linear output index WITHIN head blockIdx.y with d fastest: i = t / D, d = t % D. D >= 64 so a
// warp never straddles a row: i is warp-uniform, hence the P[i,j] load is a 32-lane broadcast
// and the V[j, d..d+31] load is 32 consecutive elements = one fully used 128-byte line (fp32)
// or 64 bytes (fp16) — this kernel's loads ARE coalesced, which is what makes its ncu report
// the control for fa_s1_qk's: same shape of loop, different bytes-per-sector figure.
//
// Expected ncu symptom, stated per memory level because they differ: at L1 (l1tex__t_bytes.sum)
// it is one FFMA (2 FLOP) per 8 B requested (P + V), ~0.25 FLOP/B — LSU/L1-throughput bound with
// stalls dominated by long_scoreboard. At DRAM the picture inverts: a P row is one broadcast
// sector per warp (read from DRAM once, 4*N^2 B per head) and V (N*D, 512 KB at C64) is served
// from L2, so the DRAM-level intensity scripts/roofline.py plots is ~D/2 FLOP/B, RIGHT of the
// T4 ridge (8.1e12 / 320e9 ~ 25 FLOP/B), with dram__throughput low. The N^2*D L2 read traffic
// (lts__t_bytes.sum) while the tensor is N*D is S3's argument. The O store is coalesced, once.
// ============================================================================================
template <typename T, int D>
__global__ void __launch_bounds__(kThreads)
    fa_s1_pv(const float* __restrict__ P, const T* __restrict__ v, T* __restrict__ o, int H,
             int Hkv, int N, int h_per_kv) {
  static_assert(D == 64 || D == 128, "head dim must be 64 or 128 (CONTRACT §1)");
  const int nd = N * D;                                                   // outputs per head
  const int t = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);  // output index in head
  if (t >= nd) return;
  const int i = t / D;      // query row (warp-uniform; D is a power of two -> shift)
  const int d = t - i * D;  // head-dim column (lane-varying, contiguous)
  const int bh = static_cast<int>(blockIdx.y);  // b*H + h
  const int b = bh / H;
  const int h = bh - b * H;
  const int hkv = h / h_per_kv;  // GQA/MQA: query head h reads kv head h / (H/Hkv)

  const float* pi = P + (static_cast<std::size_t>(bh) * N + i) * N;  // P[b,h,i,:]
  // V[b,hkv,0,d]: head base plus the column; row j is at stride D.
  const T* vd = v + (static_cast<std::size_t>(b) * Hkv + hkv) * N * D + d;
  float acc = 0.0f;
#pragma unroll 4
  for (int j = 0; j < N; ++j)
    acc = fmaf(pi[j], to_f32(vd[static_cast<std::size_t>(j) * D]), acc);
  o[(static_cast<std::size_t>(bh) * N + i) * D + d] = from_f32<T>(acc);
}

// ============================================================================================
// Host launcher: the stage ABI of src/dispatch.cu (fa::stage1::forward / supports).
// ============================================================================================
namespace fa {
namespace stage1 {

namespace {

// Grid formulas (all three kernels use kThreads = 256 threads per block):
//   fa_s1_qk       grid (ceil(N*N / 256), B*H)   one thread per score
//   fa_s1_softmax  grid (B*H*N)                  one block per row
//   fa_s1_pv       grid (ceil(N*D / 256), B*H)   one thread per output element
// Limits these formulas rely on, checked by forward() before any launch: B*H <= 65535
// (gridDim.y), N*N < 2^32 (the in-head score index is 32-bit; N = 65536 would already need a
// 32 GB workspace), B*H*N <= INT_MAX (gridDim.x of the softmax grid).
template <typename T, int D>
cudaError_t launch(const void* q, const void* k, const void* v, void* o, float* lse,
                   void* workspace, const Params& p, cudaStream_t stream) {
  const std::size_t nn = static_cast<std::size_t>(p.N) * static_cast<std::size_t>(p.N);
  const std::size_t bh = static_cast<std::size_t>(p.B) * static_cast<std::size_t>(p.H);
  float* S = static_cast<float*>(workspace);  // first B*H*N*N floats
  float* P = S + bh * nn;                     // next  B*H*N*N floats
  const int h_per_kv = p.H / p.Hkv;           // query heads per kv head (>= 1; H % Hkv == 0)
  const auto* qt = static_cast<const T*>(q);
  const auto* kt = static_cast<const T*>(k);
  const auto* vt = static_cast<const T*>(v);
  auto* ot = static_cast<T*>(o);

  const dim3 grid_qk(static_cast<unsigned>((nn + kThreads - 1) / kThreads),
                     static_cast<unsigned>(bh));
  fa_s1_qk<T, D>
      <<<grid_qk, kThreads, 0, stream>>>(qt, kt, S, p.H, p.Hkv, p.N, h_per_kv, p.scale, p.causal);
  FA_CUDA_TRY_LAUNCH();

  const unsigned rows = static_cast<unsigned>(bh * static_cast<std::size_t>(p.N));
  fa_s1_softmax<kThreads><<<rows, kThreads, 0, stream>>>(S, P, lse, p.N);
  FA_CUDA_TRY_LAUNCH();

  const std::size_t nd = static_cast<std::size_t>(p.N) * D;
  const dim3 grid_pv(static_cast<unsigned>((nd + kThreads - 1) / kThreads),
                     static_cast<unsigned>(bh));
  fa_s1_pv<T, D><<<grid_pv, kThreads, 0, stream>>>(P, vt, ot, p.H, p.Hkv, p.N, h_per_kv);
  FA_CUDA_TRY_LAUNCH();
  return cudaSuccess;
}

template <typename T>
cudaError_t launch_d(const void* q, const void* k, const void* v, void* o, float* lse,
                     void* workspace, const Params& p, cudaStream_t stream) {
  switch (p.D) {
    case 64:
      return launch<T, 64>(q, k, v, o, lse, workspace, p, stream);
    case 128:
      return launch<T, 128>(q, k, v, o, lse, workspace, p, stream);
    default:
      return cudaErrorNotSupported;  // dispatch.cu validated D already; belt and braces
  }
}

}  // namespace

// F32 and F16 for D in {64,128} on every arch. BF16 is NOT offered by S1: attention.cuh says
// Naive/Tiled/Fused support F32 and F16, dispatch.cu's contract_allows() rejects bf16 for them,
// and instantiating unreachable bf16 kernels would only pad the ptxas report; the bf16 path
// (software conversion on sm_75, native on sm_80+) is a one-line change if the contract relaxes.
bool supports(Dtype dtype, int D, int sm) {
  if (D != 64 && D != 128) return false;
  switch (dtype) {
    case Dtype::F32:
    case Dtype::F16:
      return true;
    case Dtype::BF16:
      return false;  // unreachable in this build, see the comment above
  }
  return false;
}

// Inputs were validated by fa::forward (shapes, N % 32, non-null pointers, workspace present);
// only the grid-limit invariants documented above launch() are re-checked here. `dbg` is
// ignored: S1 has no tiles (see the file header). Returns the first launch error, never syncs.
cudaError_t forward(Dtype dtype, const void* q, const void* k, const void* v, void* o, float* lse,
                    void* workspace, const Params& p, const DebugOverrides& /*dbg*/, int sm,
                    cudaStream_t stream) {
  if (!supports(dtype, p.D, sm)) return cudaErrorNotSupported;
  if (workspace == nullptr) return cudaErrorInvalidValue;
  const std::size_t bh = static_cast<std::size_t>(p.B) * static_cast<std::size_t>(p.H);
  const std::size_t nn = static_cast<std::size_t>(p.N) * static_cast<std::size_t>(p.N);
  if (bh > 65535u) return cudaErrorInvalidConfiguration;  // gridDim.y ceiling
  if (nn > static_cast<std::size_t>(UINT_MAX)) return cudaErrorInvalidConfiguration;  // 32-bit t
  if (bh * static_cast<std::size_t>(p.N) > static_cast<std::size_t>(INT_MAX))
    return cudaErrorInvalidConfiguration;  // softmax grid.x
  switch (dtype) {
    case Dtype::F32:
      return launch_d<float>(q, k, v, o, lse, workspace, p, stream);
    case Dtype::F16:
      return launch_d<__half>(q, k, v, o, lse, workspace, p, stream);
    case Dtype::BF16:
      break;  // not offered by S1 (see supports()); the registry never routes bf16 here
  }
  return cudaErrorNotSupported;
}

}  // namespace stage1
}  // namespace fa
