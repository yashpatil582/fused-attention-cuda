// experiments/wmma_gemm/wmma_gemm.cu — standalone S4 de-risking microkernel.
//
// WHAT: one fused-attention inner iteration for ONE K/V block, on tensor cores:
//   S = A·Bᵀ  (A = Q tile [64×K], B = K tile [64×K], fp16/bf16 in, fp32 accumulate, nvcuda::wmma)
//   store_matrix_sync(S) → padded fp32 smem → row max / row sum via warp shuffles
//   P = exp2((S − rowmax)·scale·log2e) as fp16 → reload as matrix_a → O = P·V → /rowsum.
// WHY: WMMA fragment layouts are opaque (CUDA C++ Programming Guide, "Warp Matrix Functions":
//   "the way matrix elements map to fragment internal storage is unspecified"), so a softmax
//   cannot index acc.x[i] by (row, col). The only portable route is the smem round trip
//   below. This file proves that route compiles, is numerically right vs an fp64 CPU oracle,
//   fits the T4's 64 KB smem/block, and exercises the tensor pipe — BEFORE it is fused into
//   src/stage4_wmma.cu. No torch, no cuBLAS (optional comparison behind -DUSE_CUBLAS).
//
// Build/run: ./run.sh [M N K] [--iters n] [--bf16] [--dump file] [--ref-only]   (see README.md)
// Exit codes: 0 pass, 1 S check failed, 2 P·V ratio rule failed, 3 usage / unsupported / CUDA
// error.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <type_traits>
#include <vector>
#ifdef USE_CUBLAS
#include <cublas_v2.h>
#endif

using namespace nvcuda;

// ---------------------------------------------------------------------------------------------
// Tile configuration (fixed: the S4 d=64 configuration from .research/PLAN.md)
// ---------------------------------------------------------------------------------------------
constexpr int BM = 64;    // query rows per block  (Br)
constexpr int BN = 64;    // keys per block        (Bc)
constexpr int BK = 16;    // WMMA k-step
constexpr int D = 64;     // head dim of V (P·V output width); K (= Q/K head dim) is a runtime arg
constexpr int WARPS = 4;  // 2×2 warp grid, each warp owns a 32×32 sub-tile = 2×2 fragments
constexpr int THREADS = WARPS * 32;
constexpr int ROWS_PER_WARP = BM / WARPS;  // softmax/epilogue: warp w owns rows [16w, 16w+16)
// Padding. +8 halves = +16 bytes per row. WHY 8 and not 1: load/store_matrix_sync require
// ldm to be a multiple of 8 halves / 4 floats (16 B) and every fragment pointer to be 256-bit
// aligned, so the pad must itself be a multiple of 8 halves. WHY pad at all: an unpadded
// 64-half row is exactly 128 B = one full sweep of the 32×4 B banks, so the same column of 8
// consecutive rows lands in the same bank (8-way conflict for 16-B row-chunk loads). With a
// 144-B stride (36 words) row r starts at bank (4·r) mod 32: 8 rows × 4 banks = all 32 banks
// once → conflict-free for ldmatrix-style 16-B-per-row chunks. (Expected from the bank
// arithmetic; measured by S5a with l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum.)
constexpr int PAD_H = 8;         // halves/bf16
constexpr int PAD_F = 4;         // floats (same 16 B)
constexpr int LDS = BN + PAD_F;  // fp32 S / O tile stride: 68 floats = 272 B (bank shift 4/row)
constexpr int LDV = D + PAD_H;   // V tile stride:          72 halves = 144 B
constexpr int LDP = BN + PAD_H;  // P tile stride:          72 halves = 144 B
constexpr int STATIC_SMEM_LIMIT = 48 * 1024;  // above this: dynamic smem + cudaFuncSetAttribute

#define CUDA_CHECK(call)                                                                      \
  do {                                                                                        \
    cudaError_t err__ = (call);                                                               \
    if (err__ != cudaSuccess) {                                                               \
      std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorName(err__), __FILE__, \
                   __LINE__, cudaGetErrorString(err__));                                      \
      std::exit(3);                                                                           \
    }                                                                                         \
  } while (0)

// dtype traits: the ONLY place fp16 and bf16 differ (conversions + the packed 2-vector).
template <typename T>
struct Cvt;
template <>
struct Cvt<__half> {
  using vec2 = __half2;
  static constexpr const char* name = "fp16";
  static __host__ __device__ __forceinline__ __half from(float f) {
    return __float2half(f);
  }
  static __host__ __device__ __forceinline__ float to(__half h) {
    return __half2float(h);
  }
  static __device__ __forceinline__ __half2 pack2(float a, float b) {
    return __floats2half2_rn(a, b);  // .x = a at the lower address
  }
};
template <>
struct Cvt<__nv_bfloat16> {
  using vec2 = __nv_bfloat162;
  static constexpr const char* name = "bf16";
  static __host__ __device__ __forceinline__ __nv_bfloat16 from(float f) {
    return __float2bfloat16(f);
  }
  static __host__ __device__ __forceinline__ float to(__nv_bfloat16 h) {
    return __bfloat162float(h);
  }
  static __device__ __forceinline__ __nv_bfloat162 pack2(float a, float b) {
    return __floats2bfloat162_rn(a, b);
  }
};

// ---------------------------------------------------------------------------------------------
// Shared-memory carve-up, computed identically on host (for the launch) and device (pointers).
// P aliases the B (= K) tile: B is dead once S is stored, exactly as the K tile is dead in S4
// once S for that KV block exists. That aliasing is what makes the S4 d=64 budget fit: without
// it Q+K+V+S+P = 54 KB (1 block/SM on T4 and no room for D=128 at all).
// ---------------------------------------------------------------------------------------------
struct SmemLayout {
  int lda;                          // A/B tile row stride in elements (K + PAD_H)
  size_t a, b, v, s, stats, total;  // byte offsets; stats = row_l[BM] then row_m[BM]
};
__host__ __device__ inline size_t align128(size_t x) {
  return (x + 127) & ~static_cast<size_t>(127);
}
__host__ __device__ inline SmemLayout smem_layout(int K, int elem_bytes) {
  SmemLayout L;
  L.lda = K + PAD_H;
  const size_t bt = static_cast<size_t>(BN) * L.lda, pt = static_cast<size_t>(BM) * LDP;
  const size_t b_elems = bt > pt ? bt : pt;  // (no std::max: host+device function)
  L.a = 0;
  L.b = align128(L.a + static_cast<size_t>(BM) * L.lda * elem_bytes);
  L.v = align128(L.b + b_elems * elem_bytes);  // P (BM×LDP) aliases B; K<64 makes P the larger
  L.s = align128(L.v + static_cast<size_t>(BN) * LDV * elem_bytes);
  L.stats = align128(L.s + static_cast<size_t>(BM) * LDS * sizeof(float));
  L.total = align128(L.stats + 2 * BM * sizeof(float));
  return L;
}

// Global → shared, 16-byte chunks. Coalesced: consecutive threads copy consecutive 16-B chunks
// of one row (a warp moves 4 full 128-B rows for K=64). Requires cols % 8 == 0 and 16-B aligned
// row starts on both sides — the +8 pad keeps the smem row stride a multiple of 16 B too.
template <typename T>
__device__ __forceinline__ void load_tile_16B(T* dst, const T* __restrict__ src, int rows, int cols,
                                              int ld_src, int ld_dst, int tid) {
  constexpr int EPC = 16 / sizeof(T);  // elements per 16-byte chunk (8 for 16-bit types)
  const int chunks_per_row = cols / EPC;
  for (int t = tid; t < rows * chunks_per_row; t += THREADS) {
    const int r = t / chunks_per_row, c = (t - r * chunks_per_row) * EPC;
    *reinterpret_cast<uint4*>(dst + r * ld_dst + c) =
        *reinterpret_cast<const uint4*>(src + static_cast<size_t>(r) * ld_src + c);
  }
}

// Butterfly reductions: every lane ends with the full-row value (no smem, no barrier).
__device__ __forceinline__ float warp_max(float v) {
  for (int o = 16; o > 0; o >>= 1)
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}
__device__ __forceinline__ float warp_sum(float v) {
  for (int o = 16; o > 0; o >>= 1)
    v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

using AccFrag = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;
__device__ __forceinline__ void zero_acc(AccFrag (&acc)[2][2]) {
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j)
      wmma::fill_fragment(acc[i][j], 0.0f);
}
// The opaque-layout escape hatch: dump the 2×2 accumulators into smem in a KNOWN row-major layout
// at this warp's 32×32 position. Fragment pointer = base + 16-row × 272-B + 16-col × 4-B multiples,
// so it stays 256-bit aligned as required.
__device__ __forceinline__ void store_acc(float* tile, const AccFrag (&acc)[2][2], int warp_m,
                                          int warp_n) {
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j)
      wmma::store_matrix_sync(tile + (warp_m * 32 + i * 16) * LDS + warp_n * 32 + j * 16, acc[i][j],
                              LDS, wmma::mem_row_major);
}

// ---------------------------------------------------------------------------------------------
// The block body. Grid = (N/BN, M/BM); block (bn, bm) owns query rows [bm·64, +64) and keys
// [bn·64, +64) — one (Q tile, KV block) pair of the S4 loop, run in isolation.
// ---------------------------------------------------------------------------------------------
template <typename T>
__device__ __forceinline__ void block_attention(const T* __restrict__ A, const T* __restrict__ B,
                                                const T* __restrict__ V, float* out_s, float* out_o,
                                                float* sink, int N, int K, float c_log2) {
  extern __shared__ __align__(128) unsigned char smem_raw[];  // 128-B base ⇒ 256-bit fragment ptrs
  const SmemLayout L = smem_layout(K, sizeof(T));
  T* As = reinterpret_cast<T*>(smem_raw + L.a);
  T* Bs = reinterpret_cast<T*>(smem_raw + L.b);
  T* Ps = Bs;  // alias: B is dead after the S GEMM + barrier (see SmemLayout)
  T* Vs = reinterpret_cast<T*>(smem_raw + L.v);
  float* Ss = reinterpret_cast<float*>(smem_raw + L.s);  // S, then reused for O
  float* row_l = reinterpret_cast<float*>(smem_raw + L.stats);
  float* row_m = row_l + BM;

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int warp_m = warp >> 1, warp_n = warp & 1;  // 2×2 warp grid over the 64×64 tile
  const int bn = blockIdx.x, bm = blockIdx.y, lda = L.lda;

  // (1) Stage the three operand tiles. In S4, A (=Q) stays resident; B (=K) and V stream.
  load_tile_16B<T>(As, A + static_cast<size_t>(bm) * BM * K, BM, K, K, lda, tid);
  load_tile_16B<T>(Bs, B + static_cast<size_t>(bn) * BN * K, BN, K, K, lda, tid);
  load_tile_16B<T>(Vs, V + static_cast<size_t>(bn) * BN * D, BN, D, D, LDV, tid);
  __syncthreads();

  // (2) S = A·Bᵀ. B is loaded as matrix_b COL_MAJOR: matrix_b is K×N and col_major means element
  // (k, n) lives at ptr[n·ldm + k] — exactly where row-major B[n][k] already is. Kᵀ is free.
  AccFrag acc[2][2];
  zero_acc(acc);
  for (int kk = 0; kk < K; kk += BK) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> a[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::col_major> b[2];
#pragma unroll
    for (int i = 0; i < 2; ++i)
      wmma::load_matrix_sync(a[i], As + (warp_m * 32 + i * 16) * lda + kk, lda);
#pragma unroll
    for (int j = 0; j < 2; ++j)
      wmma::load_matrix_sync(b[j], Bs + (warp_n * 32 + j * 16) * lda + kk, lda);
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int j = 0; j < 2; ++j)
        wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
  }
  store_acc(Ss, acc, warp_m, warp_n);  // (3) S → smem in a known layout
  __syncthreads();  // S complete and visible; B tile is dead from here on (P may overwrite it)

  // (4) Row softmax from smem. Each lane holds 2 adjacent columns (float2 read: 32 lanes × 8 B
  // = 256 B contiguous → conflict-free); reductions are shuffles. P is written as packed
  // 16-bit pairs into the (dead) B tile.
  for (int i = 0; i < ROWS_PER_WARP; ++i) {
    const int r = warp * ROWS_PER_WARP + i;
    const float2 s2 = *reinterpret_cast<const float2*>(Ss + r * LDS + 2 * lane);
    if (out_s != nullptr)  // check mode: expose S so the host can verify the first GEMM alone
      *reinterpret_cast<float2*>(out_s + static_cast<size_t>(bm * BM + r) * N + bn * BN +
                                 2 * lane) = s2;
    const float m = warp_max(fmaxf(s2.x, s2.y));
    // exp2 with the softmax scale folded into c_log2 = scale·log2(e): one FMUL + EX2 per element
    // instead of expf. (s − m) ≤ 0 so P ∈ (0, 1]; no -inf guard needed without masking.
    const float p0 = exp2f((s2.x - m) * c_log2), p1 = exp2f((s2.y - m) * c_log2);
    const float l = warp_sum(p0 + p1);  // row sum from the fp32 P, as FA2 does (before rounding)
    *reinterpret_cast<typename Cvt<T>::vec2*>(Ps + r * LDP + 2 * lane) = Cvt<T>::pack2(p0, p1);
    if (lane == 0) {
      row_l[r] = l;
      row_m[r] = m;  // unused here; it is S4's running-max slot (m_i) — kept so the budget is real
    }
  }
  __syncthreads();

  // (5) O = P·V: P re-enters the tensor pipe as matrix_a (row-major, ldm = LDP), V is matrix_b
  // row_major (K×N = keys × d). Same 2×2 warp tiling, 4 k-steps of 16 keys.
  zero_acc(acc);
#pragma unroll
  for (int kk = 0; kk < BN; kk += BK) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> a[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::row_major> b[2];
#pragma unroll
    for (int i = 0; i < 2; ++i)
      wmma::load_matrix_sync(a[i], Ps + (warp_m * 32 + i * 16) * LDP + kk, LDP);
#pragma unroll
    for (int j = 0; j < 2; ++j)
      wmma::load_matrix_sync(b[j], Vs + kk * LDV + warp_n * 32 + j * 16, LDV);
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int j = 0; j < 2; ++j)
        wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
  }
  // (6) The row-wise divide by l is position-dependent too, so O also goes through smem.
  // S4 consequence: the online-softmax rescale O ← α·O + P·V is row-wise as well, so the O
  // accumulator cannot live in an opaque fragment across KV blocks — see README.
  store_acc(Ss, acc, warp_m, warp_n);
  __syncthreads();

  // (7) Epilogue: O / l. A checksum always reaches global memory so the timed (no-output) runs
  // cannot be dead-code-eliminated. l ≥ 1 here (the row max contributes exp2(0)); S4 needs an
  // l == 0 guard only for fully-masked rows.
  float csum = 0.0f;
  const int nbn = N / BN;
  for (int i = 0; i < ROWS_PER_WARP; ++i) {
    const int r = warp * ROWS_PER_WARP + i;
    float2 o2 = *reinterpret_cast<const float2*>(Ss + r * LDS + 2 * lane);
    const float inv_l = 1.0f / row_l[r];
    o2.x *= inv_l;
    o2.y *= inv_l;
    csum += o2.x + o2.y;
    if (out_o != nullptr)  // layout [M/64][N/64][64][64]: one 64×64 partial per (bm, bn)
      *reinterpret_cast<float2*>(out_o + (static_cast<size_t>(bm * nbn + bn) * BM + r) * D +
                                 2 * lane) = o2;
  }
  csum = warp_sum(csum);
  if (lane == 0) atomicAdd(sink, csum);
}

__global__ void __launch_bounds__(THREADS)
    fa_wmma_f16_kernel(const __half* A, const __half* B, const __half* V, float* out_s,
                       float* out_o, float* sink, int N, int K, float c_log2) {
  block_attention<__half>(A, B, V, out_s, out_o, sink, N, K, c_log2);
}
// bf16 WMMA fragments exist only for compute capability ≥ 8.0 (Programming Guide, "Alternate
// Floating Point"); Turing tensor cores are fp16/int only. The body is compiled for sm_80+ passes
// only; an sm_75 build gets a trap and the HOST refuses to launch it (binaryVersion check).
__global__ void __launch_bounds__(THREADS)
    fa_wmma_bf16_kernel(const __nv_bfloat16* A, const __nv_bfloat16* B, const __nv_bfloat16* V,
                        float* out_s, float* out_o, float* sink, int N, int K, float c_log2) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  block_attention<__nv_bfloat16>(A, B, V, out_s, out_o, sink, N, K, c_log2);
#else
  __trap();
#endif
}

template <typename T>
static const void* kernel_ptr() {
  if constexpr (std::is_same_v<T, __half>)
    return reinterpret_cast<const void*>(&fa_wmma_f16_kernel);
  else
    return reinterpret_cast<const void*>(&fa_wmma_bf16_kernel);
}
template <typename T>
static void launch(dim3 grid, size_t smem, const T* A, const T* B, const T* V, float* S, float* O,
                   float* sink, int N, int K, float c) {
  if constexpr (std::is_same_v<T, __half>)
    fa_wmma_f16_kernel<<<grid, THREADS, smem>>>(A, B, V, S, O, sink, N, K, c);
  else
    fa_wmma_bf16_kernel<<<grid, THREADS, smem>>>(A, B, V, S, O, sink, N, K, c);
}

// ---------------------------------------------------------------------------------------------
// Host side: deterministic inputs, fp64 oracle, checks, timing.
// ---------------------------------------------------------------------------------------------
struct Args {
  int M = 2048, N = 2048, K = 64, iters = 50;
  bool bf16 = false, ref_only = false;
  std::string dump;
};

// Knuth MMIX 64-bit LCG; the top 24 bits give a value in [-1, 1). Mirrored in check_vs_torch.py.
static double lcg_next(uint64_t& s) {
  s = s * 6364136223846793005ULL + 1442695040888963407ULL;
  return static_cast<double>(s >> 40) / 16777216.0 * 2.0 - 1.0;
}
template <typename T>
static void fill(std::vector<T>& h, std::vector<double>& hd, size_t n, uint64_t seed) {
  h.resize(n);
  hd.resize(n);
  for (size_t i = 0; i < n; ++i) {
    h[i] = Cvt<T>::from(static_cast<float>(lcg_next(seed)));
    hd[i] = static_cast<double>(Cvt<T>::to(h[i]));  // the oracle sees exactly the rounded inputs
  }
}

// fp64 oracle on the rounded inputs. S: plain GEMM. O: one-block attention softmax(S_blk)·V_blk
// in the kernel's [M/64][N/64][64][64] layout, plus the "same-dtype" reference O16 that rounds P
// to T before P·V (what any 16-bit tensor-core kernel must do). err_ref = max|O16 − O| is the
// yardstick of the FA test-suite ratio rule used in docs/NUMERICS.md: err_mine ≤ 2·err_ref + 1e-6.
struct Oracle {
  std::vector<double> S, O, O16;
  double s_max = 0, s_mean = 0, o_mean = 0, err_ref = 0;
};
template <typename T>
static Oracle oracle(int M, int N, int K, const std::vector<double>& A,
                     const std::vector<double>& B, const std::vector<double>& V, double scale) {
  Oracle R;
  const size_t nS = static_cast<size_t>(M) * N;
  R.S.resize(nS);
  R.O.resize(nS);
  R.O16.resize(nS);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      const double *arow = &A[static_cast<size_t>(m) * K], *brow = &B[static_cast<size_t>(n) * K];
      double acc = 0.0;
      for (int k = 0; k < K; ++k)
        acc += arow[k] * brow[k];
      R.S[static_cast<size_t>(m) * N + n] = acc;
      R.s_max = std::max(R.s_max, std::fabs(acc));
      R.s_mean += std::fabs(acc);
    }
  R.s_mean /= static_cast<double>(nS);
  const int nbn = N / BN;
  std::vector<double> p(BN), p16(BN);
  for (int bm = 0; bm < M / BM; ++bm)
    for (int bn = 0; bn < nbn; ++bn)
      for (int r = 0; r < BM; ++r) {
        const double* srow = &R.S[static_cast<size_t>(bm * BM + r) * N + bn * BN];
        const double mx = *std::max_element(srow, srow + BN);
        double l = 0.0;
        for (int j = 0; j < BN; ++j) {
          p[j] = std::exp((srow[j] - mx) * scale);
          l += p[j];  // from unrounded P, matching the kernel's fp32 row sum
          p16[j] = static_cast<double>(Cvt<T>::to(Cvt<T>::from(static_cast<float>(p[j]))));
        }
        for (int d = 0; d < D; ++d) {
          double o = 0.0, o16 = 0.0;
          for (int j = 0; j < BN; ++j) {
            const double v = V[static_cast<size_t>(bn * BN + j) * D + d];
            o += p[j] * v;
            o16 += p16[j] * v;
          }
          const size_t idx = (static_cast<size_t>(bm * nbn + bn) * BM + r) * D + d;
          R.O[idx] = o / l;
          R.O16[idx] = o16 / l;
          R.err_ref = std::max(R.err_ref, std::fabs(R.O16[idx] - R.O[idx]));
          R.o_mean += std::fabs(R.O[idx]);
        }
      }
  R.o_mean /= static_cast<double>(nS);
  return R;
}

static float median_ms(std::vector<float> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

template <typename T>
static int run(const Args& a) {
  const int M = a.M, N = a.N, K = a.K;
  const size_t nA = static_cast<size_t>(M) * K, nB = static_cast<size_t>(N) * K,
               nV = static_cast<size_t>(N) * D, nS = static_cast<size_t>(M) * N;
  const double scale = 1.0 / std::sqrt(static_cast<double>(K));
  const float c_log2 = static_cast<float>(scale * 1.4426950408889634);  // scale·log2(e)

  std::vector<T> hA, hB, hV;
  std::vector<double> dA64, dB64, dV64;
  fill<T>(hA, dA64, nA, 1);
  fill<T>(hB, dB64, nB, 2);
  fill<T>(hV, dV64, nV, 3);
  const Oracle R = oracle<T>(M, N, K, dA64, dB64, dV64, scale);
  std::printf("wmma_gemm: M=%d N=%d K=%d D=%d dtype=%s iters=%d\n", M, N, K, D, Cvt<T>::name,
              a.iters);
  std::printf("[ref] S = A·Bᵀ: max|S_ref| %.4f mean|S_ref| %.4f\n", R.s_max, R.s_mean);
  std::printf("[ref] O: mean|O_ref| %.4f err_ref(%s-rounded P) %.3e\n", R.o_mean, Cvt<T>::name,
              R.err_ref);
  if (a.ref_only) return 0;  // no CUDA call was made: usable on a GPU-less box to diff vs torch

  cudaDeviceProp prop;
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  const SmemLayout L = smem_layout(K, sizeof(T));
  std::printf("device: %s (sm_%d%d), smem/block opt-in max %zu B, %d warps/SM max\n", prop.name,
              prop.major, prop.minor, prop.sharedMemPerBlockOptin,
              prop.maxThreadsPerMultiProcessor / 32);
  std::printf("smem layout (B): A %zu | B/P %zu | V %zu | S/O %zu | stats %zu | total %zu (%s)\n",
              L.b - L.a, L.v - L.b, L.s - L.v, L.stats - L.s, L.total - L.stats, L.total,
              L.total > STATIC_SMEM_LIMIT ? "dynamic, >48 KB opt-in" : "dynamic, <=48 KB");
  if (a.bf16 && prop.major < 8) {  // runtime refusal on Turing (T4): bf16 WMMA does not exist there
    std::fprintf(stderr, "error: --bf16 needs compute capability >= 8.0; this device is sm_%d%d\n",
                 prop.major, prop.minor);
    return 3;
  }
  if (L.total > prop.sharedMemPerBlockOptin) {
    std::fprintf(stderr, "error: needs %zu B smem/block, device allows %zu B\n", L.total,
                 prop.sharedMemPerBlockOptin);
    return 3;
  }
  cudaFuncAttributes fattr;
  CUDA_CHECK(cudaFuncGetAttributes(&fattr, kernel_ptr<T>()));
  if (a.bf16 && fattr.binaryVersion < 80) {  // e.g. an sm_75 build JIT-run on an A100: trap stub
    std::fprintf(stderr,
                 "error: bf16 kernel in this binary was built for sm_%d (< 80); rebuild with "
                 "ARCH=sm_80 or newer\n",
                 fattr.binaryVersion);
    return 3;
  }
  if (L.total > STATIC_SMEM_LIMIT)  // opt-in can FAIL on a device with less smem: checked.
    CUDA_CHECK(cudaFuncSetAttribute(kernel_ptr<T>(), cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    static_cast<int>(L.total)));
  int blocks_per_sm = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel_ptr<T>(), THREADS,
                                                           L.total));
  std::printf("ptxas: %d regs/thread, %zu B static smem; occupancy %d block(s)/SM = %d warps/SM\n",
              fattr.numRegs, fattr.sharedSizeBytes, blocks_per_sm, blocks_per_sm * WARPS);

  T *dA, *dB, *dV;
  float *dS, *dO, *dSink;
  CUDA_CHECK(cudaMalloc(&dA, nA * sizeof(T)));
  CUDA_CHECK(cudaMalloc(&dB, nB * sizeof(T)));
  CUDA_CHECK(cudaMalloc(&dV, nV * sizeof(T)));
  CUDA_CHECK(cudaMalloc(&dS, nS * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dO, nS * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dSink, sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), nA * sizeof(T), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), nB * sizeof(T), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dV, hV.data(), nV * sizeof(T), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dSink, 0, sizeof(float)));

  const dim3 grid(N / BN, M / BM);
  launch<T>(grid, L.total, dA, dB, dV, dS, dO, dSink, N, K, c_log2);  // check run (writes S, O)
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> hS(nS), hO(nS);
  CUDA_CHECK(cudaMemcpy(hS.data(), dS, nS * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hO.data(), dO, nS * sizeof(float), cudaMemcpyDeviceToHost));

  double s_err = 0.0, err_mine = 0.0;
  bool finite = true;
  for (size_t i = 0; i < nS; ++i) {
    finite = finite && std::isfinite(hS[i]) && std::isfinite(hO[i]);
    s_err = std::max(s_err, std::fabs(static_cast<double>(hS[i]) - R.S[i]));
    err_mine = std::max(err_mine, std::fabs(static_cast<double>(hO[i]) - R.O[i]));
  }
  const bool s_pass = finite && s_err <= 5e-2;
  const bool o_pass = finite && err_mine <= 2.0 * R.err_ref + 1e-6;
  std::printf("[check] S max abs err %.3e (limit 5e-2) %s\n", s_err, s_pass ? "PASS" : "FAIL");
  std::printf(
      "[check] O err_mine %.3e err_ref %.3e ratio %.3f (rule err_mine <= 2*err_ref + 1e-6) %s\n",
      err_mine, R.err_ref, R.err_ref > 0 ? err_mine / R.err_ref : 0.0, o_pass ? "PASS" : "FAIL");

  if (!a.dump.empty()) {  // raw dump for check_vs_torch.py --compare
    FILE* f = std::fopen(a.dump.c_str(), "wb");
    if (f == nullptr) {
      std::perror("dump");
      return 3;
    }
    const int hdr[4] = {M, N, K, a.bf16 ? 1 : 0};
    std::fwrite("WMMAGEMM", 1, 8, f);
    std::fwrite(hdr, sizeof(int), 4, f);
    std::fwrite(hS.data(), sizeof(float), nS, f);
    std::fwrite(hO.data(), sizeof(float), nS, f);
    std::fclose(f);
    std::printf("[dump] wrote S and O to %s\n", a.dump.c_str());
  }

  // Timing: no S/O writes (only the 4-B checksum sink), one cudaEvent pair per iteration, single
  // sync after the loop, median. No L2 flush on purpose: A/B/V total 0.75 MB at the default size
  // and live in L2 — this measures the tensor-core inner loop, not HBM (bench/ does the flush).
  std::vector<cudaEvent_t> ev(2 * a.iters);
  for (auto& e : ev)
    CUDA_CHECK(cudaEventCreate(&e));
  std::vector<float> ms(a.iters);
  auto time_loop = [&](auto&& fn) {  // returns median ms of a.iters launches after 5 warm-ups
    for (int i = 0; i < 5; ++i)
      fn();
    for (int i = 0; i < a.iters; ++i) {
      CUDA_CHECK(cudaEventRecord(ev[2 * i]));
      fn();
      CUDA_CHECK(cudaEventRecord(ev[2 * i + 1]));
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int i = 0; i < a.iters; ++i)
      CUDA_CHECK(cudaEventElapsedTime(&ms[i], ev[2 * i], ev[2 * i + 1]));
    return static_cast<double>(median_ms(ms));
  };
  const double med = time_loop([&] {
    launch<T>(grid, L.total, dA, dB, dV, nullptr, nullptr, dSink, N, K, c_log2);
  });
  const double mn = *std::min_element(ms.begin(), ms.end());
  const double flops = 2.0 * M * N * K + 2.0 * M * N * D;  // both GEMMs; the exps are not counted
  std::printf(
      "[time] median %.2f us, min %.2f us (5 warmup + %d timed, no L2 flush); %.3e FLOP "
      "(2MNK + 2MND) -> %.2f TFLOPS\n",
      med * 1e3, mn * 1e3, a.iters, flops, flops / (med * 1e-3) / 1e12);

#ifdef USE_CUBLAS
  {  // S GEMM only, same inputs, for a tensor-core-throughput reference point.
    // Row-major S[M×N] = A·Bᵀ  ⇔  column-major Sᵀ[N×M] = B·Aᵀ, where the buffers ARE col-major
    // Bᵀ[K×N] (ld K, so op=T) and Aᵀ[K×M] (ld K, op=N); C with ld N is S row-major in memory.
    cublasHandle_t h;
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) return 3;
    const cudaDataType_t dt = std::is_same_v<T, __half> ? CUDA_R_16F : CUDA_R_16BF;
    const float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t st = CUBLAS_STATUS_SUCCESS;
    const double cb_med = time_loop([&] {
      st = cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, dB, dt, K, dA, dt, K, &beta,
                        dS, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    });
    if (st != CUBLAS_STATUS_SUCCESS) return 3;
    CUDA_CHECK(cudaMemcpy(hS.data(), dS, nS * sizeof(float), cudaMemcpyDeviceToHost));
    double cb_err = 0.0;
    for (size_t i = 0; i < nS; ++i)
      cb_err = std::max(cb_err, std::fabs(hS[i] - R.S[i]));
    std::printf(
        "[cublas] S GEMM only: median %.2f us -> %.2f TFLOPS (2MNK), max abs err vs fp64 %.3e\n",
        cb_med * 1e3, 2.0 * M * N * K / (cb_med * 1e-3) / 1e12, cb_err);
    cublasDestroy(h);
  }
#endif
  for (auto& e : ev)
    CUDA_CHECK(cudaEventDestroy(e));
  for (void* p : {static_cast<void*>(dA), static_cast<void*>(dB), static_cast<void*>(dV),
                  static_cast<void*>(dS), static_cast<void*>(dO), static_cast<void*>(dSink)})
    CUDA_CHECK(cudaFree(p));

  const char* verdict = !s_pass ? "FAIL (S GEMM)" : !o_pass ? "FAIL (P·V ratio rule)" : "PASS";
  std::printf("RESULT: %s\n", verdict);
  return !s_pass ? 1 : !o_pass ? 2 : 0;
}

int main(int argc, char** argv) {
  Args a;
  int pos = 0;
  bool bad = false, help = false;
  for (int i = 1; i < argc && !bad; ++i) {
    const std::string s = argv[i];
    if (s == "--iters" && i + 1 < argc)
      a.iters = std::atoi(argv[++i]);
    else if (s == "--dump" && i + 1 < argc)
      a.dump = argv[++i];
    else if (s == "--bf16")
      a.bf16 = true;
    else if (s == "--ref-only")
      a.ref_only = true;
    else if (s == "--help" || s == "-h")
      help = true;
    else if (s.empty() || s[0] == '-')
      bad = true;
    else if (pos < 3) {
      int* const dst[3] = {&a.M, &a.N, &a.K};
      *dst[pos++] = std::atoi(argv[i]);
    } else
      bad = true;
  }
  if (help || bad || a.M <= 0 || a.N <= 0 || a.K <= 0 || a.iters <= 0 || a.M % BM || a.N % BN ||
      a.K % BK) {
    std::fprintf(help ? stdout : stderr,
                 "usage: wmma_gemm [M N K] [--iters n] [--bf16] [--dump file] [--ref-only]\n"
                 "  defaults M=2048 N=2048 K=64; M%%64==0, N%%64==0, K%%16==0; D is fixed at 64\n");
    return help ? 0 : 3;
  }
  return a.bf16 ? run<__nv_bfloat16>(a) : run<__half>(a);
}
