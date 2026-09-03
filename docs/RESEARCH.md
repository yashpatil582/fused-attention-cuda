# Phase-0 research bibliography

Compiled 2026-09-02 from the research bundle in `.research/` (nine technical references and six
GPU-platform fact-checks, each followed by an adversarial refutation pass). Every URL below was
fetched on 2026-09-02 by that workflow unless a different date is given. Nothing in this file is a
measurement of this repo's kernels; numbers quoted from papers and vendor docs are theirs. Anything
the bundle marked UNVERIFIED stays UNVERIFIED here and names the probe that settles it.

Conventions: **borrow** = a pattern re-implemented here from scratch, with attribution;
**cite** = context only, nothing reused. The license table is in section 9.

## 1. NVIDIA cuda-samples: exact paths and what is borrowed

The top-level `Samples/` directory was renamed to `cpp/` in the CUDA 13.2 update (CHANGELOG.md:
"Renamed top-level `Samples` directory to `cpp`"), so any pre-2026 link using `Samples/...` 404s.
Paths are `cpp/<category>/<sample>/` on `master` (pushed 2026-05-27; the README says the tree
supports CUDA Toolkit 13.3 and needs CMake >= 3.20).

| Sample | Path | What it demonstrates | Here |
|---|---|---|---|
| matrixMul | `cpp/0_Introduction/matrixMul/matrixMul.cu` | `MatrixMulCUDA<BLOCK_SIZE>`: `__shared__ As/Bs` tiles, `aBegin/aStep` index arithmetic, two `__syncthreads()` per k-step, scalar `Csub` accumulator | **borrow**: S2 tile-loop shape |
| reduction | `cpp/2_Concepts_and_Techniques/reduction/reduction_kernel.cu` | `warpReduceSum` with `__shfl_down_sync`; the reduce0..reduce7 ladder; the int `__reduce_add_sync` path is sm_80+ | **borrow**: S2 shuffle softmax, S5b butterflies |
| cudaTensorCoreGemm | `cpp/3_CUDA_Features/cudaTensorCoreGemm/cudaTensorCoreGemm.cu` | WMMA 16x16x16 fp16 GEMM, `#define SKEW_HALF 16` padding, `cudaFuncAttributeMaxDynamicSharedMemorySize`; README lists SM 7.0-9.0, so it runs on T4 | **borrow**: S4 fragment usage, S5a padding idea |
| globalToShmemAsyncCopy | `cpp/3_CUDA_Features/globalToShmemAsyncCopy/globalToShmemAsyncCopy.cu` | `cuda::memcpy_async` / `cuda::pipeline` / `cuda::barrier` ladder (Naive -> LargeChunk -> SingleStage -> MultiStage); README: applies "when on compute capability 8.0 or higher" | **borrow** on the `__CUDA_ARCH__ >= 800` branch only; cite on T4 |
| simpleOccupancy | `cpp/0_Introduction/simpleOccupancy/` | `cudaOccupancyMaxPotentialBlockSize`; companion API `cudaOccupancyMaxActiveBlocksPerMultiprocessor` | **borrow**: degraded-mode occupancy |
| bf16/tf32TensorCoreGemm | `cpp/3_CUDA_Features/{bf16,tf32}TensorCoreGemm/` | READMEs list SM 8.0/8.6/8.7/8.9/9.0 only | cite: why bf16 is compile-only on T4 |
| transpose | `cpp/6_Performance/transpose/` | bank-conflict / coalescing microbenchmark | cite |

cuda-samples never drops below `nvcuda::wmma`: a GitHub code search for `ldmatrix` and for
`mma.sync` in the repo returns 0 hits. If this repo ever uses raw `mma.sync`, the citation is
CUTLASS/CuTe (section 2), not cuda-samples.

- https://github.com/NVIDIA/cuda-samples and https://github.com/NVIDIA/cuda-samples/blob/master/CHANGELOG.md
- https://github.com/NVIDIA/cuda-samples/blob/master/cpp/3_CUDA_Features/cudaTensorCoreGemm/README.md
- https://github.com/NVIDIA/cuda-samples/blob/master/cpp/3_CUDA_Features/globalToShmemAsyncCopy/README.md

## 2. CUTLASS / CuTe

- **Example 41, `examples/41_fused_multi_head_attention/`** (`kernel_forward.h`, `kernel_backward.h`,
  `gemm_kernel_utils.h`, `fused_multihead_attention_fixed_seqlen.cu`). The file header says it is
  based on "Self-Attention Does Not Need O(n^2) Memory" (arXiv 2112.05682) and is "very similar to"
  FlashAttention, upstreamed from Meta xFormers. Stated design: S "is stored in shared memory between
  the 2 back to back gemms", the output accumulates "directly in registers if we can (head_size_v <= 128)",
  blocks are parallelised over batch, heads and query length. `gemm_kernel_utils.h` has a
  `DISPATCH_ARCHTAG` over `Sm80/Sm75/Sm70/Sm50` and a tensor-op `DefaultGemmType` for f16 on Sm75+,
  so the kernel headers reach T4; the shipped runner hard-codes `cutlass::arch::Sm80`.
  **borrow**: the online-softmax rescale structure and the "S in smem between the two GEMMs" layout,
  which is exactly the WMMA-safe S4 design. Never the code.
- **Example 88, `examples/88_hopper_fmha/`** (README: "implement fused multi-head attention (FMHA)
  using CUTLASS for the NVIDIA Hopper architecture", refers to the FA3 paper; "All kernels use the
  Tensor Memory Accelerator for loads"; forward head dims 32/64/128/256 fp16/bf16, 128/256 fp8;
  warp-specialised cooperative schedules for 128/256). **cite only** (TMA + WGMMA are Hopper).
  Its README's GQA-as-strides recipe is worth quoting: for GQA the KV head layout is
  `(numHeads/numGroups,numGroups:headStride,0)`, for MQA `(numHeads:0)`.
- `examples/77_blackwell_fmha` and `examples/93_blackwell_low_latency_gqa` are CMake-gated on
  `CUTLASS_NVCC_ARCHS MATCHES 100a OR 103a`. **cite only**.
- **`include/cute/swizzle.hpp`**: `template <int BBits, int MBase, int SShift> struct Swizzle` with
  `static_assert(abs(num_shft) >= num_bits)`; the operation is one XOR,
  `offset ^ shiftr(offset & yyy_msk{}, msk_sft{})`. Pure index arithmetic, so it is portable to
  sm_75/86/89 and is the principled alternative to `SKEW_HALF` padding (costs no smem). CuTe's
  `sgemm_sm80.cu` tutorial uses `composition(Swizzle<3,3,3>{}, ...)` under the comment "Swizzles for
  LDSM and 128b k-major loads". **borrow** (optional S5a second commit: XOR of the 16-byte chunk
  index by `(row & 7)`).
- Reusable on sm_75/86/89: `Swizzle`, the `SM75_U32x4_LDSM_N` / `SM75_U16x8_LDSM_T` ldmatrix copy
  atoms (`include/cute/arch/copy_sm75.hpp`, `include/cute/atom/copy_traits_sm75.hpp`), the
  `SM75_16x8x8_F32F16F16F32_TN` MMA atom, `SM80_CP_ASYNC_*` and `SM80_16x8x16_*` on Ampere+.
  Hopper-only: `examples/cute/tutorial/hopper/{wgmma_sm90.cu,wgmma_tma_sm90.cu}`, anything
  `cp.async.bulk` / `wgmma`.
- `media/docs/cpp/functionality.md`: SM75 tables list WMMA shapes 16x16x16 / 8x32x16 / 32x8x16 and
  f16/s8/s4/b1; native TensorOp `m16n8k16` f16 appears only in SM80+ tables; SM80 adds bf16, tf32,
  f64. CuTe docs live under `media/docs/cpp/cute/` (`01_layout.md`, `02_layout_algebra.md`,
  `0t_mma_atom.md`, `0x_gemm_tutorial.md`, `0z_tma_tensors.md`); there is no standalone swizzle doc.
- https://github.com/NVIDIA/cutlass/blob/main/examples/41_fused_multi_head_attention/fused_multihead_attention_fixed_seqlen.cu
- https://github.com/NVIDIA/cutlass/blob/main/examples/41_fused_multi_head_attention/kernel_forward.h
- https://github.com/NVIDIA/cutlass/blob/main/examples/88_hopper_fmha/README.md
- https://github.com/NVIDIA/cutlass/blob/main/include/cute/swizzle.hpp
- https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/sgemm_sm80.cu
- https://raw.githubusercontent.com/NVIDIA/cutlass/main/media/docs/cpp/functionality.md
- https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cutlass/arch/mma_sm80.h (m16n8k16 under `__CUDA_ARCH__ >= 800`),
  `.../mma_sm89.h` (FP8 `m16n8k32` under `>= 890`), `.../memory_sm75.h` (ldmatrix under `>= 750`)

## 3. FlashAttention 1 / 2 / 3, in my own words

**The identity (FA1 Sec 3.1; originally Milakov & Gimelshein, arXiv 1805.02867).** For a row split
into two pieces `x = [x1, x2]`: `m = max(m1, m2)`, `l = e^(m1-m) l1 + e^(m2-m) l2`, and the unnormalised
numerators rescale by the same factors. Everything else is bookkeeping around this.

**The recurrence as shipped (FA2 form, deferred normalisation).** Per K/V tile j:
`m_new = max(m, rowmax(S_j))`; `P = exp(S_j - m_new)`; `alpha = exp(m - m_new)`;
`l = alpha*l + rowsum(P)`; `acc = alpha*acc + P V_j`; once at the end `O = acc / l`,
`LSE = m + log(l)`. `alpha` is a factor in (0, 1], multiplied in, never an inverse; the shipped code is
`scores_scale = exp2f((scores_max_prev - scores_max_cur) * softmax_scale_log2)` in
`csrc/flash_attn/src/softmax.h::softmax_rescale_o` (the FA2 paper's `diag(...)^-1` typesetting is a
notation artefact; implement from the code). FA1's Algorithm 1 divides by `l` on *every* tile
(`diag(l_new)^-1 (diag(l) e^(m - m_new) O + ...)`), which is the non-matmul FLOP waste FA2 removes.

**The exp2 trick.** The kernels call `exp2f`, never `expf`, with `softmax_scale * log2(e)` folded
into one constant; the running max is kept in raw `QK^T` units and the true `LSE` is reassembled as
`row_max * softmax_scale + log(sum)` (`normalize_softmax_lse`). Reason: `ex2.approx.f32` is one SFU
instruction.

**Guards.** A fully masked row has `m = -inf`, and `-inf - (-inf) = NaN`, so the shipped code clamps
`scores_max_cur = (row_max == -INFINITY ? 0.f : row_max)`; the final divide uses
`inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f/sum`; in split-KV mode the empty-split LSE sentinel
is `-INFINITY` so the cross-split max ignores it.

**IO complexity (FA1 Theorem 2).** Standard attention makes `Theta(N d + N^2)` HBM accesses because
S and P are `N x N` objects that round-trip HBM. With SRAM of `M` elements and tiles
`B_c = ceil(M/(4d))`, `B_r = min(ceil(M/(4d)), d)`, FlashAttention makes `Theta(N^2 d^2 / M)`
accesses: K and V stream once (`Theta(Nd)`), Q/O are re-touched once per K/V tile,
`T_c = ceil(N/B_c) ~ 4Nd/M` times, giving `Theta(N^2 d^2 / M)`. Versus the `N^2` term the win is
`~M/d^2`, which is a win because on real hardware `M` (tens to hundreds of KB) is comparable to or
larger than `d^2` (`d = 128 -> 64 KB` of fp32). More arithmetic, less traffic, and attention is
memory-bound, so that is a good trade. FA1 also proves a matching lower bound for `M in [d, Nd]`.

**FA1 -> FA2 (arXiv 2307.08691).** (1) *Loop order*: FA1's outer loop is over K/V tiles and inner
over Q tiles, so `O_i, m_i, l_i` bounce through HBM every outer step; FA2 makes one thread block own
one Q tile for its whole life and loops over K/V inside, so the accumulator and statistics live in
registers and are written once. (2) *Fewer non-matmul FLOPs*: drop the per-tile `diag(l)^-1`, keep an
unnormalised accumulator, divide once; store one statistic `L = m + log l`. Motivation stated in the
paper: A100 has 312 TFLOPs/s of fp16 matmul but 19.5 TFLOPs/s of non-matmul fp32, so each
non-matmul FLOP is ~16x more expensive. (3) *Warp partitioning, split-K -> split-Q*: FA1 splits K/V
across the four warps, so each warp holds a partial sum of the same output rows and they must meet
in shared memory and synchronise; FA2 splits Q across warps, so each warp owns whole rows and the
row reductions are warp-local. Visible in `softmax.h` as `quad_allreduce_(row_sum, row_sum, sum_op)`,
a 4-lane shuffle butterfly, because the `m16n8k16` accumulator puts one row across 4 lanes.
(4) *Sequence-length parallelism*: grid `(ceil(N/B_r), batch, heads)` instead of `(batch, heads)`, so
small-batch long-context runs still fill the SMs. Paper: "50-73% of the theoretical maximum FLOPs/s on
A100", "up to 225 TFLOPs/s per A100 GPU (72% model FLOPs utilization)". Causal block skipping:
"around 1.7-1.8x speedup compared to attention without the causal mask" (p. 6); forward FLOPs are
`4 * seqlen^2 * head_dim * nheads`, halved for causal (Sec 4.1).

**FA3 (arXiv 2407.08608), why it is Hopper-only.** (1) *Warp specialisation*: producer warpgroups only
issue TMA loads into an s-stage circular smem buffer and call `setmaxnreg` to give their registers
away; consumer warpgroups take those registers and run WGMMA + softmax; handshake via mbarriers.
(2) *Pingpong*: two consumer warpgroups alternate so one's softmax runs under the other's WGMMA, named
barriers `FwdNamedBarriers::WarpSchedulerWG1/2/3` in `hopper/named_barrier.hpp`; motivation is H100's
~989 TFLOPS fp16 matmul vs ~3.9 TFLOPS special-function throughput (256x). (3) *Intra-warpgroup
2-stage pipelining* (Algorithm 2): issue `S_next` before softmaxing `S_cur`. (4) *FP8*: block
quantisation (one scale per `B_r x d` / `B_c x d` block) and incoherent processing (random-sign
Hadamard on Q and K); Table 3 reports FP8 RMSE 9.1e-3 vs 2.4e-2 baseline (2.6x), with block
quantisation doing nearly all the work; plus WGMMA layout conformance (in-kernel V transpose, byte
permutes). Headline: 740 TFLOPs/s fp16 (75% of H100 peak), ~1.2 PFLOPs/s FP8, 1.5-2.0x over FA2;
Table 2 ablation 661 / 582 (no pipelining) / 570 (no warp specialisation) TFLOPs/s. Mechanically,
it needs `wgmma.mma_async` (async, sources operands from smem; compiled with
`-gencode arch=compute_sm90a,code=sm_90a` per `hopper/setup.py`), TMA (`cp.async.bulk.tensor`,
sm_90+), `setmaxnreg`, and clusters/DSMEM. Pre-Hopper `mma.sync` is synchronous with register
operands, so there is nothing to overlap softmax against; that is the interview answer. Nuance: the
`hopper/` tree now also ships `mainloop_fwd_sm80.hpp` and 140 `*_sm80.cu` instantiations, an Ampere
fallback path, not the WS/pingpong algorithm.

**Flash-decoding (PyTorch blog, last updated 2024-11-16).** With `seq_q = 1` the FA2 grid is
`(1, batch, heads)`; the blog says at batch size 1 FlashAttention uses "less than 1% of the GPU".
Three steps: split K/V into chunks (views, free); run attention per chunk in parallel writing a
partial O and one LSE per row per split; reduce across splits with LSE weights:
`O = sum_s e^(L_s - L*) O_s / sum_s e^(L_s - L*)`, `L* = max_s L_s` (the same identity one level up).
Two kernels. Blog micro-benchmark (A100, f16, 16 q-heads / 2 kv-heads, d = 128): B=1, seqlen 65536:
PyTorch eager 1335.6 us, FA v2.0.9 2300.6 us, Flash-Decoding 64.4 us. In flash-attn >= 2.2.

**Where the pieces live in Dao-AILab/flash-attention (read 2026-09-02).**
`csrc/flash_attn/src/softmax.h`: `scale_apply_exp2` (~67), `max_scale_exp2_sum` (~96),
`struct Softmax<kNRows>` (~129) with `softmax_rescale_o<Is_first, Check_inf>` (~137) and
`normalize_softmax_lse<Is_dropout, Split>` (~170). `kernel_traits.h`: `Flash_kernel_traits` picks
`MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>` (or bf16) under `__CUDA_ARCH__ >= 800` else
`SM75_16x8x8_F32F16F16F32_TN`; `Has_cp_async = true` only >= 800; `Element = half_t` forced
pre-Ampere; ldmatrix copy atoms under `>= 750`; `Flash_fwd_kernel_traits` holds
`kBlockM/kBlockN/kHeadDim`, `kBlockKSmem = (kHeadDim % 64 == 0 ? 64 : 32)`,
`kSwizzle = (kBlockKSmem == 32 ? 2 : 3)`, `Share_Q_K_smem`, `Is_Q_in_regs`. `mask.h`: lane-derived
indices `(lane_id % 4) * 2`, `row_idx_base + i * 8`. `utils.h`: `convert_layout_acc_rowcol`,
`convert_layout_acc_Aregs` (identity for m16n8k8). `block_info.h` (~50 lines): padded vs varlen
addressing. `flash.h`: `softmax_lse_ptr` (~62), `num_splits` (~140), `dq_accum_ptr` (~160),
`deterministic` (~187). `flash_bwd_kernel.h`: `atomicAdd` into fp32 `dq_accum` (~678), per-block
buffers when deterministic (~124). `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp`
(`warp_scheduler_barrier_sync` ~915), `hopper/named_barrier.hpp` (lines 52-54),
`hopper/flash_fwd_combine_kernel.h` (split-KV combine). Support matrix (README): FA2 needs
"Ampere, Ada, or Hopper GPUs", fp16/bf16, head dim <= 256, CUDA >= 12.0, PyTorch >= 2.2; "For
Turing GPUs (T4, RTX 2080), see the separate flash-attention-turing repo"; latest FA2 tag v2.8.3.
Community Turing port: https://github.com/ssiu/flash-attention-turing (fwd+bwd, hdim 64/128,
causal, GQA, varlen; reports up to 2.19x non-causal / 1.95x causal over PyTorch SDPA forward on T4;
splits the backward into separate dQ and dK/dV kernels).

- https://arxiv.org/abs/2205.14135 (FA1; full text https://ar5iv.labs.arxiv.org/html/2205.14135)
- https://arxiv.org/abs/2307.08691 (FA2; https://ar5iv.labs.arxiv.org/html/2307.08691)
- https://arxiv.org/abs/2407.08608 (FA3; https://ar5iv.labs.arxiv.org/html/2407.08608)
- https://arxiv.org/abs/1805.02867 (online normalizer calculation for softmax)
- http://arxiv.org/abs/2112.05682 (Self-Attention Does Not Need O(n^2) Memory; cited by CUTLASS example 41's header)
- https://pytorch.org/blog/flash-decoding/
- https://github.com/Dao-AILab/flash-attention and https://raw.githubusercontent.com/Dao-AILab/flash-attention/main/README.md
- https://raw.githubusercontent.com/Dao-AILab/flash-attention/main/csrc/flash_attn/src/kernel_traits.h ,
  `.../softmax.h` , `.../mask.h` , `.../utils.h` , `.../flash_fwd_kernel.h`
- https://raw.githubusercontent.com/Dao-AILab/flash-attention/main/tests/test_flash_attn.py (tolerance pattern)
- https://raw.githubusercontent.com/Dao-AILab/flash-attention/main/benchmarks/benchmark_flash_attention.py (FLOP formula)
- https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html (FA2 recurrence + 3-stage causal loop)

## 4. CUDA Programming Guide, Best Practices Guide, PTX, tuning guides

| Topic | Fact (verbatim where quoted) | Source |
|---|---|---|
| Coalescing | Best Practices 10.2.1, High Priority: on CC 6.0+ a warp's accesses coalesce into "the number of 32-byte transactions necessary to service all of the threads of the warp"; strided and misaligned patterns waste bandwidth | https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html |
| Bank conflicts | 10.2.3.1: "each bank has a bandwidth of 32 bits every clock cycle... The warp size is 32 threads and the number of banks is also 32"; "successive 32-bit words are assigned to successive banks"; conflicting accesses "are serialized" (same-address broadcast excepted) | same; https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/ |
| Occupancy | Best Practices 11.1; ncu defines it as "the ratio of the number of active warps per multiprocessor to the maximum number of possible active warps" | same; Nsight Profiling Guide |
| Timing | 9.1.1: CPU timers need `cudaDeviceSynchronize()`; 9.1.2: `cudaEventRecord`/`cudaEventElapsedTime`, ~0.5 us resolution | same |
| APOD | Sec 2.2 Assess, Parallelize, Optimize, Deploy | same |
| WMMA opacity | Programming Guide, Warp Matrix Functions: "the way matrix elements map to fragment internal storage is unspecified"; fragment exposes `x[]` with `num_elements`; NVIDIA blog: "The mapping between individual matrix parameters to their fragments is opaque" but "element-wise operations don't have to know this mapping" | https://docs.nvidia.com/cuda/archive/10.2/cuda-c-programming-guide/index.html#wmma-description ; https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/ |
| mma.sync lane map | PTX ISA 9.7.15.5: `groupID = %laneid >> 2`, `threadID_in_group = %laneid % 4`; m16n8k16 f32 C/D: `row = groupID` and `groupID + 8`, `col = threadID_in_group * 2 + i` | https://docs.nvidia.com/cuda/parallel-thread-execution/index.html |
| cp.async | PTX 9.7.9.26.3.1: `cp.async.{ca,cg}.shared.global`, cp-size 4/8/16, ".cg is only supported for cp-size 16", "Requires sm_80 or higher"; `commit_group` / `wait_group N`. Ampere Tuning Guide: async copy "avoid[s] using extra registers for memory copies and can also bypass the L1 cache", exposed via the pipeline API. Turing guide does not mention it | same; https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html |
| smem per arch | Turing cc 7.5: 96 KB unified L1/shared per SM, carveouts 64/32 or 32/64, "Turing allows a single thread block to address the full 64 KB of shared memory", static limited to 48 KB. Ampere: A100 192 KB combined, "up to 163 KB" per block; cc 8.6 128 KB combined, "up to 99 KB"; "CUDA reserves 1 KB of shared memory per thread block". Ada cc 8.9: 128 KB combined, 99 KB per block. Hopper: 256 KB combined, 227 KB per block. >48 KB dynamic needs `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, bytes)` | https://docs.nvidia.com/cuda/turing-tuning-guide/index.html ; https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html ; https://docs.nvidia.com/cuda/ada-tuning-guide/index.html ; https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html; Volta guide for the 96 KB/block figure that belongs to sm_70 |
| Registers / warps | 64K 32-bit registers per SM and 255 per thread on 7.5/8.0/8.6/8.9/9.0; concurrent warps per SM: 32 on Turing "(versus 64 on Volta)", 64 on cc 8.0, 48 on 8.6/8.9, 64 on 9.0 | tuning guides above |
| Tensor-core shapes | Turing: fp16 in, fp16/fp32 accumulate, WMMA 16x16x16 / 32x8x16 / 8x32x16; no bf16, tf32 or async copy in the guide. Ampere adds bf16, tf32, m16n8k16. Ada adds FP8 (`mma.sync.m16n8k32 e4m3/e5m2`, `__CUDA_ARCH__ >= 890`, CUDA >= 12.4 for f32 accumulate). `__nv_bfloat16` WMMA "for devices with compute capability compute_80 and higher" | tuning guides; https://developer.nvidia.com/blog/cuda-11-features-revealed/ ; CUTLASS `mma_sm89.h` |
| Hopper-only | TMA "1D to 5D tensors", clusters (portable size 8, 16 non-portable on H100), distributed shared memory; `wgmma.mma_async` needs `sm_90a` | https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html |
| nvcc | `-Xptxas -v` / `--resource-usage` print registers and spill stores/loads; `-lineinfo` is the profiling-safe alternative to `-G` ("not intended for profiling"); `__CUDA_ARCH__` is "undefined in host code"; `-arch=native` emits "no PTX program"; last `-gencode` should be `code=compute_XX` for forward compat; CUDA 13.0 removed Maxwell/Pascal/Volta | https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html ; https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html |
| Occupancy API | `cudaOccupancyMaxActiveBlocksPerMultiprocessor(int*, const void*, int blockSize, size_t dynSmem)`; no profiling permission needed | https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__OCCUPANCY.html |
| SASS tools | `cuobjdump -sass`, `cuobjdump -res-usage`, `nvdisasm -cfg`, `nvdisasm -plr` (register liveness) | https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html |
| Turing trap | 32 warps/SM is half of A100; a 128-thread block at 128 regs/thread and 32 KB smem gets 2 blocks = 8 warps = 25% on T4 (derived arithmetic from the constants above) | derived |

## 5. Nsight Compute

- **Sections** (Profiling Guide identifiers): `SpeedOfLight`, `SpeedOfLight_RooflineChart`,
  `SpeedOfLight_HierarchicalDoubleRooflineChart` (both roofline names confirmed by the 2020-11-18 NVIDIA
  blog; `SpeedOfLight_HierarchicalTensorRooflineChart` / `...HalfRooflineChart` per the reality-check
  reference), `MemoryWorkloadAnalysis`, `ComputeWorkloadAnalysis`, `Occupancy`, `LaunchStats`,
  `WarpStateStats`, `SchedulerStats`, `SourceCounters`, `InstructionStats`. Sets:
  basic (default) / default / detailed / full / roofline / source. `MemoryWorkloadAnalysis_Tables`
  as an identifier: **verified by PROBE 2 (2026-09-02)**, `ncu --list-sections` on the Colab T4
  listed `ComputeWorkloadAnalysis LaunchStats MemoryWorkloadAnalysis MemoryWorkloadAnalysis_Chart
  MemoryWorkloadAnalysis_Tables Occupancy SchedulerStats SpeedOfLight
  SpeedOfLight_HierarchicalDoubleRooflineChart SpeedOfLight_HierarchicalHalfRooflineChart
  SpeedOfLight_HierarchicalSingleRooflineChart SpeedOfLight_HierarchicalTensorRooflineChart
  SpeedOfLight_RooflineChart WarpStateStats`. Set membership remains UNVERIFIED (not needed:
  profile.sh names sections and metrics, never a set).
- **Metric naming**: `unit__(subunit)_(pipestage)_quantity_(qualifiers)` with rollups `.sum .avg .min .max`
  and sub-metrics `.peak_sustained`, `.peak_sustained_active/elapsed`, `.per_second`,
  `.per_cycle_active/elapsed`, `.pct_of_peak_sustained_active/elapsed`. Units: `gpu`, `sm`, `smsp`
  (4 per SM), `dram`, `l1tex`, `lts` (one L2 slice), `ltc`, `fbpa`.
- **Verified IDs** (Nsight Compute 2022.4 CLI doc, Table 10 nvprof mapping): occupancy
  `sm__warps_active.avg.pct_of_peak_sustained_active`; DRAM % `dram__throughput.avg.pct_of_peak_sustained_elapsed`;
  bytes `dram__bytes_read.sum`, `dram__bytes_write.sum`, `dram__bytes.sum`; L2 hit `lts__t_sector_hit_rate.pct`;
  L1 hit `l1tex__t_sector_hit_rate.pct`; bank conflicts `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_{ld,st}.sum`;
  tensor pipe `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active`;
  instructions `smsp__inst_executed.sum`; coalescing `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct`;
  FLOPs `smsp__sass_thread_inst_executed_op_{fadd,fmul,ffma,hadd,hmul,hfma}_pred_on.sum` with
  `flop = fadd + fmul + 2*ffma`; stalls `smsp__warp_issue_stalled_<reason>_per_warp_active.pct`;
  `launch__registers_per_thread` needs no replay pass. PROBE 2 (2026-09-02) `--query-metrics`
  listed `sm__throughput`, `gpu__time_duration`, `sm__pipe_tensor_op_hmma_cycles_active`,
  `smsp__warp_issue_stalled_long_scoreboard_per_warp_active` and
  `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld` on the Colab T4, so
  `sm__throughput.avg.pct_of_peak_sustained_elapsed` and `gpu__time_duration.sum` are enabled in
  `scripts/ncu_metrics.txt`. **Still UNVERIFIED**: `gpu__compute_memory_throughput...` and
  `lts__throughput...` (not in the probe's grep output; profile.sh prunes them at run time).
- **Stall-reason map** (old nvprof name -> ncu): memory dependency = `long_scoreboard`; exec
  dependency = `short_scoreboard` + `wait`; sync = `barrier` + `membar`; pipe busy =
  `math_pipe_throttle` + `mio_throttle`; memory throttle = `drain` + `lg_throttle`; inst fetch =
  `no_instruction`; not selected = `not_selected`. For attention: `long_scoreboard` (K/V loads) and
  `mio_throttle` + `barrier` (smem pressure and `__syncthreads` around softmax).
- **Roofline** (NERSC recipe): `time = sm__cycles_elapsed.avg / sm__cycles_elapsed.avg.per_second`;
  AI = FLOPs / `dram__bytes.sum` (L2: `lts__t_bytes.sum`, L1: `l1tex__t_bytes.sum`);
  FMA roof = `2 * <op>_ffma_pred_on.sum.peak_sustained * sm__cycles_elapsed.avg.per_second`;
  DRAM roof = `dram__bytes.sum.peak_sustained * dram__cycles_elapsed.avg.per_second`. NERSC's
  `512 * sm__inst_executed_pipe_tensor.sum` tensor FLOP constant is V100-only; count attention FLOPs
  analytically and use the tensor roof from the datasheet (`docs/ENV.md`, "spec, not measured").
- **CSV / import**: `--section` is ignored on `--import` with `--page raw` or `--page source`, so
  `ncu -i x.ncu-rep --csv --page raw` dumps every collected metric as one wide row per launch. That is
  the pandas path. `--page details` gives the section tables.
- **Replay**: kernel replay (default) saves all accessible GPU memory on pass 1 and restores the
  written subset before each later pass ("If ... only a single replay pass is necessary ... no
  save-and-restore is performed"). Application replay requires the app to be "deterministic with
  respect to its kernel activities and their assignment to GPUs, contexts, streams" and matches
  launches by name and grid size in execution order; PyTorch breaks that (autotuning, dynamic shapes),
  so profile the torch-free `fa_bench` under kernel replay. `--clock-control` defaults to `base`
  (GPC and memory clocks locked to base: "This is the default" in the CLI docs; values
  `base|boost|force-boost|none|reset`); profile.sh passes `base` explicitly anyway;
  `--cache-control` defaults to `all` (flush between passes).
- **ERR_NVGPUCTRPERM**: since driver 418.43+ (Linux) counters are admin-only by default;
  `NVreg_RestrictProfilingToAdminUsers=1` restricts to `CAP_SYS_ADMIN` / `CAP_PERFMON`. Fixes on a
  host you control: `options nvidia NVreg_RestrictProfilingToAdminUsers=0` in `/etc/modprobe.d/*.conf`,
  `update-initramfs -u -k all`, reboot; or run under `sudo`; or `/dev/nvidia-caps` grants on R610+.
  Containers: "access must be enabled on the host, or the container must be started with the
  appropriate permissions by passing --cap-add=SYS_ADMIN as an admin user." Root inside an
  unprivileged container is not sufficient.
- **Versions / paths**: CUDA 12.8 -> ncu 2025.2, 13.0 -> 2025.3, 13.2 -> 2026.1, 13.3 -> 2026.2;
  `.run` installs under `/usr/local/cuda-<ver>/nsight-compute-<ver>`, `.deb` under
  `/opt/nvidia/nsight-compute/<ver>`; `nv-nsight-cu-cli` deprecated for `ncu` since 2022.3.
- **nsys fallback**: `nsys profile -t cuda -o rep -f true --stats=true ./app`; `nsys stats rep.nsys-rep`.
  Whether CUPTI activity tracing is exempt from the counter gate is moot on Colab: **nsys is not
  installed there (PROBE 2, 2026-09-02)**, so degraded mode on Colab is CUDA events + `-Xptxas -v`;
  `--gpu-metrics-device` sampling is not exempt anywhere.
- https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html ; https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html
- https://archive.docs.nvidia.com/nsight-compute/2022.4/NsightComputeCli/index.html ; https://archive.docs.nvidia.com/nsight-compute/2022.4/ProfilingGuide/index.html
- https://docs.nvidia.com/nsight-compute/ReleaseNotes/index.html ; https://docs.nvidia.com/nsight-compute/NsightCompute/index.html
- https://developer.nvidia.com/ERR_NVGPUCTRPERM (long form: https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters)
- https://developer.nvidia.com/blog/accelerating-hpc-applications-with-nsight-compute-roofline-analysis/
- https://gitlab.com/NERSC/roofline-on-nvidia-gpus/-/raw/master/README.md ; https://docs.nersc.gov/tools/performance/roofline/
- https://docs.nvidia.com/nsight-systems/UserGuide/index.html ; https://docs.pytorch.org/docs/2.14/profiler.html
- https://leimao.github.io/blog/Docker-Nsight-Compute/ (container flags) ; https://tspeterkim.github.io/posts/nsight-setup-on-ec2 (EC2 modprobe recipe)

## 6. Production context

**TensorRT-LLM** (docs "Multi-Head, Multi-Query and Group-Query Attention"). One operator,
`tensorrt_llm.functional.gpt_attention`, takes a packed QKV tensor of shape `[num_tokens, 3 * hidden_dim]`
and serves two phases. The *context* (prefill) phase runs a FlashAttention-style fused kernel selected
by `context_fmha_type`, with `use_fp8_context_fmha = enable` on Ada/Hopper. The *generation* (decode)
phase runs "a single kernel called the masked multi-head attention" that fuses the QKV bias, RoPE and
quantise/dequantise, with `multi_block_mode` (default on since v0.13) splitting one head's K/V across
thread blocks when occupancy is low, the same idea as flash-decoding. KV cache is contiguous or paged
(`tensorrt_llm.runtime.KVCacheManager`); `kv_cache_quant_mode = QuantMode.INT8_KV_CACHE / FP8_KV_CACHE`
with a per-tensor `kv_cache_scaling_factor` of shape `[1]`. XQA is a separate generation-phase kernel
for MQA/GQA (fp16/bf16 compute, fp16/bf16/fp8/int8 KV), toggled by `--disable_xqa` or forced with
`TRTLLM_FORCE_XQA=1`. Relevance: this repo's S3 kernel is the context-phase shape; split-KV decode is
the week-after item. https://nvidia.github.io/TensorRT-LLM/advanced/gpt-attention.html

**cuDNN 9 fused SDPA.** The runtime-fusion engine exposes patterns named "Fused Flash Attention fprop",
"... fprop (FP8)", "... bprop", "... bprop (FP8)"; head_dim must be "a multiple of 8 with a maximum value
of 128 for Ampere GPUs and 256 for Hopper GPUs"; FP8 is Hopper-only; NVIDIA recommends the frontend
SDPA nodes over the raw graph API. cudnn-frontend `docs/operations/Attention.md`: "cudnn SDPA operation
requires SM80 (Ampere) or newer architectures and cuda toolkit 12.x or newer"; C++ entry
`sdpa(q, k, v, SDPA_attributes)` with setters `set_attn_scale`, `set_causal_mask`,
`set_causal_mask_bottom_right`, `set_generate_stats`, `set_paged_attention_k_table`, `set_score_mod`,
`set_diagonal_alignment`, ...; samples under `samples/cpp/sdpa/` (`fp16_fwd.cpp`, `fp8_fwd.cpp`,
`fp16_fwd_with_paged_caches.cpp`). So on a T4 there is no cuDNN fused SDPA at all, which is the
strongest one-sentence reason this repo exists.
https://docs.nvidia.com/deeplearning/cudnn/latest/developer/graph-api.html ;
https://github.com/NVIDIA/cudnn-frontend/blob/develop/docs/operations/Attention.md

**PyTorch SDPA dispatch** (`aten/src/ATen/native/transformers/cuda/sdp_utils.cpp` at v2.14.0):
FLASH requires `[sm80, sm121]` ("Flash attention only supports gpu architectures in the range
[sm80, sm121]"), head dim <= 256, fp16/bf16; EFFICIENT requires `[sm50, sm121]`, fp16/fp32 on
pre-sm80 (bf16 only on sm80+); CUDNN requires sm80+. Default priority `[flash, efficient, math]`
with cudnn promoted when `check_prefer_cudnn_attention()` is true. The math backend "preserves
torch.float" for half inputs (it upcasts) and is the only backend that accepts float64.
https://raw.githubusercontent.com/pytorch/pytorch/v2.14.0/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp ;
https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html ;
https://docs.pytorch.org/docs/2.14/generated/torch.nn.attention.sdpa_kernel.html ;
https://docs.pytorch.org/docs/2.14/library.html (`register_fake`, `opcheck`) ;
https://docs.pytorch.org/docs/2.14/cpp_extension.html ; https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html

## 7. Blog posts and comparable repos

| Post | Author / date | Why it matters here |
|---|---|---|
| https://siboehm.com/articles/22/CUDA-MMM | Simon Boehm, 2022 | The narrative template: 10 numbered kernels, GFLOPs and % of cuBLAS each (naive 1.3% -> warptiling 93.7%), and honesty about kernels 7-8 that removed bank conflicts but were "overall still slower". Calibration: "two weekends" for the first 6 kernels. |
| https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/ | Appleyard & Yokim, 2017-10-17 | `nvcuda::wmma` API, 16x16x16, fragment opacity quote |
| https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/ | Lin & Grover, 2018-01-15 | `*_sync` shuffles, `__activemask` warning, Volta independent thread scheduling |
| https://developer.nvidia.com/blog/controlling-data-movement-to-boost-performance-on-ampere-architecture/ | 2020-09-22 | `cuda::memcpy_async` / `cuda::pipeline` / `cuda::barrier`; qualitative only, no speedups |
| https://developer.nvidia.com/blog/cuda-11-features-revealed/ | 2020 | bf16 WMMA is compute_80+ |
| https://developer.nvidia.com/blog/analysis-driven-optimization-preparing-for-analysis-with-nvidia-nsight-compute-part-1/ | Bob Crovella | SOL definition: "the achieved percentage of utilization with respect to the theoretical maximum" per unit; the analysis loop this repo copies |
| https://developer.nvidia.com/blog/accelerating-hpc-applications-with-nsight-compute-roofline-analysis/ | 2020-11-18 | roofline section names, `--set detailed/full` |
| https://developer.nvidia.com/blog/kernel-fusion-in-nvidia-cuda-optimizing-memory-traffic-and-launch-overhead/ | 2026-07-10 | attention *is* a fusion story |
| https://developer.nvidia.com/blog/advanced-nvidia-cuda-kernel-optimization-techniques-handwritten-ptx/ | Bentz & Scudiero, 2025-07-24 | tone calibration: NVIDIA reports "7%-14%" and hedges |
| https://princeton-nlp.github.io/flash-atttention-2/ | Tri Dao, 2023-07-17 | FA2 announcement (note the triple-t URL; the tridao.me copy 404s) |
| https://tridao.me/blog/2024/flash3/ ; https://pytorch.org/blog/flashattention-3/ | 2024-07-11 | FA3 progression 350 -> ~570 -> ~620 -> ~640-660 TFLOPS; 989 vs 3.9 TFLOPS |
| https://alexarmbr.github.io/2024/08/10/How-To-Write-A-Fast-Matrix-Multiplication-From-Scratch-With-Tensor-Cores.html | 2024-08-10 (from the URL path; the bundle carries no other date for it) | T4 HGEMM, 6 kernels 8% -> 96% of cuBLAS; abandoned WMMA for `mma.sync.m16n8k8`; XOR swizzle ~24% -> ~50%; index math 92% of instructions in one kernel. Calibrates S4/S5 |
| https://www.bitsand.cloud/posts/profiling-gpus | 2025-07-20 | `ncu --set full` on a Colab T4 (see section 8) |

Comparable repos (star counts as of 2026-09-02, from the bundle): sonnyli/flash_attention_from_scratch
(195, 16 Ampere kernels, 99.2% of FA2 on A100, SASS committed, non-causal, d=128 only);
tspeterkim/flash-attention-minimal (1,179, ~100 lines fp32, T4 profiler numbers, "No backward pass",
slower at long N); lavawolfiee/mini-flash-attention (26, best bench hygiene: warmup=10, iters=50,
input_buffers=4, flush_l2_mb=256, median); fms-zth/BlackFlash (23, SM120 TMA/mbarrier, honestly
reports 0.62-0.73x vs cuDNN); Roger-Fpeng/ampere_flash_attention_from_scratch (22, swizzling was the
single biggest jump); kilianhae/FlashAttention.C; Mog9/FlashAttention-CuPy; keith2018/TinyFA;
66RING/tiny-flash-attention. None ship `.ncu-rep` or metric CSVs; only lavawolfiee flushes L2.
https://github.com/sonnyli/flash_attention_from_scratch ; https://lubits.ch/flash/ ;
https://github.com/tspeterkim/flash-attention-minimal ; https://github.com/lavawolfiee/mini-flash-attention ;
https://github.com/fms-zth/BlackFlash ; https://github.com/Roger-Fpeng/ampere_flash_attention_from_scratch ;
https://github.com/keith2018/TinyFA ; https://github.com/66RING/tiny-flash-attention

## 8. Empirical profiler notes per platform (from the gpu-*.json fact-checks)

| Platform | ncu hardware counters | Evidence and date | Verdict for this repo |
|---|---|---|---|
| Google Colab free T4 | **Works per-instance, not guaranteed** | Committed ncu output with `dram__bytes_read.sum` etc., driver 580.82.07, nvcc 12.8, env timestamp 2026-07-15 (Mojtaba-Alehosseini/hpc-gemm-openmp-mpi-cuda `bench/colab_t4/cuda_ncu.txt`); 2026-08-23 run, Nsight Compute 2025.1.1.0, status COMPLETE (Arwin-K/CUDA-Attention-Softmax); 2023 notebook with `--set full`, 31 passes (weedge/doraemon-nb); bitsand.cloud 2025-07-20. **Counter-evidence**: Rahu378/hardware-aware-gateway (Aug 2026): "ncu fails on Colab with ERR_NVGPUCTRPERM"; Danialjfz/Opti-GEMM: "permission depends on the instance you land on", factory-reset and retry. `ncu` may be absent from PATH (StackOverflow 75972459, 2023). 2026 successes used `--set basic` / explicit `--metrics` | Track A. PROBE 2 every session; reset <= 3x; degraded mode otherwise. Named sections + explicit metrics, never `--set full` |
| Kaggle T4x2 / P100 | **Blocked** | `==ERROR== ERR_NVGPUCTRPERM` committed for both `bench/kaggle_t4` and `bench/kaggle_p100` (same Mojtaba repo); gwenland-ai Aug-2026 notes "NCU counters: unavailable on Kaggle" twice. phone verification reportedly required (UNVERIFIED); 30 GPU-h/week (ultralytics docs corroborate) | build + bench fallback only; zero counters |
| Modal | **Off by default, unverified** | "you must have a payment method on file" (modal.com/docs/guide/billing); all compute under gVisor (docs/guide/security); gVisor `nvproxy` gained a `CapProfiling` ("GPU hardware performance counter access (Nsight Compute/Systems)") in a 2026-04-27 commit but it is excluded from `DefaultDriverCaps` and from `all`, enabled only by the host-side `--nvproxy-allowed-driver-capabilities` flag; gvisor issue #14146 (2026-08-17) open. Modal documents only `torch.profiler` | week-after timing matrix only; $30/mo Starter credits |
| Lightning AI Studios | **Unverifiable, leaning no** | 47 docs pages contain zero mentions of nsight/ncu/CAP_SYS_ADMIN; personal-email verification "may take 2-3 business days"; "5 free credits ... Add a card for 25 more"; spot is paid-tier only; peer evidence of tenant-container blocks (huggingface/hub-docs PR #2714, kopernik278 PR #1, 2026-08) | excluded for the deadline |
| NVIDIA Brev | **Plausible on VM mode, untested; card required** | "You need a credit card to provision an instance" (AI Workbench FAQ); NVIDIA moderator 2026-06-22: "not aware of any free credits"; `brev shell --host` reveals a container-vs-host split, so `ssh <name>` may land in a container; SSH-cert migration merged 2026-08-20..28 (brev-cli PRs #452/#460); every `sudo` stalls ~10 s (issue #399) | excluded in-window |
| RunPod Pods | **Blocked** | pod-create API schema has no privileged/capabilities field (docs.runpod.io/api-reference/pods/POST/pods); Bare Metal product is separate | excluded |
| Vast.ai | Docker mode **blocked** (templates expose no cap-add); **VM launch mode plausible** | docs list "Process tracing (ptrace) support" and "CUDA performance profiling" as VM use cases; SSH only; keys must be added before launch; `vastai/kvm` template; RTX 4090 ~$0.27/hr Docker-mode aggregate (getdeploying, 2026-09-02) | Track B candidate #3 after AWS quota and Lambda |
| Lambda Cloud | **Works** (real VM with sudo; documented modprobe recipe applies) | 1x A10 $1.29/hr, A100 40GB $1.99/hr, excl. tax (lambda.ai/service/gpu-cloud, 2026-09-02); no signup credit | Track B candidate #2 |
| AWS EC2 g4dn/g5 | **Works with the modprobe fix + reboot** | tspeterkim's EC2 write-up is a worked example; needs G-instance quota (PROBE 3) | Track B candidate #1 if quota exists |
| Thunder Compute | **Blocked by vendor statement** | compatibility page lists "GPU profilers" as incompatible (GPU-over-TCP) | excluded |
| TensorDock | ncu plausible, reliability refuted | uptime monitor paused since ~June 2026 after a 0-50% winter; Trustpilot 1.6/5 | excluded |
| GitHub Actions | no GPU on free runners; GPU larger runners are Team/Enterprise-only and always billed | Jimver/cuda-toolkit v0.2.36 installs nvcc; compiling needs no GPU | compile-only CI (`build.yml`) |

PROBE 2's actual output, Colab free tier, 2026-09-02 ~16:15 PDT, runtime T4 (verbatim from
`.research/colab_probe_2026-09-02.txt`, committed as `bench/results/t4/probe_2026-09-02.txt`):

```
GPU: Tesla T4, driver 580.82.07, compute_cap 7.5, 15360 MiB, clocks.max.sm 1590 MHz (idle 300 MHz)
nvcc: 12.8.93 (cuda_12.8.r12.8)   user: root (uid 0)
ncu: /usr/local/cuda/bin/ncu (Nsight Compute, (c) 2018-2025)   compute-sanitizer: /usr/local/cuda/bin/compute-sanitizer
cmake: /usr/local/bin/cmake   ninja: MISSING (pip install ninja)   nsys: MISSING   git: present
torch: 2.11.0+cu128, cuda 12.8, arch list ['sm_75','sm_80','sm_86','sm_90','sm_100','sm_120']
PROBE 2 (ncu --metrics on a 10-line kernel, no sudo, no modprobe): NCU_VERDICT=PASS
  dram__bytes.sum 296.54 KB | launch__registers_per_thread 16 | sm__cycles_elapsed.avg 3589.80 | sm__warps_active.avg.pct_of_peak_sustained_active 84.51 %
  metric IDs confirmed present: sm__throughput, gpu__time_duration, sm__pipe_tensor_op_hmma_cycles_active,
    smsp__warp_issue_stalled_long_scoreboard_per_warp_active, l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld
  sections present: ComputeWorkloadAnalysis LaunchStats MemoryWorkloadAnalysis MemoryWorkloadAnalysis_Chart MemoryWorkloadAnalysis_Tables
    Occupancy SchedulerStats SpeedOfLight SpeedOfLight_HierarchicalDoubleRooflineChart SpeedOfLight_HierarchicalHalfRooflineChart
    SpeedOfLight_HierarchicalSingleRooflineChart SpeedOfLight_HierarchicalTensorRooflineChart SpeedOfLight_RooflineChart WarpStateStats
  --clock-control base: accepted
  nvidia-smi -lgc 585,585: WORKS ("GPU clocks set to (gpuClkMin 585, gpuClkMax 585)"); -rgc works
  nsys: missing (not needed while counters work)
Caveat (from research): counter access is per-instance on Colab; re-run PROBE 2 at the start of every session.
```

Consequences applied: `sm__throughput...` / `gpu__time_duration.sum` enabled in
`scripts/ncu_metrics.txt`; `gpu_run.sh` locks clocks with `-lgc` before the bench step; degraded
mode on Colab has no nsys timeline; ninja is pip-installed by the runner.

- https://research.google.com/colaboratory/faq.html (free tier "at most 12 hours"; prohibited: "remote control such as SSH shells", "connecting to remote proxies")
- https://github.com/Mojtaba-Alehosseini/hpc-gemm-openmp-mpi-cuda/blob/main/bench/colab_t4/cuda_ncu.txt ; `.../bench/kaggle_t4/cuda_ncu.txt`
- https://github.com/Arwin-K/CUDA-Attention-Softmax/blob/main/results/runs/2026-08-23_tesla-t4_ca87722/artifacts/profiler/nsight/nsight_status.json
- https://github.com/Rahu378/hardware-aware-gateway/blob/main/notebooks/colab_bootstrap.ipynb ; https://github.com/Danialjfz/Opti-GEMM/blob/main/profiling/nsight_notes.md
- https://stackoverflow.com/questions/75972459/trouble-using-nsight-compute-on-google-colab-command-not-found-error-with-ncu
- https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md ; https://github.com/ultralytics/ultralytics/blob/main/docs/en/integrations/kaggle.md
- https://modal.com/docs/guide/billing ; https://modal.com/docs/guide/security ; https://modal.com/pricing ;
  https://raw.githubusercontent.com/google/gvisor/master/pkg/sentry/devices/nvproxy/nvconf/caps.go ; https://gvisor.dev/docs/user_guide/gpu/
- https://lightning.ai/docs/platform/overview/faq/create-account.md ; https://lightning.ai/docs/platform/build/ai-studio.md ; https://github.com/huggingface/hub-docs/pull/2714
- https://docs.nvidia.com/ai-workbench/user-guide/latest/reference/faq.html ; https://forums.developer.nvidia.com/t/any-possibilities-to-get-free-credits-from-nvidia/373993 ; https://github.com/brevdev/brev-cli/issues/399
- https://docs.runpod.io/api-reference/pods/POST/pods ; https://docs.vast.ai/guides/instances/virtual-machines ; https://getdeploying.com/gpus/nvidia-rtx-4090
- https://lambda.ai/service/gpu-cloud ; https://www.thundercompute.com/docs/compatibility ; https://github.com/Jimver/cuda-toolkit

## 9. Borrowed pattern vs cited only, with licenses

| Source | License (GitHub license API / LICENSE text, 2026-09-02) | Borrowed pattern (re-implemented, attributed) | Cited only |
|---|---|---|---|
| NVIDIA/cuda-samples | BSD-3-Clause ("Copyright (c) 2022, NVIDIA CORPORATION") | matrixMul tile loop; reduction shuffle tree; WMMA fragment usage and `SKEW_HALF`-style padding; occupancy API; cp.async pipeline on sm_80+ | transpose microbench, bf16/tf32 samples |
| NVIDIA/cutlass | BSD-3-Clause (`SPDX-License-Identifier: BSD-3-Clause` in LICENSE.txt; API reports NOASSERTION) | example 41's S-in-smem structure; `Swizzle<B,M,S>` XOR remap (optional S5a) | examples 88/77/93, CuTe atoms, wgmma/TMA tutorials |
| Dao-AILab/flash-attention | BSD-3-Clause | the recurrence, `exp2` scale folding, `-inf` / `l == 0` guards, `LSE = m*scale + log(l)`, FA2 loop order, split-Q partitioning, causal skip predicate, `kv_head = q_head / (H/Hkv)`, the 2x error-ratio test rule, the FLOP formula | `mma.sync` lane maps, backward, hopper/ |
| NVIDIA/cudnn-frontend | Apache-2.0 | none | SM80+ requirement, SDPA attribute surface |
| NVIDIA/TensorRT-LLM | NOASSERTION (Apache-2.0 text with NVIDIA additions; do not vendor) | none | context/generation phases, paged KV, XQA |
| triton-lang/triton | (not vendored) | 256 MB L2-flush benchmark convention; 3-regime causal loop shape | autotune key design |
| karpathy/nanoGPT | (not vendored) | the `Block` shape and SDPA call site for `bench/gpt_block.py` | `estimate_mfu` |
| pytorch/pytorch | (not vendored) | `TORCH_LIBRARY` + `register_fake` + `opcheck` pattern from the official tutorial | `sdp_utils.cpp` gating |

Borrowing a structural idea and citing a URL creates no attribution obligation; pasting a kernel body
does (all four NVIDIA/Dao repos are attribution-required). No kernel bodies are copied.
