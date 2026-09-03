# fused-attention-cuda

[![build](https://github.com/yashpatil582/fused-attention-cuda/actions/workflows/build.yml/badge.svg)](https://github.com/yashpatil582/fused-attention-cuda/actions/workflows/build.yml)
[![lint](https://github.com/yashpatil582/fused-attention-cuda/actions/workflows/lint.yml/badge.svg)](https://github.com/yashpatil582/fused-attention-cuda/actions/workflows/lint.yml)
[![license](https://img.shields.io/github/license/yashpatil582/fused-attention-cuda)](LICENSE)

A FlashAttention-style fused scaled-dot-product attention (SDPA) forward kernel written from
scratch in CUDA C++ and optimized in stages: three naive kernels, shared-memory tiling, one fused
kernel with online softmax, fp16 tensor cores (WMMA), then Nsight-guided tuning. Every stage is
benchmarked against PyTorch SDPA under a fixed protocol, and every optimization is justified by a
committed Nsight Compute profile (`.ncu-rep` + raw CSV) whose named metric moved. The kernel is
exposed as a PyTorch custom op (`torch.ops.fused_attn.forward`) registered with `TORCH_LIBRARY`
and a `register_fake` kernel; `torch.library.opcheck` and a `torch.compile(fullgraph=True)`
no-graph-break test are in `tests/test_op_gpu.py` (not yet run on a GPU).

**Status (2026-09-02): S0 scaffold, results pending.** The frozen API, build, benchmark harness,
CPU model of the algorithm, protocols and docs exist; no optimized kernel has run on a GPU yet. The
tables below are generated from committed CSVs and are empty by design. There are no performance
numbers anywhere in this repository until a GPU run commits them.

> **Scope and honesty**
> - **Forward pass only.** No backward, no dropout, no KV cache, no paged attention in-window.
> - **Primary GPU: NVIDIA T4 (Turing, sm_75).** Every number is measured there unless a row says
>   otherwise. Free Colab T4 is the only zero-cost platform with independent 2026 evidence that
>   Nsight Compute hardware counters work (see `docs/RESEARCH.md` §8).
> - **Speed baseline: `F.scaled_dot_product_attention` pinned to `SDPBackend.EFFICIENT_ATTENTION`.**
>   On sm_75 the FLASH backend refuses to run — PyTorch's `sdp_utils.cpp` says
>   *"Flash attention only supports gpu architectures in the range [sm80, sm121]"* — and the cuDNN
>   backend has the same sm80 floor, so the memory-efficient kernel is the fastest vendor attention
>   that exists on this GPU and the only honest denominator. `SDPBackend.MATH` in fp32/fp64 is the
>   correctness oracle, never a speed baseline (it upcasts half inputs to float).
> - **fp16 tensor cores only on T4.** bf16 WMMA and `cp.async` are Ampere (sm_80+) features. Those
>   paths compile under `#if __CUDA_ARCH__ >= 800` in CI and are labelled *compile-only* until an
>   sm_80+ run lands.
> - **`N % 32 == 0` and `Nq == Nkv` are enforced** by every stage in-window; ragged sequence
>   lengths are a stated limitation, not a silent bug. Head dims 64 and 128; fp32 and fp16 inputs;
>   scores, running max/sum and the output accumulator are fp32 in every kernel.
> - **Causal mask is top-left aligned** (PyTorch `is_causal` semantics for `L == S`).

## The optimization ladder

One stage per optimization, one named Nsight Compute metric per stage. The metric is chosen
*before* the stage is written; `docs/STAGES.md` records hypothesis, before/after metric, bench
delta and the keep/revert decision. Kernel names are frozen in `CONTRACT.md` §2.

| Stage | Kernel(s) | What changes | ncu metric that should move |
|---|---|---|---|
| S1 | `fa_s1_qk`, `fa_s1_softmax`, `fa_s1_pv` | three fp32 kernels, global memory only, S and P materialized in HBM | baseline: `dram__bytes.sum` grows with N² (S and P round-trip HBM); `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct` low on the strided K^T read |
| S2 | `fa_s2_qk_tiled`, `fa_s2_softmax`, `fa_s2_pv_tiled` | shared-memory tiled GEMMs, `float4` coalesced loads, `__shfl_down_sync` softmax; S/P still materialized | coalescing % toward ~100; `dram__bytes.sum` down by the tile-reuse factor; `l1tex__t_sector_hit_rate.pct` / `lts__t_sector_hit_rate.pct` up; bank conflicts recorded as S5a's "before" |
| S3 | `fa_s3_fused` | one kernel: a block owns a Q tile (FA2 loop order), K/V tiles stream through smem, online softmax with deferred normalization, LSE written | `dram__bytes.sum` collapses from O(N²) to O(B·H·N·D); `dram__throughput...pct` falls; stalls shift to `mio_throttle` / `barrier` |
| S4 | `fa_s4_wmma` | fp16 in / fp32 accumulate on tensor cores via `nvcuda::wmma`; S round-trips padded smem because the fragment layout is opaque | `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active` from 0 to non-zero; `sm__inst_executed_pipe_tensor.sum > 0`; ffma count down; zero spills in `-Xptxas -v` |
| S5a | `fa_s5_tuned` | bank-conflict-free smem: +4 floats / +8 halves padding, optional XOR swizzle as a second commit | `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` toward 0; `mio_throttle` down |
| S5b | `fa_s5_tuned` | `__shfl_xor_sync` butterflies replace smem tree reductions and their barriers | `smsp__warp_issue_stalled_barrier_per_warp_active.pct` down; shared-load instruction count down |
| S6 | op + `bench/gpt_block.py` | dispatch to the best kernel per (dtype, D); `opcheck`; `torch.compile` smoke; GPT-block tokens/sec; causal block skipping if time allows | causal: `smsp__inst_executed.sum` and time should roughly halve at fixed N (FA2 reports ~1.7-1.8x from skipping; to be measured here, never assumed); op: kernel time equals the standalone bench within noise |

## Benchmarks (generated by `scripts/bench_to_md.py`)

<!-- BENCH:START -->
_No benchmark results committed yet._

<!-- BENCH:END -->

## Nsight Compute evidence (generated by `scripts/ncu_to_md.py`)

<!-- NCU:START -->
_No profiles committed yet._

<!-- NCU:END -->

## End-to-end GPT block (generated from `bench/results/<gpu>/gpt_block.csv`)

<!-- GPT:START -->
_No GPT-block results yet._

<!-- GPT:END -->

Roofline: `docs/img/roofline_t4.png` is produced by `scripts/roofline.py` from
`profiles/t4/*.raw.csv` and does not exist until the first profile is committed. The FMA and DRAM
roofs come from ncu's own `.peak_sustained` counters; the fp16 tensor-core roof comes from the T4
datasheet and is labelled "spec, not measured" in `docs/ENV.md`.

`tests/test_docs_cpu.py` regenerates the three blocks above and fails CI if the README is stale,
so a number cannot appear here without a CSV behind it.

## What the CI badges prove (and do not prove)

`build` proves every `.cu` compiles for `sm_75, sm_80, sm_86, sm_89, sm_90` plus `compute_90` PTX
with CUDA 12.8.1 on a GPU-less runner (plus an informational CUDA 13.0.1 leg that may fail without
breaking the badge) and uploads the `-Xptxas -v` register/spill report. `lint`
proves clang-format, ruff, gersemi and the CPU-only pytest suite (CPU model of the algorithm,
docs staleness, CSV sanity) pass. Neither badge proves a kernel ran or was correct on a GPU:
that evidence is the committed `bench/results/<gpu>/`, `profiles/<gpu>/` and `docs/STAGES.md`.

> **What T4 (sm_75) cannot do — hardware facts, cited**
> - **64 KB shared memory per block**, 48 KB static without an opt-in; unified L1/shared is 96 KB
>   per SM with 64/32 or 32/64 carveouts. "Turing allows a single thread block to address the full
>   64 KB of shared memory." — [Turing Tuning Guide](https://docs.nvidia.com/cuda/turing-tuning-guide/index.html)
> - **32 concurrent warps per SM** ("versus 64 on Volta"), 64K 32-bit registers per SM, 255 per
>   thread — same guide. Half the occupancy ceiling of A100 (64 warps) per the
>   [Ampere Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html).
> - **fp16 WMMA yes**: Turing tensor cores take fp16 inputs with fp16/fp32 accumulate, WMMA shapes
>   16x16x16 / 32x8x16 / 8x32x16 (Turing Tuning Guide); the cuda-samples
>   [cudaTensorCoreGemm README](https://github.com/NVIDIA/cuda-samples/blob/master/cpp/3_CUDA_Features/cudaTensorCoreGemm/README.md)
>   lists SM 7.5 as supported.
> - **No bf16 tensor cores, no TF32, no `cp.async`**: all three are listed as *new* in the Ampere
>   Tuning Guide; `__nv_bfloat16` WMMA is "for devices with compute capability compute_80 and
>   higher" ([CUDA 11 features blog](https://developer.nvidia.com/blog/cuda-11-features-revealed/));
>   the [globalToShmemAsyncCopy README](https://github.com/NVIDIA/cuda-samples/blob/master/cpp/3_CUDA_Features/globalToShmemAsyncCopy/README.md)
>   says async copy applies "when on compute capability 8.0 or higher"; the PTX ISA marks
>   `cp.async` "Requires sm_80 or higher". On T4, K/V prefetch is a register-staged
>   `ld.global -> st.shared` double buffer.
> - **`mma.sync` m16n8k8 only** (no m16n8k16): FlashAttention's
>   [kernel_traits.h](https://raw.githubusercontent.com/Dao-AILab/flash-attention/main/csrc/flash_attn/src/kernel_traits.h)
>   selects `SM75_16x8x8_F32F16F16F32_TN` below `__CUDA_ARCH__ >= 800`; CUTLASS's
>   [functionality.md](https://raw.githubusercontent.com/NVIDIA/cutlass/main/media/docs/cpp/functionality.md)
>   lists m16n8k16 f16 only for SM80+. `ldmatrix` *is* available on sm_75.
> - **No vendor FlashAttention kernel exists for T4**: PyTorch's FLASH backend is sm80+
>   ([sdp_utils.cpp](https://raw.githubusercontent.com/pytorch/pytorch/v2.14.0/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp)),
>   upstream flash-attention supports "Ampere, Ada, or Hopper GPUs" and redirects Turing users to a
>   separate community repo ([README](https://raw.githubusercontent.com/Dao-AILab/flash-attention/main/README.md)),
>   and cuDNN's fused SDPA "requires SM80 (Ampere) or newer"
>   ([cudnn-frontend Attention.md](https://github.com/NVIDIA/cudnn-frontend/blob/develop/docs/operations/Attention.md)).
>   The memory-efficient SDPA backend on pre-sm80 accepts only fp16 and fp32 (no bf16). This gap is
>   the reason the repo exists, and it is also why "% of SDPA" here means "% of EFFICIENT_ATTENTION".
> - **No FP8** (Ada sm_89 / Hopper per the [Ada Tuning Guide](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html));
>   **no WGMMA / TMA / clusters** (Hopper, [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)).

## Hardware and software versions

Every CSV row carries `commit,gpu,driver,nvcc,torch,ncu,clock_mhz`. The per-run environment
(`bench/results/<gpu>/env.txt`) and the datasheet peaks are tabulated in
[`docs/ENV.md`](docs/ENV.md). Its cells are placeholders until the first run.

## Reproduce

**Mac / any CPU box (no GPU): the algorithm model and the docs checks.**
```
make cpu-test            # CMake `cpu-only` preset: fp64 CPU reference + ctest, no CUDA toolkit needed
pytest -q -m "not gpu"   # CPU model tile-invariance, causal alignment, README staleness, CSV sanity
make reproduce           # prints the full Mac -> Colab -> VM sequence that produces every number
```
`python/fused_attn/reference.py` is the fp64 tiled online-softmax recurrence with tunable
`block_m`/`block_n`; the tests assert tile invariance before any CUDA is written.

**Colab free T4 (the guaranteed track).** Open `colab/runner.ipynb` on a T4 runtime and run its
three cells in order: (1) environment probe, (2) Nsight Compute go/no-go, (3) runner. The runner
clones this repo, builds with `-DCMAKE_CUDA_ARCHITECTURES=75-real`, runs ctest, compute-sanitizer,
the GPU pytest suite, then (in `MODE=full`) locks the SM clock to base with `nvidia-smi -lgc`,
benchmarks the two canonical configs (the CONTRACT §5 48-config sweep, causal 0 and 1, only with
`--sweep`) and runs the ncu protocol, then pushes `bench/results/t4/` and `profiles/t4/` as a
`results(t4): stage N @ <sha>` commit. Outbound git only; no tunnels.

**An sm_80+ VM you control (optional second column).**
```
bash scripts/gate_gpu.sh     # 15-minute go/no-go: nvidia-smi, nvcc, ncu as user then sudo, nsys
bash scripts/sync.sh --stage N --gpu a10   # rsync (never --delete) + gpu_run.sh + pull artifacts
```
If `ncu` only works under `sudo`, the documented fix is
`options nvidia NVreg_RestrictProfilingToAdminUsers=0` in `/etc/modprobe.d/` plus a reboot
([NVIDIA ERR_NVGPUCTRPERM page](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters)).

Protocols: [`docs/BENCHMARK_PROTOCOL.md`](docs/BENCHMARK_PROTOCOL.md),
[`docs/PROFILING_PROTOCOL.md`](docs/PROFILING_PROTOCOL.md), [`docs/NUMERICS.md`](docs/NUMERICS.md).

## Repository layout

```
include/fa/attention.cuh   frozen kernel API (Impl, Dtype, Params, fa::forward, workspace_bytes, impl_supports)
include/fa/                cuda_check.cuh, tile_config.h, softmax_math.cuh
src/stage{1..5}_*.cu       one file per stage; dispatch.cu; reference/cpu_reference.cpp (fp64 oracle)
bench/                     bench_main.cu (torch-free fa_bench), bench.py, gpt_block.py, configs.yaml, results/<gpu>/
profiles/<gpu>/            stageN_<kernel>.ncu-rep + .raw.csv (+ nsys / profiler.txt in degraded mode)
python/fused_attn/         ext.cpp (TORCH_LIBRARY), ops.py (register_fake, opcheck), reference.py (CPU model)
tests/                     CPU tests (model, docs staleness, CSV sanity) and GPU tests (oracle, tile invariance, op)
scripts/                   gpu_run.sh, gate_gpu.sh, profile.sh, ncu_metrics.txt, bench_to_md.py, ncu_to_md.py,
                           roofline.py, perf_trailer.py, sync.sh, local_compile.sh, kernel_regex.py, fa_common.py
colab/runner.ipynb         probe, ncu go/no-go, runner
docs/                      RESEARCH, STAGES, INTERVIEW, BENCHMARK_PROTOCOL, PROFILING_PROTOCOL, NUMERICS, ENV, WORKLOG
CONTRACT.md                frozen interfaces: API, kernel names, CSV schema, op schemas, configs, generated blocks
```

Git story: one branch and PR per stage, squash-merged with `perf:` and `ncu:` trailers generated
from the CSVs, tagged `v0.1-naive` … `v0.5-sep4`; `git log --grep='^perf:'` prints the ladder.

## Prior art and references

Full bibliography with what was borrowed versus cited only: [`docs/RESEARCH.md`](docs/RESEARCH.md).

- FlashAttention 1 / 2 / 3: [arXiv 2205.14135](https://arxiv.org/abs/2205.14135),
  [arXiv 2307.08691](https://arxiv.org/abs/2307.08691), [arXiv 2407.08608](https://arxiv.org/abs/2407.08608);
  the online-softmax identity itself is Milakov & Gimelshein, [arXiv 1805.02867](https://arxiv.org/abs/1805.02867).
  Reference implementation: [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)
  (`csrc/flash_attn/src/softmax.h` for the recurrence and guards).
- siboehm, [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM):
  the numbered-kernel narrative format this README copies, including reporting steps that did not help.
- [tspeterkim/flash-attention-minimal](https://github.com/tspeterkim/flash-attention-minimal) (fp32,
  ~100 lines, honest about losing at long N) and
  [sonnyli/flash_attention_from_scratch](https://github.com/sonnyli/flash_attention_from_scratch)
  (Ampere, 16 kernels, SASS committed) bracket the difficulty of this ladder.
- NVIDIA cuda-samples: [matrixMul](https://github.com/NVIDIA/cuda-samples/blob/master/cpp/0_Introduction/matrixMul/matrixMul.cu),
  [reduction](https://github.com/NVIDIA/cuda-samples/blob/master/cpp/2_Concepts_and_Techniques/reduction/reduction_kernel.cu),
  [cudaTensorCoreGemm](https://github.com/NVIDIA/cuda-samples/blob/master/cpp/3_CUDA_Features/cudaTensorCoreGemm/cudaTensorCoreGemm.cu),
  [globalToShmemAsyncCopy](https://github.com/NVIDIA/cuda-samples/blob/master/cpp/3_CUDA_Features/globalToShmemAsyncCopy/globalToShmemAsyncCopy.cu).
- CUTLASS: [example 41 fused multi-head attention](https://github.com/NVIDIA/cutlass/blob/main/examples/41_fused_multi_head_attention/fused_multihead_attention_fixed_seqlen.cu)
  (sm_75+ kernel headers), [example 88 Hopper FMHA](https://github.com/NVIDIA/cutlass/blob/main/examples/88_hopper_fmha/README.md)
  (cited only), [cute/swizzle.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/swizzle.hpp).
- NVIDIA blogs: [Programming Tensor Cores in CUDA 9](https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/),
  [Using CUDA Warp-Level Primitives](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/),
  [Controlling Data Movement on Ampere](https://developer.nvidia.com/blog/controlling-data-movement-to-boost-performance-on-ampere-architecture/),
  [Analysis-Driven Optimization with Nsight Compute](https://developer.nvidia.com/blog/analysis-driven-optimization-preparing-for-analysis-with-nvidia-nsight-compute-part-1/),
  [Nsight Compute roofline](https://developer.nvidia.com/blog/accelerating-hpc-applications-with-nsight-compute-roofline-analysis/),
  [Kernel Fusion in NVIDIA CUDA](https://developer.nvidia.com/blog/kernel-fusion-in-nvidia-cuda-optimizing-memory-traffic-and-launch-overhead/).
- Benchmark convention: Triton's `do_bench` 256 MB L2 flush
  ([triton/testing.py](https://github.com/triton-lang/triton/blob/main/python/triton/testing.py),
  [driver.py](https://github.com/triton-lang/triton/blob/main/third_party/nvidia/backend/driver.py)).

## Limitations and next steps

Deferred, in the order they would be picked up (each its own PR with a STOP summary):

1. **Ampere/Ada rows**: bf16 WMMA, `cp.async` double buffering, `SDPBackend.FLASH_ATTENTION` and
   `flash_attn_func` as baselines, per-arch `TileCfg`, locked clocks — needs an sm_80+ VM.
2. **Causal block skipping** (if not landed in S6) with the exact `(n+1)/(2n)` FLOP fraction, and
   **GQA/MQA** via `kv_head = q_head / (H / Hkv)` tested against `enable_gqa=True`.
3. **FA2 split-Q warp partitioning with `mma.sync` + `ldmatrix`**: keeps S in registers and removes
   the WMMA smem round trip; needs the documented m16n8k8 lane map on sm_75.
4. **Tile sweep and `__launch_bounds__`** into a generated `tile_config.h` (FA2 leaves autotuning as
   future work).
5. **Numerical error vs N and vs tile size** study (cf. [arXiv 2405.02803](https://arxiv.org/abs/2405.02803)).
6. **Split-KV decode (flash-decoding)** — partial O + LSE per split, LSE-weighted combine — or the
   **backward pass** (FA2 Algorithm 2, fp32 dQ accumulation); one of the two, not both.
7. **Ragged N** (`N % 32 != 0`), paged KV, FP8.

Dropped with a reason: **FA3 warp specialization / pingpong / TMA / WGMMA** — Hopper-only; pre-Hopper
`mma.sync` is synchronous with register operands, so there is nothing to overlap against. **FP8** —
Ada/Hopper only, and FA3 needs block quantization plus incoherent processing to make it accurate.
**Multi-GPU** — no hardware; scaling numbers are never simulated.

## License

MIT — see [LICENSE](LICENSE). Borrowed *patterns* from cuda-samples and CUTLASS (BSD-3-Clause),
flash-attention (BSD-3-Clause) and cudnn-frontend (Apache-2.0) are cited in `docs/RESEARCH.md` §9;
no kernel bodies are copied.
