# CONTRACT.md — frozen interfaces (S0, 2026-09-02)

Everything below is shared by the build system, the benchmark harness, the PyTorch op,
the tests, the profiling scripts and the docs generators. Changing any of it requires
updating every consumer in the same commit. Agents working in parallel must read this
file first and never edit files outside their ownership list.

## 1. Kernel API
`include/fa/attention.cuh` (verbatim, frozen): `fa::forward(Impl, Dtype, q, k, v, o, lse, workspace, Params, stream)`,
`fa::workspace_bytes`, `fa::impl_supports`, `fa::set_debug_overrides`, `fa::impl_name`, `fa::dtype_name`.
Layouts: q/k/v/o `[B, H, N, D]` contiguous; lse `[B, H, N]` fp32. Tensor bases must be 16-byte aligned (S2+ use float4/uint2 vector loads; ext.cpp checks this). Causal mask is top-left aligned (torch SDPA
`is_causal` semantics for Nq == Nkv). Default scale `1/sqrt(D)`. `N % 32 == 0` enforced in-window.

## 2. Kernel names (for `ncu --kernel-name regex:`)
| Stage | kernels |
|---|---|
| S1 | `fa_s1_qk`, `fa_s1_softmax`, `fa_s1_pv` |
| S2 | `fa_s2_qk_tiled`, `fa_s2_softmax`, `fa_s2_pv_tiled` |
| S3 | `fa_s3_fused` |
| S4 | `fa_s4_wmma` |
| S5 | `fa_s5_tuned` |
| S0 smoke | `fa_smoke` |
All kernels are templates; the regex matches the mangled name prefix, so keep these exact identifiers.

## 3. Benchmark CSV schema (frozen; `bench/results/<gpu>/stageN.csv`, `summary.csv`, `gpt_block.csv`)
```
commit,gpu,driver,nvcc,torch,ncu,clock_mhz,impl,stage,dtype,B,H,N,D,causal,ms_median,ms_p10,ms_p90,tflops,gbps,notes
```
- `impl` ∈ {naive, tiled, fused, wmma, tuned, sdpa_efficient, sdpa_flash, sdpa_math_fp32, sdpa_cudnn, torch_unfused, flash_attn}
  for the README tables; additionally `smoke` (stage 0: the S0 `fa_bench --smoke` round-trip row, never rendered) and
  `gpt_sdpa_efficient` / `gpt_sdpa_flash` / `gpt_fused` in `gpt_block.csv` only (`ms_median` = block time, tokens/s in `notes`).
- `stage` ∈ {1..5} for our kernels, 0 for baselines. `dtype` ∈ {fp32, fp16, bf16}. `causal` ∈ {0,1}.
- `tflops` = FLOPs / time with FLOPs = 4·B·H·N²·D (causal rows use the full count until block skipping exists; `notes` says `flops=full`).
- `gbps` = modelled minimum bytes / time (fused: Q+K+V+O+LSE; materializing impls add one write and one read each of S and P);
  measured traffic is the profile's `dram__bytes.sum`.
- `notes` free text: `oom_by_design`, `flops=full`, `clock=unlocked`, `clock_lock=<mhz>`, `noisy`, `iters=<n>`, `run=<utc>`,
  `backend_refused:<msg>`, `degraded_profiling`, etc.
- `ms_*` from CUDA events, median of ≥100 timed iterations after ≥10 warmup, 256 MB L2 flush before every timed iteration.
- `gpu` directory name: `t4`, `a10g`, `a10`, `a100`, `4090`, … (lowercase, no spaces); `gpu` column = `nvidia-smi` name verbatim.

## 4. PyTorch ops (`python/fused_attn/ext.cpp`)
```
fused_attn::forward(Tensor q, Tensor k, Tensor v, bool is_causal, float scale) -> Tensor
fused_attn::forward_stage(Tensor q, Tensor k, Tensor v, bool is_causal, float scale, int stage) -> (Tensor, Tensor)
```
`forward` dispatches to the best implemented stage for (dtype, D). `forward_stage` returns `(o, lse)` for an explicit
stage (1..5); `lse` is an empty tensor for stages without it. Inputs must be CUDA, contiguous `[B,H,N,D]`, same dtype
(fp32/fp16/bf16). Both ops have `torch.library.register_fake` implementations in `python/fused_attn/ops.py`.
Python import name: `fused_attn` (package in `python/fused_attn`, extension module `fused_attn._C`).

## 5. Canonical configs (`bench/configs.yaml`)
- `C64` = B2 H8 N2048 D64, `C128` = B2 H8 N2048 D128, causal 0 (and 1 from S6) — the profiling configs.
- Sweep: N ∈ {128, 256, 512, 1024, 2048, 4096} × D ∈ {64, 128} × (B,H) ∈ {(1,8), (8,32)} × causal ∈ {0,1}.
- Config string format `B8_H8_N4096_D64` is accepted by `fa_bench --cfg` and `bench.py --cfg`.

## 6. README generated blocks (`scripts/bench_to_md.py`, `scripts/ncu_to_md.py`)
`<!-- BENCH:START -->…<!-- BENCH:END -->`, `<!-- NCU:START -->…<!-- NCU:END -->`, `<!-- GPT:START -->…<!-- GPT:END -->`.
`tests/test_docs_cpu.py` regenerates and diffs; stale README fails CI.

## 7. Directory ownership for parallel work (S0)
- **build**: `CMakeLists.txt`, `CMakePresets.json`, `Makefile`, `include/fa/cuda_check.cuh`, `include/fa/tile_config.h`, `include/fa/softmax_math.cuh`, `src/dispatch.cu`, `src/smoke.cu`, `src/reference/*`, `bench/bench_main.cu`, `scripts/local_compile.sh`
- **python**: `python/**`, `pyproject.toml`, `tests/conftest.py`, `tests/test_reference_cpu.py`, `tests/test_kernels_gpu.py`, `tests/test_op_gpu.py`
- **scripts**: `bench/bench.py`, `bench/configs.yaml`, `bench/gpt_block.py`, `scripts/*.py`, `scripts/gpu_run.sh`, `scripts/gate_gpu.sh`, `scripts/profile.sh`, `scripts/sync.sh`, `scripts/ncu_metrics.txt`, `tests/test_bench_sanity.py`, `tests/test_docs_cpu.py`, `colab/runner.ipynb`
- **docs**: `README.md`, `docs/**`
- **ci**: `.github/**`, `.clang-format`, `.pre-commit-config.yaml`, `.gersemirc`
- **kernels** (S1+): `src/stage*.cu` only.

## 8. Style and commit rules
- C++/CUDA: clang-format Google style, 100 columns; `.cu/.cuh` formatted as C++. Every kernel has a header comment
  explaining WHY (tile choice, layout, bottleneck removed) and cites the metric that motivated it.
- Python: ruff (check + format), line length 100. Tests: `pytest`, markers `gpu` (skipped without CUDA).
- Commits: conventional (`feat(s3): …`, `bench(t4): …`, `docs: …`). No AI attribution trailers of any kind.
- No fabricated numbers anywhere. Placeholders are literally `TBD` and fail `test_docs_cpu.py` if they reach README tables.
