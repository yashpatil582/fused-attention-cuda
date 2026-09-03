# Environment record

Every benchmark row already carries `commit, gpu, driver, nvcc, torch, ncu, clock_mhz`; this file is
the human-readable version, filled by the lead from `bench/results/<gpu>/env.txt` after PROBE 1/2
(Colab) and PROBE 4 (VM). Nothing here is typed from memory: every cell is either copied from a
probe's or a run's output (`env.txt`, the profile CSVs) or is a datasheet value with its URL; a
gap is written down as a gap.

## Per-platform toolchain

| Field | Colab T4 (Track A) | VM (Track B) |
|---|---|---|
| Runs recorded | S0 @ 023cfb8 (2026-09-03 00:35 UTC); S1-S4 @ 3add37c (06:00-08:00 UTC, canonical + 48-config sweep + Nsight); S5 @ d835758 (08:17-08:29 UTC, + `tune_s5.sh` A/B + GPT block) and @ 15b83a5 (09:12 UTC, tuned D=128 default); `bench/results/t4/env.txt` is the last run's record | none — Track B was never started (AWS quota probe failed on invalid credentials; no Lambda/Vast account; **$0 spent**) |
| GPU (`nvidia-smi --query-gpu=name`) | Tesla T4 | — |
| Compute capability | 7.5 | — |
| Driver version | 580.82.07 | — |
| `nvcc --version` | 12.8.93 (`cuda_12.8.r12.8/compiler.35583870_0`) | — |
| CUDA runtime seen by torch (`torch.version.cuda`) | 12.8 | — |
| torch version | 2.11.0+cu128 (arch list `sm_75 sm_80 sm_86 sm_90 sm_100 sm_120`) | — |
| Nsight Compute (`ncu --version`) and path | 2025.1.1.0, `/usr/local/cuda/bin/ncu` (version column of every profile row) | — |
| Nsight Systems (`nsys --version`) | **missing** on every run (degraded mode = CUDA events + `-Xptxas -v`, no timeline) | — |
| `compute-sanitizer` present | yes, `/usr/local/cuda/bin/compute-sanitizer`; memcheck + racecheck clean for every stage on `B1_H2_N288_D{64,128}` (`bench/results/t4/stage*_sanitizer.txt`) | — |
| cmake / ninja | cmake 3.31.10 (Colab image); ninja from pip (gpu_run.sh) | — |
| Host compiler (`g++ --version`) | not recorded — the env step prints nvcc/torch/cmake only (a gap, noted rather than typed from memory) | — |
| Clock policy | `nvidia-smi -lgc 585,585` works as root; every bench row says `clock_lock=585` (bench.py samples the SM clock through pynvml in a 20 ms background thread and records the median); ncu runs `--clock-control base`, so profiles and bench rows share the 585 MHz clock | — |
| Base / max SM clock (`nvidia-smi -q -d CLOCK`) | 585 MHz (`clocks.default_applications.graphics`) / 1590 MHz (`clocks.max.sm`); idle 300 MHz | — |
| `ncu --clock-control base` accepted | yes, every capture | — |
| ncu counter verdict (PROBE 2) | `PASS (counters readable)` on every session (2026-09-02 probe and each stage run) | — |
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` listed | yes, captured in every profile (`profiles/t4/ncu_metrics.used.txt`) | — |
| `gpu__time_duration.sum` listed | yes, captured | — |
| `MemoryWorkloadAnalysis_Tables` listed by `--list-sections` | yes (`profiles/t4/ncu_list_sections.txt`) | — |
| SDPA FLASH refusal string captured | yes, `env.txt` (`sdpa_flash: refused (fp16): Memory efficient kernel not used because: ... Flash attention k…`, truncated at 200 chars by the notes cleaner; the cuDNN backend refuses with the same text). Both are `refused` in every README row | n/a (FLASH runs on sm_80+) |
| Device memory (`cudaDeviceProp.totalGlobalMem`) | 15360 MiB (`nvidia-smi memory.total`) | — |
| Shared memory per block opt-in | 64 KB (Turing cap, `ArchCaps<750>`); verified in practice: `fa_s3_fused<float,128,16>` launches with 49,408 B and S5's D=128 default with 59,904 B of dynamic shared memory after `cudaFuncSetAttribute` (both > 48 KB) | — |
| Provider / instance type / hourly price | Colab free tier, T4 runtime, $0 (~5 GPU-hours used across 2026-09-03) | — |

CI (`.github/workflows/build.yml`): ubuntu-24.04 runner, `Jimver/cuda-toolkit@v0.2.36` with
`cuda: '12.8.1'`, `method: 'network'`, `sub-packages: '["nvcc","cudart-dev"]'`; compile-only for
`75-real;80-real;86-real;89-real;90-real;90-virtual`. Measured (PROBE 6): the whole `build` job,
toolkit install included, takes about 5 minutes wall on the hosted runner (e.g. run for d835758:
08:08:57 -> 08:14:07 UTC; ba3292a: 06:55:32 -> 07:00:23 UTC), well inside the 6-hour job cap.

Mac (docs, CPU model, CI scripts): Darwin arm64, `/opt/anaconda3/bin/python3` 3.12, torch 2.5.1
CPU, pytest, matplotlib, pandas; cmake 4.4.3 + ninja (brew); no nvcc, no GPU. Local CUDA compile
loop via colima + `nvidia/cuda:12.8.1-devel-ubuntu22.04` (arm64): PROBE L done 2026-09-02, nvcc
V12.8.93, `scripts/local_compile.sh` builds `fa_bench` for sm_75 with `-Xptxas -v`
(`docs/generated/S0_INTEGRATION.md`).

## Datasheet peaks ("spec, not measured")

Used only for the fp16 tensor-core roof in `docs/img/roofline_<gpu>.png` and for the
`test_bench_sanity` cap. The FMA and DRAM roofs are taken from ncu's own `.peak_sustained` counters
(see `docs/PROFILING_PROTOCOL.md` section 3), not from here. **Clock caveat:** every ncu counter,
and therefore every roofline point and roof, is taken under `ncu --clock-control base` (T4 SM at
its base clock, 585 MHz per the `-lgc` probe), and the figure prints that clock; bench rows carry
their own `clock_mhz` (locked to the same base clock by `gpu_run.sh` when `nvidia-smi -lgc` works,
otherwise unlocked/boost, up to 1590 MHz on T4). Do not compare a bench row's TFLOP/s with the
roofline's achieved FLOP/s directly unless the row's `clock_mhz` equals the profile clock; compare
% of roof.

| GPU | fp32 peak (TFLOP/s) | fp16 tensor peak (TFLOP/s) | Memory bandwidth (GB/s) | Memory (GB) | Source URL | Filled by |
|---|---|---|---|---|---|---|
| Tesla T4 (sm_75) | 8.1 | 65 | 320 | 16 | https://www.nvidia.com/en-us/data-center/tesla-t4/ (NVIDIA T4 product page / datasheet, read 2026-09-02) | lead, 2026-09-02 |
| Track B GPU | — | — | — | — | not used ($0; see the toolchain table) | lead |

A datasheet "peak" is a boost-clock marketing number; the run's clock is recorded per row, so any
"% of peak" quoted next to a measurement says which peak and which clock it used.

### Machine-readable roofs (parsed by `scripts/fa_common.py::read_env_roofs`)

One line per roof, format `ROOF <name> = <number> <unit> (<provenance>; <url>)`. Only the
`ROOF <name> = <number>` prefix is parsed; a `TBD` value does not parse, so no cap is applied until
the lead replaces it with the datasheet number and URL. `tests/test_bench_sanity.py` caps fp32 rows
by `fp32_tflops` and fp16/bf16 rows by `fp16_tensor_tflops` / `bf16_tensor_tflops`.

```
ROOF fp32_tflops = 8.1 TFLOP/s (spec, not measured; https://www.nvidia.com/en-us/data-center/tesla-t4/ read 2026-09-02)
ROOF fp16_tensor_tflops = 65 TFLOP/s (spec, not measured; https://www.nvidia.com/en-us/data-center/tesla-t4/ read 2026-09-02)
ROOF dram_gbps = 320 GB/s (spec, not measured; https://www.nvidia.com/en-us/data-center/tesla-t4/ read 2026-09-02)
```

## Architectural constants used by the kernels (cited, not measured)

| Property | sm_75 (T4) | sm_80 (A100) | sm_86 / sm_89 | sm_90 (H100) | Source |
|---|---|---|---|---|---|
| Max shared memory per block | 64 KB (48 KB static without opt-in) | 163 KB | 99 KB | 227 KB | Turing / Ampere / Ada / Hopper Tuning Guides |
| Unified L1 + shared per SM | 96 KB (carveouts 64/32 or 32/64) | 192 KB | 128 KB | 256 KB | same |
| Max concurrent warps per SM | 32 | 64 | 48 | 64 | same |
| 32-bit registers per SM / max per thread | 64K / 255 | 64K / 255 | 64K / 255 | 64K / 255 | same |
| fp16 WMMA | yes (16x16x16, 32x8x16, 8x32x16) | yes | yes | yes | Turing Tuning Guide |
| bf16 / TF32 tensor cores | no | yes | yes | yes | Ampere Tuning Guide; CUDA 11 features blog |
| `cp.async` | no ("Requires sm_80 or higher") | yes | yes | yes | PTX ISA; globalToShmemAsyncCopy README |
| `mma.sync` f16 shape | m16n8k8 | m16n8k16 | m16n8k16 | m16n8k16 | flash-attention `kernel_traits.h`; CUTLASS `functionality.md` |
| `ldmatrix` | yes | yes | yes | yes | CUTLASS `memory_sm75.h` |
| FP8 tensor cores | no | no | sm_89 only | yes | Ada Tuning Guide; CUTLASS `mma_sm89.h` |
| TMA / clusters / `wgmma` | no | no | no | yes (`sm_90a` for wgmma) | Hopper Tuning Guide; nvcc docs |

URLs for the sources are in `docs/RESEARCH.md` section 4.
