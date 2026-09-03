# Environment record

Every benchmark row already carries `commit, gpu, driver, nvcc, torch, ncu, clock_mhz`; this file is
the human-readable version, filled by the lead from `bench/results/<gpu>/env.txt` after PROBE 1/2
(Colab) and PROBE 4 (VM). Cells are `TBD` until a run has happened. Nothing here is typed from
memory: every cell is either copied from a probe's output or is a datasheet value with its URL.

## Per-platform toolchain

| Field | Colab T4 (Track A) | VM (Track B) |
|---|---|---|
| Runs recorded | probe only (2026-09-02, `.research/colab_probe_2026-09-02.txt` = `bench/results/t4/probe_2026-09-02.txt`); no stage run yet | TBD |
| GPU (`nvidia-smi --query-gpu=name`) | Tesla T4 | TBD |
| Compute capability | 7.5 | TBD |
| Driver version | 580.82.07 | TBD |
| `nvcc --version` | 12.8.93 (`cuda_12.8.r12.8`) | TBD |
| CUDA runtime seen by torch (`torch.version.cuda`) | 12.8 | TBD |
| torch version | 2.11.0+cu128 | TBD |
| `torch.cuda.get_arch_list()` includes sm_75 | yes (`['sm_75','sm_80','sm_86','sm_90','sm_100','sm_120']`) | n/a |
| Nsight Compute (`ncu --version`) and path | present, `/usr/local/cuda/bin/ncu` ("Nsight Compute, (c) 2018-2025"; exact version string recorded by gpu_run.sh into `env.txt` on the first run) | TBD |
| Nsight Systems (`nsys --version`) | **missing** (degraded mode on Colab = CUDA events + `-Xptxas -v`, no timeline) | TBD |
| `compute-sanitizer` present | yes, `/usr/local/cuda/bin/compute-sanitizer` | TBD |
| cmake / ninja | cmake `/usr/local/bin/cmake` (version checked >= 3.25 by gpu_run.sh, else pip); ninja **missing**, `pip install ninja` (gpu_run.sh) | TBD |
| Host compiler (`g++ --version`) | TBD (recorded on the first stage run) | TBD |
| Clock policy | `nvidia-smi -lgc 585,585` WORKS as root (probe); gpu_run.sh locks to the driver's default application clock before the bench step, `-rgc` on exit, and bench.py verifies the lock per row (`clock_lock=<mhz>` in notes) | TBD (locked to base: value) |
| Base / max SM clock (`nvidia-smi -q -d CLOCK`) | max 1590 MHz, idle 300 MHz (probe); base = default application clock, read by gpu_run.sh at run time (585 MHz per the `-lgc` probe) | TBD |
| `ncu --clock-control base` accepted | yes (probe) | TBD |
| ncu counter verdict (PROBE 2 / PROBE 4) | PASS, 0 resets (2026-09-02); per-instance, re-probed every session | TBD |
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` listed by `--query-metrics` | `sm__throughput` listed (probe) -> ID enabled in `scripts/ncu_metrics.txt` | TBD |
| `gpu__time_duration.sum` listed | `gpu__time_duration` listed (probe) -> enabled | TBD |
| `MemoryWorkloadAnalysis_Tables` listed by `--list-sections` | yes (probe) | TBD |
| SDPA FLASH refusal string captured | TBD (bench.py writes it to `env.txt` on the first stage run) | n/a (FLASH runs on sm_80+) |
| Device memory (`cudaDeviceProp.totalGlobalMem`) | 15360 MiB per nvidia-smi (probe); the runtime value is printed by `fa_bench` on the first run | TBD |
| Shared memory per block opt-in (`cudaDevAttrMaxSharedMemoryPerBlockOptin`) | TBD (first run) | TBD |
| Provider / instance type / hourly price | Colab free tier, $0 | TBD |

CI (`.github/workflows/build.yml`): ubuntu-24.04 runner, `Jimver/cuda-toolkit@v0.2.36` with
`cuda: '12.8.1'`, `method: 'network'`, `sub-packages: '["nvcc","cudart-dev"]'`; compile-only for
`75-real;80-real;86-real;89-real;90-real;90-virtual`. Actual install time on the first run: TBD
(PROBE 6).

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
| NVIDIA T4 (sm_75) | TBD | TBD | TBD | TBD | TBD (T4 datasheet URL, not in the research bundle; the lead pastes it) | lead, S0/S7 |
| Track B GPU | TBD | TBD | TBD | TBD | TBD | lead |

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
