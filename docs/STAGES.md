# Stages

One entry per optimization stage, written at the stage's STOP point and never rewritten afterwards
(corrections are appended, dated). Every claim of a metric moving points at a committed file under
`profiles/<gpu>/` or `bench/results/<gpu>/`; if the file does not exist the claim is `TBD`.

## Entry format

```
## S<N> — <name>   (branch <branch>, tag <tag>, merged <date>, GPU <gpu>)
Hypothesis        what is slow, which metric says so, what should move and in which direction
ncu before        metric values with the raw-CSV path they come from (previous stage's profile)
Change            what the kernel does differently, in one paragraph, with the file:line
ncu after         same metrics, new profile path; SOL SM% / Mem%, top-3 stall reasons, occupancy,
                  registers/spills from -Xptxas -v
Bench delta       the `perf:` trailer line at the canonical config, plus where it lost (if it did)
Decision          kept / reverted / kept-with-regression, and why, in the interviewer's vocabulary
Known limitations what this stage does not do yet
Go / no-go        for the next stage
5-line summary    the version I would say out loud in an interview
Two questions     the two most likely interviewer questions, with my answers
```

The "tried and rejected" experiments of a stage get their own sub-entry with the same shape. Metrics
are quoted by their exact ncu ID (`docs/PROFILING_PROTOCOL.md` section 3) so a reader can
`grep -o '<id>[^,]*,[^,]*' profiles/<gpu>/stageN_<kernel>.raw.csv` and see the same value.

---

## S0 — probes and scaffold   (branch `stage0-scaffold`, tag `v0.0-scaffold`, date 2026-09-02, GPU: none yet)

**Hypothesis.** None to test. S0 exists so that S1..S6 can each be a one-variable experiment: the API,
CSV schema, kernel names, canonical configs, benchmark and profiling protocols, tolerance rule and
docs skeleton are frozen before any kernel is written, and the GPU route is proven with a stub
kernel before it is trusted with a real one.

**What was built.** (Owners worked in parallel against `CONTRACT.md`.)
- Frozen API `include/fa/attention.cuh`: `fa::Impl {Naive=1, Tiled, Fused, Wmma, Tuned}`,
  `fa::Dtype {F32, F16, BF16}`, `fa::Params {B, H, Hkv, N, D, scale, causal}`, `fa::forward(...)`,
  `workspace_bytes`, `impl_supports`, `set_debug_overrides` (tile-invariance hooks `FA_FORCE_BR/BC`).
- Build: CMake >= 3.24 (>= 3.25 to use `CMakePresets.json` v6), presets `colab` (75-real), `vm` (native), `ci` (75/80/86/89/90-real +
  90-virtual); `-O3 -lineinfo -Xptxas -v`; `cuda_check.cuh`, `tile_config.h` with per-arch
  `static_assert` on the shared-memory cap, `softmax_math.cuh` (exp2 with folded scale, `-inf` and
  `l == 0` guards, `LSE = m*scale + log(l)`); `dispatch.cu`; `smoke.cu` stub kernel `fa_smoke`;
  fp64 CPU oracle `src/reference/cpu_reference.cpp` that builds without CUDA.
- Bench: torch-free `bench/bench_main.cu` (`fa_bench`) and `bench/bench.py` implementing
  `docs/BENCHMARK_PROTOCOL.md` identically; `bench/configs.yaml`; `bench/gpt_block.py`.
- PyTorch op: `python/fused_attn/ext.cpp` with `TORCH_LIBRARY(fused_attn)` `forward` /
  `forward_stage`, `ops.py` with `register_fake`, `reference.py` with `math_reference` (fp64) and
  `tiled_online_softmax_reference(block_m, block_n)`, the CPU model of the recurrence.
- Tests: `tests/test_reference_cpu.py` (tile invariance, causal alignment, guards, on the Mac),
  `test_kernels_gpu.py`, `test_op_gpu.py` (`gpu` marker), `test_bench_sanity.py`,
  `test_docs_cpu.py` (README generated blocks fresh vs CSVs).
- Scripts: `gpu_run.sh` (Colab/VM entry point; prints a <= 40-line summary), `gate_gpu.sh`
  (PROBE 4), `profile.sh`, `ncu_metrics.txt`, `bench_to_md.py`, `ncu_to_md.py`, `roofline.py`,
  `perf_trailer.py`, `sync.sh`, `local_compile.sh`, `kernel_regex.py`; `colab/runner.ipynb`
  (env probe, ncu go/no-go, runner).
- CI: `build.yml` (compile-only, five archs, ptxas report artifact), `lint.yml` (clang-format,
  ruff, gersemi, `pytest -m "not gpu"`); `.clang-format`, `.pre-commit-config.yaml`, `.gersemirc`.
- Docs: README with the generated blocks and the honesty box, `RESEARCH.md`, this file,
  `INTERVIEW.md`, `BENCHMARK_PROTOCOL.md`, `PROFILING_PROTOCOL.md`, `NUMERICS.md`, `ENV.md`,
  `WORKLOG.md`.

**Probes (the lead fills each result; a probe that was not run stays TBD).**

| Probe | What it settles | Command / where | Result |
|---|---|---|---|
| PROBE L | arm64 `nvcc` in a container on the Mac, so compile errors and `-Xptxas -v` never cost a Colab round trip | `colima start` + `docker run --platform linux/arm64 nvidia/cuda:12.8.1-devel-ubuntu22.04 nvcc --version`; `scripts/local_compile.sh` | **Done 2026-09-02**: `nvidia/cuda:12.8.1-devel-ubuntu22.04` arm64 -> `nvcc V12.8.93` (same as Colab); Docker Hub pulls through colima corrupted two layers (repeatable digest mismatch), fixed with `crane pull --platform linux/arm64` + `docker load`. `scripts/local_compile.sh` compiles the tree for sm_75 (and the CI list) in seconds with `-Xptxas -v` (`docs/generated/S0_INTEGRATION.md` section 2) |
| PROBE 1 | Colab env: GPU name, driver, compute cap, nvcc, ncu/nsys/compute-sanitizer presence, torch + `torch.version.cuda`, cmake/ninja | `colab/runner.ipynb` cell 1 -> `bench/results/t4/env.txt` | **Done 2026-09-02** (`.research/colab_probe_2026-09-02.txt`, copied to `bench/results/t4/probe_2026-09-02.txt`): Tesla T4, driver 580.82.07, CC 7.5, 15360 MiB, `clocks.max.sm` 1590 MHz (idle 300); nvcc 12.8.93; user root; ncu `/usr/local/cuda/bin/ncu` (2018-2025 build); compute-sanitizer present; cmake `/usr/local/bin/cmake`; **ninja MISSING** (`pip install ninja`, gpu_run.sh does it); **nsys MISSING**; torch 2.11.0+cu128, `torch.version.cuda` 12.8, arch list includes sm_75 |
| PROBE 2 | ncu hardware counters on this Colab instance (PASS = four metric values; FAIL = `ERR_NVGPUCTRPERM`, reset <= 3x); `--query-metrics` / `--list-sections` pinned; `--clock-control base` and `nvidia-smi -lgc` accepted?; nsys usable? | `colab/runner.ipynb` cell 2 -> `scripts/ncu_metrics.txt`, `docs/PROFILING_PROTOCOL.md`, `docs/ENV.md` | **PASS 2026-09-02, 0 resets**: `dram__bytes.sum` 296.54 KB, `launch__registers_per_thread` 16, `sm__cycles_elapsed.avg` 3589.80, `sm__warps_active.avg.pct_of_peak_sustained_active` 84.51 % on the 10-line kernel; `--query-metrics` lists `sm__throughput`, `gpu__time_duration`, `sm__pipe_tensor_op_hmma_cycles_active`, `smsp__warp_issue_stalled_long_scoreboard_per_warp_active`, `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld` (`gpu__compute_memory_throughput` not reported by the probe: still pruned at run time); `--list-sections` has every section profile.sh asks for incl. `MemoryWorkloadAnalysis_Tables`; `--clock-control base` accepted; `nvidia-smi -lgc 585,585` WORKS as root (`-rgc` works); nsys missing. Counter access is per-instance: PROBE 2 is re-run at the start of every session |
| PROBE 3 | AWS G-instance quota in us-east-1 (read-only) | `aws sts get-caller-identity`; `aws service-quotas list-service-quotas --service-code ec2 ... 'G and VT'` | NOT RUN — the Mac's AWS credentials are invalid (`InvalidClientTokenId` on both profiles), so no quota query was possible; Track B needs a Lambda/Vast.ai signup by the user |
| PROBE 4 | VM ncu as user then `sudo`; modprobe fix + reboot if needed; nsys; clock lock | `scripts/gate_gpu.sh` over `ssh gpu` | NOT RUN — no Track B VM yet (user decision pending) |
| PROBE 5 | Bash permission prompts for `gh pr checks` and `ssh gpu nvidia-smi` under the project allow-list | one call each | n/a — the lead runs in auto mode; no project allow-list was needed |
| PROBE 6 | CI nvcc-only network install time and `dpkg -l \| grep cuda-` on the first `build.yml` run | GitHub Actions log | PASS 2026-09-03 — run 33700104128: `Jimver/cuda-toolkit@v0.2.36` network install of nvcc + cudart-dev took 18 s (12.8.1) / 17 s (13.0.1); configure 4-6 s; full multi-arch build 8 s; both CUDA legs + cpu-only preset + lint green on the first counted push (023cfb8); the earlier pair at 83cb449 was cancelled by the concurrency group |
| PROBE 7 | macOS Nsight Compute host GUI available for the version Colab ships | developer.nvidia.com/tools-overview/nsight-compute | PROBE result: TBD |
| Round trip | stub kernel `fa_smoke` built, run, benched and pushed by `gpu_run.sh` from the runner cell as a `results(t4): stage 0 @ <sha>` commit | runner cell, `MODE=quick` then `full` | PASS 2026-09-03 00:35 UTC on Colab T4 @ 023cfb8, `MODE=quick`: configure 4 s, build 9 s, ctest 2/2 (test_cpu_reference, test_smoke), compute-sanitizer memcheck 0 / racecheck 0 errors, ncu verdict PASS, torch extension built and `fused_attn.forward` registered, GPU pytest 8 passed / 8 skipped (no stage kernels yet); `--push` not exercised (PAT not yet in Colab Secrets) |

**ncu before / after.** Not applicable: the only kernel is the stub. The S1 entry's "before" is the S1
profile itself (the baseline of the ladder).

**Bench delta.** None. No timing exists yet and none is claimed.

**Decision.** Ship S0 as `v0.0-scaffold` once CI is green, `torch.ops.fused_attn.forward` is
callable (stub), `pytest -m "not gpu"` passes on the Mac, and the runner round trip has pushed an
`env.txt` with the ncu verdict. If PROBE 2 fails after three resets, S1 proceeds in degraded
profiling mode with the reason recorded; the ladder does not wait.

**Known limitations.** Nothing runs on a GPU yet. bf16 and `cp.async` paths are compile-only.
Track B is not provisioned. `N % 32 == 0`, `Nq == Nkv`, forward only.

**Go / no-go for S1.** Go when the round-trip probe result above is filled in and green.

**5-line summary.**
1. I froze the kernel API, kernel names, CSV schema, canonical configs and the two protocols before
   writing any kernel, so every later stage changes exactly one thing and its metric is comparable.
2. I picked the free Colab T4 as the guaranteed GPU because it is the only zero-cost platform with
   independent 2026 evidence that Nsight Compute counters work, and I probe that at every session
   start because access is per-instance.
3. On T4 there is no vendor FlashAttention: SDPA's FLASH and cuDNN backends refuse `sm < 80` and
   upstream flash-attention excludes Turing, so the honest baseline is SDPA `EFFICIENT_ATTENTION`,
   named on every table.
4. I wrote the tiled online-softmax recurrence as a CPU model with a tile-invariance test first, so
   the rescale bug that kills most from-scratch kernels is caught on my laptop, not on the GPU.
5. Numbers only enter the README through CSVs and a staleness test; the scaffold's tables are empty by
   design and say so.

**Two questions.**

*Q1: Why a torch-free benchmark binary when you already have a PyTorch op?*
Two reasons. First, profiling: `ncu --kernel-name regex:fa_s3_fused` on `fa_bench` sees only my
kernels, under kernel replay, with no allocator, dispatcher or autotuner launches in the way;
application replay under PyTorch would need the whole Python process to be deterministic, which
cuDNN/cuBLAS heuristics and dynamic shapes break. Second, trust: the C++ harness implements the same
protocol as `bench.py` (events per iteration, 256 MB L2 flush, rotating inputs, checksum sink), so if
the op's kernel time in `bench.py` matches `fa_bench` within noise I have shown the op adds no hidden
copies; if they disagree I have found a bug in the binding, not in the kernel.

*Q2: Why bother with a CPU model of the tiled recurrence?*
Because the algorithm and the CUDA are separable failure modes. The dangerous bug in a fused kernel
is silent: forget `acc *= alpha` (or apply the old max to the current tile) and the output is
finite, plausible and wrong only when the running max changes across tiles. In Python I can force
`block_n` to N, N/2 and N/4 on the same input and require identical results and agreement with the
fp64 reference; that test runs in seconds on a Mac with no GPU. Once it passes, every discrepancy on
the GPU is an indexing, layout or synchronisation bug, which is what `compute-sanitizer` and the
tile-invariance test on the device are for. It also gives me the whiteboard artefact: the recurrence
in ten lines with the guards for `m == -inf` and `l == 0` visible.

---

## S1 — naive three kernels, fp32   (TBD)
## S2 — shared-memory tiled GEMMs + coalesced loads   (TBD)
## S3 — fused kernel, online softmax, FA2 loop order   (TBD)
## S4 — fp16 WMMA, fp32 accumulate   (TBD)
## S5a — bank-conflict-free shared memory   (TBD)
## S5b — warp-shuffle softmax   (TBD)
## S6 — torch op dispatch, GPT block, causal block skipping   (TBD)
