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

## S1 — naive three kernels, fp32   (branch `main`, commit 6e5bb87, date 2026-09-03, GPU Tesla T4)

**Hypothesis.** A first CUDA port of attention is three global-memory kernels: `fa_s1_qk` (one thread
per score, K rows read at stride D), `fa_s1_softmax` (one block per row, three passes) and `fa_s1_pv`
(one thread per output element). Two metrics should indict it: `smsp__sass_average_data_bytes_per_
sector_mem_global_op_ld.pct` on `fa_s1_qk` (only `sizeof(T)` of every 32-byte sector is used on the
K loads: 4/32 fp32, 2/32 fp16) and `dram__bytes.sum` on `fa_s1_softmax`, which grows as B·H·N² because
S and P are materialised in HBM (`workspace_bytes = 2·B·H·N·N·4`). This stage is the "before" of the
whole ladder; nothing in it is tuned on purpose.

**ncu before.** None — S1 is the baseline (`profiles/t4/stage1_fa_s1_qk.raw.csv`,
`stage1_fa_s1_softmax.raw.csv`, `stage1_fa_s1_pv.raw.csv`).

**Change.** `src/stage1_naive.cu`: per-head grids (`blockIdx.y = b·H + h`) keep in-kernel indices
32-bit; causal is a `-inf` write for `j > i` (top-left); the softmax normalises P before PV (P =
exp(s − m)/l) and writes `lse = m + log(l)` on the already-scaled scores; fp16 inputs are widened
on load and narrowed on store, everything else fp32. GQA head mapping `kv_head = h / (H/Hkv)`. Index
math was emulated on the CPU against `reference.py` (max |o − ref| ≤ 2e-15) before the GPU run.

**ncu after.** TBD — filled from `profiles/t4/stage1_*.raw.csv` (`ncu_to_md.py` table in README):
SOL SM% / Mem%, bytes-per-sector on `fa_s1_qk`, `dram__bytes.sum` on `fa_s1_softmax`, top-3 stall
reasons, achieved occupancy, registers (ptxas sm_75: qk 34, softmax 42, pv 30; 0 spills).

**Bench delta.** TBD — `bench/results/t4/stage1.csv` (median of ≥100 timed iterations after warm-up,
256 MB L2 flush per iteration, clocks locked to 585 MHz); expected to lose to SDPA
`EFFICIENT_ATTENTION` by a wide margin at every N, and to hit "OOM by design" for the fp32 N×N
intermediates at B=8, H=32, N=4096 (17.2 GB of S alone on a 15 GB T4).

**Decision.** Keep as the baseline whatever the numbers are; the value of S1 is the metric table,
not the time.

**Known limitations.** Materialises S and P; K read at stride D; PV sums over all N keys even under
causal (the masked P entries are exactly 0, so `flops=full` accounting stays honest); bf16 not
offered (contract: Naive is fp32/fp16).

**Go / no-go for S2.** Go once the fp64 ratio-rule tests pass for fp32 and fp16 at D = 64 and 128,
causal and non-causal, and the three profiles are committed.

**5-line summary.**
1. Three kernels, one thread per element, everything through global memory — the way a first port
   looks, so every later stage is a one-variable change against it.
2. The K loads are strided by D on purpose: adjacent threads own adjacent keys, so a warp touches 32
   sectors and uses 4 bytes of each; that is what "uncoalesced" means and it shows up as
   bytes-per-sector, not as an error.
3. The N×N score and probability matrices round-trip HBM, so `dram__bytes.sum` grows quadratically in
   the sequence length while the tensors themselves are linear in it — the number FlashAttention's
   IO argument is about.
4. Correctness is a differential test against an fp64 oracle with the FlashAttention ratio rule,
   plus a CPU emulation of the index math, so the T4 run only had to catch layout and sync bugs.
5. It is a baseline, not a product: its job is to make the S2 and S3 deltas measurable.

**Two questions.**

*Q1: Why is the thread-per-(i, j) mapping uncoalesced when each thread reads a whole K row?*
Coalescing is a warp-level property of one load instruction, not of one thread's loop. At iteration
d every lane loads `K[j_lane][d]`; the 32 lanes have consecutive j, so their addresses are D·sizeof(T)
apart — 32 different 32-byte sectors for 128 requested bytes. The next iteration touches the same 32
sectors one element further along, which is why L1 hit rate is high but the sector traffic is 8×
(fp32) / 16× (fp16) what a walk along d would cost. S2 fixes it by making consecutive lanes walk d.

*Q2: Where does S1's time actually go, and what would a roofline say?*
Different kernels, different regimes. `fa_s1_qk` is latency/L1-bound on the strided K pattern;
`fa_s1_softmax` is pure bandwidth: 2–5 N×4 bytes per row at L2 and DRAM level, no math worth
counting, far left on the roofline. `fa_s1_pv` is the interesting one: at the L1 level it is one
FFMA per 8 bytes (LSU-bound, `long_scoreboard` stalls), but at the DRAM level P is read once (one
sector broadcast per warp) and V is served from L2, so its DRAM intensity is ~D/2 FLOP/B — right of
the T4 ridge (~25 FLOP/B) — with low DRAM throughput. The lesson is that the roofline level has to be
named; I state both in the kernel headers and plot the DRAM point.

---

## S2 — shared-memory tiled GEMMs + coalesced loads   (branch `main`, commit be6b6c8, date 2026-09-03, GPU Tesla T4)

**Hypothesis.** Keep S and P materialised (so the S1→S2 delta isolates memory-access pattern from
fusion) but compute both GEMMs from shared-memory tiles with coalesced vector loads and 2-D register
blocking. `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct` on the QKᵀ kernel should
rise toward 100 %, `dram__bytes.sum` should drop by the tile-reuse factor, and L1/L2 hit rates should
rise. Shared-memory bank conflicts are expected and RECORDED here
(`l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`) as S5a's "before".

**ncu before.** S1's three profiles.

**Change.** `src/stage2_tiled.cu`: `fa_s2_qk_tiled` / `fa_s2_pv_tiled` — 64×64 output tile per
block, 256 threads as a 16×16 grid, 4×4 register tile per thread, reduction chunks of 16 through
`Qs[64][16]`, `Ks[64][16]` (8 KB static smem per block), `float4` (fp32) / 8-byte (fp16) loads with
consecutive threads walking the contiguous D axis, two `__syncthreads` per k-step (RAW after the
cooperative load, WAR before the next), scale and causal mask applied at store time, block-uniform
early exit for fully masked tiles. `fa_s2_softmax`: `__shfl_down_sync` warp trees plus one smem fold
(2 barriers per row versus S1's 4). `#pragma unroll 2` keeps qk/pv at 61 registers so 4 blocks of 256
threads fit per SM (32 warps); full unrolling under `__launch_bounds__(256, 4)` spilled and was
rejected (recorded in the file header).

**ncu after.** TBD — `profiles/t4/stage2_*.raw.csv`: bytes-per-sector, `dram__bytes.sum`,
`l1tex__t_sector_hit_rate.pct`, `lts__t_sector_hit_rate.pct`, bank conflicts (the S5a "before"),
occupancy, stalls.

**Bench delta.** TBD — `perf:` trailer from `bench/results/t4/stage2.csv` versus stage1 at
`B2_H8_N2048_D64` fp32.

**Decision.** TBD after measurement (expected: kept).

**Known limitations.** Still materialises S and P (the N² traffic remains — that is S3's job); one
tile configuration, no sweep; `Ks` column reads on a row-major tile are bank-conflicted by design.

**Go / no-go for S3.** Go once the coalescing metric moved as predicted and the tests pass.

**5-line summary.**
1. Same math and the same N×N intermediates as S1; only the way bytes reach the SM changed.
2. Each block loads a 64×16 slice of Q and K into shared memory with 16-byte vector loads where
   consecutive lanes are consecutive addresses, so a warp's request is four full 128-byte lines.
3. Every value that reaches a register feeds four FMAs (4×4 register tile), so the inner loop is
   8 shared loads per 32 FMAs instead of 2 loads per FMA.
4. The unroll factor was chosen by the ptxas register report, not by taste: unroll 2 keeps 61
   registers and four resident blocks; full unroll spills under a launch bound.
5. The bank conflicts on the K tile are left in deliberately — they are the measured "before" for the
   S5a padding change.

**Two questions.**

*Q1: What is a shared-memory bank conflict and where exactly is it in this kernel?*
Shared memory is 32 banks of 4 bytes; a warp's load is serialised into as many wavefronts as there
are distinct addresses in the same bank. In `fa_s2_qk_tiled` the inner loop reads `Ks[col][k]` with
`col` varying across lanes and the row stride 16 floats: 16 columns × 16 floats = addresses 64 bytes
apart, i.e. every second bank, so 16 lanes hit 8 banks two ways — a multi-way conflict on every
k-step. Padding the row to 20 floats (S5a) makes consecutive columns land in consecutive banks. The
counter is `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`.

*Q2: If S2 is faster, why not stop here and tune it?*
Because the ceiling is wrong. Tiling reduces how many times each Q/K element crosses the L1
boundary, but the N×N score matrix still crosses HBM twice (write, read) and P again, so the traffic
is Θ(N²) no matter how well the GEMMs run. The FlashAttention argument is that with on-chip
memory M you can make the HBM traffic Θ(N²d²/M) by never writing S; on a T4 with 64 KB per block
that is the only way the N=4096 configurations stop being memory-bound. S3 changes the algorithm's
IO complexity; S2 only changed constants.

---

## S3 — fused kernel, online softmax, FA2 loop order   (branch `main`, commit adf94d0, date 2026-09-03, GPU Tesla T4)

**Hypothesis.** One kernel per Q tile that streams K/V tiles through shared memory and keeps the
softmax statistics and the un-normalised output in registers never materialises S or P, so
`dram__bytes.sum` should collapse from O(B·H·N²) to O(B·H·N·D) (FA1 Theorem 2) and DRAM throughput
should stop being the limiter. The honest expectation is that fp32 thread-per-row S3 is NOT fast:
low `sm__throughput`, `mio_throttle` and `barrier` stalls and 1–2 blocks per SM are the recorded
motivation for S4 (tensor cores) and S5 (occupancy, shuffles).

**ncu before.** S2's profiles.

**Change.** `src/stage3_fused.cu`: grid `(ceil(N/64), B·H)`, 64 threads per block, thread-per-row;
running max `m` (raw score units), running sum `l` and `acc[D]` in registers; `alpha =
exp2((m − m_new)·scale·log2e)` (a multiply, never an inverse; 0 when `m = −inf`) rescales BOTH `l`
and `acc`; `p = exp2(s·c − m_new·c)` with the scale folded once into `c`; one divide per row after
the last tile; `lse = m·scale + ln(l)`. Q tile padded to D+1 floats so thread-per-row column reads
are conflict-free; K/V tiles unpadded because every lane reads the same element (broadcast
`float4`). Causal: fully masked K/V tiles are skipped by the loop bound, straddling tiles masked per
element with global indices (`j > i`). BC ∈ {16, 32, 64} selectable through `FA_FORCE_BC` for the
tile-invariance test; defaults 32 (D=64) and 16 (D=128); dynamic-smem opt-in above 48 KB.

**ncu after.** TBD — `profiles/t4/stage3_fa_s3_fused.raw.csv`: `dram__bytes.sum` (the headline),
`dram__throughput.avg.pct_of_peak_sustained_elapsed`, SOL SM%, top stalls, occupancy (expected
6–12 %: the padded Q tile is 16.6–33 KB), registers (ptxas sm_75: 146–255 depending on D and BC,
0 spills).

**Bench delta.** TBD — `perf:` trailer versus stage2 at `B2_H8_N2048_D64` fp32; and the first row
that is not OOM-by-design at B=8, H=32, N=4096.

**Decision.** TBD after measurement; kept even if slower than SDPA(EFFICIENT) at N=4096, because
the traffic collapse is the point of the stage and the speed comes from S4/S5.

**Known limitations.** Thread-per-row means `acc[D]` in registers and 2 warps per block: occupancy is
bound by the Q tile's shared memory; no tensor cores; `N % 32 == 0`; forward only.

**Go / no-go for S4.** Go once `dram__bytes.sum` is confirmed O(N) in the profile and the
tile-invariance and causal tests pass.

**5-line summary.**
1. The block owns a 64-row Q tile for its whole life and walks the keys in tiles; the score tile
   lives in registers, so nothing of size N×N ever leaves the SM.
2. The softmax is computed online: keep a running max and a running sum per row; when a new tile
   raises the max, multiply the old sum and the old accumulator by exp(old − new) — a factor in (0, 1]
   — and add the new tile's contributions.
3. Normalisation is deferred to one divide per row at the end (FA2), which also means the `l = 0` guard
   exists in exactly one place.
4. The measured effect is in DRAM bytes, not time: fp32 thread-per-row is register-heavy and
   low-occupancy, and the profile says so.
5. Everything the kernel does is mirrored by a ten-line Python model with a tile-invariance test, so
   the recurrence is defended on a whiteboard before it is defended in SASS.

**Two questions.**

*Q1: Derive the online-softmax rescale and explain why it is numerically safe.*
For a row split into tiles, softmax(x) = exp(x − m)/Σexp(x − m) for any m; using the running max
m_j of the tiles seen so far, the partial sum l_j = Σ_{k≤j} exp(x_k − m_j). When tile j+1 raises the
max to m_{j+1} ≥ m_j, l_{j+1} = exp(m_j − m_{j+1})·l_j + Σ exp(x_{k∈j+1} − m_{j+1}); the same factor
multiplies the un-normalised output accumulator because it is linear in the same exponentials. The
factor exp(m_j − m_{j+1}) is ≤ 1, so it can underflow to 0 harmlessly but never overflow, and each
exp argument is ≤ 0 so p ∈ [0, 1]. The remaining hazard is a row with no unmasked key (m = −inf):
exp(−inf − (−inf)) is NaN, so the factor is defined as 0 and the final 1/l is defined as 0 for l = 0.

*Q2: FlashAttention's theorem says HBM traffic is Θ(N²d²/M). What is M on a T4 and what does the
ratio actually buy you at d = 64 and d = 128?*
M is the on-chip memory a block can use for its tiles: 64 KB on Turing (48 KB without opt-in), i.e.
16 K fp32 elements. The N² term's coefficient is d²/M, so the reduction over the materialised
algorithm's Θ(N² + Nd) is about M/d² ≈ 4 at d = 64 and ≈ 1 at d = 128 in element units — which is
why the profile is the right artefact: on a small-SRAM GPU the win is real but not the 10–20× an
A100 gets, and fp16 tiles (S4) double the effective M.

---
## S4 — fp16 WMMA, fp32 accumulate   (TBD)
## S5a — bank-conflict-free shared memory   (TBD)
## S5b — warp-shuffle softmax   (TBD)
## S6 — torch op dispatch, GPT block, causal block skipping   (TBD)
