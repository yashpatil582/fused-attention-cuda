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

**Bench delta.** `bench/results/t4/stage1.csv` @ 3add37c (median of 100 timed iterations after
warm-up, 256 MB L2 flush per iteration, SM clock locked to 585 MHz): naive fp32 **410.60 ms** at
`B2_H8_N2048_D64` and **817.09 ms** at `B2_H8_N2048_D128` versus SDPA `EFFICIENT_ATTENTION`
11.61 ms / 20.87 ms and `torch_unfused` 12.16 ms / 17.84 ms — 35× and 39× slower than the vendor
kernel, 0.042 TFLOP/s. The B=8, H=32, N=4096 rows are "OOM by design" (fp32 S alone would be
17.2 GB on a 15 GB T4). The two canonical medians reproduced to within 0.01 ms across three runs
(be6b6c8 dirty, f686962 dirty, 3add37c clean) — the protocol is repeatable.

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

**Bench delta.** `perf: 410.6038ms -> 59.1586ms (6.94x) cfg=B2_H8_N2048_D64_fp32 gpu=Tesla-T4 stage=2`
and `817.0893ms -> 115.8372ms (7.05x)` at D=128 (`bench/results/t4/stage2.csv` @ 3add37c, 585 MHz).
Still 5.1× / 5.6× slower than SDPA `EFFICIENT_ATTENTION` (11.61 / 20.87 ms): the N² traffic is
untouched. At N=128 the tiled kernels lose to SDPA by 2.7× (0.199 vs 0.074 ms, B1 H8): a 64×64 tile
per block is too coarse for 8 heads × 2 tiles of work.

**Decision.** Kept: 7× from access pattern alone, with the algorithm and its N² intermediates unchanged.

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

**Bench delta.** `perf: 59.1586ms -> 25.1035ms (2.36x) cfg=B2_H8_N2048_D64_fp32 gpu=Tesla-T4 stage=3`;
at D=128 only `115.8372ms -> 94.3486ms (1.23x)` (`bench/results/t4/stage3.csv` @ 3add37c). Versus SDPA
`EFFICIENT_ATTENTION` fp32 the fused fp32 kernel is 2.16× slower at D=64 and 4.5× slower at D=128 —
the predicted loss: thread-per-row with `acc[128]` in registers runs 2 warps per block at 1 block/SM.
B=8, H=32, N=4096 now runs instead of being OOM by design (no N×N workspace).

**Decision.** Kept. The stage removes the N² intermediates (the workspace is 0 bytes and the largest
sweep shape runs); the fp32 thread-per-row mapping is deliberately simple and its low occupancy is
the recorded reason for S4.

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
## S4 — fp16 WMMA, fp32 accumulate   (branch `main`, commit 1bb0e1a, date 2026-09-03, GPU Tesla T4)

**Hypothesis.** S3 is bound by its FFMA stream: every FMA has a shared-memory operand and the
thread-per-row mapping caps occupancy. Moving both inner products onto the tensor cores
(`nvcuda::wmma` 16×16×16, fp16 operands, fp32 accumulate) should make
`sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active` go from 0 to clearly
non-zero, `sm__inst_executed_pipe_tensor.sum` > 0, and the FFMA count fall; the new top stalls
should be `barrier` and `mio_throttle` from the shared-memory round trips that the opaque fragment
layout forces — S5's targets.

**ncu before.** S3's profile (`profiles/t4/stage3_fa_s3_fused*.raw.csv`).

**Change.** `src/stage4_wmma.cu` (`fa_s4_wmma<T, D, BC>`): block = BR rows = 16·warps (BR 64 / 4
warps at D=64, BR 32 / 2 warps at D=128 because the fp32 O staging strip is D wide); warp w owns
rows 16w..16w+15 for everything (FA2's partition), so the softmax bookkeeping never crosses warps.
Q fragments (`matrix_a row_major`, stride D+8 halves) are loaded once per block and stay in
registers; K is read as `matrix_b col_major` so Kᵀ costs nothing; S is `store_matrix_sync`'d to a
padded fp32 strip because the fragment→(row, col) mapping is opaque; the two lanes of a row do the
online-softmax update on halves of the strip and combine with `__shfl_xor_sync(…, 1)`; P is written
as packed `__half2` into the dead Q tile and re-loaded as `matrix_a`; O accumulators live in
registers and are rescaled through a second round trip that a warp vote skips whenever every row's
alpha is exactly 1 (the running max stabilised). Shared memory 31–60 KB per block (padding +8 halves /
+4 floats keeps every fragment pointer 256-bit aligned and `ldm` a multiple of 16 B); 0 spills on
sm_75 (92–151 registers at D=64, 168–253 at D=128). bf16 compiles only for sm_80+ (trap stubs below).

**ncu after.** TBD — `profiles/t4/stage4_fa_s4_wmma*.raw.csv`: tensor-pipe utilisation, FFMA
count, stall breakdown, occupancy (1 block/SM = 4 warps at D=64/BC=64 is the honest T4 number).

**Bench delta.** `perf: 25.1035ms -> 3.9442ms (6.36x) cfg=B2_H8_N2048_D64 (fp32 S3 -> fp16 S4) gpu=Tesla-T4 stage=4`
and `94.3486ms -> 11.1697ms (8.45x)` at D=128 (`bench/results/t4/stage4.csv` @ 3add37c). Against SDPA
`EFFICIENT_ATTENTION` in fp16 (2.4863 ms / 5.0038 ms) the WMMA kernel reaches **63.0 %** of the vendor
throughput at D=64 (4.36 vs 6.91 TFLOP/s) and **44.8 %** at D=128 (3.08 vs 6.87 TFLOP/s). Whole
ladder at C64: 410.6 → 59.2 → 25.1 → 3.94 ms = 104× from naive.

**Decision.** Kept. First stage that is within 2× of the vendor kernel; the remaining gap is the two
shared-memory round trips and the single resident block per SM, which S5 measures knob by knob.

**Known limitations.** Two shared-memory round trips per K/V tile (S store/reload, O rescale);
one block per SM on T4 at the default tiles; no cp.async (Ampere+), no swizzle (S5a), no
`ldmatrix`/`mma.sync` (the register-resident path FA2 uses is week-after work).

**Go / no-go for S5.** Go once the tensor-pipe metric is non-zero and the fp16 ratio-rule tests pass.

**5-line summary.**
1. Both matmuls run on the tensor cores as 16×16×16 fp16 fragment products with fp32 accumulation;
   Q fragments are hoisted out of the K/V loop and Kᵀ is free by declaring K column-major.
2. WMMA's fragment layout is opaque, so the softmax cannot index the accumulator; the score tile is
   stored to shared memory, reduced there, and the probabilities go back in as an fp16 operand.
3. Each warp owns 16 rows end to end, which keeps the running max and sum warp-local and removes
   every cross-warp barrier from the softmax.
4. The output accumulator must be rescaled by alpha per row, which is another round trip — skipped
   by a warp vote once the max stops changing.
5. The tensor pipe metric is the proof this stage is real; the barrier and MIO stalls it introduces
   are what S5 is for.

**Two questions.**

*Q1: Why not read the row max straight out of the accumulator fragment?*
Because NVIDIA documents the fragment element mapping as opaque ("your program should not make
assumptions about it"); it differs across architectures and even compiler versions. The portable
answer is `store_matrix_sync` to shared memory and index there, which is what this stage does and
what makes the round trips show up in the profile. FlashAttention-2 avoids the round trips by using
`mma.sync` PTX directly, whose accumulator layout IS documented (lane `groupID = lane >> 2` owns rows
`groupID` and `groupID + 8`), so the softmax can be done on register fragments — that is the S5/week-after
path and the reason production kernels do not use WMMA.

*Q2: Why +8 halves of padding on every shared-memory row?*
Two reasons, one required and one measured. Required: `load_matrix_sync` needs `ldm` to be a multiple
of 8 halves (16 B) and fragment pointers to be 256-bit aligned; D+8 keeps every 16-row/16-column
fragment origin aligned because 16·(D+8)·2 B is a multiple of 32 B. Measured: a 64-element fp16 row
is exactly 128 B, so rows start in the same bank and column-wise fragment loads serialise; shifting
each row by 16 B rotates the bank of successive rows — the same idea as the `SKEW_HALF` in
cuda-samples' tensor-core GEMM. S5a records the conflict counter with and without it.

---
## S5 — Nsight-guided tuning: register-staged K/V prefetch + block height   (branch `main`, commits 3ca2d8a / fe02818 (kernel + A/B script), 15b83a5 → 371f406 (D=128 tile chosen by N), date 2026-09-03, GPU Tesla T4)

The plan's S5a (bank-conflict-free smem) and S5b (warp-shuffle softmax) were not built. S4's
profile put neither at the top: the +8-half padding had already taken the S2-style 32-way conflicts
off the table (4.9 M load conflicts against 47.7 M warp instructions at C64, values in
`docs/INTERVIEW.md` §4) and S4's softmax already reduces over the row pair with `__shfl_xor_sync`.
The two costs the profile did name are what S5 attacks.

**Hypothesis.** (1) `smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct` = 11.4 % at
C64 / 17.4 % at C128: S4 loads every K/V tile synchronously (LDG → STS → `__syncthreads`) and the
whole block waits on DRAM latency at the top of every tile with nothing to issue. `cp.async` is the
Ampere answer and does not exist on sm_75, so a register-staged double buffer — issue tile t+1's
16-B loads into a `uint4` array right after tile t's RAW barrier, run tile t's S GEMM / softmax /
P·V while they are in flight, store the registers to Ks/Vs after tile t's WAR barrier — should
drive that stall toward 0 at the price of `2·BC·D/threads` halves of registers per thread, with 0
spills as the gate. (2) `sm__warps_active.avg.pct_of_peak_sustained_active` = 12.5 % at C64 /
6.25 % at C128: shared memory admits one block per SM, so the only occupancy lever left on a
64 KB part is a taller block (more warps inside the one block), which also loads each K/V tile once
for more query rows. Both knobs are template parameters, `fa_s5_tuned<T, D, WARPS, BC, PREFETCH>`
(54 instantiations, every one its own `-Xptxas -v` line, `PREFETCH=false` at S4's WARPS reproduces
S4's register count exactly — the proof that the off path is S4's code). Keep rule: ≥ 3 % on the
canonical median with p10/p90 beside it, one knob per commit; a losing variant stays instantiated
(`FA_FORCE_BR`, `FA_FORCE_BC`, `FA_S5_PREFETCH=0`) but is not the default.

**ncu before.** S4 @ 3add37c (`profiles/t4/stage4_wmma*.raw.csv`, `--clock-control base`, 585 MHz):

| | C64 (BR 64 / 4 warps / BC 64) | C128 (BR 32 / 2 warps / BC 32) |
|---|---|---|
| kernel time | 3.957 ms | 11.105 ms |
| `long_scoreboard` stall | 11.38 % | 17.42 % |
| `mio_throttle` / `barrier` stall | 19.61 % / 7.13 % | 8.68 % / 7.54 % |
| tensor pipe active | 18.61 % | 13.36 % |
| `sm__warps_active` | 12.52 % | 6.25 % |
| registers / dynamic smem | 150 / 45,056 B | 168 / 43,008 B |
| `dram__bytes.sum` | 19.13 MB | 38.41 MB |

**Change 1 — PREFETCH, S4's tiles (kept).** `profiles/t4/stage5_tuned*.raw.csv` @ d835758:

| | C64 | C128 (S4's tile + prefetch) |
|---|---|---|
| kernel time | 3.638 ms (−8.1 %) | 9.887 ms (−11.0 %) |
| `long_scoreboard` stall | **0.85 %** | **0.41 %** |
| `mio_throttle` / `barrier` stall | 22.09 % / 6.08 % | 13.55 % / 7.44 % |
| tensor pipe active | 20.33 % | 14.87 % |
| `sm__warps_active` | 12.49 % | 6.25 % |
| registers | 168 (+18) | 254 (+86: at the ceiling, 0 spills) |
| `smsp__inst_executed.sum` | 49.3 M (+3.3 %) | 85.5 M (+1.7 %) |
| `dram__bytes.sum` | 19.25 MB | 38.37 MB |

Bench (`bench/results/t4/stage5.csv` @ d835758, 585 MHz): C64 3.9442 → **3.6286 ms** (−8.0 %),
C128 11.1697 → 9.8732 ms (−11.6 %). The stall moved exactly where the hypothesis said: the
DRAM-latency wait is gone and the time went to the shared-memory pipe (`mio_throttle` up) — the S4
round trips are now the top of the list. DRAM bytes are unchanged (same tiles, same traffic); the
instruction count grows by the staging moves only.

**Change 2 — block height, chosen by A/B.** `scripts/tune_s5.sh` →
`bench/results/t4/stage5_tuning.csv` @ d835758: 22 variants at the canonical configs, fp16,
585 MHz, ≥ 100 iterations each, ratio-rule check passing in every row. Medians in ms as
`PREFETCH=0 / PREFETCH=1`:

D=64 (S4 = 3.9442 ms):

| BC \ BR | 32 (2 warps) | 64 (4 warps) | 128 (8 warps) |
|---|---|---|---|
| 16 | 4.878 / 4.724 | 4.394 / 4.298 | 4.936 / 4.759 |
| 32 | 4.477 / 4.363 | 4.467 / 4.154 | 4.016 / 3.837 |
| 64 | 3.954 / 3.838 | 3.943 / **3.638** | — (71,680 B > 64 KB) |

D=128 (S4 = 11.1697 ms):

| BC \ BR | 32 (2 warps) | 64 (4 warps) |
|---|---|---|
| 16 | 13.372 / 12.094 | 8.995 / **8.453** |
| 32 | 11.164 / 9.882 | — (68,608 B > 64 KB) |

Reading. Prefetch wins in all 11 pairs (3–12 %). At D=64 the tallest block (8 warps, BR 128) can
only afford BC ≤ 32, and halving BC doubles the K/V tiles, the barriers and the S round trips per
query row; 8 resident warps instead of 4 do not pay for that (3.837 vs 3.638 ms), so S4's
BR 64 / BC 64 with prefetch stays the default (`kDefaultWarps64 = 4`). At D=128 S4's block was only
2 warps: BR 64 / BC 16 (4 warps, 59,904 B, opt-in) beats S4's tile with prefetch by **14.5 %**
(8.453 vs 9.882 ms) and S4 itself by 24.3 %, past the keep rule, so 15b83a5 made it the default —
at 168 registers instead of the 254 the S4 tile needed once prefetch was on.

**Change 2b — the sweep says the tall tile is a size trade-off (371f406).** The A/B ran at the
canonical N=2048 only. The 48-config sweep at 15b83a5 (tall tile everywhere) against the same
sweep at d835758 (S4's tile everywhere), both with prefetch, `bench/results/t4/stage5.csv`,
D=128, dense rows, ms:

| N | B1_H8: S4 tile → tall | B8_H32: S4 tile → tall |
|---|---|---|
| 128 | 0.0447 → 0.0620 (+39 %) | 0.938 → 1.047 (+12 %) |
| 256 | 0.1409 → 0.1577 (+12 %) | 3.328 → 3.652 (+10 %) |
| 512 | 0.4713 → 0.4937 (+5 %) | 11.99 → 12.28 (+2.5 %) |
| 1024 | 1.4845 → 1.5557 (+5 %) | 43.19 → 40.25 (−7 %) |
| 2048 | 4.954 → 4.532 (−8.5 %) | 154.7 → 132.2 (−14.5 %) |
| 4096 | 17.85 → 14.23 (−20 %) | 562.1 → 445.6 (−21 %) |

Causal rows move the same way (+29…+46 % at N ≤ 256, −2.7…−16 % at N ≥ 2048). Why: at small N the
tall block halves the grid (16 blocks < 40 SMs at B1_H8_N128) and BC 16 doubles the K/V tiles —
twice the barriers and S round trips per query row — with too few tiles for the prefetch to
amortise; from N = 2048 the grid is full and the halved per-row K/V traffic and 4 resident warps
win. The crossover is between 1024 and 2048 (mixed at 1024), so 371f406 picks the tile by N:
`kTallTileMinN128 = 2048` — WARPS 4 / BC 16 from N = 2048 up, S4's WARPS 2 / BC 32 below
(`default_warps(D, N)`, `default_bc(D, N)`; `FA_FORCE_BR/BC` override both). No autotuner: one
threshold, read off a committed sweep, and every row of the 371f406 sweep is the tile the rule
picks. S4's tile stays reachable at any N through `FA_FORCE_BR=32 FA_FORCE_BC=32`.

**ncu after (the D=128 tile 371f406 selects from N = 2048 up; captured at 15b83a5, re-captured at
371f406).** The committed `profiles/t4/stage5_tuned_C128.raw.csv` is the 371f406 re-capture
(2026-09-04 00:52 UTC); the table was written from the 15b83a5 capture and every metric in it
moved by less than 0.2 % between the two (kernel time 8.437 → 8.439 ms, warps active 12.51 →
12.49 %, `mio_throttle` 18.19 → 18.21 %). C64 is unchanged by the rule (3.639 ms, 0.85 %
`long_scoreboard`, 12.49 % warps active):

| C128 | S4's tile + prefetch (d835758) | tall tile (15b83a5) |
|---|---|---|
| kernel time | 9.887 ms | **8.437 ms** at 15b83a5, 8.439 ms at 371f406 (−14.6 %; −24.0 % vs S4's 11.105) |
| `sm__warps_active` | 6.25 % | **12.51 %** |
| `lts__t_bytes.sum` | 1,091 MB | **554 MB** (K/V re-reads per head halve with BR) |
| `dram__bytes.sum` | 38.37 MB | 38.28 MB |
| `long_scoreboard` stall | 0.41 % | 0.61 % |
| `barrier` / `mio_throttle` stall | 7.44 % / 13.55 % | 18.32 % / 18.19 % |
| tensor pipe active | 14.87 % | 17.42 % |
| registers / dynamic smem | 254 / 43,008 B | 168 / 59,904 B |
| `smsp__inst_executed.sum` | 85.5 M | 94.5 M (+10.5 %: twice the tiles) |

Reading: the win is L2 traffic (halved) plus four resident warps instead of two; the price is the
barrier share (four warps meet at twice as many barriers, 7.4 → 18.3 %), which is exactly what
turns into a loss when N is small and the tiles are few.

**Bench delta.** `bench/results/t4/stage5.csv`, 585 MHz. C64: S4 3.9442 → **3.6374 ms** at
371f406 (−7.8 %, 4.72 TFLOP/s, **68 %** of SDPA EFFICIENT fp16 at 2.4850 ms); the size rule does
not touch D=64, and the d835758 run of the same path read 3.6286 ms against 2.4860 (69 %) — a 0.2 %
run-to-run spread that happens to sit on the rounding boundary, so the README quotes what the HEAD
row divides to. C128 at the canonical config: S4 11.1697 → 9.8732 ms with S4's tile + prefetch
(the d835758 default row, −11.6 %) → **8.4527 ms** with the tile 371f406 selects there (the
371f406 `bench.py --cfg canonical` row, run 2026-09-04T00:33:58Z, −24.3 % vs S4; the same tile read
8.4526 ms in the `stage5_tuning.csv` A/B and 8.439 ms in the 371f406 Nsight capture) = **59 %** of
SDPA (4.9972 ms). That canonical row was the last open measurement of this entry: the 2026-09-03
run at 371f406 was cut at the bench step by the Colab session limit, the free GPU quota was then
exhausted, and the row landed on 2026-09-04 once a T4 came back (`docs/WORKLOG.md`). The full
48-config sweep and the GPT block were then re-run at 371f406 as well (b8bc949), so every sweep
row the README shows is the shipping kernel's own measurement, not a d835758/15b83a5 stand-in for
the tile the size rule would pick. Whole ladder at C64: 410.6 → 59.2 → 25.1 → 3.94 → 3.64 ms = **113×** from
naive; at C128: 817.1 → 115.8 → 94.3 → 11.17 → 8.45 ms = 97×.

**Decision.** Kept, both knobs, defaults as above. Rejected and kept instantiated: WARPS 8 at D=64
(−5.5 % against the default), WARPS 2 everywhere, every `PREFETCH=0` variant.

**Known limitations.** `mio_throttle` is now the top stall at both configs: the S store/reload and
the O rescale round trip of the WMMA design are the wall, and only the `mma.sync` + `ldmatrix`
rewrite (register-resident S with the documented lane map) removes them — out of window. One block
per SM at every default tile; the 22-variant grid is the whole search space a 64 KB part allows, no
autotuner. Prefetch holds 16–64 halves per thread across the tile body, which is why the D=128 S4
tile went to 254 registers with it and why WARPS=1 / BC=64 at D=64 is the one dropped variant
(ptxas 255 registers, 56 B spill stores; `kPrefetchOk<64, 1, 64> = false`).

**Go / no-go for S6.** Go: `torch.ops.fused_attn.forward` dispatches fp16 to S5 and the GPU op
tests pass at this commit.

**5-line summary.**
1. The S4 profile said DRAM latency (`long_scoreboard`) and a single resident block — not the
   tensor pipe — were the next costs; S5 attacks exactly those two, each as a template knob so
   every variant has its own ptxas line and its own profile.
2. Turing has no `cp.async`, so the double buffer is hand-rolled through registers: tile t+1's
   loads are issued before tile t's math and stored to shared memory after its last reader passed;
   the stall fell from 11–17 % to under 1 %.
3. Registers are the price (150 → 168, and 168 → 254 on S4's D=128 tile) and 0 spills is the gate;
   the A/B, not the reasoning, chose the defaults.
4. Taller blocks lose at D=64 because they force smaller K/V tiles (more barriers per row) and win
   at D=128 where S4's block was only 2 warps — but only once the grid is full (−14.5 % at N=2048,
   −20 % at 4096, +39 % at N=128), so the D=128 tile is chosen by N with one sweep-derived threshold.
5. What is left is `mio_throttle` — the WMMA round trips — which is the register-resident
   `mma.sync` design FlashAttention-2 uses and the stated next step.

**Two questions.**

*Q1: Why not double-buffer in shared memory like cuda-samples' pipelined GEMMs?*
The budget. A second K/V buffer at the default tiles is +22.5 KB (D=64) / +18 KB (D=128) on top of
45 KB / 43 KB — past the 64 KB Turing cap at D=64 and leaving no room for a taller block at
D=128; registers were the only free store. `cp.async` on Ampere gets both (no register cost, L1
bypass), which is why that path is the first item on the sm_80+ list rather than a Turing knob.

*Q2: The prefetch loads are issued in program order before the MMA section — how do you know
ptxas kept them there under register pressure?*
By the metric, not the source. If ptxas had sunk the LDGs toward their STS the overlap would vanish
and `long_scoreboard` would stay; it went to 0.4–0.85 %, and it is 0.41 % even at the 254-register
D=128 variant. `cuobjdump -sass` is the second check and would have been the first thing to open
had the metric not moved.

---

## S6 — torch op dispatch, GPT block, causal accounting   (branch `main`, commits 83cb449 (op, S0) → 15b83a5 (measurements), date 2026-09-03, GPU Tesla T4)

**What.** `python/fused_attn/ext.cpp` registers `fused_attn::forward(Tensor q, Tensor k, Tensor v,
bool is_causal, float scale) -> Tensor` and `forward_stage(..., int stage) -> (Tensor, Tensor)`
(O and LSE) with `TORCH_LIBRARY`, a `register_fake` meta kernel, and
`impl_supports(stage, dtype, D)`. `forward` picks the highest stage that
`fa::impl_supports(stage, dtype, D, sm)` accepts on the current device — fp16 at D 64/128 → S5,
fp32 → S3, bf16 → S5 on sm_80+ only — so the op is the ladder's best kernel for the dtype without
the caller naming a stage. Inputs must be contiguous `[B, H, N, D]` (16-B rows); the op never
copies, the caller sees the copy.

**Tests (`tests/test_op_gpu.py`, T4).** Per-stage ratio-rule accuracy against the fp64 oracle and
the LSE check through the op, `torch.library.opcheck` (schema, fake kernel, autograd-not-implemented),
`torch.compile(fullgraph=True)` with no graph break, and the dispatch table per (dtype, D). Green on
every stage run (`pytest-gpu` line of `build/gpu_run_stage<N>.log`: 15b83a5 and 371f406: 47 passed, 31 skipped — the bf16 cases, absent on sm_75 by design — 1 xpassed: GQA heads already work through `kv_head = h / (H / Hkv)`).

**GPT block (`bench/gpt_block.py` → `bench/results/t4/gpt_block.csv`, README "End-to-end GPT
block").** nanoGPT-style block, n_embd 768, 12 heads (D 64), T 1024, batch 8, fp16, causal,
inference, 50 timed iterations after warmup, clocks locked at 585 MHz, one CUDA-event pair around
the block and a second around the attention call. Measured twice: @ d835758 SDPA 855,987 tok/s (block 9.570 ms, attention 2.036 ms = 21.3 %) vs
our op 702,728 tok/s (11.657 ms, attention 4.116 ms = 35.3 %); @ 15b83a5 815,921 vs 703,077 tok/s
(10.040 / 11.652 ms; attention 2.148 / 4.118 ms). SDPA's block moved 5 % between runs, ours did
not (the D=64 path, untouched by the D=128 tile rule). The attention window around our
call includes the three `.contiguous()` copies the op needs after the head-split transpose (SDPA
consumes the strided views directly) — on purpose: that is the binding-boundary cost the number is
meant to expose. Honest reading: inside a real block our kernel costs about a fifth of the tokens/s
against the vendor kernel on this GPU, consistent with the standalone causal rows (S5 at
`B8_H32_N1024_D64` causal is 61 % of SDPA EFFICIENT, `B1_H8_N1024_D64` 77 %).

**Causal accounting.** Skipping fully-masked K/V tiles by loop bound has been in every fused stage
since S3 (`kv_end = causal ? min(i0 + BR, N) : N`); the diagonal tile is masked element-wise, the
interior runs with no mask instruction. Measured on the sweep (`stage5.csv` @ d835758): causal /
dense time at fixed shape = **1.73–1.87×** over N ∈ {1024, 2048, 4096}, D ∈ {64, 128}, B·H ∈
{8, 256} — FA2's "around 1.7–1.8×" reproduced, below 2× for the `(n+1)/(2n)` + diagonal-tile
reason. At the shipping kernel (`stage5.csv` @ 371f406, size-aware tile) the same ratio is
1.76–1.86× at D=64 and 1.73–1.81× at D=128 for N = 1024 (S4's tile), but 1.64–1.73× at D=128 for
N ≥ 2048 where the tall tile applies: it halves the number of query tiles, so a causal grid at
B·H = 8 is small and uneven (the last, longest tiles finish alone — a load-balance hypothesis, not
profiled; the rows are committed either way). Accounting per CONTRACT §5: causal rows of tile-skipping kernels are credited half of
`4·B·H·N²·D` (`flops=causal_half`, from a9681a6 on); rows benchmarked before that convention
(S1–S4 sweeps at 3add37c) still say `flops=full`; materialising kernels always keep the full count.

**Decision.** Kept as measured. Not done: a separate causal kernel variant (the last-tile masking
cost was not profiled on its own), GQA through the op (the kernels accept `Hkv`, the op exposes
`Nq == Nkv`, `H == Hkv` only).

**5-line summary.**
1. One op, one schema, dispatch by capability: the highest stage that supports (dtype, D, sm) runs,
   so the ladder's best kernel is what PyTorch users get without naming a stage.
2. `register_fake` + `opcheck` + a `fullgraph=True` compile test are the difference between "it
   runs" and "it composes with torch.compile"; all three pass on the T4.
3. The GPT-block number is measured with our binding-boundary copies inside the window, so it is
   the cost a user would pay, not the kernel's best case.
4. Causal skipping is a loop bound, not a branch per element; it gives 1.73–1.87× and is credited
   half the FLOPs only from the commit where the convention was written down.
5. The block still runs faster on SDPA than on our kernel on this GPU; the number is committed
   because the docs promise measurements, not wins.

**Two questions.**

*Q1: Why does `torch.compile` need a fake kernel for a custom op?*
Dynamo traces with FakeTensors: it has to know the output's shape/dtype/device without running the
CUDA kernel. Without `register_fake` the op is a graph break (the compiled region ends before the
call and restarts after it) and `fullgraph=True` raises. `opcheck` exercises exactly that contract —
schema, fake kernel, and that autograd is declared unimplemented rather than silently wrong.

*Q2: Why are the `.contiguous()` copies inside the timed window instead of excluded as "not the
kernel"?*
Because a user of the op pays them. SDPA's memory-efficient kernel takes the strided views nanoGPT
produces; ours needs `[B, H, N, D]` rows 16-B aligned and contiguous. Excluding the copies would
report a kernel that does not exist in the block. The fix — accepting strided inputs, or fusing
the head split — is the next step and would be measured by the same window shrinking.

---

## S7 — docs, roofline, evidence   (branch `main`, commits 05ee981 → HEAD, date 2026-09-03)

**What.** `docs/img/roofline_t4.png` from `scripts/roofline.py` (one point per stage and
canonical config; FMA and DRAM roofs from ncu's `.peak_sustained` counters at the 585 MHz profile
clock: 2.99 TFLOP/s fp32, 320 GB/s; fp16 tensor roof from the datasheet, labelled spec). Reading:
S1–S3 sit under the FMA roof (S1 at 1.4 % of it), S4/S5 cross it because their FLOPs run on the
tensor pipe, whose roof is ~22× higher; at AI ≈ 900 FLOP/byte every fused stage is compute-side
of the ridge, so the remaining gap to SDPA is issue and latency, not bandwidth. The FFMA-counter
cross-check of the analytic FLOP count holds for the fp32 stages (ratio 1.02–1.11) and is labelled
inapplicable for S4/S5 (HMMA FLOPs are invisible to it). Evidence: 5 stage CSVs + sweeps, 20 Nsight
raw captures with `.meta` provenance, sanitizer logs, the S5 A/B, the GPT block; README tables,
`docs/generated/ncu_stage*.md` and the figure are regenerated from those files and
`tests/test_docs_cpu.py` fails CI if they drift. Transport: no PAT on the runtime, so results left
the Colab VM as a sha256-verified xz/base64 blob (`docs/WORKLOG.md`); the `.ncu-rep` binaries
stayed on the VM.

**Known limitations.** No Nsight Systems timeline (missing on Colab); no sm_80+ row; the
`.ncu-rep` files are not in the repo, the raw CSVs are.

