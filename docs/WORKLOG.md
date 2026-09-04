# Worklog

Dated entries in the siboehm / sonnyli style: decisions, bugs, dead ends and kept regressions, written
as they happen. Numbers appear here only after they exist in a committed CSV or profile; until then
the entry says what was decided and why. Spend on rented GPUs is logged at the bottom.

## 2026-09-02 — S0: probes and scaffold

**Context.** Application deadline 2026-09-04 EOD. Mentor-approved project: a FlashAttention-style
fused attention forward kernel in CUDA C++, optimised in stages, each stage justified by a committed
Nsight Compute metric, benchmarked against PyTorch SDPA, exposed as a PyTorch custom op. The Mac
(M2 Air) has no NVIDIA GPU and no nvcc, so every compile/run/profile is remote.

**GPU-route research outcome (27-agent research workflow, six platforms fact-checked adversarially;
details and URLs in `docs/RESEARCH.md` section 8).**
- Colab free T4 is the guaranteed track: $0, no card, and the only free platform with independent 2026
  evidence that `ncu` hardware counters work (committed logs dated 2026-07-15 and 2026-08-23). Access
  is per-instance: one Aug-2026 `ERR_NVGPUCTRPERM` report exists, and "factory-reset the runtime and
  retry" is the documented remedy. So PROBE 2 runs at the start of every session, resets up to 3x,
  then falls back to degraded profiling with the reason recorded.
- Kaggle counters are affirmatively blocked (`ERR_NVGPUCTRPERM` committed for both T4 and P100).
  Modal needs a payment method and runs under gVisor with the profiling capability off by default.
  Lightning's personal-email verification can take 2-3 business days. Brev needs a card and has no
  free credits. RunPod/Vast Docker pods cannot add `SYS_ADMIN`. Thunder Compute documents "GPU
  profilers" as incompatible. None of these is a primary route in-window.
- Track B (optional, <= ~$25): one real sm_80+ VM with sudo, gated by a 15-minute `sudo ncu` probe
  before any project code. Order: AWS G-instance quota probe, then Lambda A10/A100, then Vast.ai
  VM-launch mode. Terminate whenever idle > 1 h.
- Transport: outbound git only from Colab (a fine-grained PAT, single repo, Contents RW, 7-day
  expiry, stored in Colab Secrets, revoked on Sep 4). No SSH/ngrok tunnels, ever (the Colab FAQ
  disallows "remote control such as SSH shells" on free runtimes).
- Claude Code stays on the Mac and drives Colab through the notebook's runner cell and Track B
  through `ssh gpu`; it is never installed on a rented box.

**Design decisions frozen today.**
- `include/fa/attention.cuh` is frozen: `fa::forward(Impl, Dtype, q, k, v, o, lse, workspace,
  Params, stream)`, layouts `[B, H, N, D]`, `lse [B, H, N]` fp32, top-left causal, `N % 32 == 0`.
- CSV schema, kernel names, op schemas, canonical configs and README generated blocks are frozen in
  `CONTRACT.md`; the S0 scaffold was built by six parallel owners against that contract.
- Speed baseline on T4 is SDPA `EFFICIENT_ATTENTION`, named on every table, because FLASH refuses
  `sm < 80` and cuDNN fused SDPA needs SM80+. "No vendor FlashAttention kernel exists for T4" is the
  honest reason the repo exists.
- fp16 WMMA is the only tensor-core path that runs on T4; bf16 and `cp.async` compile under
  `#if __CUDA_ARCH__ >= 800` in CI and are labelled compile-only until an sm_80+ run.
- S4 is correctness-first WMMA with the S tile round-tripping padded shared memory (fragment layout is
  documented as opaque); no swizzle / `ldmatrix` / `mma.sync` in-window; hard cut line 12:00 Sep 4.
- Tolerance is the flash-attention ratio rule against an fp64 oracle, not `assert_close` defaults
  (`docs/NUMERICS.md`).
- The CPU model of the tiled online-softmax recurrence (`python/fused_attn/reference.py`) is tested
  on the Mac before any CUDA is written, so GPU round trips are spent on indexing and sync bugs.

**Scaffolded in S0 (no GPU has run any of it yet).** Repo, LICENSE (MIT), `.gitignore`, CMake with
presets (`colab` = 75-real, `vm` = native, `ci` = five real archs + PTX), frozen API header,
`cuda_check.cuh`, `tile_config.h`, `softmax_math.cuh`, `dispatch.cu` + `smoke.cu` stub kernel, the
fp64 CPU oracle, torch-free `fa_bench`, `bench.py`, `gpt_block.py`, `configs.yaml`, the PyTorch
extension with `TORCH_LIBRARY` + `register_fake`, `reference.py` + CPU tests, `scripts/`
(`gpu_run.sh`, `gate_gpu.sh`, `profile.sh`, `ncu_metrics.txt`, `bench_to_md.py`, `ncu_to_md.py`,
`roofline.py`, `perf_trailer.py`, `sync.sh`, `local_compile.sh`), `colab/runner.ipynb`, CI
(`build.yml`, `lint.yml`), pre-commit, and these docs.

**Probes.** PROBE L (local arm64 nvcc via colima), PROBE 1 (Colab env), PROBE 2 (Colab ncu),
PROBE 3 (AWS quota), PROBE 4 (VM ncu), PROBE 5 (permission prompts), PROBE 6 (CI install time),
PROBE 7 (macOS ncu GUI): results recorded in `docs/STAGES.md` S0 and `docs/ENV.md` as they came
in (PROBE L, 1, 2, 5, 6 done; PROBE 3 failed on invalid AWS credentials; PROBE 4 and 7 not run —
no VM, no host GUI needed for the CLI workflow).

**Open risks carried into S1.** Colab may hand out a blocked instance or no T4; the S4 WMMA overrun
(opaque fragments, `ldm` alignment, 64 KB cap); Colab nvcc/torch versions differing from the
July-2026 evidence (12.8); Claude Max usage windows during long autonomous runs.

**Bugs / dead ends.** None yet: nothing has compiled on a GPU.

## Spend log (Track B)

| Date | Provider | Instance | Hours | USD | Purpose |
|---|---|---|---|---|---|
| — | none | none | 0 | 0.00 | Track B never started: the AWS quota probe failed (invalid credentials) and no Lambda/Vast account was opened; every number in the repo is from the free Colab T4 |

Running total: **$0.00** (cap ~$25). Colab T4 time used: ~5 GPU-hours across 2026-09-03.

### 2026-09-03 00:35 UTC — S0 round trip on Colab T4

- `gpu_run.sh --stage 0 --gpu t4 --mode quick` at 023cfb8: build 9 s, smoke ctest green, sanitizer clean, ncu PASS,
  torch extension built (torch 2.11.0+cu128), GPU pytest 8 passed / 8 skipped. Whole cell ~70 s.
- CI on the first push: the `feat(s0)` runs were cancelled by concurrency when the `docs(env)` push landed 50 s later;
  the second pair is the one that counts (result recorded in STAGES.md PROBE 6 once finished).
- Colab quirks worth remembering: the runtime idles out in ~30 min between cells ("Runtime disconnected" dialog →
  Reconnect); coordinate clicks from Claude-in-Chrome are unreliable, element refs and one-line cells are not.

### 2026-09-03 05:20–05:50 UTC — S1 validated on the T4; S2 run interrupted; sanitizer resized

- S1 at be6b6c8: 47 GPU tests passed + 2 xpassed (GQA already works through `kv_head = h/(H/Hkv)`),
  memcheck/racecheck clean, clocks locked to 585 MHz, 8 bench rows, 6 Nsight captures (32 metrics).
  Naive fp32 at C64: 410.6 ms vs SDPA(EFFICIENT) 11.6 ms; C128: 817.1 ms vs 20.9 ms.
- Lesson: the compute-sanitizer gate ran `fa_bench --smoke --stage N` at C64 (B2 H8 N2048). memcheck
  on the tiled stage took ~1 s per iteration, racecheck (shared-memory instrumentation) far longer —
  the stage-2 run sat in racecheck for ~15 min before I interrupted it. Fixed in f686962: the sanitizer
  now runs B1_H2_N288_D64 and _D128 (multi-tile, partial last tile for BC=64, both head dims).
- Also: gpu_run.sh now tees its whole output to build/gpu_run_stage<N>.log (a deleted notebook cell
  cost the S1 summary once; Colab's ctrl+m z restored it).
- Still no GH_PAT in Colab Secrets: results are read from the notebook; the runner push waits.

### 2026-09-03 05:50–06:05 UTC — S2, S3, S4 validated on the T4 in one 9-minute batch

- All three stages: GPU tests green (S2 47 passed; S3 75 passed incl. tile invariance and LSE; S4 47 passed
  + 31 fp32 cases skipped by design), compute-sanitizer clean on the N=288 shapes, clocks locked to 585 MHz,
  Nsight captures written (6 kernels for S2, 2 for S3, 2 for S4).
- Canonical-config medians from that batch (fp32 for S1–S3, fp16 for S4; SDPA = EFFICIENT_ATTENTION backend):
  C64 = B2 H8 N2048 D64: naive 410.6 ms → tiled 59.2 ms → fused 25.1 ms → wmma 3.94 ms (SDPA fp32 11.6 ms,
  SDPA fp16 2.49 ms); C128: 817.1 → 115.8 → 94.3 → 11.17 ms (SDPA fp32 20.9 ms, fp16 5.00 ms).
  S4 = 63 % of SDPA fp16 throughput at D=64, 45 % at D=128; fp32 S3 loses to SDPA as predicted in STAGES.md.
- These rows were stamped `f686962-dirty` because a docs generator had run inside the Colab checkout before
  the stage runs; `test_bench_sanity` rejects dirty commits, so the batch is being re-run from a clean tree at
  3add37c (canonical configs + the 48-config sweep for every stage). Only the clean rows will be committed.

### 2026-09-03 06:00–08:00 UTC — clean S1–S4 re-run at 3add37c (canonical + 48-config sweep + Nsight)

- Every stage re-benchmarked from a clean checkout so no row is `-dirty`: 8 + 8 + 8 + 8 canonical rows, the
  full CONTRACT §5 sweep per stage (causal 0 and 1), 16 Nsight captures at `--clock-control base`.
  Two findings only the sweep shows: (a) S3 fp32 at D=128 is barely better than S2 (94.3 vs 115.8 ms) because
  the thread-per-row kernel runs 2 warps per SM (`sm__warps_active` 6.25 %); (b) the causal rows are 1.73–1.87×
  faster than dense at the same shape in every fused stage — the loop-bound tile skip has been there since S3.
- `gpu_run.sh` died with exit 141 after the sweep (a `grep | head` under `pipefail` — SIGPIPE) with all rows
  already written; fixed in ba3292a (`|| true` on every `| head`). The sweep rows were kept.

### 2026-09-03 08:08–08:30 UTC — S5 at d835758: prefetch + block height, A/B, GPT block

- S5 = the S4 kernel with two template knobs (WARPS = block height, PREFETCH = register-staged K/V double
  buffer), 54 instantiations, 0 spills on sm_75. GPU tests green, sanitizer clean, 2 Nsight captures.
- Prefetch at S4's tiles: `long_scoreboard` stall 11.4 % → 0.85 % (C64) and 17.4 % → 0.41 % (C128), medians
  3.944 → 3.629 ms and 11.170 → 9.873 ms. `tune_s5.sh` A/B (22 variants): at D=64 nothing beats S4's tile
  with prefetch (WARPS=8 needs BC ≤ 32 and loses 5.5 %); at D=128 WARPS=4 × BC=16 wins 14.5 % over S4's tile
  with prefetch (8.453 vs 9.882 ms) → adopted as the default in 15b83a5.
- GPT block (nanoGPT shape, B8 H12 T1024 D64 fp16 causal): 703 K tok/s with our op vs 856 K with SDPA; our
  attention window (op + the three `.contiguous()` copies) is 4.12 ms vs 2.04 ms.

### 2026-09-03 08:30–09:15 UTC — evidence transport without a PAT; S5 default changed

- No `GH_PAT` in Colab Secrets, so `gpu_run.sh --push` could not run. Results (17 CSV/TXT files, 41 profile
  files, 1.7 MB) came out as an xz tarball → base64 in the notebook output (49,812 B, 23 lines of 3,000
  chars, read in two chunks by Claude in Chrome), decoded on the Mac with a sha256 check
  (`.research/transport/`), then committed as 05ee981. `.ncu-rep` binaries stay on the VM (not transportable
  this way; the raw CSVs carry every metric the docs quote).
- Data-hygiene bug found while committing: `stage_csvs()` globbed `stage5_tuning.csv`, so `summary.csv` and
  the README's S5 column showed the last A/B variant instead of the default. Fixed in the same commit.
- 15b83a5: S5 D=128 default → WARPS=4 × BC=16 (168 registers, 59,904 B smem, 0 spills); local nvcc build
  clean; S5 re-run started on the T4 at 09:12 UTC to bench and profile the new default.

### 2026-09-03 09:12–10:00 UTC — S5 re-run at 15b83a5 shows the tall D=128 tile is size-dependent; 371f406 picks by N

- 15b83a5 (WARPS=4 × BC=16 at D=128 everywhere): GPU tests 47 passed, sanitizer clean, 48-config sweep,
  2 Nsight captures (C128 kernel time 8.437 ms vs 9.887 ms with S4's tile), GPT block re-measured.
- Against the d835758 sweep (S4's tile + prefetch) the tall tile is −2.7…−21 % at N ≥ 2048 but +2.5…+46 %
  at N ≤ 1024 (B1_H8_N128: 0.062 vs 0.045 ms): half the grid, twice the K/V tiles, nothing for the prefetch
  to amortise. An A/B at one config would have shipped a regression for every short sequence.
- 371f406: `default_warps(D, N)` / `default_bc(D, N)` switch at `kTallTileMinN128 = 2048`; local nvcc build
  clean; final S5 run (sweep + GPT + canonical pair under the 585 MHz lock) started 09:45 UTC.
- Lesson recorded in STAGES.md S5 "Change 2b": `--sweep` replaces the canonical run in `gpu_run.sh`, so the
  canonical pair is benchmarked separately after it (same clock lock) — the README canonical row and the
  sweep rows must come from the same commit.

### 2026-09-04 00:24–00:45 UTC — Colab quota back: the 371f406 canonical pair and two re-captures land

- The 2026-09-03 09:45 UTC run at 371f406 was cut at the bench step by the session limit and the free
  GPU quota was exhausted afterwards. A 23:59 UTC retry got a T4 and reached the sweep step, but the
  runtime was gone by 00:24 UTC with nothing harvested (fresh VM on reconnect). Lesson applied
  immediately: measure the row that matters first and harvest after it, not once at the end.
- 00:28 UTC, fresh T4, `gpu_run.sh --stage 5 --gpu t4 --mode full` at 371f406 with the canonical pair
  first: build 1:50, ctest, compute-sanitizer clean, ncu PASS, pytest-gpu 47 passed / 31 skipped /
  1 xpassed, `bench.py` canonical rows under the 585 MHz lock, `profile.sh` 2 captures — 16 min end to
  end. Committed as c03a8a2.
- Rows (`stage5.csv` @ 371f406): C64 tuned **3.6374 ms** (4.72 TFLOP/s) vs SDPA EFFICIENT fp16
  2.4850 ms = 68 %; the d835758 run read 3.6286 / 2.4860 = 69 % (0.2 % spread; the rounded percent
  flips, the README now says 68 % because that is what the HEAD row divides to). C128 tuned
  **8.4527 ms** (4.07 TFLOP/s) vs 4.9972 ms = 59 %, matching the A/B's 8.4526 ms; the re-captured
  Nsight kernel time read 8.449 ms in this run (15b83a5: 8.437). The size-aware default is now backed
  by a canonical `bench.py` row, the A/B and a capture, all at the shipping commit.
- Transport without a PAT, cheaper than 2026-09-03: the harvest cell prints the xz+base64 blob, a page
  script selects that output element, ⌘C copies it into the Mac clipboard, `pbpaste` → `ingest.sh`
  (sha256 verified). Only works with the notebook editor blurred — with the cell in edit mode ⌘C copies
  the cell source instead of the selected output.
- 00:42–00:52 UTC: `--sweep --gpt` at 371f406 on the same VM (build reused): 176 sweep rows, GPT block
  703.3 K tok/s on our op vs 855.9 K on SDPA EFFICIENT (identical to 15b83a5), and the two captures
  re-taken once more — C128 8.439 ms, C64 3.639 ms are the committed files now, and every metric in
  the STAGES S5 "ncu after" table is within 0.2 % of the 15b83a5 capture it was written from.
  Committed as b8bc949. Every README sweep row is now the shipping kernel's own measurement.
- Causal skip at the size-aware tile (371f406 sweep): 1.76–1.86× at D=64, 1.73–1.81× at D=128 for
  N = 1024, 1.64–1.73× at D=128 for N ≥ 2048 (tall tile: half the query tiles, a smaller and more
  uneven causal grid at B·H = 8 — hypothesis, not profiled). STAGES S6 / INTERVIEW carry the numbers.
- Colab: three runs on one T4 session (00:28–00:52 UTC), $0; the runtime was deleted afterwards.
