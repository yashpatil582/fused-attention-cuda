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
PROBE 7 (macOS ncu GUI): all results TBD, recorded in `docs/STAGES.md` S0 and `docs/ENV.md`.

**Open risks carried into S1.** Colab may hand out a blocked instance or no T4; the S4 WMMA overrun
(opaque fragments, `ldm` alignment, 64 KB cap); Colab nvcc/torch versions differing from the
July-2026 evidence (12.8); Claude Max usage windows during long autonomous runs.

**Bugs / dead ends.** None yet: nothing has compiled on a GPU.

## Spend log (Track B)

| Date | Provider | Instance | Hours | USD | Purpose |
|---|---|---|---|---|---|
| TBD | TBD | TBD | TBD | TBD | TBD |

Running total: TBD (cap ~$25, <= 12 GPU-hours in-window).

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
