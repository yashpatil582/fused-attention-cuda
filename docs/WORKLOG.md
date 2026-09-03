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
