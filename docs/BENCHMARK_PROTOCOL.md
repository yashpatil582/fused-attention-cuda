# Benchmark protocol

This protocol is implemented identically in `bench/bench_main.cu` (the torch-free `fa_bench` binary)
and `bench/bench.py` (the PyTorch side). A timing that was not produced under every rule below is not
a result and must not reach a CSV. Rules are numbered so `docs/STAGES.md` and code comments can cite
them ("per BENCHMARK_PROTOCOL rule 2").

## Rules

1. **Warmup and repetitions.** At least 10 untimed warmup iterations, then at least 100 timed
   iterations (or as many as fit in >= 500 ms of timed work, whichever is larger; the count is
   sized from three launches taken *after* the warmup, never from the first launch, which carries
   lazy module loading). One `cudaEvent` pair is recorded **per iteration**; a single device
   synchronize follows the loop; the per-iteration times are sorted and reported as `ms_median`,
   `ms_p10`, `ms_p90`; the row states `iters=<n>;warmup=<n>` and `tests/test_bench_sanity.py`
   rejects a kernel row with fewer than 100 iterations or with `p90/p10 >= 1.15` and no `noisy`
   flag (`fa_common.NOISE_RATIO`, the same constant in both harnesses). Why: kernel launches are
   asynchronous, so a host timer without a sync measures launch overhead (CUDA Best Practices
   9.1.1); events give ~0.5 us resolution on the GPU clock (9.1.2); the median rejects clock and
   scheduling noise that a mean averages in.
   *Stated exception:* `bench/gpt_block.py` is a warm end-to-end measurement of a whole
   transformer block, not a kernel benchmark: 50 iterations, no L2 flush, no rotating inputs, one
   outer event pair per iteration plus one inner pair per attention call, all read after a single
   synchronize. Its rows say `warm;no_flush` and live in `gpt_block.csv` only.
2. **L2 flush.** A 256 MB buffer of `int32` (`cudaMalloc` in C++;
   `torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int, device="cuda")` in Python) is zeroed before
   **every** timed iteration. This is Triton's `do_bench` convention: its driver keeps "a buffer of
   256 MB that we clear before each kernel call to make sure that the L2 cache doesn't contain any
   input data before the run" (`third_party/nvidia/backend/driver.py`, `clear_cache = cache.zero_()`).
   The buffer is allocated once, outside the loop. Inputs rotate over 4 pre-allocated buffers so the
   same Q/K/V bytes are not L2-resident from the previous iteration. Why: at small N the whole K/V fits
   in L2 and a warm-cache loop reports fictitious bandwidth.
3. **Anti-DCE and correctness coupling.** After the loop the output is reduced into a `volatile`
   checksum sink (C++) or `.sum().item()` (Python). A bench row is invalid unless the same process
   passed the correctness check of `docs/NUMERICS.md` on the same shapes first; `bench.py` refuses to
   write a row otherwise.
4. **Clocks.** `nvidia-smi -lgc` WORKS on Colab as root (PROBE 2, 2026-09-02: `-lgc 585,585`
   accepted, `-rgc` works). `scripts/gpu_run.sh` therefore locks the SM clock before the bench step
   to the driver's default application clock (`clocks.default_applications.graphics`, the base
   clock `ncu --clock-control base` also uses; 585 MHz on T4; read from the driver, never typed),
   writes `clock_lock=<mhz>` to `env.txt`, resets with `-rgc` in its exit trap, and passes
   `--clock-locked <mhz>` to `bench.py` / `gpt_block.py`. The harnesses do not trust the flag: the
   SM clock is sampled *during* every timed loop (`bench.py`: a background thread on
   `pynvml.nvmlDeviceGetClockInfo`, nvidia-smi as the fallback; `fa_bench`: NVML through dlopen
   at four points of the loop), `clock_mhz` is the median sample, `clock=<min>-<max>` records the
   spread, `noisy_clock` flags a spread above 10%, and a lock whose samples disagree with the
   requested value is dropped for that row (`clock=unlocked;clock_lock_failed=<mhz>`). A verified
   lock is `clock_lock=<mhz>` in notes. When the lock is refused (no root, another platform) the rows
   say `clock=unlocked` and the p10/p90 columns carry the resulting spread. Locking to base means
   bench rows and ncu profiles are at the same clock and their FLOP/s are directly comparable; an
   unlocked (boost) run may be added as a second, labelled row, never mixed into the same cell. On
   the VM (Track B) the same script does the same. ncu runs use `--clock-control base` (or `none`
   if refused, recorded in `env.txt` and the stage `.meta`).
5. **FLOP convention.** `FLOPs = 4 * B * H * N^2 * D` for the forward pass (two GEMMs of
   `2 * N * N * D` each), the same formula as flash-attention's
   `benchmarks/benchmark_flash_attention.py`
   (`f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)`). A causal row of a kernel
   that skips the fully masked K/V tiles (the fused stages S3+, whose loop bound is
   `ceil(min(i0+BR, N)/BC)`, and the SDPA/flash baselines) uses the **halved** count and says
   `notes=flops=causal_half`; a causal row of a materialising kernel (S1, S2, `torch_unfused`, the
   MATH oracle) keeps the **full** count and says `flops=full`, because it really computes every
   block. The halved count is conservative: a tile-skipping kernel actually executes the fraction
   `(n + 1) / (2n)` of the full work with `n = ceil(N / BC)` (the diagonal tiles are computed and
   masked), i.e. 3 % more than half at N = 2048, BC = 64, so the reported TFLOP/s never flatters
   the kernel. Never halve the numerator for a kernel that still computes every block: that would
   fabricate a ~2x. `tflops = FLOPs / (ms_median * 1e-3) / 1e12`.
6. **Bandwidth convention.** `gbps` = **modelled minimum bytes** / time, not measured traffic.
   Fused stages (S3+): Q + K + V + O + LSE = `4 * B * H * N * D * sizeof(dtype) + 4 * B * H * N`.
   Materialising impls add the minimum S and P traffic: write S, read S, write P, read P, each
   `B * H * N^2 * sizeof(intermediate)` (fp32 for our naive/tiled workspace and the MATH oracle,
   the input dtype for `torch_unfused`). The model undercounts impls that make extra N x N
   passes (torch's unfused path scales and masks S in separate kernels; the S1 three-pass softmax
   reads S more than once), so `gbps` is comparable *within* an impl across shapes, not between
   impls; the measured bytes are `dram__bytes.sum` in the NCU table. Rows whose N x N
   intermediates (S and P together) would exceed 25% of device memory (`cudaDeviceProp.totalGlobalMem`
   in `fa_bench`, `torch.cuda.get_device_properties().total_memory` in `bench.py`, never typed and
   never `free` memory, so both harnesses classify a shape identically) are emitted with
   `notes=oom_by_design` and no timing; e.g. B=8, H=32, N=4096 fp32 S alone is
   `8 * 32 * 4096^2 * 4 B ~ 17.2 GB` (derived), which no 16 GB card holds.
7. **Baselines on every sweep.** (a) `F.scaled_dot_product_attention` pinned with
   `sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION])`: the headline denominator on T4, because FLASH
   refuses `sm < 80` ("Flash attention only supports gpu architectures in the range [sm80, sm121]",
   `sdp_utils.cpp` v2.14.0) and cuDNN attention has the same sm80 floor. The refusal string is
   captured into `bench/results/<gpu>/env.txt` as evidence. (b) SDPA `MATH` in fp32, labelled
   "oracle, not a speed baseline": it upcasts half inputs to float, so beating it proves nothing.
   (c) Unfused torch `softmax(Q K^T * s) @ V`. On Track B add (d) `SDPBackend.FLASH_ATTENTION`, the
   headline there, and (e) `flash_attn_func` only if a release wheel matches the box's torch (assets
   stop at torch2.9). Every forced backend is wrapped in `try/except RuntimeError`; a refused backend
   yields a row with `notes=backend_refused` and the message, never a silent fallback. fp32 baselines
   pin `torch.backends.cuda.matmul.fp32_precision = "ieee"` (the `allow_tf32` booleans are
   deprecated in 2.14) so TF32 cannot leak into an "fp32" row.
8. **Provenance.** Every row carries `commit` (source SHA under test; `-dirty` is appended when a
   tracked source file differs from HEAD, `bench/results` and `profiles` excluded, and a `-dirty`
   row is rejected by the sanity test so an rsync'd uncommitted tree cannot name a clean SHA),
   `gpu` (nvidia-smi name verbatim), `driver`, `nvcc` (full `V12.8.93` form in both harnesses),
   `torch` (blank in `fa_bench` rows by design: it is torch-free), `ncu` (`ncu --version`),
   `clock_mhz` (median of the in-loop samples, rule 4) and a `run=<utc>` stamp in `notes` that
   `bench_to_md.py` uses to pick the newest measurement per (impl, shape) across stage files, so a
   baseline re-measured by a later stage never silently shadows a newer one; when the SDPA row the
   README divides by comes from a different commit than the kernel row, the "% of SDPA" cell says
   so. Inputs differ between the harnesses on purpose and are stated: `fa_bench` fills uniform
   `[-1, 1)` (hash-based, no curand), `bench.py` uses `torch.randn`; both put the softmax in the
   O(1)-score regime and both gates use the same fp64 oracle. README tables are regenerated by
   `scripts/bench_to_md.py`; `tests/test_docs_cpu.py` fails when the README is stale;
   `tests/test_bench_sanity.py` checks p90/p10 spreads, `iters>=100`, that `tflops` recomputes
   from `ms_median` and the shape, that our kernels' rows carry a `check=`/`max_err=` note, and
   that no `tflops` exceeds the recorded roof in `docs/ENV.md` (it warns while the roof is `TBD`).
9. **Timing source.** Only CUDA-event medians appear in tables. Nsight Compute durations are never
   used for speedups: ncu serialises kernels, flushes caches between passes and replays, so every
   number it reports is for an isolated kernel. ncu answers "why"; events answer "how fast".
10. **Reproducibility spot-check.** Once per GPU day the runner re-runs a tagged commit with
    `MODE=full` and no push; medians must land inside the committed p10/p90 band. Drift is written to
    `docs/WORKLOG.md`, never overwritten.

## CSV schema (frozen, CONTRACT section 3)

```
commit,gpu,driver,nvcc,torch,ncu,clock_mhz,impl,stage,dtype,B,H,N,D,causal,ms_median,ms_p10,ms_p90,tflops,gbps,notes
```

- `impl` in {naive, tiled, fused, wmma, tuned, sdpa_efficient, sdpa_flash, sdpa_math_fp32,
  sdpa_cudnn, torch_unfused, flash_attn}, plus `smoke` (stage 0, the S0 `fa_bench --smoke` round
  trip, never rendered) and `gpt_sdpa_efficient` / `gpt_sdpa_flash` / `gpt_fused` in
  `gpt_block.csv` only; `stage` 1..5 for our kernels, 0 for baselines; `dtype` in {fp32, fp16,
  bf16}; `causal` in {0, 1}.
- `notes` free text: `oom_by_design`, `flops=full`, `clock=unlocked`, `clock_lock=<mhz>`,
  `clock=<min>-<max>`, `noisy`, `noisy_clock`, `iters=<n>`, `warmup=<n>`, `check=rows<n>`,
  `run=<utc>`, `degraded_profiling`, `backend_refused:<msg>`, `warm;no_flush` (GPT rows).
- Files: `bench/results/<gpu>/stageN.csv`, `summary.csv`, `gpt_block.csv`; `<gpu>` directory name is
  lowercase (`t4`, `a10g`, `a10`, `a100`, `4090`).

## Canonical configs (CONTRACT section 5)

- `C64 = B2_H8_N2048_D64`, `C128 = B2_H8_N2048_D128`, causal 0 (and 1 from S6): the profiling
  configs, used at the same SHA as the bench row they explain.
- Sweep: N in {128, 256, 512, 1024, 2048, 4096} x D in {64, 128} x (B, H) in {(1, 8), (8, 32)} x
  causal in {0, 1}. Config strings `B8_H8_N4096_D64` are accepted by `fa_bench --cfg` and
  `bench.py --cfg`.
- GPT block (`bench/gpt_block.py`): nanoGPT-shaped block, `n_embd 768`, 12 heads, `T 1024`, fp16,
  tokens/sec with SDPA vs `torch.ops.fused_attn.forward`, plus the attention share of block time.

## What "% of SDPA" means in this repo

On T4 it means "% of `SDPBackend.EFFICIENT_ATTENTION` at the same shape, dtype and causal flag,
measured in the same process under rules 1-4". It never means % of FlashAttention-2, which does not
run on Turing (upstream flash-attention supports "Ampere, Ada, or Hopper GPUs"). Rows from an sm_80+
machine say so in `gpu` and use FLASH as their denominator. A stage that loses to the baseline at
some N is reported at that N with the loss visible; the sweep is never trimmed to the winning shapes.
