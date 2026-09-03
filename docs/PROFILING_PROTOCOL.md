# Profiling protocol

Every optimization stage ships a committed Nsight Compute report for the canonical configs, taken at
the same commit as its benchmark row, with the raw metric CSV next to it so nobody needs the GUI to
check a number quoted in `docs/STAGES.md`. This file is the procedure; `scripts/profile.sh`
implements it and `scripts/ncu_metrics.txt` is the single source of truth for the metric list.

## 1. What gets profiled

- Configs: `C64 = B2_H8_N2048_D64` and `C128 = B2_H8_N2048_D128`, causal 0 (plus causal 1 from S6);
  fp32 for S1-S3, fp16 for S4+. Same SHA as the bench row.
- Binary: the torch-free `./build/bench/fa_bench --stage N --cfg C64 --iters 12`, so only our kernels
  appear and there is no PyTorch allocator, autotuner or dispatcher noise in the report.
- Kernels: one report per kernel name from CONTRACT section 2 (`fa_s1_qk`, `fa_s1_softmax`,
  `fa_s1_pv`, `fa_s2_*`, `fa_s3_fused`, `fa_s4_wmma`, `fa_s5_tuned`), selected with
  `--kernel-name regex:<name>`; the regex matches the mangled template prefix.
- Replay: kernel replay (the default). Application replay requires the application to be
  "deterministic with respect to its kernel activities and their assignment to GPUs, contexts,
  streams" (Nsight Compute Profiling Guide) and is never used under PyTorch.

## 2. Commands

Per kernel, per config (from `scripts/profile.sh`):

```
ncu --kernel-name regex:<kernel> --launch-skip 10 --launch-count 1 \
    --clock-control base --cache-control all \
    --section SpeedOfLight --section SpeedOfLight_RooflineChart \
    --section MemoryWorkloadAnalysis --section Occupancy --section LaunchStats \
    --section WarpStateStats --section SchedulerStats --section ComputeWorkloadAnalysis \
    --metrics $(paste -sd, scripts/ncu_metrics.txt) \
    -f -o profiles/<gpu>/stageN_<kernel> \
    ./build/bench/fa_bench --stage N --cfg C64 --iters 12
ncu --import profiles/<gpu>/stageN_<kernel>.ncu-rep --csv --page raw > profiles/<gpu>/stageN_<kernel>.raw.csv
```

- `--launch-skip 10 --launch-count 1`: `fa_bench --iters 12` makes 23 matching launches per kernel
  (1 pre-flight + 10 warmup + 12 timed; with `--iters` given the harness does not auto-size), so
  `--launch-skip 10` profiles the 11th, the last warmup launch, in steady state; without
  `--launch-count` all 23 would be replayed at full cost.
- `--clock-control base` (ncu's default, passed explicitly): GPC and memory clocks locked to base
  so two reports are comparable (recorded as `ncu_clock_control=none` in `env.txt` and the stage
  `.meta` if the runtime refuses it). **Every counter below is therefore at base clock (T4: 585 MHz
  SM).** Bench rows carry their own `clock_mhz`; `gpu_run.sh` locks them to the same base clock when
  `nvidia-smi -lgc` works (verified on Colab), otherwise they are boost/unlocked and their TFLOP/s
  is not directly comparable with the roofline's achieved FLOP/s: compare % of roof instead.
  `scripts/roofline.py` prints the profile clock in the figure title.
- Provenance: `profiles/<gpu>/stage<N>.meta` records commit (`FA_COMMIT`, exported by
  `gpu_run.sh` and stamped into the fa_bench rows), date, ncu version, clock control, dtype, whether
  `--print-units base` was accepted on import and the exact ncu command line per report;
  `scripts/ncu_to_md.py` prints the commit next to each README row and
  `tests/test_bench_sanity.py` fails when it differs from the stage's bench rows.
- `--cache-control all` (the default) flushes caches between replay passes; it makes L2 hit rates
  pessimistic versus a warm loop, which is the comparable choice and is stated in STAGES.md.
- `--page raw` on import ignores `--section` by design and dumps every collected metric as one wide
  row per launch: the pandas-readable artefact. `--page details` is the human view.
- Named sections + explicit `--metrics`, never `--set full`: the 2026 Colab successes used
  `--set basic` / explicit metrics, and `--set full` multiplies replay passes.
- Output: `profiles/<gpu>/stageN_<kernel>.ncu-rep` + `.raw.csv` are committed (small: one launch).
  `scripts/ncu_to_md.py` renders the README `<!-- NCU:START -->` block; `scripts/roofline.py` draws
  `docs/img/roofline_<gpu>.png`; `scripts/perf_trailer.py` emits the `ncu:` commit trailer.
- Build flags for any profiled binary: `-O3 -lineinfo -Xptxas -v`; never `-G` (nvcc docs: "not
  intended for profiling"). The `-Xptxas -v` registers/spills/smem line is saved as
  `bench/results/<gpu>/stageN_ptxas.txt`.

## 3. Metric glossary: what each ID proves

Verified IDs are quoted verbatim from the Nsight Compute 2022.4 CLI nvprof-mapping table (Table 10)
and the Kernel Profiling Guide. UNVERIFIED IDs are added to `scripts/ncu_metrics.txt` only after
PROBE 2's `ncu --query-metrics` prints them.

| Metric ID | Status | What it proves in this ladder |
|---|---|---|
| `dram__bytes.sum`, `dram__bytes_read.sum`, `dram__bytes_write.sum` | verified | Bytes between L2 and DRAM. S1/S2 grow with N^2 (S and P round-trip HBM); S3 collapses to O(B*H*N*D). This is FA1 Theorem 2 as a counter. |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | verified | Memory Speed-of-Light %. High = memory-bound; drops after S3 as the kernel becomes latency/compute-bound. |
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | verified (PROBE 2, 2026-09-02: `sm__throughput` listed by `--query-metrics` on the Colab T4) | Compute Speed-of-Light %. |
| `gpu__time_duration.sum` | verified (PROBE 2, 2026-09-02: `gpu__time_duration` listed) | Kernel duration per ncu; never used for speedups (rule 9 of the benchmark protocol). |
| `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` | UNVERIFIED (not in the probe's grep output; pruned at run time by profile.sh) | Memory Speed-of-Light % as ncu's SOL section reports it; `dram__throughput...` is the verified stand-in. |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | verified | Achieved occupancy; compared with the theoretical value from registers/smem per block (T4 ceiling is 32 warps/SM). |
| `launch__registers_per_thread` | verified (no replay pass needed) | Register footprint; cross-checked with `-Xptxas -v`; drives the `__launch_bounds__` decision in S5c. |
| `lts__t_sector_hit_rate.pct`, `l1tex__t_sector_hit_rate.pct` | verified | L2 / L1 hit rates: rise from S1 to S2 as tiles are reused from shared memory instead of re-fetched. |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`, `..._op_st.sum` | verified | Shared-memory bank conflicts. Recorded in S2 (column-wise K^T reads on a row-major tile), the "before" for S5a's padding/swizzle, expected to approach 0 after. |
| `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct` | verified | Global-load coalescing: fraction of each 32-byte sector actually used. Low on S1's strided K reads, toward ~100 after S2's `float4` loads along the contiguous axis. |
| `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active` | verified | HMMA (fp16 tensor) pipe utilisation. Exactly 0 for S1-S3; must be clearly non-zero for S4 or the WMMA path is not running. |
| `sm__inst_executed_pipe_tensor.sum` | verified | Tensor-pipe instruction count; `> 0` is the S4 existence proof. Not multiplied by 512 for FLOPs (that constant is V100-only). |
| `smsp__inst_executed.sum` | verified | Warp instructions executed. Causal block skipping (S6) should roughly halve it at fixed N; S5b removes shared-load and barrier instructions. |
| `smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct` | verified | Stalled on global/L2 latency (K/V tile loads). Falls with prefetch/double buffering (S5d, cp.async on sm_80+). |
| `smsp__warp_issue_stalled_barrier_per_warp_active.pct` | verified | Stalled at `__syncthreads`. Rises after S3 (two barriers per K/V tile), falls in S5b when shuffles replace smem tree reductions. |
| `smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct` | verified | Shared-memory / MIO queue pressure. The S3 -> S4/S5 motivation; falls after S5a. |
| `sm__cycles_elapsed.avg`, `sm__cycles_elapsed.avg.per_second` | verified | Kernel time for the roofline: `time = cycles / rate`. |
| `smsp__sass_thread_inst_executed_op_{fadd,fmul,ffma,hadd,hmul,hfma}_pred_on.sum` | verified | Executed FLOPs (`fadd + fmul + 2*ffma`; same for `h*`). Cross-checks the analytic `4BHN^2D` for S1-S3; `ffma` count falls in S4 as GEMM work moves to the tensor pipe. |
| `sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained`, `dram__bytes.sum.peak_sustained`, `dram__cycles_elapsed.avg.per_second` | verified (NERSC section file) | The roofs: FMA roof `= 2 * ffma.peak_sustained * sm__cycles_elapsed.avg.per_second`; DRAM roof `= dram__bytes.sum.peak_sustained * dram__cycles_elapsed.avg.per_second`. No spec sheet needed for these two. |
| `lts__t_bytes.sum`, `l1tex__t_bytes.sum` | verified | Hierarchical arithmetic-intensity points (L2, L1) next to the DRAM point. |

The fp16 tensor-core roof for S4 is not derivable from these counters on Turing; it is read from
the T4 datasheet, recorded in `docs/ENV.md` with its URL and labelled "spec, not measured".

## 4. Reading a report (SOL first, then stalls)

1. `SpeedOfLight`: SM % and Memory %. Both low: latency-bound, go to `WarpStateStats` and
   `Occupancy`. Memory high: memory-bound, go to `MemoryWorkloadAnalysis`. SM high: compute-bound, go
   to `ComputeWorkloadAnalysis` and the pipe utilisations.
2. `WarpStateStats`: top-3 `smsp__warp_issue_stalled_*` reasons, in ncu vocabulary (Stall Long
   Scoreboard, Stall MIO Throttle, Stall Barrier, Stall Short Scoreboard, ...).
3. `Occupancy` + `LaunchStats`: theoretical vs achieved, and which limiter (registers, smem, warps).
4. Classify the stage as memory-, compute- or latency-bound in STAGES.md and name the metric that
   the next stage should move.

## 5. Degraded mode (counters blocked)

PROBE 2 (`colab/runner.ipynb` cell 2) runs `ncu --metrics sm__cycles_elapsed.avg,dram__bytes.sum,...`
on a 10-line kernel at the start of every session. `ERR_NVGPUCTRPERM` means the runtime does not
grant counter access; the fix (`NVreg_RestrictProfilingToAdminUsers=0` on the host, or
`--cap-add=SYS_ADMIN` at container start) cannot be applied from inside a notebook. Procedure:
Runtime -> Disconnect and delete runtime -> reconnect T4 -> retry, up to 3 times. If still blocked:

- `nsys profile -t cuda -o profiles/<gpu>/stageN_timeline -f true --stats=true ./build/bench/fa_bench ...`
  on a box that has nsys. **nsys is absent on Colab (PROBE 2, 2026-09-02)**, so degraded mode there
  is CUDA events + `-Xptxas -v` only, no timeline; `profile.sh` records "nsys missing" in
  `profiler.txt`.
- CUDA-event timing (the bench rows are unaffected).
- `-Xptxas -v` registers / spills / smem, and `cudaOccupancyMaxActiveBlocksPerMultiprocessor`
  for theoretical occupancy (pure driver queries, no permission needed).
- An analytic roofline point: FLOPs from the formula, bytes from the byte model in the benchmark
  protocol, roofs from the datasheet, all labelled "analytic".
- `profiles/<gpu>/stageN.profiler.txt` records the exact error text and the reset count; the README
  and STAGES.md entry say "hardware counters unavailable on this runtime"; the CSV row carries
  `notes=degraded_profiling`.

Never invent, estimate or back-fill a counter value. A stage whose metric could not be measured says
so and keeps its hypothesis open.

## 6. Reading `.ncu-rep` on the Mac

The committed `.raw.csv` is the guaranteed-readable form (pandas). Whether a macOS Nsight Compute
host GUI is available for the version Colab ships is UNVERIFIED (PROBE 7).
