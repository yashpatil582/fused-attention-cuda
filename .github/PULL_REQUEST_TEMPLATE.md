<!--
Per-stage PR body (CONTRACT.md §8, docs/STAGES.md). Keep every heading, even if a
section only says "none". Reviewers reward stated limitations and punish discovered
ones. Numbers are pasted from generated output, never typed; an unmeasured value is
the literal TBD.
-->

## What and why

<!-- One paragraph: the bottleneck the previous stage's ncu profile showed, the ONE
change this stage makes, and why that change should move the named metric. -->

Stage `S<N>` · kernel(s) `fa_s<N>_…` · branch `stage<N>-…` · tag on merge `v0.<N>-…`

## Before / after benchmark

<!-- Paste the block printed by:
       python scripts/bench_to_md.py --stage <N> --gpu t4
     It reads bench/results/t4/stage<N>.csv (frozen schema, CONTRACT.md §3).
     Baseline named on the T4 rows: SDPA EFFICIENT_ATTENTION (FLASH refuses sm_75).
     FLOPs = 4·B·H·N²·D; causal rows use the full count until block skipping exists. -->

| impl | dtype | cfg | ms_median (p10 / p90) | TFLOP/s | GB/s | vs previous stage | % of SDPA(EFFICIENT) |
|------|-------|-----|-----------------------|---------|------|-------------------|----------------------|
| TBD  | TBD   | TBD | TBD                   | TBD     | TBD  | TBD               | TBD                  |

GPU / driver / nvcc / torch / ncu / clock: TBD (from `bench/results/t4/env.txt`, same SHA as the rows)

## ncu Speed-Of-Light delta

<!-- Canonical config C64 = B2_H8_N2048_D64 (and C128) at the SAME SHA as the bench
     rows. Values come from profiles/t4/stage<N>_<kernel>.raw.csv via
       python scripts/ncu_to_md.py --stage <N> --gpu t4
     Two lines minimum (SpeedOfLight section), plus the metric this stage targeted. -->

- SOL SM throughput:     TBD % → TBD %
- SOL memory throughput: TBD % → TBD %
- Targeted metric `<metric id>`: TBD → TBD
- Profiler: `ncu <version>` on `<GPU>` — or *degraded mode* (reason recorded in `profiles/t4/stage<N>.profiler.txt`)

## Known limitations

<!-- Explicit list, flash-attention-minimal style: what this stage does NOT do
     (e.g. N % 32 == 0 only, no causal block skipping, fp32 only, T4 has no bf16). -->

-

## Tests added / changed

- [ ] `tests/…` — <what it checks>
- [ ] `compute-sanitizer --tool memcheck` and `--tool racecheck` clean on `fa_bench --smoke` (paste the summary line)
- [ ] `pytest -q -m "not gpu"` green on the Mac (CPU model, README staleness, CSV sanity)

## Merge checklist

- [ ] `docs/STAGES.md` entry: hypothesis → metric before → change → metric after → decision → 5-line summary → 2 interview Q&A
- [ ] `docs/WORKLOG.md` dated entry (bugs hit, dead ends, regressions kept)
- [ ] README generated blocks regenerated; `tests/test_docs_cpu.py` passes
- [ ] Squash-merge message ends with the `perf:` and `ncu:` trailers printed by `scripts/perf_trailer.py`
- [ ] Every number above traces to a committed CSV / raw.csv at this SHA; no fabricated values
