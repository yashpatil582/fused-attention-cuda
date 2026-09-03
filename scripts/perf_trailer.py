#!/usr/bin/env python3
"""Commit-message trailers from CSVs, for the squash-merge of a stage PR:

    perf: 12.40ms -> 5.31ms (2.34x) cfg=B8_H8_N4096_D64_fp32 gpu=Tesla-T4 stage=3
    ncu: dram__bytes.sum 8.6GB -> 0.42GB

"before" = the previous stage's row on the same config (bench/results/<gpu>/stage<N-1>.csv);
"after" = this stage's row. Everything is read from committed CSVs — if a value is missing
the trailer says so in words instead of inventing one. `git log --grep='^perf:'` then prints
the whole ladder.

    python scripts/perf_trailer.py 3                 # stage 3, canonical C64, gpu t4
    python scripts/perf_trailer.py stage3 --cfg C128 --gpu t4 --dtype fp32
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fa_common import (  # noqa: E402
    OUR_IMPLS,
    fmt_bytes,
    latest_rows,
    parse_cfg,
    parse_ncu_raw_csv,
    raw_csv_files,
    read_rows,
    repo_root,
    to_float,
)
from kernel_regex import canonical_cfg_string  # noqa: E402

# The metric each stage claims (README ladder); sum-type metrics add across a stage's
# kernels, percentage metrics take the max (the worst kernel is the honest number).
STAGE_METRICS = {
    1: ["dram__bytes.sum", "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct"],
    2: [
        "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct",
        "dram__bytes.sum",
        "lts__t_sector_hit_rate.pct",
    ],
    3: ["dram__bytes.sum", "dram__throughput.avg.pct_of_peak_sustained_elapsed"],
    4: [
        "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active",
        "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
    ],
    5: [
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
        "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct",
    ],
}


def stage_row(
    gpu_dir: Path, stage: int, shape: dict, dtype: str | None, causal: int
) -> dict | None:
    f = gpu_dir / f"stage{stage}.csv"
    if not f.exists():
        return None
    rows = [
        r
        for r in latest_rows(read_rows(f)[1])
        if r["impl"] == OUR_IMPLS[stage]
        and to_float(r["ms_median"]) is not None
        and all(int(r[k]) == shape[k] for k in "BHND")
        and int(r["causal"]) == causal
    ]
    same = [r for r in rows if dtype is None or r["dtype"] == dtype]
    return (same or rows or [None])[0]


def stage_metric(profiles: Path, gpu: str, stage: int, cfg: str, metric: str) -> float | None:
    vals = []
    for f in raw_csv_files(profiles):
        if f["gpu"] == gpu and f["stage"] == stage and f["cfg"] == cfg:
            v = parse_ncu_raw_csv(f["path"])["values"].get(metric)
            if v is not None:
                vals.append(v)
    if not vals:
        return None
    return sum(vals) if metric.endswith(".sum") else max(vals)


def fmt_metric(metric: str, v: float | None) -> str:
    if v is None:
        return "n/a"
    if metric.startswith("dram__bytes") or metric.endswith("_bytes.sum"):
        return fmt_bytes(v).replace(" ", "")
    return f"{v:.1f}%" if metric.endswith(".pct") or ".pct_of_peak" in metric else f"{v:.4g}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("stage", help="N or stageN")
    ap.add_argument("--gpu", default="t4")
    ap.add_argument("--cfg", default="C64", help="C64 | C128 | B2_H8_N2048_D64")
    ap.add_argument("--dtype", help="prefer this dtype row (default: this stage's row, any dtype)")
    ap.add_argument("--causal", type=int, default=0)
    ap.add_argument("--metric", action="append", help="override the ncu metric(s) to report")
    ap.add_argument(
        "--root", default=str(repo_root()), help="repo root (tests point this at fixtures)"
    )
    a = ap.parse_args(argv)
    stage = int(re.sub(r"\D", "", a.stage))
    cfg_name = a.cfg if a.cfg.startswith("C") else None
    shape = parse_cfg(canonical_cfg_string(a.cfg) if cfg_name else a.cfg)
    cfg_str = "_".join(f"{k}{shape[k]}" for k in "BHND")
    root = Path(a.root)
    gpu_dir = root / "bench" / "results" / a.gpu
    after = stage_row(gpu_dir, stage, shape, a.dtype, a.causal)
    if after is None:
        print(
            f"perf: unavailable (no {OUR_IMPLS[stage]} row for cfg={cfg_str} causal={a.causal} "
            f"in {gpu_dir / f'stage{stage}.csv'})"
        )
    else:
        gpu_name = re.sub(r"\s+", "-", after["gpu"].strip()) or a.gpu
        tag = f"cfg={cfg_str}_{after['dtype']} gpu={gpu_name} stage={stage}"
        before = (
            stage_row(gpu_dir, stage - 1, shape, after["dtype"], a.causal) if stage > 1 else None
        )
        a_ms = to_float(after["ms_median"])
        if before is None:
            print(
                f"perf: {a_ms:.2f}ms {tag}"
                + (
                    " (first stage, no predecessor)"
                    if stage == 1
                    else f" (no stage {stage - 1} row to compare)"
                )
            )
        else:
            b_ms = to_float(before["ms_median"])
            dt = "" if before["dtype"] == after["dtype"] else f" (before in {before['dtype']})"
            print(f"perf: {b_ms:.2f}ms -> {a_ms:.2f}ms ({b_ms / a_ms:.2f}x) {tag}{dt}")
    profiles = root / "profiles"
    for metric in a.metric or STAGE_METRICS.get(stage, []):
        v_after = stage_metric(profiles, a.gpu, stage, cfg_name or "C64", metric)
        v_before = (
            stage_metric(profiles, a.gpu, stage - 1, cfg_name or "C64", metric)
            if stage > 1
            else None
        )
        if v_after is None:
            print(f"ncu: {metric} unavailable (no profiles/{a.gpu}/stage{stage}_*.raw.csv column)")
        elif v_before is None:
            print(
                f"ncu: {metric} {fmt_metric(metric, v_after)}"
                + (" (baseline)" if stage == 1 else f" (no stage {stage - 1} profile)")
            )
        else:
            print(f"ncu: {metric} {fmt_metric(metric, v_before)} -> {fmt_metric(metric, v_after)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
