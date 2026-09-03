#!/usr/bin/env python3
"""Roofline per GPU from the committed ncu raw CSVs -> docs/img/roofline_<gpu>.png.

NERSC recipe (judge_plan §4.5), no datasheet needed for the two measured roofs:
  time_s      = sm__cycles_elapsed.avg / sm__cycles_elapsed.avg.per_second
  FLOPs       = 4·B·H·N²·D analytic for the stage (cross-checked against the fadd+fmul+2·ffma
                counters and printed, never substituted)
  AI (DRAM)   = FLOPs / dram__bytes.sum        (hollow markers: FLOPs / lts__t_bytes.sum, L2)
  FMA roof    = 2 · sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained
                  · sm__cycles_elapsed.avg.per_second
  DRAM roof   = dram__bytes.sum.peak_sustained · dram__cycles_elapsed.avg.per_second
The fp16 tensor-core roof is NOT derivable from counters on Turing+, so it is read from
docs/ENV.md (`ROOF fp16_tensor_tflops = <n>`) and drawn dashed, labelled "spec, not measured".

One point per (stage, cfg): kernels of a multi-kernel stage (S1/S2) are summed in time and
bytes; a stage with a missing kernel file is drawn hollow-red and labelled "partial".
Skips gracefully (exit 0, no PNG) when there are no profiles. A kernel whose cycle or rate
column carries a unit that could not be normalised to base (fa_common `unparsed_units`) is
refused with a printed reason, never plotted in the wrong unit.

Clock: every counter here was taken under `ncu --clock-control base` (T4: 585 MHz SM), so
"achieved FLOP/s" is at BASE clock. The README BENCH table's TFLOP/s comes from bench rows at
their own `clock_mhz` (locked to the same base clock by gpu_run.sh when nvidia-smi -lgc works,
else unlocked/boost). The figure prints the profiled SM clock; compare % of roof, not TFLOP/s,
across the two unless the row's clock_mhz equals the profile clock.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fa_common import (  # noqa: E402
    attention_flops,
    parse_ncu_raw_csv,
    raw_csv_files,
    read_env_roofs,
    repo_root,
)
from kernel_regex import KERNELS, canonical_cfg_string  # noqa: E402

M = {
    "cyc": "sm__cycles_elapsed.avg",
    "rate": "sm__cycles_elapsed.avg.per_second",
    "dram": "dram__bytes.sum",
    "l2": "lts__t_bytes.sum",
    "ffma_peak": "sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained",
    "dram_peak": "dram__bytes.sum.peak_sustained",
    "dram_rate": "dram__cycles_elapsed.avg.per_second",
    "fadd": "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
    "fmul": "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum",
    "ffma": "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
}


def stage_points(profiles_dir: Path) -> dict[str, list[dict]]:
    """{gpu: [{stage, cfg, time_s, flops, dram_bytes, l2_bytes, partial, counter_flops}]}."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for f in raw_csv_files(profiles_dir):
        d = parse_ncu_raw_csv(f["path"])
        bad = {k: d.get("unparsed_units", {}).get(k) for k in (M["cyc"], M["rate"])}
        bad = {k: u for k, u in bad.items() if u is not None}
        if bad:
            print(
                f"roofline: refusing {f['path']}: non-base units {bad}; "
                "re-import with --print-units base"
            )
            continue
        if d["values"].get(M["cyc"]) and d["values"].get(M["rate"]):
            groups[(f["gpu"], f["stage"], f["cfg"])].append(dict(f, **d))
    out: dict[str, list[dict]] = defaultdict(list)
    for (gpu, stage, cfg), ks in sorted(groups.items()):
        v = [k["values"] for k in ks]
        shape = canonical_cfg_string(cfg)
        B, H, N, D = (int(x[1:]) for x in shape.split("_"))
        expected = {s for s, _ in KERNELS.get(stage, [])}
        t = sum(x[M["cyc"]] / x[M["rate"]] for x in v)
        cf = sum(
            (x.get(M["fadd"]) or 0) + (x.get(M["fmul"]) or 0) + 2 * (x.get(M["ffma"]) or 0)
            for x in v
        )
        out[gpu].append(
            {
                "stage": stage,
                "cfg": cfg,
                "time_s": t,
                "flops": attention_flops(B, H, N, D),
                "counter_flops": cf,
                "dram_bytes": sum(x.get(M["dram"]) or 0 for x in v),
                "l2_bytes": sum(x.get(M["l2"]) or 0 for x in v),
                "partial": expected - {k["kernel"] for k in ks},
                "peaks": v[0],
            }
        )
    return out


def roofs_from(peaks: dict) -> dict[str, float]:
    r = {}
    if peaks.get(M["ffma_peak"]) and peaks.get(M["rate"]):
        r["fp32 FMA (measured peak_sustained)"] = 2 * peaks[M["ffma_peak"]] * peaks[M["rate"]]
    if peaks.get(M["dram_peak"]) and peaks.get(M["dram_rate"]):
        r["DRAM"] = peaks[M["dram_peak"]] * peaks[M["dram_rate"]]
    return r


def plot(
    gpu: str, pts: list[dict], roofs: dict[str, float], spec_tensor_tflops: float | None, out: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.5))
    dram_bw = roofs.get("DRAM")
    peak = max(
        [v for k, v in roofs.items() if k != "DRAM"]
        + [(spec_tensor_tflops or 0) * 1e12]
        + [p["flops"] / p["time_s"] for p in pts]
    )
    ai = [1e-2, 1e4]
    for name, val in roofs.items():
        if name == "DRAM":
            continue
        ax.plot(ai, [val, val], "k-", lw=1)
        ax.text(
            ai[-1], val, f" {name}: {val / 1e12:.2f} TFLOP/s", ha="right", va="bottom", fontsize=8
        )
    if spec_tensor_tflops:
        y = spec_tensor_tflops * 1e12
        ax.plot(ai, [y, y], "k--", lw=1)
        ax.text(
            ai[-1],
            y,
            f" fp16 tensor: {spec_tensor_tflops:g} TFLOP/s (spec, not measured; docs/ENV.md)",
            ha="right",
            va="bottom",
            fontsize=8,
        )
    if dram_bw:
        xs = [ai[0], peak / dram_bw]
        ax.plot(xs, [x * dram_bw for x in xs], "k-", lw=1)
        ax.text(
            xs[0],
            xs[0] * dram_bw,
            f" DRAM: {dram_bw / 1e9:.0f} GB/s (measured peak_sustained)",
            fontsize=8,
            rotation=32,
        )
    for p in pts:
        perf = p["flops"] / p["time_s"]
        label = f"S{p['stage']} {p['cfg']}" + (
            f" (partial: missing {','.join(sorted(p['partial']))})" if p["partial"] else ""
        )
        style = dict(
            marker="o",
            ms=8,
            mfc="none" if p["partial"] else None,
            color="tab:red" if p["partial"] else None,
        )
        if p["dram_bytes"]:
            ax.plot(p["flops"] / p["dram_bytes"], perf, ls="", label=label, **style)
        if p["l2_bytes"]:
            ax.plot(
                p["flops"] / p["l2_bytes"], perf, ls="", marker="s", ms=6, mfc="none", color="gray"
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity, FLOP / byte (filled: DRAM bytes; hollow gray: L2 bytes)")
    ax.set_ylabel("achieved FLOP/s (4·B·H·N²·D / measured time)")
    rate = pts[0]["peaks"].get(M["rate"])  # cycle/second after unit normalisation
    clock = f"SM clock during profile: {rate / 1e6:.0f} MHz" if rate else "SM clock: n/a"
    ax.set_title(
        f"Roofline — {gpu} — one point per stage, canonical configs, ncu counters\n"
        f"{clock} (ncu --clock-control base); bench rows carry their own clock_mhz",
        fontsize=10,
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--profiles", default=str(repo_root() / "profiles"))
    ap.add_argument("--out-dir", default=str(repo_root() / "docs" / "img"))
    ap.add_argument("--env-md", default=str(repo_root() / "docs" / "ENV.md"))
    a = ap.parse_args(argv)
    per_gpu = stage_points(Path(a.profiles))
    if not per_gpu:
        print("roofline: no profiles/<gpu>/*.raw.csv with cycle counters; nothing to plot")
        return 0
    spec = read_env_roofs(a.env_md).get("fp16_tensor_tflops")
    for gpu, pts in per_gpu.items():
        roofs = roofs_from(pts[0]["peaks"])
        if not roofs:
            print(f"roofline[{gpu}]: peak_sustained counters missing; cannot draw measured roofs")
        for p in pts:
            perf = p["flops"] / p["time_s"]
            ai = p["flops"] / p["dram_bytes"] if p["dram_bytes"] else float("nan")
            ratio = p["counter_flops"] / p["flops"] if p["counter_flops"] else float("nan")
            print(
                f"roofline[{gpu}] S{p['stage']} {p['cfg']}: time={p['time_s'] * 1e3:.3f} ms "
                f"achieved={perf / 1e12:.3f} TFLOP/s AI_dram={ai:.2f} "
                f"counter_flops/analytic={ratio:.2f}"
                + (f" PARTIAL missing {sorted(p['partial'])}" if p["partial"] else "")
            )
        out = Path(a.out_dir) / f"roofline_{gpu}.png"
        plot(gpu, pts, roofs, spec, out)
        print(f"roofline[{gpu}]: wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
