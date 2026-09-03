#!/usr/bin/env python3
"""profiles/<gpu>/stage<N>_<kernel>[_C128].raw.csv -> README NCU block + docs/generated/ncu_*.md

The README block is the per-stage evidence table (SOL SM/Mem %, DRAM bytes and %, L1/L2 hit,
bank conflicts, occupancy, registers, tensor-pipe %, top-3 stall reasons) plus the
`-Xptxas -v` registers/spills line from bench/results/<gpu>/stage<N>_ptxas.txt. The
docs/generated file is the complete metric dump of every profiled launch, so a metric quoted
in docs/STAGES.md can be grepped back to its raw CSV.

Degraded runs (counters blocked) are shown as such — the first line of
profiles/<gpu>/stage<N>.profiler.txt — never as numbers. A metric whose unit could not be
normalised to base (fa_common.parse_ncu_raw_csv `unparsed_units`) renders as `n/a (unit X)`,
never as a number in the wrong unit. The commit column comes from profiles/<gpu>/stage<N>.meta
(written by profile.sh). `--check` exits 1 when stale.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fa_common import (  # noqa: E402
    EMPTY_TEXT,
    fmt_bytes,
    fmt_num,
    get_block,
    parse_ncu_raw_csv,
    parse_ptxas,
    raw_csv_files,
    replace_block,
    repo_root,
)

# Column -> metric ID. The two SOL IDs are UNVERIFIED (judge_plan §4.5); the SpeedOfLight
# section puts them in the raw page when they exist, and the cell says n/a when they do not.
COLS = [
    ("SOL SM %", "sm__throughput.avg.pct_of_peak_sustained_elapsed", 1),
    ("SOL Mem %", "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed", 1),
    ("DRAM bytes", "dram__bytes.sum", "bytes"),
    ("DRAM %", "dram__throughput.avg.pct_of_peak_sustained_elapsed", 1),
    ("L1 hit %", "l1tex__t_sector_hit_rate.pct", 1),
    ("L2 hit %", "lts__t_sector_hit_rate.pct", 1),
    (
        "bank conflicts ld+st",
        (
            "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
            "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
        ),
        0,
    ),
    ("occupancy %", "sm__warps_active.avg.pct_of_peak_sustained_active", 1),
    ("regs/thread", "launch__registers_per_thread", 0),
    ("tensor pipe %", "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active", 2),
]
STALL_PREFIX, STALL_SUFFIX = "smsp__warp_issue_stalled_", "_per_warp_active.pct"


def col_value(values: dict, spec, fmt, unparsed: dict | None = None) -> str:
    unparsed = unparsed or {}
    ids = spec if isinstance(spec, tuple) else (spec,)
    bad = [f"{i.split('__')[0]}: unit {unparsed[i]!r}" for i in ids if i in unparsed]
    if bad:
        return "n/a (" + "; ".join(bad) + ")"
    if isinstance(spec, tuple):
        parts = [values.get(s) for s in spec]
        v = None if all(p is None for p in parts) else sum(p for p in parts if p is not None)
    else:
        v = values.get(spec)
    if fmt == "bytes":
        return fmt_bytes(v)
    return fmt_num(v, fmt)


def read_meta(profiles_dir: Path, gpu: str, stage: int) -> dict[str, str]:
    """profiles/<gpu>/stage<N>.meta: `key=value` lines written by profile.sh (commit, date,
    ncu version, clock control, dtype, the exact ncu command lines). {} when absent."""
    p = profiles_dir / gpu / f"stage{stage}.meta"
    if not p.exists():
        return {}
    out: dict[str, str] = {}
    for ln in p.read_text(errors="replace").splitlines():
        if "=" in ln and not ln.startswith("#"):
            k, v = ln.split("=", 1)
            out.setdefault(k.strip(), v.strip())
    return out


def top_stalls(values: dict, n: int = 3) -> str:
    stalls = [
        (v, k[len(STALL_PREFIX) : -len(STALL_SUFFIX)])
        for k, v in values.items()
        if k.startswith(STALL_PREFIX) and k.endswith(STALL_SUFFIX) and v is not None
    ]
    stalls.sort(reverse=True)
    return ", ".join(f"{name} {v:.1f}" for v, name in stalls[:n]) or "n/a"


def load_profiles(profiles_dir: Path) -> list[dict]:
    out = []
    for f in raw_csv_files(profiles_dir):
        out.append(dict(f, **parse_ncu_raw_csv(f["path"])))
    return out


def degraded_notes(profiles_dir: Path, have: set[tuple[str, int]]) -> list[str]:
    notes = []
    for p in sorted(profiles_dir.glob("*/stage*.profiler.txt")):
        stage = int("".join(ch for ch in p.stem if ch.isdigit()) or 0)
        if (p.parent.name, stage) in have:
            continue
        first = next(
            (ln for ln in p.read_text(errors="replace").splitlines() if ln.strip()), ""
        ).strip()
        notes.append(
            f"- **S{stage} on `{p.parent.name}`: hardware counters unavailable on this runtime** — "
            f"`{p.relative_to(profiles_dir.parent)}`: {first[:200]}"
        )
    return notes


def render_readme(profiles_dir: Path, results_dir: Path) -> str:
    profs = load_profiles(profiles_dir)
    have = {(p["gpu"], p["stage"]) for p in profs if p["values"]}
    notes = degraded_notes(profiles_dir, have)
    if not profs and not notes:
        return EMPTY_TEXT["NCU"]
    out = [
        "One ncu launch per kernel on the canonical configs (`--launch-skip 10 --launch-count 1`,",
        "base clocks, caches flushed). Every value below is a column of the committed",
        "`profiles/<gpu>/*.raw.csv`; `docs/generated/ncu_stage<N>.md` has the full dump of",
        "each launch.",
        "",
    ]
    by_gpu: dict[str, list[dict]] = defaultdict(list)
    for p in profs:
        by_gpu[p["gpu"]].append(p)
    for gpu, ps in sorted(by_gpu.items()):
        out += [
            f"**`{gpu}`**",
            "",
            "| stage | kernel | cfg | commit | "
            + " | ".join(c[0] for c in COLS)
            + " | top-3 stalls (% of warp-active cycles) |",
            "|---|---|---|---|" + "---|" * (len(COLS) + 1),
        ]
        for p in sorted(ps, key=lambda p: (p["stage"], p["cfg"], p["kernel"])):
            commit = read_meta(profiles_dir, gpu, p["stage"]).get("commit", "n/a")
            if not p["values"]:
                out.append(
                    f"| S{p['stage']} | `{p['kernel']}` | {p['cfg']} | `{commit}` | "
                    + " | ".join(["n/a"] * len(COLS))
                    + f" | (unparseable: {p.get('error', '?')}) |"
                )
                continue
            unparsed = p.get("unparsed_units", {})
            cells = [col_value(p["values"], spec, fmt, unparsed) for _, spec, fmt in COLS]
            out.append(
                f"| S{p['stage']} | `{p['kernel']}` | {p['cfg']} | `{commit}` | "
                + " | ".join(cells)
                + f" | {top_stalls(p['values'])} |"
            )
        out.append("")
        ptx_rows = []

        def spill(v):
            return "n/a" if v is None else str(v)

        for stage in sorted({p["stage"] for p in ps}):
            for e in parse_ptxas(results_dir / gpu / f"stage{stage}_ptxas.txt"):
                ptx_rows.append(
                    f"| S{stage} | `{e['kernel']}` | {e['sm']} | {e['regs']} | {e['smem']} "
                    f"| {spill(e['spill_stores'])} / {spill(e['spill_loads'])} |"
                )
        if ptx_rows:
            out += [
                "`-Xptxas -v` at the same SHA:",
                "",
                "| stage | kernel | arch | registers | static smem B | spill stores / loads B |",
                "|---|---|---|---|---|---|",
                *ptx_rows,
                "",
            ]
    out += notes
    return "\n".join(out).rstrip("\n")


def render_stage_dump(profs: list[dict], stage: int) -> str:
    out = [
        f"# Nsight Compute raw metrics — stage {stage}",
        "",
        "Generated by `scripts/ncu_to_md.py` from `profiles/<gpu>/stage*_*.raw.csv`; do not edit.",
        "",
    ]
    for p in sorted(profs, key=lambda p: (p["gpu"], p["cfg"], p["kernel"])):
        m = p["meta"]
        out += [
            f"## `{p['gpu']}` · `{p['kernel']}` · {p['cfg']}",
            "",
            f"launch: `{m.get('Kernel Name', '?')}` grid {m.get('Grid Size', '?')} "
            f"block {m.get('Block Size', '?')} on {m.get('Device', '?')} (CC {m.get('CC', '?')})",
            "",
        ]
        if p["scaled"]:
            out += [f"note: prefixed units scaled to base for: {', '.join(p['scaled'])}", ""]
        unparsed = p.get("unparsed_units", {})
        if unparsed:
            out += [
                "note: value withheld (unit not normalisable to base) for: "
                + ", ".join(f"{k} [{u}]" for k, u in sorted(unparsed.items())),
                "",
            ]
        out += ["| metric | value | unit |", "|---|---|---|"]
        for k in sorted(p["values"]):
            v = p["values"][k]
            if k in unparsed:
                shown = f"n/a (unit {unparsed[k]!r})"
            else:
                shown = "n/a" if v is None else f"{v:.6g}"
            out.append(f"| `{k}` | {shown} | {p['units'].get(k, '')} |")
        out.append("")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--readme", default=str(repo_root() / "README.md"))
    ap.add_argument("--profiles", default=str(repo_root() / "profiles"))
    ap.add_argument("--results", default=str(repo_root() / "bench" / "results"))
    ap.add_argument("--generated", default=str(repo_root() / "docs" / "generated"))
    a = ap.parse_args(argv)
    profiles, results, gen = Path(a.profiles), Path(a.results), Path(a.generated)
    stale = []
    profs = load_profiles(profiles)
    for stage in sorted({p["stage"] for p in profs if p["values"]}):
        target = gen / f"ncu_stage{stage}.md"
        body = render_stage_dump([p for p in profs if p["stage"] == stage and p["values"]], stage)
        if a.check:
            if not target.exists() or target.read_text() != body:
                stale.append(str(target))
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(body)
            print(f"ncu_to_md: wrote {target}")
    readme = Path(a.readme)
    if not readme.exists():
        print(f"ncu_to_md: {readme} missing; README block skipped")
    else:
        text = readme.read_text()
        if get_block(text, "NCU") is None:
            print("ncu_to_md: README has no NCU block; skipped")
        else:
            new = replace_block(text, "NCU", render_readme(profiles, results))
            if a.check and new != text:
                stale.append(str(readme))
            elif not a.check and new != text:
                readme.write_text(new)
                print(f"ncu_to_md: updated {readme}")
    if a.check:
        if stale:
            print(
                "ncu_to_md --check: stale: "
                + ", ".join(stale)
                + "; run `python scripts/ncu_to_md.py`"
            )
            return 1
        print("ncu_to_md --check: up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
