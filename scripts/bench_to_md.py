#!/usr/bin/env python3
"""bench/results/<gpu>/stage*.csv + gpt_block.csv -> README tables (CONTRACT §6).

Why generated: a number can only reach the README through a committed CSV row that carries
its commit/gpu/driver/nvcc/torch/ncu/clock provenance. `--check` (run by
tests/test_docs_cpu.py and CI) exits 1 when the README differs from what the CSVs say, so a
stale or hand-edited table fails the build.

Output is deterministic (no timestamps): rows sorted by shape, columns in IMPL_ORDER, the
newest row per (impl, stage, dtype, shape, causal) wins (newest = the `run=<utc>` stamp in
notes, then file order). Rows whose impl is outside IMPL_ORDER (`smoke`, S0 round trip) never
reach the README. A baseline whose backend refused the device renders as `refused`; a "% of
SDPA" cell whose baseline row was measured at another commit says so.

    python scripts/bench_to_md.py            # rewrite README.md blocks in place
    python scripts/bench_to_md.py --check    # exit 1 if README is stale
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fa_common import (  # noqa: E402
    EMPTY_TEXT,
    IMPL_ORDER,
    OUR_IMPLS,
    cfg_string,
    get_block,
    load_stage_rows,
    read_rows,
    replace_block,
    repo_root,
    to_float,
)

DISPLAY = {
    "naive": "S1 naive",
    "tiled": "S2 tiled",
    "fused": "S3 fused",
    "wmma": "S4 wmma",
    "tuned": "S5 tuned",
    "sdpa_efficient": "SDPA EFFICIENT",
    "sdpa_flash": "SDPA FLASH",
    "sdpa_cudnn": "SDPA cuDNN",
    "sdpa_math_fp32": "SDPA MATH fp32 (oracle)",
    "torch_unfused": "torch unfused",
    "flash_attn": "flash_attn",
}


def cell(r: dict | None) -> str:
    if r is None:
        return "—"
    notes = r.get("notes", "")
    if "backend_refused" in notes:
        return "refused"
    if "oom_by_design" in notes:
        return "OOM by design"
    if "correctness_failed" in notes:
        return "**correctness failed**"
    if "oom_runtime" in notes:
        return "OOM (runtime)"
    ms = to_float(r.get("ms_median"))
    return "—" if ms is None else f"{ms:.3f}"


def shape_key(r: dict) -> tuple:
    return tuple(int(r[k]) for k in ("N", "D", "B", "H"))


def env_line(gpu_dir: str, rows: list[dict]) -> str:
    def uniq(col):
        return ", ".join(sorted({r.get(col, "") for r in rows if r.get(col)}))

    timed = [r for r in rows if to_float(r.get("ms_median")) is not None]
    locks = sorted({m for r in timed for m in re.findall(r"clock_lock=(\d+)", r.get("notes", ""))})
    if timed and all("clock=unlocked" not in r.get("notes", "") for r in timed):
        clock = (
            f"clocks locked to {', '.join(locks)} MHz (nvidia-smi -lgc, verified per row)"
            if locks
            else "clocks locked"
        )
    else:
        clock = "clocks unlocked (SM clock sampled during each timed loop; `clock_mhz` = median)"
    return (
        f"**`{gpu_dir}`** — GPU `{uniq('gpu')}`, driver {uniq('driver')}, nvcc {uniq('nvcc')}, "
        f"torch {uniq('torch')}, ncu {uniq('ncu')}, commit(s) `{uniq('commit')}`, {clock}."
    )


def headline_table(rows: list[dict]) -> list[str]:
    impls = [i for i in IMPL_ORDER if any(r["impl"] == i for r in rows)]
    by = {(shape_key(r), r["impl"]): r for r in rows}
    shapes = sorted({shape_key(r) for r in rows})
    out = [
        "| config | "
        + " | ".join(DISPLAY.get(i, i) for i in impls)
        + " | best ours vs S1 naive | best ours, % of SDPA EFFICIENT |",
        "|---|" + "---|" * (len(impls) + 2),
    ]
    for s in shapes:
        cells = [cell(by.get((s, i))) for i in impls]
        ours = [
            (to_float(by[(s, i)]["ms_median"]), i)
            for i in OUR_IMPLS.values()
            if (s, i) in by and to_float(by[(s, i)].get("ms_median")) is not None
        ]
        best = min(ours) if ours else None
        naive = to_float(by[(s, "naive")]["ms_median"]) if (s, "naive") in by else None
        sdpa = (
            to_float(by[(s, "sdpa_efficient")]["ms_median"])
            if (s, "sdpa_efficient") in by
            else None
        )
        vs_naive = f"{naive / best[0]:.2f}x ({best[1]})" if best and naive else "—"
        if best and sdpa:
            vs_sdpa = f"{100.0 * sdpa / best[0]:.0f}% ({best[1]})"
            # The baseline is re-measured by every stage run; say so when the row the README
            # divides by was not measured at the same commit as the kernel row.
            bc, kc = by[(s, "sdpa_efficient")]["commit"], by[(s, best[1])]["commit"]
            if bc != kc:
                vs_sdpa += f" (baseline @ `{bc}`)"
        elif best:
            vs_sdpa = "no baseline row"
        else:
            vs_sdpa = "—"
        first = next((by[(s, i)] for i in impls if (s, i) in by), None)
        if first is None:  # no rendered impl for this shape (e.g. smoke-only rows)
            continue
        out.append(
            f"| `{cfg_string(first)}` | " + " | ".join(cells) + f" | {vs_naive} | {vs_sdpa} |"
        )
    return out


def details_table(rows: list[dict]) -> list[str]:
    out = [
        "| config | impl | stage | ms median | p10 | p90 | TFLOP/s | GB/s | clock MHz | commit "
        "| notes |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    def order(r):
        return shape_key(r), IMPL_ORDER.index(r["impl"]) if r["impl"] in IMPL_ORDER else 99

    for r in sorted(rows, key=order):
        nums = " | ".join(r[k] or "—" for k in ("ms_median", "ms_p10", "ms_p90", "tflops", "gbps"))
        out.append(
            f"| `{cfg_string(r)}` | {r['impl']} | {r['stage']} | {nums} | {r['clock_mhz'] or '—'} "
            f"| `{r['commit']}` | {r['notes']} |"
        )
    return out


def render_bench(results_dir: Path) -> str:
    per_gpu = load_stage_rows(results_dir)
    if not per_gpu:
        return EMPTY_TEXT["BENCH"]
    out = [
        "Median ms per forward call (lower is better); one CUDA-event pair per iteration, 256 MB",
        "L2 flush before every timed iteration, >=100 iterations after >=10 warmup. Every column",
        "is the stage's DEFAULT tile configuration (the S5 A/B sweep lives in `stage5_tuning.csv`).",
        "TFLOP/s in the CSVs use 4·B·H·N²·D; causal rows are credited half of that",
        "(`flops=causal_half`, CONTRACT §5) only in rows benchmarked after that convention landed",
        "(S5 onward) — earlier sweep rows carry `flops=full`. Speed baseline is SDPA pinned to",
        "`EFFICIENT_ATTENTION`; `SDPA MATH fp32` is the correctness oracle, not a speed baseline.",
        "",
    ]
    rendered = 0
    for gpu_dir in sorted(per_gpu):
        # `smoke` (S0 round trip) and any other non-contract impl never enter the README.
        rows = [r for r in per_gpu[gpu_dir] if r["impl"] in IMPL_ORDER]
        if not rows:
            continue
        rendered += 1
        out += [env_line(gpu_dir, rows), ""]
        groups: dict[tuple, list[dict]] = defaultdict(list)
        for r in rows:
            groups[(r["dtype"], r["causal"])].append(r)
        for (dtype, causal), g in sorted(groups.items()):
            out += [f"**{dtype}, causal={causal}**", ""] + headline_table(g) + [""]
        out += (
            [
                "<details><summary>p10/p90, TFLOP/s, GB/s, clock and notes for every row</summary>",
                "",
            ]
            + details_table(rows)
            + ["", "</details>", ""]
        )
    if rendered == 0:
        return EMPTY_TEXT["BENCH"]
    return "\n".join(out).rstrip("\n")


def render_gpt(results_dir: Path) -> str:
    files = sorted(results_dir.glob("*/gpt_block.csv"))
    if not files:
        return EMPTY_TEXT["GPT"]
    out = [
        "nanoGPT-style block (n_embd 768, 12 heads, T 1024, batch 8, fp16, inference): tokens/s",
        "over 50 iterations after warmup; attention share from a second CUDA-event pair around the",
        "attention call (the `.contiguous()` copies our op needs are in that window on purpose).",
        "",
        "| gpu | impl | stage | tokens/s | block ms (median) | attention ms | attention share "
        "| commit |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for f in files:
        _, rows = read_rows(f)
        seen = {}
        for r in rows:
            seen[(r["impl"], r["stage"])] = r
        for r in seen.values():
            n = dict(kv.split("=", 1) for kv in r["notes"].split(";") if "=" in kv)
            out.append(
                f"| {f.parent.name} | {r['impl']} | {r['stage']} | {n.get('tok_s', '—')} "
                f"| {r['ms_median']} | {n.get('attn_ms', '—')} | {n.get('attn_pct', '—')}% "
                f"| `{r['commit']}` |"
            )
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--check", action="store_true", help="exit 1 if README differs from regenerated"
    )
    ap.add_argument("--readme", default=str(repo_root() / "README.md"))
    ap.add_argument("--results", default=str(repo_root() / "bench" / "results"))
    a = ap.parse_args(argv)
    readme = Path(a.readme)
    if not readme.exists():
        print(f"bench_to_md: {readme} missing; nothing to {'check' if a.check else 'write'}")
        return 0
    text = readme.read_text()
    new = text
    for tag, body in (
        ("BENCH", render_bench(Path(a.results))),
        ("GPT", render_gpt(Path(a.results))),
    ):
        if get_block(text, tag) is None:
            print(f"bench_to_md: README has no {tag} block; skipped")
            continue
        new = replace_block(new, tag, body)
    if a.check:
        if new != text:
            print(
                "bench_to_md --check: README BENCH/GPT blocks are stale; "
                "run `python scripts/bench_to_md.py`"
            )
            return 1
        print("bench_to_md --check: README up to date")
        return 0
    if new != text:
        readme.write_text(new)
        print(f"bench_to_md: updated {readme}")
    else:
        print("bench_to_md: no change")
    return 0


if __name__ == "__main__":
    sys.exit(main())
