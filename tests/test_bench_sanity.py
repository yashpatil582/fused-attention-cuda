"""CPU sanity checks over every committed benchmark CSV (bench/results/<gpu>/*.csv).

Why: the README is generated from these files, so a malformed or implausible row is a
fabricated number waiting to happen. Each row must carry the frozen schema exactly, a clean
commit SHA (no `-dirty`), a tight p10/p90 band (or be flagged `noisy`), a `tflops` value that
recomputes from its own ms_median and shape, `iters>=100` for every timed kernel row, a
correctness note on every row of our kernels, and a TFLOP/s value at or below the roof
recorded in docs/ENV.md (`ROOF fp32_tflops = ...`, `ROOF fp16_tensor_tflops = ...`; a
missing roof is warned about, not silently skipped). Profiles must match their bench rows'
commit. Passes trivially (skips) when no CSV has been committed yet.
"""

from __future__ import annotations

import glob
import os
import sys
import warnings

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from fa_common import (  # noqa: E402
    DTYPES,
    EXTRA_IMPLS,
    IMPL_ORDER,
    NOISE_RATIO,
    OUR_IMPLS,
    SCHEMA,
    attention_flops,
    read_env_roofs,
    read_rows,
    roof_for_dtype,
    to_float,
)

CSVS = sorted(glob.glob(os.path.join(ROOT, "bench", "results", "*", "*.csv")))
METAS = sorted(glob.glob(os.path.join(ROOT, "profiles", "*", "stage*.meta")))
# Rows without timings must say why.
NO_TIMING_NOTES = ("oom_by_design", "correctness_failed", "oom_runtime", "backend_refused")
MIN_ITERS = 100
TFLOPS_TOL = 0.01  # 1%: the CSV rounds ms to 4 decimals and tflops to 3


def _rows(path):
    header, rows = read_rows(path)
    return header, rows


def _notes(r: dict) -> dict[str, str]:
    """`k=v` pairs of the notes column (bare tokens map to '')."""
    out = {}
    for tok in (r.get("notes") or "").split(";"):
        k, _, v = tok.partition("=")
        out.setdefault(k.strip(), v.strip())
    return out


@pytest.mark.skipif(not CSVS, reason="no benchmark CSV committed yet")
@pytest.mark.parametrize("path", CSVS, ids=[os.path.relpath(p, ROOT) for p in CSVS])
def test_schema_exact(path):
    header, _ = _rows(path)
    assert header == SCHEMA, f"{path}: header {header} != CONTRACT schema"


@pytest.mark.skipif(not CSVS, reason="no benchmark CSV committed yet")
@pytest.mark.parametrize("path", CSVS, ids=[os.path.relpath(p, ROOT) for p in CSVS])
def test_rows_are_plausible(path):
    _, rows = _rows(path)
    roofs = read_env_roofs()
    if not roofs:
        warnings.warn(
            "docs/ENV.md has no parseable `ROOF <name> = <number>` line: the TFLOP/s cap is "
            "inert until the datasheet peaks are pasted there",
            stacklevel=1,
        )
    for i, r in enumerate(rows, start=2):
        where = f"{os.path.relpath(path, ROOT)}:{i} impl={r['impl']}"
        notes = _notes(r)
        assert r["commit"].strip(), f"{where}: empty commit column"
        assert "-dirty" not in r["commit"], (
            f"{where}: commit {r['commit']!r} was measured on a modified tree; committed CSVs "
            "must come from a clean checkout"
        )
        # CONTRACT §3: the 11 table impls + `smoke` (S0 round trip) + `gpt_*` (gpt_block.csv)
        assert r["impl"] in IMPL_ORDER or r["impl"] in EXTRA_IMPLS, f"{where}: unknown impl"
        assert r["dtype"] in DTYPES, f"{where}: dtype {r['dtype']!r}"
        assert r["stage"] in {"0", "1", "2", "3", "4", "5"}, f"{where}: stage {r['stage']!r}"
        assert r["causal"] in {"0", "1"}, f"{where}: causal {r['causal']!r}"
        for k in ("B", "H", "N", "D"):
            assert int(r[k]) > 0, f"{where}: {k}={r[k]!r}"
        med, p10, p90 = (to_float(r[k]) for k in ("ms_median", "ms_p10", "ms_p90"))
        if med is None:
            assert any(t in r["notes"] for t in NO_TIMING_NOTES), (
                f"{where}: empty ms_median without an explaining note {NO_TIMING_NOTES}"
            )
            continue
        assert p10 is not None and p90 is not None and 0 < p10 <= med <= p90, (
            f"{where}: p10/median/p90 order"
        )
        if p90 / p10 >= NOISE_RATIO:
            assert "noisy" in r["notes"], (
                f"{where}: p90/p10 = {p90 / p10:.3f} >= {NOISE_RATIO} and not flagged `noisy`"
            )
        is_kernel_row = r["impl"] in IMPL_ORDER  # not smoke, not gpt_*
        if is_kernel_row:
            # BENCHMARK_PROTOCOL rule 1: >= 100 timed iterations, stated in the row itself.
            assert notes.get("iters", "").isdigit(), f"{where}: no `iters=<n>` note"
            assert int(notes["iters"]) >= MIN_ITERS, (
                f"{where}: iters={notes['iters']} < {MIN_ITERS} (protocol rule 1)"
            )
            # Formula drift guard: tflops must be 4*B*H*N^2*D / ms_median (halved only when the
            # row says flops=causal_half), nothing else.
            B, H, N, D = (int(r[k]) for k in "BHND")
            causal_half = "flops=causal_half" in r["notes"]
            expect = attention_flops(B, H, N, D, causal_half=causal_half) / (med * 1e-3) / 1e12
            tflops = to_float(r["tflops"])
            assert tflops is not None and abs(tflops - expect) <= TFLOPS_TOL * expect + 1e-3, (
                f"{where}: tflops={tflops} but 4BHN^2D/ms gives {expect:.3f}"
            )
        if r["impl"] in OUR_IMPLS.values():
            assert "check=" in r["notes"] or "max_err=" in r["notes"], (
                f"{where}: a row of our kernel carries no correctness note (`check=`/`max_err=`)"
            )
        tflops, roof = to_float(r["tflops"]), roof_for_dtype(roofs, r["dtype"])
        if tflops is not None and roof is not None:
            assert tflops <= roof, (
                f"{where}: {tflops} TFLOP/s exceeds the docs/ENV.md roof {roof} for {r['dtype']}"
            )


@pytest.mark.skipif(not METAS, reason="no profiles/<gpu>/stage<N>.meta committed yet")
@pytest.mark.parametrize("meta", METAS, ids=[os.path.relpath(p, ROOT) for p in METAS])
def test_profile_commit_matches_bench_rows(meta):
    """PROFILING_PROTOCOL: a profile is taken at the same commit as its benchmark row."""
    kv = {}
    for ln in open(meta, errors="replace"):
        if "=" in ln and not ln.startswith("#"):
            k, v = ln.split("=", 1)
            kv.setdefault(k.strip(), v.strip())
    commit, stage = kv.get("commit", ""), kv.get("stage", "")
    assert commit and "-dirty" not in commit, f"{meta}: commit {commit!r}"
    gpu = os.path.basename(os.path.dirname(meta))
    csv_path = os.path.join(ROOT, "bench", "results", gpu, f"stage{stage}.csv")
    if not os.path.exists(csv_path):
        pytest.skip(f"no {csv_path} yet for this profile")
    _, rows = read_rows(csv_path)
    ours = {r["commit"] for r in rows if r["impl"] == OUR_IMPLS.get(int(stage or 0))}
    if not ours:
        pytest.skip(f"{csv_path} has no row of our stage-{stage} kernel yet")
    assert commit in ours, (
        f"{meta}: profiled at {commit} but stage {stage} bench rows are at {sorted(ours)}"
    )
