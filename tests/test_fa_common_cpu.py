"""CPU tests for the shared CSV/ncu/ptxas helpers and the two benchmark harnesses' shared rules.

Why: these helpers decide what number reaches the README. Each test pins a case a review found
(or nearly found) wrong: ncu units printed in auto mode, a stage CSV with only the S0 `smoke`
row, the `used K barriers` clause and the indented spill line of `-Xptxas -v`, the OOM-by-design
rule implemented twice (C++ and Python), and the noise constants shared with bench_main.cu.
"""

from __future__ import annotations

import math
import os
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from fa_common import (  # noqa: E402
    CLOCK_SPREAD,
    NOISE_RATIO,
    SCHEMA,
    append_rows,
    latest_rows,
    normalize_unit,
    parse_ncu_raw_csv,
    parse_ptxas,
)

pytestmark = pytest.mark.filterwarnings("ignore::pytest.PytestUnknownMarkWarning")


# ---------------------------------------------------------------- ncu units --------------
@pytest.mark.parametrize(
    "unit, expected",
    [
        ("byte", (1.0, "byte")),
        ("Mbyte", (1e6, "byte")),
        ("Gbyte/second", (1e9, "byte/second")),
        ("cycle/nsecond", (1e9, "cycle/second")),
        ("cycle/usecond", (1e6, "cycle/second")),
        ("usecond", (1e-6, "second")),
        ("byte/cycle", (1.0, "byte/cycle")),
        ("inst/cycle", (1.0, "inst/cycle")),
        ("%", (1.0, "%")),
        ("register/thread", (1.0, "register/thread")),
        ("", (1.0, "")),
    ],
)
def test_normalize_unit_table(unit, expected):
    mult, base = normalize_unit(unit)
    assert base == expected[1]
    assert math.isclose(mult, expected[0], rel_tol=1e-12)


def test_normalize_unit_refuses_unknown_prefixed_forms():
    assert normalize_unit("Kbyte/Kcycle") is None


def _raw_csv(tmp_path: Path, units: dict[str, str], values: dict[str, str]) -> Path:
    cols = ["ID", "Kernel Name", "Grid Size", "Block Size", "Device", "CC", *units]
    unit_row = ["", "", "", "", "", "", *units.values()]
    data = ["0", "fa_s1_qk", "(64, 8, 2)", "(128, 1, 1)", "Tesla T4", "7.5", *values.values()]
    p = tmp_path / "stage1_qk.raw.csv"
    p.write_text(
        "==PROF== Connected to process 1\n"
        + ",".join(f'"{c}"' for c in cols)
        + "\n"
        + ",".join(f'"{u}"' for u in unit_row)
        + "\n"
        + ",".join(f'"{v}"' for v in data)
        + "\n"
    )
    return p


def test_parse_ncu_raw_csv_auto_units_are_scaled_to_base(tmp_path):
    """The plain `ncu --import --csv --page raw` (no --print-units base) output."""
    p = _raw_csv(
        tmp_path,
        {
            "dram__bytes.sum": "Mbyte",
            "sm__cycles_elapsed.avg": "cycle",
            "sm__cycles_elapsed.avg.per_second": "cycle/nsecond",
            "gpu__time_duration.sum": "usecond",
            "dram__bytes.sum.peak_sustained": "byte/cycle",
            "sm__warps_active.avg.pct_of_peak_sustained_active": "%",
        },
        {
            "dram__bytes.sum": "1,234.50",
            "sm__cycles_elapsed.avg": "3,589.80",
            "sm__cycles_elapsed.avg.per_second": "0.585",
            "gpu__time_duration.sum": "6.14",
            "dram__bytes.sum.peak_sustained": "320",
            "sm__warps_active.avg.pct_of_peak_sustained_active": "84.51",
        },
    )
    d = parse_ncu_raw_csv(p)
    v, u = d["values"], d["units"]
    assert d["unparsed_units"] == {}
    assert math.isclose(v["dram__bytes.sum"], 1234.5e6) and u["dram__bytes.sum"] == "byte"
    assert math.isclose(v["sm__cycles_elapsed.avg.per_second"], 0.585e9)
    assert u["sm__cycles_elapsed.avg.per_second"] == "cycle/second"
    assert (
        math.isclose(v["gpu__time_duration.sum"], 6.14e-6)
        and u["gpu__time_duration.sum"] == "second"
    )
    assert (
        v["dram__bytes.sum.peak_sustained"] == 320
        and u["dram__bytes.sum.peak_sustained"] == "byte/cycle"
    )
    # time = cycles / rate must come out in seconds, which is what roofline.py divides by
    t = v["sm__cycles_elapsed.avg"] / v["sm__cycles_elapsed.avg.per_second"]
    assert math.isclose(t, 3589.8 / 0.585e9)
    assert set(d["scaled"]) == {
        "dram__bytes.sum",
        "sm__cycles_elapsed.avg.per_second",
        "gpu__time_duration.sum",
    }
    assert d["meta"]["Kernel Name"] == "fa_s1_qk"


def test_parse_ncu_raw_csv_withholds_unparseable_units(tmp_path):
    p = _raw_csv(
        tmp_path,
        {"dram__bytes.sum": "Kbyte/Kcycle", "sm__cycles_elapsed.avg": "cycle"},
        {"dram__bytes.sum": "12", "sm__cycles_elapsed.avg": "100"},
    )
    d = parse_ncu_raw_csv(p)
    assert d["values"]["dram__bytes.sum"] is None
    assert d["unparsed_units"] == {"dram__bytes.sum": "Kbyte/Kcycle"}
    assert d["values"]["sm__cycles_elapsed.avg"] == 100


# ---------------------------------------------------------------- ptxas ------------------
PTXAS_NVCC_12_8 = """\
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z8fa_smokePKfPffm' for 'sm_75'
ptxas info    : Function properties for _Z8fa_smokePKfPffm
    0 bytes stack frame, 8 bytes spill stores, 4 bytes spill loads
ptxas info    : Used 10 registers, used 1 barriers, 2048 bytes smem, 384 bytes cmem[0]
ptxas info    : Compile time = 1.579 ms
ptxas info    : Compiling entry function '_Z10fa_s3_fusedILi64EEvPKfS1_S1_Pf' for 'sm_75'
ptxas info    : Function properties for _Z10fa_s3_fusedILi64EEvPKfS1_S1_Pf
ptxas info    : Used 96 registers, used 2 barriers, 384 bytes cmem[0]
"""


def test_parse_ptxas_handles_barriers_clause_and_indented_spill_line(tmp_path):
    p = tmp_path / "stage0_ptxas.txt"
    p.write_text(PTXAS_NVCC_12_8)
    e = parse_ptxas(p)
    assert [x["kernel"] for x in e] == ["fa_smoke", "fa_s3_fused"]
    assert e[0]["regs"] == 10 and e[0]["smem"] == 2048
    assert e[0]["spill_stores"] == 8 and e[0]["spill_loads"] == 4
    # second entry: no spill line in the report -> None (n/a), never 0; no smem -> 0
    assert e[1]["regs"] == 96 and e[1]["smem"] == 0
    assert e[1]["spill_stores"] is None and e[1]["spill_loads"] is None


def test_ptxas_summary_script_agrees(tmp_path):
    sys.path.insert(0, str(ROOT / ".github" / "scripts"))
    import ptxas_summary

    entries = ptxas_summary.parse(PTXAS_NVCC_12_8)
    assert [(e.regs, e.smem, e.spill_st, e.spill_ld) for e in entries] == [
        (10, 2048, 8, 4),
        (96, 0, None, None),
    ]
    assert "n/a / n/a" in ptxas_summary.render(entries, demangler="no-such-demangler")


# ---------------------------------------------------------------- README from a smoke CSV --
def _row(**kw) -> dict:
    base = {k: "" for k in SCHEMA}
    base.update(commit="abc1234", gpu="Tesla T4", driver="580.82.07", nvcc="12.8.93")
    base.update(kw)
    return base


def test_bench_to_md_smoke_only_stage0_renders_empty_block(tmp_path):
    """A stage0.csv holding only the fa_bench --smoke row must not crash headline_table()."""
    import bench_to_md

    results = tmp_path / "results"
    append_rows(
        results / "t4" / "stage0.csv",
        [
            _row(
                impl="smoke",
                stage="0",
                dtype="fp32",
                B="2",
                H="8",
                N="2048",
                D="64",
                causal="0",
                ms_median="0.0500",
                ms_p10="0.0490",
                ms_p90="0.0510",
                gbps="335.5",
                notes="smoke;clock=unlocked;iters=3;warmup=1",
            )
        ],
    )
    body = bench_to_md.render_bench(results)
    assert body == bench_to_md.EMPTY_TEXT["BENCH"]


def test_bench_to_md_refused_baseline_and_foreign_commit_baseline(tmp_path):
    import bench_to_md

    results = tmp_path / "results"
    shape = dict(dtype="fp32", B="2", H="8", N="2048", D="64", causal="0")
    append_rows(
        results / "t4" / "stage1.csv",
        [
            _row(
                impl="naive",
                stage="1",
                ms_median="10.0000",
                ms_p10="9.9000",
                ms_p90="10.1000",
                tflops="1.0",
                notes="iters=100;check=rows128;run=2026-09-02T10:00:00Z",
                **shape,
            ),
            _row(
                impl="sdpa_efficient",
                stage="0",
                commit="0ldc0de",
                ms_median="2.0000",
                ms_p10="1.9000",
                ms_p90="2.1000",
                tflops="5.0",
                notes="iters=100;run=2026-09-02T10:00:00Z",
                **shape,
            ),
            _row(
                impl="sdpa_flash",
                stage="0",
                notes="backend_refused:FLASH needs sm80+;run=2026-09-02T10:00:00Z",
                **shape,
            ),
        ],
    )
    body = bench_to_md.render_bench(results)
    assert "| refused |" in body
    assert "20% (naive) (baseline @ `0ldc0de`)" in body


def test_latest_rows_prefers_run_stamp_over_file_order():
    old = _row(
        impl="sdpa_efficient",
        stage="0",
        dtype="fp32",
        B="1",
        H="1",
        N="64",
        D="64",
        causal="0",
        ms_median="1",
        notes="run=2026-09-02T12:00:00Z",
    )
    newer = dict(
        old, ms_median="2", notes="run=2026-09-01T12:00:00Z"
    )  # appears later in file order, older stamp
    assert latest_rows([old, newer])[0]["ms_median"] == "1"
    unstamped_a, unstamped_b = (
        dict(old, notes="", ms_median="3"),
        dict(old, notes="", ms_median="4"),
    )
    assert latest_rows([unstamped_a, unstamped_b])[0]["ms_median"] == "4"


# ---------------------------------------------------------------- shared rules -----------
def _load_configs():
    import yaml

    return yaml.safe_load(open(ROOT / "bench" / "configs.yaml"))


def test_oom_by_design_rule_agrees_between_harnesses():
    """bench.py's rule and bench_main.cu's (`workspace_bytes > total_bytes / 4` with the fp32
    2*B*H*N*N*4 workspace) must classify every CONTRACT sweep shape identically."""
    import importlib.util
    import itertools

    spec = importlib.util.spec_from_file_location("bench_mod", ROOT / "bench" / "bench.py")
    # bench.py imports torch at module level; the CPU wheel is enough for this function.
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfgs = _load_configs()
    total = 15360 * 1024 * 1024  # T4 per the 2026-09-02 probe
    sw = cfgs["sweep"]
    for n, d, (b, h) in itertools.product(sw["N"], sw["D"], sw["BH"]):
        cpp_rule = 2 * b * h * n * n * 4 > total // 4
        for impl in ("naive", "tiled"):
            py_rule = mod.oom_by_design(impl, cfgs["oom_by_design"], b, h, n, total, 4)
            assert py_rule == cpp_rule, (impl, b, h, n, d)


def test_noise_constants_match_bench_main():
    src = (ROOT / "bench" / "bench_main.cu").read_text()
    m = re.search(r"kNoiseRatio\s*=\s*([0-9.]+)", src)
    assert m and float(m.group(1)) == NOISE_RATIO
    m = re.search(r"kClockSpread\s*=\s*([0-9.]+)", src)
    assert m and float(m.group(1)) == CLOCK_SPREAD


def test_gpu_run_and_local_compile_keep_the_spill_line():
    """Both scripts must use build.yml's ptxas pattern (the spill line has no `ptxas` prefix)."""
    for rel in ("scripts/gpu_run.sh", "scripts/local_compile.sh", ".github/workflows/build.yml"):
        text = (ROOT / rel).read_text()
        assert "bytes spill|bytes stack frame" in text, rel
    assert "grep -E '^ptxas' build/build.log" not in (ROOT / "scripts" / "gpu_run.sh").read_text()
    assert os.path.exists(ROOT / "scripts" / "gpu_run.sh")
