"""Shared, GPU-free helpers for the bench/profile/docs scripts.

Why one module: the frozen CSV schema (CONTRACT.md §3), the impl -> stage map, the
README block markers and the ENV.md roof convention are consumed by bench.py,
gpt_block.py, bench_to_md.py, ncu_to_md.py, roofline.py, perf_trailer.py and two CPU
tests. Keeping them here means the schema can only drift in one place, and that
place is checked by tests/test_bench_sanity.py against every committed CSV.

Nothing here imports torch, so every consumer that only reads CSVs runs on the Mac.
"""

from __future__ import annotations

import csv
import glob
import os
import re
import subprocess
from pathlib import Path

# CONTRACT.md §3 — frozen; order matters, the tests compare the header exactly.
SCHEMA = [
    "commit",
    "gpu",
    "driver",
    "nvcc",
    "torch",
    "ncu",
    "clock_mhz",
    "impl",
    "stage",
    "dtype",
    "B",
    "H",
    "N",
    "D",
    "causal",
    "ms_median",
    "ms_p10",
    "ms_p90",
    "tflops",
    "gbps",
    "notes",
]

# Our kernels, by stage (CONTRACT §3 `impl` values, §2 kernel names).
OUR_IMPLS = {1: "naive", 2: "tiled", 3: "fused", 4: "wmma", 5: "tuned"}
STAGE_OF_IMPL = {v: k for k, v in OUR_IMPLS.items()}
BASELINE_IMPLS = [
    "sdpa_efficient",
    "sdpa_flash",
    "sdpa_cudnn",
    "sdpa_math_fp32",
    "torch_unfused",
    "flash_attn",
]
IMPL_ORDER = list(OUR_IMPLS.values()) + BASELINE_IMPLS
# CONTRACT §3 also allows `smoke` (S0 fa_bench round trip, never rendered) and the `gpt_*`
# impls of gpt_block.csv; neither enters the README benchmark tables.
EXTRA_IMPLS = ("smoke", "gpt_sdpa_efficient", "gpt_sdpa_flash", "gpt_fused")
DTYPES = ("fp32", "fp16", "bf16")

# p90/p10 at or above this -> the row must carry `noisy` in notes. One constant for
# bench_main.cu (kNoiseRatio, kept equal by tests/test_fa_common_cpu.py), bench.py,
# gpt_block.py and tests/test_bench_sanity.py.
NOISE_RATIO = 1.15
# Clock samples spanning more than this fraction of their median -> `noisy_clock`.
CLOCK_SPREAD = 0.10

# README generated-block markers (CONTRACT §6) and the empty-state texts the README
# ships with, so `--check` passes on a repo with no committed results.
BLOCK_TAGS = ("BENCH", "NCU", "GPT")
EMPTY_TEXT = {
    "BENCH": "_No benchmark results committed yet._",
    "NCU": "_No profiles committed yet._",
    "GPT": "_No GPT-block results yet._",
}

# Row key: one row per (gpu dir, impl, stage, dtype, shape, causal); the newest row wins so a
# re-run appends and the newest measurement is what the README shows. "Newest" is the
# `run=<utc>` stamp bench.py/gpt_block.py write into notes; rows without a stamp fall back to
# file order (stage files are concatenated in stage order, so an unstamped baseline re-measured
# by a later stage run would otherwise shadow a newer one from a lower stage file).
ROW_KEY = ("impl", "stage", "dtype", "B", "H", "N", "D", "causal")
RUN_RE = re.compile(r"(?:^|;)run=([0-9T:Z-]+)")


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def git_commit(root: str | os.PathLike | None = None) -> str:
    """Short SHA of HEAD, with `-dirty` appended when a TRACKED source file differs from HEAD
    (CONTRACT §3: `commit` = the source under test). bench/results and profiles are excluded:
    a run appends to them by design. `unknown` outside a git checkout."""
    root = str(root or repo_root())

    def run(args: list[str]) -> str | None:
        try:
            r = subprocess.run(
                ["git", "-C", root, *args], capture_output=True, text=True, timeout=30
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return r.stdout.strip() if r.returncode == 0 else None

    sha = run(["rev-parse", "--short", "HEAD"])
    if not sha:
        return "unknown"
    dirty = run(
        [
            "status",
            "--porcelain",
            "--untracked-files=no",
            "--",
            ".",
            ":(exclude)bench/results",
            ":(exclude)profiles",
        ]
    )
    return f"{sha}-dirty" if dirty else sha


def utc_stamp() -> str:
    import datetime as _dt

    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_stamp(r: dict) -> str:
    m = RUN_RE.search(r.get("notes", "") or "")
    return m.group(1) if m else ""


# ---------------------------------------------------------------- CSV I/O ---------------
def read_rows(path: str | os.PathLike) -> tuple[list[str], list[dict]]:
    """Returns (header, rows). Rows are dicts keyed by the header as written."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames or [])
        rows = [dict(r) for r in reader]
    return header, rows


def append_rows(path: str | os.PathLike, rows: list[dict]) -> None:
    """Appends rows in SCHEMA order; writes the header only when the file is new/empty."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SCHEMA, extrasaction="ignore")
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in SCHEMA})


def write_rows(path: str | os.PathLike, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SCHEMA, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in SCHEMA})


def row_key(r: dict) -> tuple:
    return tuple(str(r.get(k, "")) for k in ROW_KEY)


def latest_rows(rows: list[dict]) -> list[dict]:
    """Dedupes by ROW_KEY keeping the newest occurrence: the greatest `run=<utc>` stamp, ties
    (and unstamped rows) resolved by file order (later wins)."""
    seen: dict[tuple, dict] = {}
    for r in rows:
        k = row_key(r)
        if k not in seen or run_stamp(r) >= run_stamp(seen[k]):
            seen[k] = r
    return list(seen.values())


def stage_csvs(results_dir: str | os.PathLike) -> dict[str, list[str]]:
    """{gpu_dir_name: [stage1.csv, stage2.csv, ...]} sorted by stage number."""
    out: dict[str, list[str]] = {}
    for p in sorted(glob.glob(os.path.join(str(results_dir), "*", "stage*.csv"))):
        # Only the per-stage result files: `stage5_tuning.csv` (the S5 A/B sweep, every variant
        # under FA_FORCE_BR/BC/FA_S5_PREFETCH) must not feed summary.csv or the README, whose
        # S5 column is the DEFAULT configuration only.
        if not re.fullmatch(r"stage\d+\.csv", os.path.basename(p)):
            continue
        gpu = os.path.basename(os.path.dirname(p))
        out.setdefault(gpu, []).append(p)
    for gpu in out:
        out[gpu].sort(key=lambda p: int(re.sub(r"\D", "", os.path.basename(p)) or 0))
    return out


def load_stage_rows(results_dir: str | os.PathLike) -> dict[str, list[dict]]:
    """{gpu_dir: deduped rows across every stageN.csv of that gpu dir}."""
    out = {}
    for gpu, files in stage_csvs(results_dir).items():
        rows: list[dict] = []
        for f in files:
            _, r = read_rows(f)
            rows.extend(r)
        out[gpu] = latest_rows(rows)
    return out


# ---------------------------------------------------------------- numbers ---------------
def percentile(sorted_vals: list[float], p: float) -> float:
    """Linear-interpolated percentile on an already sorted list (p in [0, 100])."""
    if not sorted_vals:
        raise ValueError("empty sample")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (len(sorted_vals) - 1) * p / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def to_float(s) -> float | None:
    """Parses ncu/CSV numerics: strips thousands separators, maps n/a to None."""
    if s is None:
        return None
    t = str(s).strip().replace(",", "")
    if t in ("", "n/a", "N/A", "-", "nan"):
        return None
    try:
        return float(t)
    except ValueError:
        return None


def fmt_bytes(n: float | None) -> str:
    if n is None:
        return "n/a"
    for unit, div in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if abs(n) >= div:
            return f"{n / div:.2f} {unit}"
    return f"{n:.0f} B"


def fmt_num(x: float | None, nd: int = 2) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def attention_flops(B: int, H: int, N: int, D: int, causal_half: bool = False) -> float:
    """Forward FLOPs = 4*B*H*N^2*D (QK^T + PV). A causal row of a kernel that skips the fully
    masked tiles (fused stages S3+, and the SDPA/flash baselines) counts half of that — the
    FlashAttention benchmark convention (flops // 2 if causal) — and carries `flops=causal_half`
    in notes; a causal row of a materialising kernel (S1, S2, torch_unfused, the MATH oracle)
    keeps the full count and says `flops=full` (CONTRACT §3, protocol rule 5)."""
    return 4.0 * B * H * N * N * D * (0.5 if causal_half else 1.0)


def cfg_string(r: dict, with_dtype: bool = False) -> str:
    s = f"B{r['B']}_H{r['H']}_N{r['N']}_D{r['D']}"
    return f"{s}_{r['dtype']}" if with_dtype else s


def parse_cfg(s: str) -> dict[str, int]:
    """'B8_H8_N4096_D64' -> {'B': 8, 'H': 8, 'N': 4096, 'D': 64} (CONTRACT §5 format)."""
    m = re.fullmatch(r"B(\d+)_H(\d+)_N(\d+)_D(\d+)", s.strip())
    if not m:
        raise ValueError(f"bad config string {s!r}; expected B<b>_H<h>_N<n>_D<d>")
    return dict(zip("BHND", map(int, m.groups()), strict=True))


# ---------------------------------------------------------------- README blocks ---------
def block_re(tag: str) -> re.Pattern:
    # group(1) = opening marker + newline, group(2) = body, group(3) = closing marker; the
    # newline before the closing marker is consumed OUTSIDE the groups so that re-rendering
    # the same body is a no-op (no newline accumulates on every run).
    return re.compile(rf"(<!-- {tag}:START -->\n)(.*?)\n?(<!-- {tag}:END -->)", re.S)


def get_block(text: str, tag: str) -> str | None:
    m = block_re(tag).search(text)
    return None if m is None else m.group(2)


def replace_block(text: str, tag: str, body: str) -> str:
    """Replaces the body between the markers; the markers themselves are preserved."""
    m = block_re(tag).search(text)
    if m is None:
        raise KeyError(f"README has no <!-- {tag}:START/END --> block")
    # Preserve the README author's padding style (a blank line before the END marker or not)
    # so regeneration never fights hand-written layout and `--check` stays whitespace-stable.
    pad = "\n" if m.group(2).endswith("\n") else ""
    return (
        text[: m.start()]
        + m.group(1)
        + body.rstrip("\n")
        + "\n"
        + pad
        + m.group(3)
        + text[m.end() :]
    )


# ---------------------------------------------------------------- docs/ENV.md roofs -----
# Convention (documented for the docs owner): docs/ENV.md carries machine-readable lines
#   ROOF <name> = <number> <unit> (<provenance: measured|spec, not measured>; <url>)
# e.g. `ROOF fp16_tensor_tflops = 65 TFLOP/s (spec, not measured; https://...)`.
# Only the `ROOF <name> = <number>` prefix is parsed; everything after is free text.
ROOF_RE = re.compile(r"^\s*ROOF\s+([A-Za-z0-9_]+)\s*=\s*([0-9]+(?:\.[0-9]+)?)", re.M)


def read_env_roofs(env_md: str | os.PathLike | None = None) -> dict[str, float]:
    path = Path(env_md) if env_md else repo_root() / "docs" / "ENV.md"
    if not path.exists():
        return {}
    return {k: float(v) for k, v in ROOF_RE.findall(path.read_text())}


def roof_for_dtype(roofs: dict[str, float], dtype: str) -> float | None:
    """fp32 rows are capped by the fp32 FMA roof; fp16/bf16 rows by the tensor roof."""
    if dtype == "fp32":
        return roofs.get("fp32_tflops")
    return roofs.get("fp16_tensor_tflops") or roofs.get("bf16_tensor_tflops")


# ---------------------------------------------------------------- ncu raw CSV -----------
# `ncu --import x.ncu-rep --csv --page raw` = header row (ID, Process ID, ..., Kernel Name,
# ..., <metric IDs>), then ONE units row, then one row per profiled launch. Values may carry
# thousands separators. profile.sh imports with `--print-units base` (bytes in bytes, rates
# per second); if that flag is refused the plain import prints auto-scaled units (`Mbyte`,
# `Gbyte/second`, `cycle/nsecond`, `usecond`, ...). Every unit is normalised to base here with
# the complete table below; a unit that cannot be normalised makes the value None and is
# reported in `unparsed_units`, so a mis-scaled number can never reach a table or a roofline.
UNIT_SCALE = {"K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12}
TIME_UNITS = {"second": 1.0, "msecond": 1e-3, "usecond": 1e-6, "nsecond": 1e-9}
PER_TIME = {"second": 1.0, "msecond": 1e3, "usecond": 1e6, "nsecond": 1e9}  # 1/TIME_UNITS, exact
META_COLS = ("Kernel Name", "Block Size", "Grid Size", "Device", "CC", "ID")
_PREFIXED = re.compile(r"^([KMGT])([a-z][a-z0-9]*)$")


def normalize_unit(unit: str) -> tuple[float, str] | None:
    """(multiplier, base_unit) so that value * multiplier is in base_unit; None if unknown.

    Rules (ncu's own vocabulary): an SI prefix K/M/G/T before a lowercase quantity scales the
    numerator (`Mbyte` -> byte x 1e6); a time denominator `msecond/usecond/nsecond` becomes
    `/second` (`cycle/nsecond` -> cycle/second x 1e9); a bare time unit becomes `second`
    (`usecond` -> second x 1e-6). Anything else with a prefix or an unknown denominator is
    refused. Prefix-free units (`byte`, `%`, `inst`, `register/thread`, `byte/cycle`, ...)
    are base already and pass through unchanged."""
    u = (unit or "").strip()
    if u == "":
        return 1.0, u
    numer, _, denom = u.partition("/")
    mult = 1.0
    if numer in TIME_UNITS and not denom:
        return TIME_UNITS[numer], "second"
    m = _PREFIXED.match(numer)
    if m:
        mult *= UNIT_SCALE[m.group(1)]
        numer = m.group(2)
    if denom:
        if denom in PER_TIME:
            mult *= PER_TIME[denom]
            denom = "second"
        elif _PREFIXED.match(denom) or denom in ("", "/"):
            return None
    if _PREFIXED.match(numer):  # a prefix survived (e.g. `MKbyte`): not a unit we understand
        return None
    return mult, (f"{numer}/{denom}" if denom else numer)


def parse_ncu_raw_csv(path: str | os.PathLike) -> dict:
    """Returns {'meta': {...}, 'values': {metric: float|None}, 'units': {metric: base unit},
    'scaled': [metrics whose unit was converted to base], 'unparsed_units': {metric: unit
    as printed} (value forced to None)} for the FIRST launch row."""
    empty = {"meta": {}, "values": {}, "units": {}, "scaled": [], "unparsed_units": {}}
    with open(path, newline="", errors="replace") as f:
        lines = [ln for ln in f.read().splitlines() if not ln.startswith("==")]
    rows = list(csv.reader(lines))
    hdr_i = next((i for i, r in enumerate(rows) if "Kernel Name" in r), None)
    if hdr_i is None or len(rows) < hdr_i + 3:
        return dict(empty, error="no launch row")
    header, units_row, data = rows[hdr_i], rows[hdr_i + 1], rows[hdr_i + 2]
    meta, values, units, scaled, unparsed = {}, {}, {}, [], {}
    for col, unit, val in zip(header, units_row, data, strict=False):  # ragged rows tolerated
        if col in META_COLS:
            meta[col] = val
            continue
        if "__" not in col:
            continue
        v = to_float(val)
        u = (unit or "").strip()
        norm = normalize_unit(u)
        if norm is None:
            unparsed[col] = u
            values[col], units[col] = None, u
            continue
        mult, base = norm
        if v is not None and mult != 1.0:
            v *= mult
            scaled.append(col)
        values[col], units[col] = v, base
    return {
        "meta": meta,
        "values": values,
        "units": units,
        "scaled": scaled,
        "unparsed_units": unparsed,
    }


RAW_NAME_RE = re.compile(r"^stage(\d+)_(.+?)(?:_(C\d+))?\.raw\.csv$")


def raw_csv_files(profiles_dir: str | os.PathLike) -> list[dict]:
    """Every profiles/<gpu>/stage<N>_<kernel>[_C128].raw.csv with its parsed name parts."""
    out = []
    for p in sorted(glob.glob(os.path.join(str(profiles_dir), "*", "stage*_*.raw.csv"))):
        m = RAW_NAME_RE.match(os.path.basename(p))
        if not m:
            continue
        out.append(
            {
                "path": p,
                "gpu": os.path.basename(os.path.dirname(p)),
                "stage": int(m.group(1)),
                "kernel": m.group(2),
                "cfg": m.group(3) or "C64",
            }
        )
    return out


# ---------------------------------------------------------------- ptxas -v report -------
PTXAS_ENTRY_RE = re.compile(r"Compiling entry function '([^']+)' for '(sm_\d+)'")


# nvcc >= 12.4 prints `Used N registers, used K barriers, M bytes smem, C bytes cmem[0]`; the
# non-greedy skip tolerates the barriers clause (and its absence on older nvcc).
PTXAS_USED_RE = re.compile(r"Used (\d+) registers(?:.*?, (\d+) bytes smem)?")
# The spill line is indented and has no `ptxas` prefix; a report filtered with `grep ^ptxas`
# lacks it, in which case spills are reported as None (n/a), never as 0.
PTXAS_SPILL_RE = re.compile(r"(\d+) bytes spill (stores|loads)")


def parse_ptxas(path: str | os.PathLike) -> list[dict]:
    """[{kernel, sm, regs, smem, spill_stores, spill_loads}] from `-Xptxas -v` output lines.
    spill_* are None when the report carries no spill line for the kernel."""
    out, cur = [], None
    if not Path(path).exists():
        return out
    for line in Path(path).read_text(errors="replace").splitlines():
        m = PTXAS_ENTRY_RE.search(line)
        if m:
            k = re.search(r"fa_s\d_[a-z_]+|fa_smoke", m.group(1))
            cur = {
                "kernel": k.group(0) if k else m.group(1)[:48],
                "sm": m.group(2),
                "regs": None,
                "smem": 0,
                "spill_stores": None,
                "spill_loads": None,
            }
            continue
        if cur is None:
            continue
        for n, kind in PTXAS_SPILL_RE.findall(line):
            cur[f"spill_{kind}"] = int(n)
        m = PTXAS_USED_RE.search(line)
        if m:
            cur["regs"], cur["smem"] = int(m.group(1)), int(m.group(2) or 0)
            out.append(cur)
            cur = None
    return out
