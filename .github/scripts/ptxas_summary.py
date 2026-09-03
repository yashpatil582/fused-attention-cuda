"""Turn `nvcc -Xptxas -v` output into a per-kernel register/spill table.

Used by .github/workflows/build.yml: the CI build log is filtered to the ptxas lines
and this script writes a Markdown table to $GITHUB_STEP_SUMMARY so a reviewer can see
registers, spills and static shared memory for every (kernel, sm_XX) pair without
downloading the artifact. It never gates the job: spills are recorded, not forbidden
(S5c is the stage that promises spill-free launch bounds, and it proves it here).

Input format (ptxas -v, one block per entry function; the spill line is indented
and does NOT start with "ptxas"):

    ptxas info    : Compiling entry function '_Z8fa_smokePf' for 'sm_75'
    ptxas info    : Function properties for _Z8fa_smokePf
        0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
    ptxas info    : Used 8 registers, 8 bytes smem, 356 bytes cmem[0]

nvcc >= 12.4 inserts a `used K barriers` clause before the smem figure
(`Used 10 registers, used 0 barriers, 384 bytes cmem[0]`), which the regex skips. A report
that was filtered with `grep ^ptxas` lacks the indented spill line; spills are then reported
as n/a, never as 0.

Usage: python .github/scripts/ptxas_summary.py <ptxas-report.txt> [--markdown OUT]
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_ENTRY = re.compile(r"Compiling entry function '([^']+)' for '(sm_\d+)'")
_SPILL = re.compile(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads")
_USED = re.compile(r"Used (\d+) registers?(?:.*?, (\d+) bytes smem)?")


@dataclass
class Entry:
    name: str
    arch: str
    stack: int | None = None
    spill_st: int | None = None
    spill_ld: int | None = None
    regs: int = 0
    smem: int = 0


def parse(text: str) -> list[Entry]:
    """Parse ptxas -v text; entries are grouped by the 'Compiling entry function' line."""
    entries: list[Entry] = []
    cur: Entry | None = None
    for line in text.splitlines():
        m = _ENTRY.search(line)
        if m:
            cur = Entry(name=m.group(1), arch=m.group(2))
            entries.append(cur)
            continue
        if cur is None:
            continue
        m = _SPILL.search(line)
        if m:
            cur.stack, cur.spill_st, cur.spill_ld = (int(g) for g in m.groups())
            continue
        m = _USED.search(line)
        if m:
            cur.regs = int(m.group(1))
            cur.smem = int(m.group(2) or 0)
    return entries


def render(entries: list[Entry], demangler: str = "c++filt") -> str:
    """Markdown table; kernel names are shown mangled unless c++filt is available."""
    names = [e.name for e in entries]
    shown = names
    if names and shutil.which(demangler):
        out = subprocess.run(
            [demangler], input="\n".join(names), capture_output=True, text=True, check=False
        )
        if out.returncode == 0:
            shown = out.stdout.splitlines()
            if len(shown) != len(names):
                shown = names
    spilled = sum(1 for e in entries if e.spill_st or e.spill_ld)
    unknown = sum(1 for e in entries if e.spill_st is None)
    lines = [
        f"**ptxas -v**: {len(entries)} (kernel, arch) entries, "
        f"{spilled} with register spills"
        + (f" ({unknown} without a spill line in the report)" if unknown else "")
        + f", max registers {max((e.regs for e in entries), default=0)}",
        "",
        "| kernel | arch | regs | spill st / ld (B) | stack (B) | static smem (B) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for e, n in zip(entries, shown, strict=True):
        # Template kernels demangle to long signatures; keep the table scannable.
        short = n if len(n) <= 90 else n[:87] + "..."
        flag = " **!**" if (e.spill_st or e.spill_ld) else ""

        def na(v: int | None) -> str:
            return "n/a" if v is None else str(v)

        lines.append(
            f"| `{short}` | {e.arch} | {e.regs} | {na(e.spill_st)} / {na(e.spill_ld)}{flag} "
            f"| {na(e.stack)} | {e.smem} |"
        )
    if not entries:
        lines.append(
            "| (no ptxas entries found — is `-Xptxas -v` set in CMakeLists.txt?) | | | | | |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("report", type=Path, help="file with the filtered ptxas lines")
    ap.add_argument(
        "--markdown", type=Path, help="append the table here (e.g. $GITHUB_STEP_SUMMARY)"
    )
    args = ap.parse_args(argv)

    entries = parse(args.report.read_text(errors="replace"))
    table = render(entries)
    if args.markdown:
        with args.markdown.open("a") as fh:
            fh.write(table)
    sys.stdout.write(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
