#!/usr/bin/env python3
"""Stage -> kernel-name regexes for `ncu --kernel-name regex:<...>` (CONTRACT.md §2).

Why a script and not a bash array: profile.sh, ncu_to_md.py and perf_trailer.py all need
the same (stage, short name, regex) triples, and the kernel identifiers are frozen in the
contract — one table, one place to change it if the contract ever changes.

The regexes are unanchored substrings of the frozen identifiers: all kernels are templates,
so the demangled name looks like `void fa_s1_qk<64>(const float*, ...)`; a substring match
works under every `--kernel-name-base` mode (function / demangled / mangled).

    python scripts/kernel_regex.py 1          -> "qk<TAB>fa_s1_qk" ... one line per kernel
    python scripts/kernel_regex.py 1 --json   -> [{"short": "qk", "regex": "fa_s1_qk"}, ...]
    python scripts/kernel_regex.py --cfg C64  -> B2_H8_N2048_D64   (canonical cfg -> string)
    python scripts/kernel_regex.py --dtype 4  -> fp16               (default dtype per stage)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fa_common import repo_root  # noqa: E402

# CONTRACT.md §2 — exact identifiers. "short" is what the profile file is named after:
# profiles/<gpu>/stage<N>_<short>.ncu-rep (judge_plan §4.5: stage1_{qk,softmax,pv}, stage3_fused).
KERNELS: dict[int, list[tuple[str, str]]] = {
    0: [("smoke", "fa_smoke")],
    1: [("qk", "fa_s1_qk"), ("softmax", "fa_s1_softmax"), ("pv", "fa_s1_pv")],
    2: [
        ("qk_tiled", "fa_s2_qk_tiled"),
        ("softmax", "fa_s2_softmax"),
        ("pv_tiled", "fa_s2_pv_tiled"),
    ],
    3: [("fused", "fa_s3_fused")],
    4: [("wmma", "fa_s4_wmma")],
    5: [("tuned", "fa_s5_tuned")],
}


def kernels_for(stage: int) -> list[tuple[str, str]]:
    if stage not in KERNELS:
        raise SystemExit(f"unknown stage {stage}; known: {sorted(KERNELS)}")
    return KERNELS[stage]


def _configs() -> dict:
    import yaml  # local import: only the --cfg/--dtype paths need it

    with open(repo_root() / "bench" / "configs.yaml") as f:
        return yaml.safe_load(f)


def canonical_cfg_string(name: str) -> str:
    c = _configs()["canonical"]
    if name not in c:
        raise SystemExit(f"unknown canonical config {name!r}; known: {sorted(c)}")
    v = c[name]
    return f"B{v['B']}_H{v['H']}_N{v['N']}_D{v['D']}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("stage", nargs="?", type=int, help="stage number 0..5")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of TSV")
    ap.add_argument("--cfg", help="print the B_H_N_D string of a canonical config (C64/C128)")
    ap.add_argument("--dtype", type=int, metavar="STAGE", help="print the default dtype of a stage")
    a = ap.parse_args(argv)
    if a.cfg:
        print(canonical_cfg_string(a.cfg))
        return 0
    if a.dtype is not None:
        print(_configs()["stage_dtype"].get(a.dtype, "fp32"))
        return 0
    if a.stage is None:
        ap.error("stage is required unless --cfg/--dtype is given")
    ks = kernels_for(a.stage)
    if a.json:
        print(json.dumps([{"short": s, "regex": r} for s, r in ks]))
    else:
        for s, r in ks:
            print(f"{s}\t{r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
