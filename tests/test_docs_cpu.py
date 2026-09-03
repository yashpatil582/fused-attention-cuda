"""CPU test: the README's generated blocks match the committed CSVs/profiles (CONTRACT §6).

`scripts/bench_to_md.py --check` and `scripts/ncu_to_md.py --check` regenerate the BENCH / GPT
and NCU blocks in memory and exit 1 on any difference, so a hand-edited or stale table fails
CI. A `TBD` inside a generated block is a placeholder that reached the README and fails too.
When README.md does not exist yet (parallel S0 scaffold) the tests skip with that reason.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README = os.path.join(ROOT, "README.md")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from fa_common import BLOCK_TAGS, get_block  # noqa: E402


def _run(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts", script), "--check"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )


def _readme_text() -> str:
    if not os.path.exists(README):
        pytest.skip("README.md not created yet; generated blocks absent")
    return open(README).read()


@pytest.mark.parametrize("script", ["bench_to_md.py", "ncu_to_md.py"])
def test_generated_blocks_are_fresh(script):
    _readme_text()
    p = _run(script)
    assert p.returncode == 0, f"{script} --check failed:\n{p.stdout}\n{p.stderr}"


@pytest.mark.parametrize("tag", BLOCK_TAGS)
def test_no_tbd_inside_generated_blocks(tag):
    text = _readme_text()
    block = get_block(text, tag)
    if block is None:
        pytest.skip(f"README has no <!-- {tag}:START/END --> block yet")
    assert "TBD" not in block, f"placeholder TBD inside the generated {tag} block"
