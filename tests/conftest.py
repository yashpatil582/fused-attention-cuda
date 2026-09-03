"""Shared pytest configuration for fused-attention-cuda.

* Puts ``<root>/python`` on ``sys.path`` so ``import fused_attn`` works from a clean checkout
  without installing the package (the CPU model needs no build at all).
* ``MPLBACKEND=Agg`` before anything imports matplotlib (headless CI / Colab / SSH).
* ``gpu`` marker: skipped automatically when ``torch.cuda.is_available()`` is False.
* ``--stage N``: restrict the stage-parametrized GPU tests to one stage (default: all).
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

os.environ.setdefault("MPLBACKEND", "Agg")

STAGES = (1, 2, 3, 4, 5)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--stage",
        action="store",
        default="all",
        help="run stage-parametrized GPU tests for one stage only (1..5); default: all",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "gpu: needs a CUDA device; auto-skipped when torch.cuda.is_available() is False"
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    have_cuda = torch.cuda.is_available()
    skip_gpu = pytest.mark.skip(reason="no CUDA device (torch.cuda.is_available() is False)")
    wanted = config.getoption("--stage")
    only_stage = None if wanted in (None, "all") else int(wanted)
    deselected: list[pytest.Item] = []
    kept: list[pytest.Item] = []
    for item in items:
        if "gpu" in item.keywords and not have_cuda:
            item.add_marker(skip_gpu)
        params = getattr(getattr(item, "callspec", None), "params", {})
        if only_stage is not None and "stage" in params and params["stage"] != only_stage:
            deselected.append(item)
        else:
            kept.append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = kept


@pytest.fixture(scope="session")
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# (B, H, N, D) shapes.  CPU shapes are small because the fp64 model is pure torch on a laptop;
# GPU shapes are subsampled from the CONTRACT.md 5 sweep and sized so the naive stages' fp32
# N x N workspace (2*B*H*N*N*4 bytes) stays far below T4 memory.
CPU_SHAPES = [(1, 2, 64, 64), (2, 2, 128, 128), (1, 3, 256, 64)]
GPU_SHAPES = [
    (1, 1, 64, 64),
    (2, 4, 256, 64),
    (1, 2, 512, 128),
    (2, 2, 1024, 64),
    (1, 2, 2048, 64),
]


def shape_id(shape: tuple[int, int, int, int]) -> str:
    """CONTRACT.md 5 config-string format, e.g. ``B8_H8_N4096_D64``."""
    b, h, n, d = shape
    return f"B{b}_H{h}_N{n}_D{d}"


@pytest.fixture(params=CPU_SHAPES, ids=shape_id)
def cpu_shape(request: pytest.FixtureRequest) -> tuple[int, int, int, int]:
    return request.param


@pytest.fixture(params=GPU_SHAPES, ids=shape_id)
def gpu_shape(request: pytest.FixtureRequest) -> tuple[int, int, int, int]:
    return request.param


@pytest.fixture
def make_qkv() -> Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Seeded random q/k/v: ``make_qkv(B, H, N, D, dtype=, device=, hkv=, seed=, scale_inputs=)``.

    Generated on the CPU with a fixed seed then moved/cast, so the same (seed, shape) gives the
    same values on every machine and dtype (only the final rounding differs)."""

    def _make(
        b: int,
        h: int,
        n: int,
        d: int,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
        hkv: int | None = None,
        seed: int = 0,
        scale_inputs: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gen = torch.Generator(device="cpu").manual_seed(seed)
        hkv = h if hkv is None else hkv
        q = torch.randn(b, h, n, d, generator=gen, dtype=torch.float32)
        k = torch.randn(b, hkv, n, d, generator=gen, dtype=torch.float32)
        v = torch.randn(b, hkv, n, d, generator=gen, dtype=torch.float32)
        out = tuple(
            (t * scale_inputs).to(device=device, dtype=dtype).contiguous() for t in (q, k, v)
        )
        return out  # type: ignore[return-value]

    return _make
