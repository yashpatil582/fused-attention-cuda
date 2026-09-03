"""fused_attn -- PyTorch bindings for fused-attention-cuda.

Import always succeeds, even on a machine without the compiled extension (e.g. the Mac running
``tests/test_reference_cpu.py``): the fp64 CPU model in :mod:`fused_attn.reference` needs no CUDA.
Calling :func:`forward` / :func:`forward_stage` without the extension raises an ``ImportError``
that says how to build it.

Build (Colab T4, from the repo root)::

    cd python && TORCH_CUDA_ARCH_LIST="7.5" python setup.py build_ext --inplace

Ops (schemas frozen in CONTRACT.md 4, registered with TORCH_LIBRARY in ``ext.cpp``)::

    torch.ops.fused_attn.forward(q, k, v, is_causal, scale) -> Tensor
    torch.ops.fused_attn.forward_stage(q, k, v, is_causal, scale, stage) -> (Tensor, Tensor)
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from . import ops, reference

__all__ = [
    "__version__",
    "forward",
    "forward_stage",
    "impl_name",
    "impl_supports",
    "is_available",
    "ops",
    "reference",
    "set_debug_overrides",
]
__version__ = "0.1.0"

_EXT_ERROR: ImportError | None
try:
    from . import _C  # noqa: F401  (loading the .so runs the TORCH_LIBRARY static initializers)
except ImportError as exc:  # pragma: no cover - depends on the machine
    _EXT_ERROR = exc
else:
    _EXT_ERROR = None
    ops.register_fakes()

_BUILD_HINT = (
    "fused_attn._C is not built. From the repo root run\n"
    '  cd python && TORCH_CUDA_ARCH_LIST="7.5" python setup.py build_ext --inplace\n'
    "(needs nvcc + a CUDA-enabled torch; set TORCH_CUDA_ARCH_LIST to your GPU's compute "
    "capability)."
)

_DTYPE_NAMES = {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}


def is_available() -> bool:
    """True when the compiled extension imported."""
    return _EXT_ERROR is None


def _require_ext() -> None:
    if _EXT_ERROR is not None:
        raise ImportError(_BUILD_HINT) from _EXT_ERROR


def _resolve_scale(q: Tensor, scale: float | None) -> float:
    # PyTorch's SDPA default; the kernels take the scale explicitly so there is exactly one place
    # (here) where the default is decided.
    return float(1.0 / math.sqrt(q.shape[-1]) if scale is None else scale)


def forward(
    q: Tensor, k: Tensor, v: Tensor, is_causal: bool = False, scale: float | None = None
) -> Tensor:
    """Fused attention forward, dispatched to the best implemented stage for (dtype, D, GPU)."""
    _require_ext()
    return torch.ops.fused_attn.forward(q, k, v, bool(is_causal), _resolve_scale(q, scale))


def forward_stage(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    stage: int,
    is_causal: bool = False,
    scale: float | None = None,
) -> tuple[Tensor, Tensor]:
    """Run one explicit stage (1..5).  Returns ``(o, lse)``; ``lse`` is empty for stages 1-2."""
    _require_ext()
    return torch.ops.fused_attn.forward_stage(
        q, k, v, bool(is_causal), _resolve_scale(q, scale), int(stage)
    )


def impl_supports(stage: int, dtype: torch.dtype, head_dim: int) -> bool:
    """Whether this build can run ``stage`` for (dtype, D) on the *current* CUDA device."""
    _require_ext()
    name = _DTYPE_NAMES.get(dtype)
    if name is None:
        return False
    return bool(torch.ops.fused_attn.impl_supports(int(stage), name, int(head_dim)))


def impl_name(stage: int) -> str:
    """ "naive" / "tiled" / "fused" / "wmma" / "tuned" (CONTRACT.md 3 ``impl`` column)."""
    _require_ext()
    return str(torch.ops.fused_attn.impl_name(int(stage)))


def set_debug_overrides(block_m: int = 0, block_n: int = 0) -> None:
    """TEST-ONLY: force kernel tile sizes (0 = kernel default) for the tile-invariance tests.

    Global, process-wide, not thread-safe by design -- always reset to ``(0, 0)`` afterwards.
    """
    _require_ext()
    torch.ops.fused_attn.set_debug_overrides(int(block_m), int(block_n))
