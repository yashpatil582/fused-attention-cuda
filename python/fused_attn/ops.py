"""Fake (meta) kernels and opcheck helper for the ``fused_attn::*`` ops.

Why this file exists: ``torch.compile`` / AOTAutograd trace ops on *fake tensors* that carry only
shape/dtype/device.  An op registered with ``TORCH_LIBRARY`` but without a fake implementation is
a graph break -- the traced graph stops, the op runs eagerly, and a new graph starts.  The fakes
below tell the tracer the output metadata of each op without touching data, which is what lets
``tests/test_op_gpu.py`` assert zero graph breaks under ``torch.compile(fullgraph=True)``.

The fakes must be registered *after* ``fused_attn._C`` is loaded (the schemas live in C++), so
``register_fakes()`` is called from ``fused_attn/__init__.py`` once the extension imports.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
SUPPORTED_HEAD_DIMS = (64, 128)
# Stages whose kernels write the logsumexp (CONTRACT.md 4: naive/tiled return an empty lse).
LSE_STAGES = (3, 4, 5)

_registered = False


def _check_inputs(q: Tensor, k: Tensor, v: Tensor) -> None:
    """Mirror of the TORCH_CHECKs in ext.cpp, expressed with ``torch._check`` so they also hold
    under symbolic shapes (each check becomes a guard / runtime assert in the compiled graph)."""
    torch._check(q.dim() == 4 and k.dim() == 4 and v.dim() == 4, lambda: "q/k/v must be [B,H,N,D]")
    torch._check(q.dtype == k.dtype and q.dtype == v.dtype, lambda: "q/k/v must share one dtype")
    torch._check(q.dtype in SUPPORTED_DTYPES, lambda: f"unsupported dtype {q.dtype}")
    b, h, n, d = q.shape
    torch._check(d == 64 or d == 128, lambda: f"D must be 64 or 128, got {d}")
    torch._check(n % 32 == 0, lambda: f"N must be a multiple of 32, got {n}")
    torch._check(k.shape[0] == b and v.shape[0] == b, lambda: "batch mismatch")
    torch._check(k.shape[2] == n and v.shape[2] == n, lambda: "Nq == Nkv is required in-window")
    torch._check(k.shape[3] == d and v.shape[3] == d, lambda: "head-dim mismatch")
    torch._check(k.shape[1] == v.shape[1], lambda: "k and v must have the same number of heads")
    torch._check(h % k.shape[1] == 0, lambda: "H must be a multiple of Hkv")


def _fake_forward(q: Tensor, k: Tensor, v: Tensor, is_causal: bool, scale: float) -> Tensor:
    _check_inputs(q, k, v)
    return torch.empty_like(q)


def _fake_forward_stage(
    q: Tensor, k: Tensor, v: Tensor, is_causal: bool, scale: float, stage: int
) -> tuple[Tensor, Tensor]:
    _check_inputs(q, k, v)
    torch._check(1 <= stage <= 5, lambda: f"stage must be in 1..5, got {stage}")
    b, h, n, _ = q.shape
    o = torch.empty_like(q)
    if stage in LSE_STAGES:
        lse = q.new_empty((b, h, n), dtype=torch.float32)
    else:
        lse = q.new_empty((0,), dtype=torch.float32)
    return o, lse


def register_fakes() -> None:
    """Register the fake kernels (idempotent; requires ``fused_attn._C`` to be loaded)."""
    global _registered
    if _registered:
        return
    torch.library.register_fake("fused_attn::forward", _fake_forward)
    torch.library.register_fake("fused_attn::forward_stage", _fake_forward_stage)
    _registered = True


# opcheck's default test_utils. test_autograd_registration needs at least one input with
# requires_grad=True (torch/testing/_internal/optests/autograd_registration.py raises
# "no inputs have requires_grad=True" otherwise), while the other three must run on plain
# inputs: with grad-requiring inputs test_aot_dispatch_dynamic would trace a backward for an
# op that has no autograd kernel (the CUDA-only impl) and fail. So the two groups run
# separately on the same values.
_NO_GRAD_TESTS = ("test_schema", "test_faketensor", "test_aot_dispatch_dynamic")
_GRAD_TESTS = ("test_autograd_registration",)


def _opcheck(op, args: tuple) -> dict[str, str]:
    """Runs opcheck in two groups and merges the result dicts (see the note above)."""
    q, k, v, *rest = args
    res = dict(torch.library.opcheck(op, (q, k, v, *rest), test_utils=_NO_GRAD_TESTS))
    grad_args = (
        q.detach().requires_grad_(),
        k.detach().requires_grad_(),
        v.detach().requires_grad_(),
        *rest,
    )
    # autograd_registration_check switches the autograd fallback to "nothing" while calling the
    # op, so an op with only a CUDA kernel passes as long as it returns no grad-requiring output.
    res.update(torch.library.opcheck(op, grad_args, test_utils=_GRAD_TESTS))
    return res


def opcheck_all(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    is_causal: bool = False,
    scale: float | None = None,
    stages: Sequence[int] = (),
) -> dict[str, dict[str, str]]:
    """Run ``torch.library.opcheck`` on ``forward`` and on ``forward_stage`` for each stage.

    opcheck exercises the schema (no undeclared aliasing/mutation), the autograd registration,
    the fake kernel against the real one, and AOT dispatch with dynamic shapes -- i.e. everything
    ``torch.compile`` relies on.  Raises on the first failure; returns the per-op result dicts
    (all four opcheck utilities, merged from the two groups ``_opcheck`` runs).
    """
    if scale is None:
        scale = 1.0 / (q.shape[-1] ** 0.5)
    results: dict[str, dict[str, str]] = {}
    results["forward"] = _opcheck(torch.ops.fused_attn.forward, (q, k, v, is_causal, float(scale)))
    for stage in stages:
        results[f"forward_stage[{stage}]"] = _opcheck(
            torch.ops.fused_attn.forward_stage, (q, k, v, is_causal, float(scale), int(stage))
        )
    return results
