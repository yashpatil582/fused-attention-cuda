"""PyTorch custom-op integration tests (marker ``gpu``): opcheck + torch.compile without breaks.

Why these exist: a kernel bound with pybind11 alone would work eagerly and silently graph-break
under ``torch.compile``.  ``TORCH_LIBRARY`` + ``register_fake`` is what makes the op a first-class
citizen; ``torch.library.opcheck`` is PyTorch's own conformance test for exactly that.
"""

from __future__ import annotations

import pytest
import torch

import fused_attn
from fused_attn.ops import LSE_STAGES, opcheck_all

pytestmark = pytest.mark.gpu

STAGES = [1, 2, 3, 4, 5]


def require_ext() -> None:
    if not fused_attn.is_available():
        pytest.skip("fused_attn._C is not built")


def supported_stages(dtype: torch.dtype, head_dim: int) -> list[int]:
    return [s for s in STAGES if fused_attn.impl_supports(s, dtype, head_dim)]


def require_any(dtype: torch.dtype, head_dim: int) -> list[int]:
    require_ext()
    stages = supported_stages(dtype, head_dim)
    if not stages:
        pytest.skip(f"no stage implemented for {dtype} D={head_dim} on this GPU")
    return stages


# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=["fp32", "fp16"])
def test_opcheck_both_ops(dtype, causal, make_qkv):
    stages = require_any(dtype, 64)
    q, k, v = make_qkv(1, 2, 64, 64, dtype=dtype, device="cuda")
    results = opcheck_all(q, k, v, is_causal=causal, stages=stages)
    assert set(results) == {"forward", *(f"forward_stage[{s}]" for s in stages)}
    # opcheck raises on failure (raise_exception=True); the dict only reports what ran.
    for name, res in results.items():
        assert all(status in ("SUCCESS", "SKIPPED") for status in res.values()), (name, res)


# --------------------------------------------------------------------------------------------
def test_fake_kernels_give_metadata_without_running(make_qkv):
    """The registered fakes must produce the right shapes/dtypes on fake tensors (no kernel run)."""
    require_ext()
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        q = torch.empty(2, 4, 256, 64, dtype=torch.float16, device="cuda")
        k = torch.empty_like(q)
        v = torch.empty_like(q)
        o = torch.ops.fused_attn.forward(q, k, v, False, 0.125)
        assert o.shape == q.shape and o.dtype == q.dtype and o.device.type == "cuda"
        for stage in STAGES:
            o_s, lse = torch.ops.fused_attn.forward_stage(q, k, v, True, 0.125, stage)
            assert o_s.shape == q.shape and o_s.dtype == q.dtype
            expected = (2, 4, 256) if stage in LSE_STAGES else (0,)
            assert tuple(lse.shape) == expected and lse.dtype == torch.float32
        bad_q = torch.empty(2, 4, 100, 64, dtype=torch.float16, device="cuda")
        with pytest.raises(RuntimeError, match="multiple of 32"):
            torch.ops.fused_attn.forward(bad_q, bad_q, bad_q, False, 0.125)


# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=["fp32", "fp16"])
def test_torch_compile_fullgraph_no_graph_break(dtype, make_qkv):
    require_any(dtype, 64)
    q, k, v = make_qkv(1, 2, 128, 64, dtype=dtype, device="cuda")

    def f(q, k, v):
        # Something before and after the op so a graph break would be observable as >1 graph.
        q = q * 1.0
        o = torch.ops.fused_attn.forward(q, k, v, True, 0.125)
        return o * 2.0

    torch._dynamo.reset()
    explanation = torch._dynamo.explain(f)(q, k, v)
    assert explanation.graph_break_count == 0, explanation.break_reasons
    assert explanation.graph_count == 1, explanation

    torch._dynamo.reset()
    from torch._dynamo.utils import counters

    counters.clear()
    compiled = torch.compile(f, fullgraph=True)  # fullgraph=True raises on any graph break
    out = compiled(q, k, v)
    assert sum(counters["graph_break"].values()) == 0, dict(counters["graph_break"])
    torch.testing.assert_close(out, f(q, k, v), atol=0, rtol=0)  # same kernel -> bitwise


def test_torch_compile_dynamic_shapes_forward_stage(make_qkv):
    """Dynamic N: the fake's torch._check(N % 32 == 0) must become a guard, not a break."""
    stages = require_any(torch.float32, 64)
    stage = max(stages)

    def g(q, k, v):
        o, lse = torch.ops.fused_attn.forward_stage(q, k, v, False, 0.125, stage)
        return o.sum(-1), lse

    torch._dynamo.reset()
    compiled = torch.compile(g, fullgraph=True, dynamic=True)
    for n in (128, 256):
        q, k, v = make_qkv(1, 2, n, 64, dtype=torch.float32, device="cuda")
        o_sum, lse = compiled(q, k, v)
        ref_sum, ref_lse = g(q, k, v)
        torch.testing.assert_close(o_sum, ref_sum, atol=1e-4, rtol=1e-5)
        assert lse.shape == ref_lse.shape


def test_public_wrapper_matches_raw_op(make_qkv):
    require_any(torch.float32, 64)
    q, k, v = make_qkv(1, 2, 64, 64, dtype=torch.float32, device="cuda")
    o_wrap = fused_attn.forward(q, k, v, is_causal=True)  # default scale = 1/sqrt(64)
    o_raw = torch.ops.fused_attn.forward(q, k, v, True, 0.125)
    assert torch.equal(o_wrap, o_raw)
    for s in supported_stages(torch.float32, 64):
        assert fused_attn.impl_name(s) in {"naive", "tiled", "fused", "wmma", "tuned"}
