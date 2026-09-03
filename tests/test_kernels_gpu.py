"""GPU correctness tests for every kernel stage (marker ``gpu``; needs the built extension).

Every stage is judged against the fp64 oracle with the FlashAttention ratio rule
(``err_mine <= 2 * err_torch_same_dtype + 1e-6``, see docs/NUMERICS.md) plus finite checks and
absolute sanity caps.  Stages a build does not implement for a (dtype, D, GPU) are *skipped*
via ``fused_attn.impl_supports`` so the file is green from the S0 stub onwards.
Run one stage only with ``pytest tests/test_kernels_gpu.py --stage 3``.
"""

from __future__ import annotations

import pytest
import torch

import fused_attn
from fused_attn.reference import error_ratio, math_reference, unfused_reference

pytestmark = pytest.mark.gpu

STAGES = [1, 2, 3, 4, 5]
LSE_STAGES = [3, 4, 5]
# Dtypes each stage is expected to support (attention.cuh: naive/tiled/fused fp32+fp16, wmma/tuned
# fp16 (+bf16 on sm_80+)).  impl_supports() still decides at runtime; this only trims the matrix.
STAGE_DTYPES = {
    1: [torch.float32, torch.float16],
    2: [torch.float32, torch.float16],
    3: [torch.float32, torch.float16],
    4: [torch.float16, torch.bfloat16],
    5: [torch.float16, torch.bfloat16],
}
STAGE_DTYPE_PAIRS = [(s, dt) for s in STAGES for dt in STAGE_DTYPES[s]]
# Absolute sanity caps on max|out - ref64| for unit-variance inputs (docs/NUMERICS.md).  These are
# policy, not measurements: they stop a kernel from passing only because torch is also bad.
ABS_CAP = {torch.float32: 1e-4, torch.float16: 1e-2, torch.bfloat16: 5e-2}
# LSE tolerance: LSE ~ log(N) + max score, accumulated in fp32 in every kernel.
LSE_TOL = {torch.float32: (1e-4, 1e-5), torch.float16: (2e-2, 1e-3), torch.bfloat16: (5e-2, 5e-3)}

_ids = {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}


def _dtype_id(dt: torch.dtype) -> str:
    return _ids[dt]


def _pair_id(p: int | torch.dtype) -> str:
    """ids for the (stage, dtype) parametrization: '3-fp16'."""
    return str(p) if isinstance(p, int) else _dtype_id(p)


def require(stage: int, dtype: torch.dtype, head_dim: int) -> None:
    if not fused_attn.is_available():
        pytest.skip("fused_attn._C is not built")
    if not fused_attn.impl_supports(stage, dtype, head_dim):
        cap = torch.cuda.get_device_capability()
        pytest.skip(
            f"stage {stage} not implemented for {_ids[dtype]} D={head_dim} on sm_{cap[0]}{cap[1]}"
        )


def check_against_reference(o, q, k, v, causal, dtype, attn_mask=None, scale=None):
    """Ratio rule vs fp64 + finite + absolute cap.  Returns the RatioResult for reporting."""
    ref64, _ = math_reference(q, k, v, is_causal=causal, scale=scale, attn_mask=attn_mask)
    pt = unfused_reference(q, k, v, is_causal=causal, scale=scale, attn_mask=attn_mask)
    r = error_ratio(o, ref64, pt)
    assert r.out_finite, "kernel produced NaN/inf"
    assert r.ok, f"ratio rule failed: {r}"
    assert r.err_mine <= ABS_CAP[dtype], f"absolute cap {ABS_CAP[dtype]} exceeded: {r}"
    return r


@pytest.fixture
def reset_overrides():
    """Guarantee the process-global tile override is cleared even when a test fails."""
    yield
    if fused_attn.is_available():
        fused_attn.set_debug_overrides(0, 0)


# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("stage,dtype", STAGE_DTYPE_PAIRS, ids=_pair_id)
def test_matches_fp64_reference(stage, dtype, causal, gpu_shape, make_qkv):
    b, h, n, d = gpu_shape
    require(stage, dtype, d)
    q, k, v = make_qkv(b, h, n, d, dtype=dtype, device="cuda")
    o, lse = fused_attn.forward_stage(q, k, v, stage, is_causal=causal)
    assert o.shape == q.shape and o.dtype == dtype and o.is_contiguous()
    expected_lse_shape = (b, h, n) if stage in LSE_STAGES else (0,)
    assert tuple(lse.shape) == expected_lse_shape and lse.dtype == torch.float32
    check_against_reference(o, q, k, v, causal, dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16], ids=_dtype_id)
@pytest.mark.parametrize("d", [64, 128])
def test_forward_dispatches_to_best_stage(dtype, d, make_qkv):
    if not fused_attn.is_available():
        pytest.skip("fused_attn._C is not built")
    supported = [s for s in STAGES if fused_attn.impl_supports(s, dtype, d)]
    q, k, v = make_qkv(1, 2, 256, d, dtype=dtype, device="cuda")
    if not supported:
        with pytest.raises(RuntimeError, match="no stage is implemented"):
            fused_attn.forward(q, k, v)
        return
    best = max(supported)
    o = fused_attn.forward(q, k, v, is_causal=True)
    o_best, _ = fused_attn.forward_stage(q, k, v, best, is_causal=True)
    # Same kernel, same inputs -> bitwise identical (kernels are deterministic, no atomics).
    assert torch.equal(o, o_best), f"forward() did not dispatch to stage {best}"
    check_against_reference(o, q, k, v, True, dtype)


# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("n", [64, 128])
@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("stage,dtype", [p for p in STAGE_DTYPE_PAIRS if p[0] >= 3], ids=_pair_id)
def test_tile_invariance(stage, dtype, causal, n, make_qkv, reset_overrides):
    """The online-softmax rescale is only exercised when a row spans several KV tiles, so the
    same problem is run with the default tiles and with block_n in {N/2, N/4}: every run must
    pass the ratio rule and all runs must agree to within the dtype cap."""
    d = 64
    require(stage, dtype, d)
    q, k, v = make_qkv(2, 2, n, d, dtype=dtype, device="cuda", seed=11)
    outs = {}
    for bm, bn in [(0, 0), (0, n // 2), (0, n // 4), (n // 4, n // 4)]:
        fused_attn.set_debug_overrides(bm, bn)
        o, _ = fused_attn.forward_stage(q, k, v, stage, is_causal=causal)
        torch.cuda.synchronize()
        check_against_reference(o, q, k, v, causal, dtype)
        outs[(bm, bn)] = o.double()
    fused_attn.set_debug_overrides(0, 0)
    base = outs[(0, 0)]
    for key, o in outs.items():
        diff = float((o - base).abs().max())
        assert diff <= 2 * ABS_CAP[dtype], f"tiles {key} differ from default by {diff}"


# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("stage,dtype", STAGE_DTYPE_PAIRS, ids=_pair_id)
def test_determinism_50_runs_bitwise_equal(stage, dtype, make_qkv):
    d = 64
    require(stage, dtype, d)
    q, k, v = make_qkv(2, 4, 256, d, dtype=dtype, device="cuda", seed=12)
    first, first_lse = fused_attn.forward_stage(q, k, v, stage, is_causal=True)
    for _ in range(49):
        o, lse = fused_attn.forward_stage(q, k, v, stage, is_causal=True)
        assert torch.equal(o, first), "non-deterministic output (atomics? uninitialized smem?)"
        assert torch.equal(lse, first_lse)


# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("factor", [1e2, 1e-3], ids=["x1e2", "x1e-3"])
@pytest.mark.parametrize(
    "stage,dtype", [p for p in STAGE_DTYPE_PAIRS if p[1] != torch.float32], ids=_pair_id
)
def test_half_precision_dynamic_range(stage, dtype, factor, make_qkv):
    """Inputs scaled by 1e2 push raw scores past fp16's 65504 (they must live in fp32) and
    exp() past fp32 without the running-max subtraction; 1e-3 probes the small end.  PyTorch's
    own fp16 unfused attention may overflow to inf/NaN here, in which case the ratio rule is not
    applicable and a relative cap is used instead."""
    d = 64
    require(stage, dtype, d)
    q, k, v = make_qkv(1, 2, 256, d, dtype=dtype, device="cuda", seed=13, scale_inputs=factor)
    o, lse = fused_attn.forward_stage(q, k, v, stage)
    assert torch.isfinite(o).all(), "kernel overflowed: scores/max/sum/acc must be fp32"
    if stage in LSE_STAGES:
        assert torch.isfinite(lse).all()
    ref64, _ = math_reference(q, k, v)
    r = error_ratio(o, ref64, unfused_reference(q, k, v))
    rel = r.err_mine / max(float(ref64.abs().max()), 1e-30)
    if r.torch_finite:
        assert r.ok, f"ratio rule failed: {r}"
    assert rel <= ABS_CAP[dtype], f"relative error {rel:.3e} exceeds {ABS_CAP[dtype]} ({r})"


# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("stage", STAGES)
def test_n_not_multiple_of_32_raises(stage, make_qkv):
    require(stage, torch.float32 if stage <= 3 else torch.float16, 64)
    dtype = torch.float32 if stage <= 3 else torch.float16
    q, k, v = make_qkv(1, 1, 100, 64, dtype=dtype, device="cuda")
    with pytest.raises(RuntimeError, match="multiple of 32"):
        fused_attn.forward_stage(q, k, v, stage)
    with pytest.raises(RuntimeError, match="multiple of 32"):
        fused_attn.forward(q, k, v)


def test_input_validation_messages(make_qkv):
    if not fused_attn.is_available():
        pytest.skip("fused_attn._C is not built")
    q, k, v = make_qkv(1, 2, 64, 64, dtype=torch.float32, device="cuda")
    with pytest.raises(RuntimeError, match="CUDA"):
        fused_attn.forward(q.cpu(), k.cpu(), v.cpu())
    with pytest.raises(RuntimeError, match="contiguous"):
        # Right shape, wrong strides ([B,N,H,D] storage viewed as [B,H,N,D]).
        fused_attn.forward(q.transpose(1, 2).contiguous().transpose(1, 2), k, v)
    with pytest.raises(RuntimeError, match="dtype"):
        fused_attn.forward(q, k.half(), v)
    with pytest.raises(RuntimeError, match="D must be 64 or 128"):
        fused_attn.forward(
            q[..., :32].contiguous(), k[..., :32].contiguous(), v[..., :32].contiguous()
        )
    with pytest.raises(RuntimeError, match="stage must be in 1..5"):
        fused_attn.forward_stage(q, k, v, 0)
    with pytest.raises(RuntimeError, match="stage must be in 1..5"):
        fused_attn.forward_stage(q, k, v, 6)
    with pytest.raises(RuntimeError, match="Nq == Nkv"):
        fused_attn.forward(q, k[:, :, :32].contiguous(), v[:, :, :32].contiguous())
    with pytest.raises(RuntimeError, match="multiple of Hkv"):
        fused_attn.forward(q[:, :1].contiguous(), k, v)
    for stage in STAGES:
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            if not fused_attn.impl_supports(stage, dtype, 64):
                with pytest.raises(RuntimeError, match="not implemented"):
                    fused_attn.forward_stage(q.to(dtype), k.to(dtype), v.to(dtype), stage)


# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("stage,dtype", STAGE_DTYPE_PAIRS, ids=_pair_id)
def test_causal_equals_explicit_tril_mask(stage, dtype, make_qkv):
    d = 64
    require(stage, dtype, d)
    n = 512
    q, k, v = make_qkv(1, 2, n, d, dtype=dtype, device="cuda", seed=14)
    tril = torch.ones(n, n, dtype=torch.bool, device="cuda").tril()
    o, _ = fused_attn.forward_stage(q, k, v, stage, is_causal=True)
    # Reference built from the explicit top-left mask, not from is_causal, so a kernel with an
    # off-by-one on the diagonal (j <= i vs j < i) or bottom-right alignment fails here.
    check_against_reference(o, q, k, v, False, dtype, attn_mask=tril)
    # Query 0 attends to key 0 only: its output is v[0] up to input rounding.
    assert torch.allclose(o[:, :, 0, :].float(), v[:, :, 0, :].float(), atol=ABS_CAP[dtype], rtol=0)


@pytest.mark.parametrize("scale", [0.05, 0.25])
@pytest.mark.parametrize("stage,dtype", STAGE_DTYPE_PAIRS, ids=_pair_id)
def test_non_default_scale(stage, dtype, scale, make_qkv):
    d = 128
    require(stage, dtype, d)
    q, k, v = make_qkv(1, 2, 256, d, dtype=dtype, device="cuda", seed=15)
    o, _ = fused_attn.forward_stage(q, k, v, stage, is_causal=False, scale=scale)
    check_against_reference(o, q, k, v, False, dtype, scale=scale)


# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize(
    "stage,dtype", [p for p in STAGE_DTYPE_PAIRS if p[0] in LSE_STAGES], ids=_pair_id
)
def test_lse_matches_fp64(stage, dtype, causal, gpu_shape, make_qkv):
    b, h, n, d = gpu_shape
    require(stage, dtype, d)
    q, k, v = make_qkv(b, h, n, d, dtype=dtype, device="cuda", seed=16)
    _, lse = fused_attn.forward_stage(q, k, v, stage, is_causal=causal)
    _, lse64 = math_reference(q, k, v, is_causal=causal)
    assert lse.shape == (b, h, n) and lse.dtype == torch.float32
    assert torch.isfinite(lse).all()
    atol, rtol = LSE_TOL[dtype]
    torch.testing.assert_close(lse.double(), lse64, atol=atol, rtol=rtol)


# --------------------------------------------------------------------------------------------
@pytest.mark.xfail(strict=False, reason="GQA (Hkv < H) is week-after work (PLAN.md W2)")
@pytest.mark.parametrize("stage,dtype", STAGE_DTYPE_PAIRS, ids=_pair_id)
def test_gqa_heads(stage, dtype, make_qkv):
    d = 64
    require(stage, dtype, d)
    q, k, v = make_qkv(1, 4, 256, d, dtype=dtype, device="cuda", hkv=2, seed=17)
    o, _ = fused_attn.forward_stage(q, k, v, stage, is_causal=True)
    check_against_reference(o, q, k, v, True, dtype)
