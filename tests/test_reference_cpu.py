"""CPU tests of the fp64 model in ``fused_attn.reference`` -- run on the Mac in seconds, no GPU.

These are the whiteboard checks: the tiled online-softmax recurrence must reproduce the
un-tiled fp64 truth for *every* tiling, and every guard must be reachable and NaN-free.  A
deliberately broken copy of the recurrence at the bottom proves the tile-invariance test has
teeth (it fails exactly when a row spans more than one KV tile).
"""

from __future__ import annotations

import math

import pytest
import torch

from fused_attn.reference import (
    LOG2E,
    default_scale,
    error_ratio,
    math_reference,
    sdpa_reference,
    tiled_online_softmax_reference,
    unfused_reference,
)

TOL64 = 1e-10  # fp64 agreement between two exact algorithms (different summation order only)
NS = [64, 128, 256]
DS = [64, 128]
CAUSALS = [False, True]
# block_n as a fraction of N (plus a fixed 16): the sizes the task pins down.
BLOCK_N_SPECS = ["N", "N/2", "N/4", "N/8", "16"]


def block_from_spec(spec: str, n: int) -> int:
    return 16 if spec == "16" else n // {"N": 1, "N/2": 2, "N/4": 4, "N/8": 8}[spec]


def qkv(b: int, h: int, n: int, d: int, seed: int = 0, hkv: int | None = None):
    gen = torch.Generator().manual_seed(seed)
    hkv = h if hkv is None else hkv
    q = torch.randn(b, h, n, d, generator=gen, dtype=torch.float64)
    k = torch.randn(b, hkv, n, d, generator=gen, dtype=torch.float64)
    v = torch.randn(b, hkv, n, d, generator=gen, dtype=torch.float64)
    return q, k, v


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max())


# --------------------------------------------------------------------------------------------
# 1. Tile invariance: the tiled recurrence == the un-tiled math for every (block_m, block_n).
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("block_m_spec", ["N", "N/4"])
@pytest.mark.parametrize("block_n_spec", BLOCK_N_SPECS)
@pytest.mark.parametrize("causal", CAUSALS, ids=["full", "causal"])
@pytest.mark.parametrize("d", DS)
@pytest.mark.parametrize("n", NS)
def test_tile_invariance_matches_math(n, d, causal, block_n_spec, block_m_spec):
    q, k, v = qkv(1, 2, n, d)
    o_ref, lse_ref = math_reference(q, k, v, is_causal=causal)
    bm, bn = block_from_spec(block_m_spec, n), block_from_spec(block_n_spec, n)
    o, lse = tiled_online_softmax_reference(q, k, v, is_causal=causal, block_m=bm, block_n=bn)
    assert o.dtype == torch.float64 and lse.dtype == torch.float64
    assert max_abs(o, o_ref) < TOL64
    assert max_abs(lse, lse_ref) < TOL64


@pytest.mark.parametrize("causal", CAUSALS, ids=["full", "causal"])
def test_tile_invariance_ragged_and_oversized_blocks(causal):
    # N not a multiple of the tile (96 = 64 + 32) and a tile larger than N must both work.
    q, k, v = qkv(1, 1, 96, 64)
    o_ref, lse_ref = math_reference(q, k, v, is_causal=causal)
    for bm, bn in [(64, 64), (32, 64), (96, 512), (512, 512), (16, 40)]:
        o, lse = tiled_online_softmax_reference(q, k, v, is_causal=causal, block_m=bm, block_n=bn)
        assert max_abs(o, o_ref) < TOL64, (bm, bn)
        assert max_abs(lse, lse_ref) < TOL64, (bm, bn)


def test_causal_block_skipping_is_exact():
    # Skipping fully masked KV tiles before any load (the S6 optimization) must be a no-op.
    q, k, v = qkv(1, 2, 128, 64)
    o_skip, lse_skip = tiled_online_softmax_reference(
        q, k, v, is_causal=True, block_m=32, block_n=16, skip_masked_blocks=True
    )
    o_noskip, lse_noskip = tiled_online_softmax_reference(
        q, k, v, is_causal=True, block_m=32, block_n=16, skip_masked_blocks=False
    )
    assert torch.equal(o_skip, o_noskip)
    assert torch.equal(lse_skip, lse_noskip)


# --------------------------------------------------------------------------------------------
# 2. The model agrees with PyTorch's own oracle (SDPA fp64, MATH backend).
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("causal", CAUSALS, ids=["full", "causal"])
@pytest.mark.parametrize("d", DS)
@pytest.mark.parametrize("n", NS)
def test_math_and_tiled_match_sdpa(n, d, causal):
    q, k, v = qkv(2, 2, n, d, seed=1)
    o_sdpa = sdpa_reference(q, k, v, is_causal=causal)
    o_math, _ = math_reference(q, k, v, is_causal=causal)
    o_tiled, _ = tiled_online_softmax_reference(q, k, v, is_causal=causal, block_m=32, block_n=32)
    assert max_abs(o_math, o_sdpa) < TOL64
    assert max_abs(o_tiled, o_sdpa) < TOL64


def test_gqa_heads_match_sdpa_enable_gqa():
    # Hkv < H: query head h reads kv head h // (H/Hkv), exactly torch's enable_gqa semantics.
    q, k, v = qkv(1, 4, 64, 64, hkv=2)
    o_sdpa = sdpa_reference(q, k, v, is_causal=True)
    o_math, _ = math_reference(q, k, v, is_causal=True)
    o_tiled, _ = tiled_online_softmax_reference(q, k, v, is_causal=True, block_m=16, block_n=16)
    assert max_abs(o_math, o_sdpa) < TOL64
    assert max_abs(o_tiled, o_sdpa) < TOL64


# --------------------------------------------------------------------------------------------
# 3. LSE == logsumexp of the scaled (masked) scores.
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("causal", CAUSALS, ids=["full", "causal"])
@pytest.mark.parametrize("scale", [None, 0.37], ids=["default_scale", "scale0.37"])
def test_lse_equals_torch_logsumexp(causal, scale):
    n, d = 128, 64
    q, k, v = qkv(1, 2, n, d, seed=2)
    s = torch.matmul(q, k.transpose(-1, -2)) * (default_scale(d) if scale is None else scale)
    if causal:
        s = s.masked_fill(~torch.ones(n, n, dtype=torch.bool).tril(), float("-inf"))
    lse_torch = torch.logsumexp(s, dim=-1)
    _, lse_math = math_reference(q, k, v, is_causal=causal, scale=scale)
    _, lse_tiled = tiled_online_softmax_reference(
        q, k, v, is_causal=causal, scale=scale, block_m=64, block_n=16
    )
    assert lse_math.shape == (1, 2, n)
    assert max_abs(lse_math, lse_torch) < TOL64
    assert max_abs(lse_tiled, lse_torch) < TOL64


# --------------------------------------------------------------------------------------------
# 4. Scale folding: exp(x*scale) == exp2(x*scale*log2e) for a non-default scale.
# --------------------------------------------------------------------------------------------
def test_scale_folding_non_default_scale():
    q, k, v = qkv(1, 2, 128, 128, seed=3)
    scale = 0.05  # far from 1/sqrt(128) = 0.088
    o_ref, lse_ref = math_reference(q, k, v, scale=scale)
    o_sdpa = sdpa_reference(q, k, v, scale=scale)
    o, lse = tiled_online_softmax_reference(q, k, v, scale=scale, block_m=32, block_n=16)
    assert max_abs(o, o_ref) < TOL64
    assert max_abs(o, o_sdpa) < TOL64
    assert max_abs(lse, lse_ref) < TOL64
    # ... and the scale actually matters (guards against a model that ignores it).
    o_default, _ = math_reference(q, k, v)
    assert max_abs(o, o_default) > 1e-3
    x = torch.linspace(-30.0, 5.0, 101, dtype=torch.float64)
    assert max_abs(torch.exp2(x * scale * LOG2E), torch.exp(x * scale)) < 1e-14


# --------------------------------------------------------------------------------------------
# 5. Causal == explicit top-left tril mask (torch is_causal semantics for L == S).
# --------------------------------------------------------------------------------------------
def test_causal_equals_explicit_tril_mask():
    n = 128
    q, k, v = qkv(1, 2, n, 64, seed=4)
    tril = torch.ones(n, n, dtype=torch.bool).tril()
    o_causal, lse_causal = math_reference(q, k, v, is_causal=True)
    o_mask, lse_mask = math_reference(q, k, v, attn_mask=tril)
    o_sdpa_causal = sdpa_reference(q, k, v, is_causal=True)
    o_sdpa_mask = sdpa_reference(q, k, v, attn_mask=tril)
    o_tiled, lse_tiled = tiled_online_softmax_reference(q, k, v, attn_mask=tril, block_n=16)
    assert torch.equal(o_causal, o_mask) and torch.equal(lse_causal, lse_mask)
    assert max_abs(o_sdpa_causal, o_sdpa_mask) < TOL64
    assert max_abs(o_causal, o_sdpa_causal) < TOL64
    assert max_abs(o_tiled, o_causal) < TOL64 and max_abs(lse_tiled, lse_causal) < TOL64
    # Row 0 attends only to key 0 -> its output is exactly v[0].
    assert max_abs(o_causal[:, :, 0, :], v[:, :, 0, :]) < TOL64


# --------------------------------------------------------------------------------------------
# 6. Guards: fully masked rows / tiles never produce NaN.
# --------------------------------------------------------------------------------------------
def test_fully_masked_row_guards_never_nan():
    n, bn = 64, 16
    q, k, v = qkv(1, 2, n, 64, seed=5)
    keep = torch.ones(n, n, dtype=torch.bool)
    keep[5, :] = False  # row 5: fully masked -> l == 0 at the end (guard 2)
    keep[7, :bn] = False  # row 7: first KV tile fully masked -> m == -inf after tile 0 (guard 1)
    keep[9, bn:] = False  # row 9: everything after tile 0 masked -> alpha == 1 path with p == 0
    o_math, lse_math = math_reference(q, k, v, attn_mask=keep)
    o_tiled, lse_tiled = tiled_online_softmax_reference(
        q, k, v, attn_mask=keep, block_m=16, block_n=bn
    )
    for o, lse in [(o_math, lse_math), (o_tiled, lse_tiled)]:
        assert not torch.isnan(o).any()
        assert not torch.isnan(lse).any()
        assert torch.equal(o[:, :, 5, :], torch.zeros_like(o[:, :, 5, :]))
        assert torch.isposinf(lse[:, :, 5]).all()
        assert torch.isfinite(o[:, :, [7, 9], :]).all() and torch.isfinite(lse[:, :, [7, 9]]).all()
    assert max_abs(o_tiled, o_math) < TOL64
    assert torch.equal(torch.isinf(lse_tiled), torch.isinf(lse_math))
    finite = torch.isfinite(lse_math)
    assert max_abs(lse_tiled[finite], lse_math[finite]) < TOL64
    # Rows 7 and 9 are ordinary partially masked rows: check them against the un-tiled truth
    # computed with the mask baked into the inputs (a second independent path).
    o_sdpa = sdpa_reference(q, k, v, attn_mask=keep)
    assert max_abs(o_tiled[:, :, [7, 9], :], o_sdpa[:, :, [7, 9], :]) < TOL64
    # A fully masked *tile* with skipping disabled hits the same guards on the causal path.
    o_a, _ = tiled_online_softmax_reference(q, k, v, is_causal=True, block_n=8, block_m=8)
    o_b, _ = tiled_online_softmax_reference(
        q, k, v, is_causal=True, block_n=8, block_m=8, skip_masked_blocks=False
    )
    assert not torch.isnan(o_b).any() and torch.equal(o_a, o_b)


# --------------------------------------------------------------------------------------------
# 7. The ratio rule helper.
# --------------------------------------------------------------------------------------------
def test_error_ratio_rule():
    ref = torch.randn(2, 2, 32, 64, dtype=torch.float64, generator=torch.Generator().manual_seed(6))
    noise = torch.randn_like(ref)
    torch_same = ref + 1e-3 * noise
    good = error_ratio(ref + 1e-3 * noise.flip(0), ref, torch_same)
    assert good.ok and good.out_finite and good.torch_finite
    assert good.err_mine <= good.bound == pytest.approx(2 * good.err_torch + 1e-6)
    bad = error_ratio(ref + 1e-2 * noise, ref, torch_same)
    assert not bad.ok and bad.err_mine > bad.bound
    nan_out = ref.clone()
    nan_out[0, 0, 0, 0] = float("nan")
    assert not error_ratio(nan_out, ref, torch_same).out_finite
    torch_nan = torch_same.clone()
    torch_nan[0, 0, 0, 0] = float("inf")
    r = error_ratio(ref, ref, torch_nan)
    assert not r.torch_finite and not r.ok  # rule not applicable -> caller falls back to a cap
    # A model of "PyTorch's own unfused fp32 attention" passes its own rule trivially.
    q, k, v = qkv(1, 2, 128, 64, seed=7)
    q32, k32, v32 = (t.to(torch.float32) for t in (q, k, v))
    pt = unfused_reference(q32, k32, v32, is_causal=True)
    o64, _ = math_reference(q, k, v, is_causal=True)
    assert pt.dtype == torch.float32
    self_check = error_ratio(pt, o64, pt)
    assert self_check.ok and self_check.err_torch < 1e-5


def test_inputs_of_any_float_dtype_are_upcast():
    q, k, v = qkv(1, 1, 64, 64, seed=8)
    o64, lse64 = math_reference(q, k, v)
    for dt in (torch.float32, torch.float16, torch.bfloat16):
        o, lse = tiled_online_softmax_reference(q.to(dt), k.to(dt), v.to(dt), block_n=16)
        assert o.dtype == torch.float64 and lse.dtype == torch.float64
        # Only the input rounding differs; bf16 has 8 mantissa bits, hence the loose bound.
        assert (
            max_abs(o, o64) < {torch.float32: 1e-5, torch.float16: 1e-2, torch.bfloat16: 1e-1}[dt]
        )


# --------------------------------------------------------------------------------------------
# 8. Teeth: a recurrence that forgets a rescale passes with one KV tile and fails with several.
# --------------------------------------------------------------------------------------------
def _tiled_with_bug(q, k, v, bug: str, block_m: int, block_n: int, scale=None):
    """Private copy of the recurrence with ONE step removed (never import this into the model).

    bug == "no_acc_rescale": acc = acc + p @ v      (forgets alpha on the accumulator)
    bug == "no_l_rescale":   l   = l + rowsum(p)    (forgets alpha on the row sum)
    """
    q, k, v = (t.to(torch.float64) for t in (q, k, v))
    b, h, n, d = q.shape
    scale = default_scale(d) if scale is None else scale
    c = scale * LOG2E
    o = torch.empty_like(q)
    for i0 in range(0, n, block_m):
        i1 = min(i0 + block_m, n)
        qt = q[:, :, i0:i1, :]
        m = torch.full((b, h, i1 - i0, 1), float("-inf"), dtype=torch.float64)
        l_row = torch.zeros((b, h, i1 - i0, 1), dtype=torch.float64)
        acc = torch.zeros((b, h, i1 - i0, d), dtype=torch.float64)
        for j0 in range(0, n, block_n):
            j1 = min(j0 + block_n, n)
            s = torch.matmul(qt, k[:, :, j0:j1, :].transpose(-1, -2))
            m_new = torch.maximum(m, s.amax(dim=-1, keepdim=True))
            alpha = torch.exp2((m - m_new) * c)
            p = torch.exp2((s - m_new) * c)
            l_row = (l_row if bug == "no_l_rescale" else alpha * l_row) + p.sum(-1, keepdim=True)
            acc = (acc if bug == "no_acc_rescale" else alpha * acc) + torch.matmul(
                p, v[:, :, j0:j1, :]
            )
            m = m_new
        o[:, :, i0:i1, :] = acc / l_row
    return o


@pytest.mark.parametrize("bug", ["no_acc_rescale", "no_l_rescale"])
def test_rescale_bug_is_caught_by_tile_invariance(bug):
    n = 128
    q, k, v = qkv(1, 2, n, 64, seed=9)
    o_ref, _ = math_reference(q, k, v)
    # One KV tile per row: no rescale ever happens, so the bug is invisible ...
    assert max_abs(_tiled_with_bug(q, k, v, bug, block_m=32, block_n=n), o_ref) < TOL64
    # ... and with several tiles per row the same code is wrong by a large margin.
    err = max_abs(_tiled_with_bug(q, k, v, bug, block_m=32, block_n=n // 4), o_ref)
    assert err > 1e-3, f"{bug}: tile invariance did not detect the missing rescale (err={err})"
    # Sanity: the real model with the same tiling is still exact.
    o_ok, _ = tiled_online_softmax_reference(q, k, v, block_m=32, block_n=n // 4)
    assert max_abs(o_ok, o_ref) < TOL64


def test_default_scale_matches_torch_convention():
    assert default_scale(64) == pytest.approx(1 / math.sqrt(64))
    assert default_scale(128) == pytest.approx(1 / math.sqrt(128))
