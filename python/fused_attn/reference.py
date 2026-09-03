"""CPU model of the fused-attention kernels (torch, float64, no CUDA required).

Three references, in decreasing order of authority:

* ``sdpa_reference``   PyTorch's own ``scaled_dot_product_attention`` in float64 under the MATH
                       backend: the oracle every kernel is judged against (docs/NUMERICS.md).
* ``math_reference``   explicit fp64 softmax attention that also returns the row-wise
                       logsumexp (LSE), which SDPA does not expose.
* ``tiled_online_softmax_reference``
                       the *whiteboard model* of stages 3-5.  Same numbers as the two above (to
                       ~1e-12 in fp64) but computed with the exact tiled recurrence the CUDA
                       kernels implement -- online softmax (Milakov & Gimelshein 2018) with the
                       FlashAttention-2 loop order and deferred normalization -- so tile-size
                       invariance and the ``m == -inf`` / ``l == 0`` guards are tested on a laptop
                       before any GPU minute is spent.

``unfused_reference`` (same dtype as the inputs, naive PyTorch) and ``error_ratio`` implement the
FlashAttention test-suite tolerance rule: a fused kernel passes when its error against the fp64
oracle is at most twice the error PyTorch's own unfused attention makes in the same dtype.

Conventions shared with ``include/fa/attention.cuh`` (frozen):
  layouts ``[B, H, N, D]``; ``k``/``v`` may carry ``Hkv`` heads with ``H % Hkv == 0`` (GQA, expanded
  with ``repeat_interleave`` exactly like ``F.scaled_dot_product_attention(enable_gqa=True)``);
  causal mask is top-left aligned (``j <= i``); default scale ``1/sqrt(D)``;
  ``lse[b, h, i] = logsumexp_j(scale * q_i . k_j)`` over unmasked ``j`` (``+inf`` for a fully
  masked row, the non-split FlashAttention convention).
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import Tensor

# Causal alignment convention (docs/NUMERICS.md section 4, CONTRACT section 1): query row i attends
# to keys j <= i, i.e. torch SDPA `is_causal` semantics for Nq == Nkv. Pinned as a constant because
# flash-attn >= 2.1 aligns bottom-right when Nq != Nkv; `_keep_mask` implements this one.
CAUSAL_ALIGN = "top_left"

# exp(x * scale) == exp2(x * scale * log2(e)).  The kernels fold scale*log2e into ONE constant so
# the inner loop uses exp2f (a single MUFU.EX2 instruction) with no extra multiply per score.
LOG2E = 1.4426950408889634

__all__ = [
    "LOG2E",
    "RatioResult",
    "default_scale",
    "error_ratio",
    "math_reference",
    "sdpa_reference",
    "tiled_online_softmax_reference",
    "unfused_reference",
]


def default_scale(head_dim: int) -> float:
    """PyTorch's default SDPA scale, ``1/sqrt(D)``."""
    return 1.0 / math.sqrt(head_dim)


def _expand_kv(q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
    """Broadcast ``Hkv`` key/value heads to ``H`` query heads (GQA/MQA).

    ``repeat_interleave`` maps query head ``h`` to kv head ``h // (H // Hkv)``, which is both what
    torch's ``enable_gqa=True`` does and what ``src/dispatch.cu`` does
    (``kv_head = q_head / (H / Hkv)``).
    """
    h, hkv = q.shape[1], k.shape[1]
    if hkv == h:
        return k, v
    if h % hkv != 0:
        raise ValueError(f"H={h} must be a multiple of Hkv={hkv}")
    rep = h // hkv
    return k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)


def _keep_mask(
    n_q: int, n_kv: int, is_causal: bool, attn_mask: Tensor | None, device: torch.device
) -> Tensor | None:
    """Boolean "attend" mask broadcastable to ``[B, H, Nq, Nkv]`` (``None`` when nothing is masked).

    Causal is top-left aligned (``CAUSAL_ALIGN == "top_left"``): query ``i`` sees keys
    ``j <= i``, i.e. ``tril()`` with diagonal 0.  For ``Nq == Nkv`` (the only case in-window) this
    equals torch's ``is_causal=True``; flash-attn >= 2.1 aligns bottom-right when ``Nq != Nkv``,
    which is why the alignment is pinned here in words and in the constant.
    ``attn_mask`` is model-only (the kernels take no mask); it exists to exercise the
    fully-masked-row guards, which a top-left causal mask with ``Nq == Nkv`` can never trigger.
    """
    keep = None
    if is_causal:
        keep = torch.ones(n_q, n_kv, dtype=torch.bool, device=device).tril()
    if attn_mask is not None:
        if attn_mask.dtype != torch.bool:
            raise TypeError("attn_mask must be a boolean 'attend' mask (True = keep)")
        keep = attn_mask if keep is None else (keep & attn_mask)
    return keep


def _prep(
    q: Tensor, k: Tensor, v: Tensor, scale: float | None
) -> tuple[Tensor, Tensor, Tensor, float]:
    """Detach, upcast to float64, expand GQA heads, resolve the default scale."""
    if not (q.dim() == 4 and k.dim() == 4 and v.dim() == 4):
        raise ValueError("q, k, v must be [B, H, N, D]")
    q64 = q.detach().to(torch.float64)
    k64 = k.detach().to(torch.float64)
    v64 = v.detach().to(torch.float64)
    k64, v64 = _expand_kv(q64, k64, v64)
    if scale is None:
        scale = default_scale(q.shape[-1])
    return q64, k64, v64, float(scale)


def math_reference(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    is_causal: bool = False,
    scale: float | None = None,
    attn_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Explicit fp64 softmax attention.  Returns ``(o, lse)`` in float64.

    ``o = softmax(scale * q k^T) v`` computed the textbook way (materialized N x N scores, one
    global row max).  This is the un-tiled truth the tiled model must reproduce exactly.
    """
    q64, k64, v64, scale = _prep(q, k, v, scale)
    n_q, n_kv = q64.shape[2], k64.shape[2]
    s = torch.matmul(q64, k64.transpose(-1, -2)) * scale  # [B, H, Nq, Nkv]
    keep = _keep_mask(n_q, n_kv, is_causal, attn_mask, s.device)
    if keep is not None:
        s = s.masked_fill(~keep, float("-inf"))
    m = s.amax(dim=-1, keepdim=True)
    # Guard: a fully masked row has m == -inf and (s - m) would be (-inf) - (-inf) = NaN.
    # Subtracting 0 instead keeps exp(-inf) = 0 for every masked entry.
    m_safe = torch.where(torch.isneginf(m), torch.zeros_like(m), m)
    p = torch.exp(s - m_safe)
    l_row = p.sum(dim=-1, keepdim=True)
    # Guard: l == 0 (fully masked row) -> output 0 instead of 0/0 = NaN.
    inv_l = torch.where(l_row == 0, torch.zeros_like(l_row), 1.0 / l_row)
    o = torch.matmul(p, v64) * inv_l
    lse = torch.where(l_row == 0, torch.full_like(l_row, float("inf")), m_safe + torch.log(l_row))
    return o, lse.squeeze(-1)


def sdpa_reference(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    is_causal: bool = False,
    scale: float | None = None,
    attn_mask: Tensor | None = None,
) -> Tensor:
    """PyTorch's SDPA in float64 pinned to the MATH backend (the only backend that runs fp64).

    This is the correctness oracle of the whole repo; it returns only ``o`` because SDPA does not
    expose the LSE.  Pinning MATH also documents that the oracle is *never* a fused kernel.
    """
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    q64 = q.detach().to(torch.float64)
    k64 = k.detach().to(torch.float64)
    v64 = v.detach().to(torch.float64)
    if scale is None:
        scale = default_scale(q.shape[-1])
    with sdpa_kernel([SDPBackend.MATH]):
        return F.scaled_dot_product_attention(
            q64,
            k64,
            v64,
            attn_mask=attn_mask,
            is_causal=is_causal,
            scale=float(scale),
            enable_gqa=k.shape[1] != q.shape[1],
        )


def tiled_online_softmax_reference(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    is_causal: bool = False,
    scale: float | None = None,
    block_m: int = 64,
    block_n: int = 64,
    attn_mask: Tensor | None = None,
    skip_masked_blocks: bool = True,
) -> tuple[Tensor, Tensor]:
    """The whiteboard model of ``fa_s3_fused`` / ``fa_s4_wmma`` / ``fa_s5_tuned``.

    One CUDA thread block owns one query tile ``q[i0:i1]`` for its whole life (FlashAttention-2
    loop order) and streams key/value tiles ``k[j0:j1]``, ``v[j0:j1]`` through shared memory.  Per
    query row it keeps three running quantities in registers -- ``m`` (max of the *raw* scores seen
    so far), ``l`` (sum of exponentials relative to ``m``) and ``acc`` (un-normalized output) -- and
    never materializes the N x N score matrix.  Line by line:

    ``c = scale * log2(e)``
        The softmax scale and the change of base to exp2 are folded into one constant so the
        inner loop is ``exp2((s - m) * c)``: one FMA + one MUFU.EX2 per score, and ``m`` can be
        tracked on the raw scores (no scaling pass over the tile).
    ``skip fully masked blocks``
        With a causal mask, a key tile that starts after the last query of this tile
        (``j0 > i1 - 1``) contributes nothing.  Skipping it *before any load* is the S6
        block-skipping optimization; mathematically it is a no-op (every ``p`` would be 0), which
        ``skip_masked_blocks=False`` lets the tests prove.  Tiles that straddle the diagonal are
        masked element-wise; tiles entirely below it are untouched.
    ``m_new = max(m, rowmax(s))``
        The running max only grows, so every previous exponential was taken relative to a value
        ``<= m_new`` and can be corrected by one multiply (next line).
    ``alpha = exp2((m - m_new) * c)``
        The rescale factor (``softmax_rescale_o`` in flash-attention's ``softmax.h``).  It is a
        multiply, never a divide: ``alpha <= 1`` so it can neither overflow nor amplify error.
    ``p = exp2((s - m_new) * c)``
        Every exponent is ``<= 0`` -> ``p in [0, 1]``: no overflow even for fp16 inputs scaled by
        1e2, which is why S/m/l/acc are fp32 in every kernel while only q/k/v/p are half.
    ``l = alpha * l + rowsum(p)``, ``acc = alpha * acc + p @ v``
        The online-softmax recurrence.  Forgetting ``alpha`` on ``acc`` is *the* classic bug: the
        result is then correct only when the whole row fits in one tile.  ``test_reference_cpu.py``
        contains a deliberately broken copy to prove the tile-invariance test catches it.
    ``o = acc / l`` (single divide, deferred normalization)
        FlashAttention-1 normalized after every tile; FlashAttention-2 defers it to the end, saving
        one multiply per output element per tile and one register per row.
    ``lse = m * scale + log(l)``
        ``m`` is the max of raw scores, so the scale is re-applied here; ``log`` (not ``log2``)
        because LSE is defined in base e.  Needed later by split-KV combine and the backward pass.
    Guards
        ``m_new == -inf`` (row fully masked so far) -> use 0 in the exponents so masked entries
        give ``exp2(-inf) = 0`` rather than ``(-inf) - (-inf) = NaN``; ``alpha`` is then
        ``exp2(-inf) = 0`` which is harmless because ``l`` and ``acc`` are still 0.
        ``l == 0`` at the end (row fully masked) -> ``o = 0`` and ``lse = +inf`` instead of 0/0.
        Neither guard can fire for a top-left causal mask with ``Nq == Nkv``; ``attn_mask`` exists
        in this model only so the tests can force them.
    """
    q64, k64, v64, scale = _prep(q, k, v, scale)
    if block_m <= 0 or block_n <= 0:
        raise ValueError("block_m and block_n must be positive")
    b, h, n_q, d = q64.shape
    n_kv = k64.shape[2]
    dev = q64.device
    c = scale * LOG2E
    keep = _keep_mask(n_q, n_kv, is_causal, attn_mask, dev)
    neg_inf = float("-inf")

    o = torch.empty_like(q64)
    lse = torch.empty(b, h, n_q, dtype=torch.float64, device=dev)

    for i0 in range(0, n_q, block_m):
        i1 = min(i0 + block_m, n_q)
        q_tile = q64[:, :, i0:i1, :]
        rows = i1 - i0
        m = torch.full((b, h, rows, 1), neg_inf, dtype=torch.float64, device=dev)
        l_row = torch.zeros((b, h, rows, 1), dtype=torch.float64, device=dev)
        acc = torch.zeros((b, h, rows, d), dtype=torch.float64, device=dev)

        for j0 in range(0, n_kv, block_n):
            j1 = min(j0 + block_n, n_kv)
            # Regime (a): key tile entirely above the causal diagonal -> nothing to do.
            if is_causal and skip_masked_blocks and j0 > i1 - 1:
                continue
            s = torch.matmul(q_tile, k64[:, :, j0:j1, :].transpose(-1, -2))  # raw, unscaled
            if keep is not None:
                blk = keep[..., i0:i1, j0:j1]
                # Regime (b): straddles the diagonal / has explicit holes -> element-wise mask.
                # Regime (c): all True -> leave the tile alone (no masking cost).
                if not bool(blk.all()):
                    s = s.masked_fill(~blk, neg_inf)
            m_new = torch.maximum(m, s.amax(dim=-1, keepdim=True))
            m_safe = torch.where(torch.isneginf(m_new), torch.zeros_like(m_new), m_new)
            alpha = torch.exp2((m - m_safe) * c)
            p = torch.exp2((s - m_safe) * c)
            l_row = alpha * l_row + p.sum(dim=-1, keepdim=True)
            acc = alpha * acc + torch.matmul(p, v64[:, :, j0:j1, :])
            m = m_new

        inv_l = torch.where(l_row == 0, torch.zeros_like(l_row), 1.0 / l_row)
        o[:, :, i0:i1, :] = acc * inv_l
        lse_tile = torch.where(
            l_row == 0, torch.full_like(l_row, float("inf")), m * scale + torch.log(l_row)
        )
        lse[:, :, i0:i1] = lse_tile.squeeze(-1)

    return o, lse


def unfused_reference(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    is_causal: bool = False,
    scale: float | None = None,
    attn_mask: Tensor | None = None,
) -> Tensor:
    """Naive PyTorch attention in the *input* dtype (the ``out_pt`` of flash-attention's tests).

    This is the "what PyTorch itself gets wrong in this dtype" term of the ratio rule.  It follows
    ``attention_ref(upcast=False, reorder_ops=True)``: scale q first, matmul, softmax, matmul, all
    in the input dtype (torch's half softmax accumulates in fp32 internally, like ours).  A row
    that is fully masked yields NaN here -- torch's behaviour, deliberately not guarded.
    """
    k, v = _expand_kv(q, k, v)
    if scale is None:
        scale = default_scale(q.shape[-1])
    n_q, n_kv = q.shape[2], k.shape[2]
    s = torch.matmul(q * scale, k.transpose(-1, -2))
    keep = _keep_mask(n_q, n_kv, is_causal, attn_mask, s.device)
    if keep is not None:
        s = s.masked_fill(~keep, float("-inf"))
    return torch.matmul(torch.softmax(s, dim=-1), v)


class RatioResult(NamedTuple):
    """Outcome of the FlashAttention ratio rule (see ``error_ratio``)."""

    err_mine: float  # max |out - ref64|
    err_torch: float  # max |torch_same_dtype - ref64|
    bound: float  # 2 * err_torch + 1e-6
    ok: bool  # err_mine <= bound (False when err_torch is NaN/inf: see torch_finite)
    out_finite: bool  # no NaN/inf in out
    torch_finite: bool  # torch's own unfused result was finite (else the rule is not applicable)

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return (
            f"err_mine={self.err_mine:.3e} err_torch={self.err_torch:.3e} bound={self.bound:.3e} "
            f"ok={self.ok} out_finite={self.out_finite} torch_finite={self.torch_finite}"
        )


def error_ratio(out: Tensor, ref64: Tensor, torch_same_dtype: Tensor) -> RatioResult:
    """FlashAttention test-suite tolerance rule: ``err_mine <= 2 * err_torch + 1e-6``.

    Why not ``torch.testing.assert_close`` defaults (fp16 rtol 1e-3 / fp32 rtol 1.3e-6)?  A fused
    kernel sums in a different order than the reference and its error grows with N, so any fixed
    atol/rtol is either too loose at N=128 or fails spuriously at N=4096.  Comparing against
    what PyTorch's own unfused attention gets wrong in the *same dtype* scales the tolerance with
    the problem (tests/test_flash_attn.py in Dao-AILab/flash-attention uses exactly 2x forward,
    3x backward).  The ``1e-6`` floor keeps the rule meaningful when ``err_torch`` is ~0 (tiny N).
    Callers still add finite checks and absolute sanity caps (fp32 1e-4, fp16 1e-2) so a kernel
    cannot pass merely because torch is also bad.
    """
    ref64 = ref64.to(torch.float64)
    err_mine_t = (out.to(torch.float64) - ref64).abs().max()
    err_torch_t = (torch_same_dtype.to(torch.float64) - ref64).abs().max()
    err_mine = float(err_mine_t)
    err_torch = float(err_torch_t)
    bound = 2.0 * err_torch + 1e-6
    out_finite = bool(torch.isfinite(out).all())
    torch_finite = math.isfinite(err_torch)
    ok = out_finite and torch_finite and err_mine <= bound
    return RatioResult(err_mine, err_torch, bound, ok, out_finite, torch_finite)
