#!/usr/bin/env python3
"""End-to-end check: a nanoGPT-style transformer block with swappable attention.

Why this exists: a kernel that wins a micro-benchmark can still lose inside a real block
(extra transposes/copies at the binding boundary, launch overhead relative to the GEMMs).
So we measure tokens/sec of one full Block (LayerNorm -> causal self-attention -> residual
-> LayerNorm -> MLP -> residual; karpathy/nanoGPT shapes: n_embd 768, 12 heads, T 1024,
batch 8, fp16, inference) with the attention call swapped between
  * F.scaled_dot_product_attention pinned to EFFICIENT_ATTENTION (or FLASH on sm_80+), and
  * torch.ops.fused_attn.forward (our best kernel for the dtype/head-dim),
and record the attention share of block time from a second CUDA-event pair around the
attention call. The `.contiguous()` our op needs after the head split is INSIDE that
attention window on purpose: it is a cost the binding imposes and hiding it would be lying.

This is a WARM end-to-end measurement, not a kernel benchmark (BENCHMARK_PROTOCOL rule 1-2
exception, stated there): 50 iterations after warmup, no L2 flush, no rotating inputs, one
synchronize after the loop (one event pair per iteration outside, one per attention call
inside, read back after the loop). Rows say `warm;no_flush`.

Rows go to bench/results/<gpu>/gpt_block.csv in the frozen schema (impl = gpt_sdpa_efficient
/ gpt_sdpa_flash / gpt_fused; ms_median = block time per iteration; tokens/sec, attention ms
and attention share live in `notes`, which is the only free-text column). The `stage` of
gpt_fused is the stage `forward()` actually dispatches to (the highest implemented stage for
fp16 / D, walked with impl_supports), never a typed argument.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "bench"))
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from fa_common import NOISE_RATIO, append_rows, percentile, utc_stamp  # noqa: E402
from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402

from bench import (  # noqa: E402
    TORCH_DTYPE,
    ClockSampler,
    clock_notes,
    env_columns,
    load_configs,
    note_env,
)


class Attention(nn.Module):
    """nanoGPT CausalSelfAttention with the SDPA call replaced by a pluggable callable.

    One (t0, t1) event pair is recorded per forward and kept in `windows`, so the caller can
    read every attention time after a single synchronize instead of syncing per iteration."""

    def __init__(self, n_embd: int, n_head: int, attn_fn):
        super().__init__()
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.n_head, self.attn_fn = n_head, attn_fn
        self.windows: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t0.record()  # attention window starts BEFORE the layout change our op requires
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = self.attn_fn(q, k, v)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        t1.record()
        self.windows.append((t0, t1))
        return self.c_proj(y)


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, attn_fn):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = Attention(n_embd, n_head, attn_fn)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.GELU(), nn.Linear(4 * n_embd, n_embd)
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


def dispatched_stage(dtype: torch.dtype, head_dim: int) -> int:
    """The stage `torch.ops.fused_attn.forward` resolves to: the highest implemented one for
    (dtype, D) on this device (ext.cpp picks the best stage the same way). 0 if none."""
    import fused_attn

    for s in (5, 4, 3, 2, 1):
        if fused_attn.impl_supports(s, dtype, head_dim):
            return s
    return 0


def attention_fns(scale: float, gpu_dir: Path, dtype: torch.dtype, head_dim: int) -> dict:
    """{impl_name: (fn, stage)}; SDPA backends that refuse the device are skipped with reason."""
    fns = {}
    for name, attr in (
        ("gpt_sdpa_efficient", "EFFICIENT_ATTENTION"),
        ("gpt_sdpa_flash", "FLASH_ATTENTION"),
    ):
        backend = getattr(SDPBackend, attr, None)
        if backend is None:
            continue

        def sdpa(q, k, v, backend=backend):
            with sdpa_kernel([backend]):
                return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)

        fns[name] = (sdpa, 0)
    try:
        import fused_attn  # noqa: F401
    except Exception as e:
        print(f"[gpt_block] fused_attn extension not importable: {str(e).splitlines()[0]}")
    else:
        stage = dispatched_stage(dtype, head_dim)
        if stage == 0:
            print(f"[gpt_block] skip gpt_fused: no stage implemented for {dtype} D={head_dim}")
        else:
            # forward(q,k,v,is_causal,scale) needs contiguous [B,H,N,D]; the copy is timed on
            # purpose.
            fns["gpt_fused"] = (
                lambda q, k, v: torch.ops.fused_attn.forward(
                    q.contiguous(), k.contiguous(), v.contiguous(), True, scale
                ),
                stage,
            )
    return fns


def measure(block: Block, x, warmup: int, iters: int, sampler: ClockSampler) -> dict:
    with torch.no_grad():
        for _ in range(warmup):
            block(x)
        torch.cuda.synchronize()
        block.attn.windows.clear()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        sampler.start()
        for i in range(iters):
            starts[i].record()
            block(x)
            ends[i].record()
        torch.cuda.synchronize()  # the single synchronize; inner windows are read after it
        clocks = sampler.stop()
    block_ms = sorted(a.elapsed_time(b) for a, b in zip(starts, ends, strict=True))
    attn_ms = sorted(t0.elapsed_time(t1) for t0, t1 in block.attn.windows[-iters:])
    return {"block": block_ms, "attn": attn_ms, "clocks": clocks}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--iters", type=int)
    ap.add_argument("--warmup", type=int)
    ap.add_argument(
        "--clock-locked",
        nargs="?",
        const=0,
        default=None,
        type=int,
        metavar="MHZ",
        help="clocks locked via nvidia-smi -lgc; with a value the lock is verified per row",
    )
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if not torch.cuda.is_available():
        print("gpt_block.py: CUDA not available", file=sys.stderr)
        return 2
    g = load_configs()["gpt_block"]
    iters, warmup = a.iters or g["iters"], a.warmup or g["warmup"]
    n_embd, n_head, T, Bsz, dtype = g["n_embd"], g["n_head"], g["T"], g["batch"], g["dtype"]
    D = n_embd // n_head
    gpu_dir = ROOT / "bench" / "results" / a.gpu
    env = env_columns()
    run_id = utc_stamp()
    sampler = ClockSampler()
    torch.manual_seed(a.seed)
    x = torch.randn(Bsz, T, n_embd, device="cuda", dtype=TORCH_DTYPE[dtype])
    rows = []
    fns = attention_fns(1.0 / math.sqrt(D), gpu_dir, TORCH_DTYPE[dtype], D)
    for name, (fn, stage) in fns.items():
        block = Block(n_embd, n_head, fn).to("cuda", TORCH_DTYPE[dtype]).eval()
        try:
            with torch.no_grad():
                block(x)
            torch.cuda.synchronize()
        except RuntimeError as e:
            reason = str(e).splitlines()[0][:200]
            print(f"[gpt_block] skip {name}: {reason}")
            note_env(gpu_dir, f"{name}: refused ({dtype}): {reason}")
            continue
        m = measure(block, x, warmup, iters, sampler)
        blk, att = percentile(m["block"], 50), percentile(m["attn"], 50)
        p10, p90 = percentile(m["block"], 10), percentile(m["block"], 90)
        tok_s = Bsz * T / (blk * 1e-3)
        clock_col, cnotes, _ = clock_notes(m["clocks"], a.clock_locked)
        notes = [
            f"tok_s={tok_s:.1f}",
            f"attn_ms={att:.4f}",
            f"attn_pct={100.0 * att / blk:.1f}",
            f"T={T}",
            f"iters={iters}",
            "warm",
            "no_flush",
            "flops=full",
            f"run={run_id}",
            *cnotes,
        ]
        if p10 > 0 and p90 / p10 >= NOISE_RATIO:
            notes.append("noisy")
        rows.append(
            dict(
                env,
                clock_mhz=clock_col or env["clock_mhz"],
                impl=name,
                stage=stage,
                dtype=dtype,
                B=Bsz,
                H=n_head,
                N=T,
                D=D,
                causal=1,
                ms_median=f"{blk:.4f}",
                ms_p10=f"{p10:.4f}",
                ms_p90=f"{p90:.4f}",
                tflops="",
                gbps="",
                notes=";".join(notes),
            )
        )
        print(
            f"GPT_ROW impl={name} stage={stage} block_ms={blk:.4f} attn_ms={att:.4f} "
            f"attn_pct={100.0 * att / blk:.1f} tok_s={tok_s:.1f}"
        )
        del block
        torch.cuda.empty_cache()
    if rows:
        append_rows(gpu_dir / "gpt_block.csv", rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
