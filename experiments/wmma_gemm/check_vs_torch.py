#!/usr/bin/env python3
"""Torch-side cross-check for experiments/wmma_gemm/wmma_gemm.cu.

Regenerates the SAME deterministic inputs (Knuth MMIX LCG, seeds 1/2/3, rounded to fp16 or
bf16), computes the fp64 references with torch.matmul, and prints the statistics the CUDA
program prints, so the two can be compared line by line on a GPU box:

    ./run.sh --dump out.bin && python3 check_vs_torch.py --compare out.bin

Without --compare it only prints the reference statistics (runs on the Mac, CPU torch).
With --compare it applies the same pass/fail rules to the dumped S and O and exits non-zero
on failure. Never fabricates: every number here is computed from the inputs.
"""

from __future__ import annotations

import argparse
import math
import struct
import sys

import numpy as np
import torch

BM = BN = D = 64  # must match wmma_gemm.cu
MASK64 = (1 << 64) - 1


def lcg_fill(n: int, seed: int) -> torch.Tensor:
    """Bit-exact mirror of lcg_next() in wmma_gemm.cu: value = (s >> 40) / 2^24 * 2 - 1."""
    s = seed
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        s = (s * 6364136223846793005 + 1442695040888963407) & MASK64
        out[i] = (s >> 40) / 16777216.0 * 2.0 - 1.0
    return torch.from_numpy(out)


def rounded(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    # The CUDA host code rounds double -> float -> half/bf16; mirror that exact chain.
    return x.float().to(dtype).double()


def references(M: int, N: int, K: int, dtype: torch.dtype):
    A = rounded(lcg_fill(M * K, 1), dtype).view(M, K)
    B = rounded(lcg_fill(N * K, 2), dtype).view(N, K)
    V = rounded(lcg_fill(N * D, 3), dtype).view(N, D)
    S = torch.matmul(A, B.T)  # fp64, [M, N]
    scale = 1.0 / math.sqrt(K)
    nbn = N // BN
    Sb = S.view(M, nbn, BN)
    m = Sb.amax(dim=-1, keepdim=True)
    P = torch.exp((Sb - m) * scale)  # [M, nbn, BN]
    row_sum = P.sum(dim=-1, keepdim=True)  # from unrounded P, as the kernel does
    P16 = rounded(P, dtype)
    Vb = V.view(nbn, BN, D)
    o_exact = torch.einsum("mbj,bjd->mbd", P, Vb) / row_sum
    o_ref16 = torch.einsum("mbj,bjd->mbd", P16, Vb) / row_sum

    def to_kernel_layout(t: torch.Tensor) -> torch.Tensor:
        # Kernel layout: [M/64][N/64][64][64] = (bm, bn, r, d)
        return t.view(M // BM, BM, nbn, D).permute(0, 2, 1, 3).contiguous().view(-1)

    return S, to_kernel_layout(o_exact), to_kernel_layout(o_ref16)


def load_dump(path: str):
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic != b"WMMAGEMM":
            sys.exit(f"{path}: bad magic {magic!r}")
        M, N, K, is_bf16 = struct.unpack("<4i", f.read(16))
        s_gpu = np.fromfile(f, dtype=np.float32, count=M * N)
        o_gpu = np.fromfile(f, dtype=np.float32, count=M * N)
    if s_gpu.size != M * N or o_gpu.size != M * N:
        sys.exit(f"{path}: truncated dump")
    s_t, o_t = torch.from_numpy(s_gpu).double(), torch.from_numpy(o_gpu).double()
    return M, N, K, bool(is_bf16), s_t, o_t


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("M", nargs="?", type=int, default=2048)
    ap.add_argument("N", nargs="?", type=int, default=2048)
    ap.add_argument("K", nargs="?", type=int, default=64)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--compare", metavar="DUMP", help="raw S/O dump written by wmma_gemm --dump")
    args = ap.parse_args()

    M, N, K, bf16 = args.M, args.N, args.K, args.bf16
    gpu = None
    if args.compare:
        M, N, K, bf16, s_gpu, o_gpu = load_dump(args.compare)
        gpu = (s_gpu, o_gpu)
        print(f"[dump] {args.compare}: M={M} N={N} K={K} dtype={'bf16' if bf16 else 'fp16'}")
    if M % BM or N % BN or K % 16:
        sys.exit("M%64, N%64 and K%16 must be 0 (same limits as wmma_gemm.cu)")
    dtype = torch.bfloat16 if bf16 else torch.float16
    name = "bf16" if bf16 else "fp16"

    S, o_exact, o_ref16 = references(M, N, K, dtype)
    err_ref = (o_ref16 - o_exact).abs().max().item()
    print(f"torch reference: M={M} N={N} K={K} D={D} dtype={name}")
    print(
        f"[ref] S = A·Bᵀ: max|S_ref| {S.abs().max().item():.4f} "
        f"mean|S_ref| {S.abs().mean().item():.4f}"
    )
    print(
        f"[ref] O: mean|O_ref| {o_exact.abs().mean().item():.4f} "
        f"err_ref({name}-rounded P) {err_ref:.3e}"
    )

    if gpu is None:
        return 0
    s_gpu, o_gpu = gpu
    finite = bool(torch.isfinite(s_gpu).all() and torch.isfinite(o_gpu).all())
    s_err = (s_gpu - S.reshape(-1)).abs().max().item()
    err_mine = (o_gpu - o_exact).abs().max().item()
    s_pass = finite and s_err <= 5e-2
    o_pass = finite and err_mine <= 2.0 * err_ref + 1e-6
    ratio = err_mine / err_ref if err_ref > 0 else float("nan")
    print(f"[check] S max abs err {s_err:.3e} (limit 5e-2) {'PASS' if s_pass else 'FAIL'}")
    print(
        f"[check] O err_mine {err_mine:.3e} err_ref {err_ref:.3e} ratio {ratio:.3f} "
        f"(rule err_mine <= 2*err_ref + 1e-6) {'PASS' if o_pass else 'FAIL'}"
    )
    if not s_pass:
        print("RESULT: FAIL (S GEMM)")
        return 1
    if not o_pass:
        print("RESULT: FAIL (P·V ratio rule)")
        return 2
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
