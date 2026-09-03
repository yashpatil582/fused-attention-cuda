"""Build the ``fused_attn._C`` PyTorch extension over the SAME kernel sources CMake compiles.

Usage (from the repo root)::

    cd python && TORCH_CUDA_ARCH_LIST="7.5" python setup.py build_ext --inplace   # Colab T4
    pip install --no-build-isolation -e python                                     # editable

Design notes (the WHY):
* Kernel sources come from ``<root>/sources.txt`` (one ``.cu`` path per line, relative to the repo
  root, ``#`` comments allowed), the single list shared with CMake so the two builds cannot drift.
  Fallback when the file is absent: ``src/*.cu``.
* ``TORCH_CUDA_ARCH_LIST`` is honoured by ``torch.utils.cpp_extension`` itself (it generates the
  ``-gencode`` flags); we never hand-write ``-gencode``.  Set it to ``"7.5"`` on Colab: without it
  torch probes the visible GPU at build time or, with no GPU, compiles for every arch it knows.
* nvcc flags live under ``extra_compile_args["nvcc"]`` -- ``extra_cuda_cflags`` only exists on the
  JIT ``load()`` path and would be silently ignored here.
* ``-lineinfo`` (never ``-G``) so Nsight Compute can map SASS back to source lines in the
  committed ``.ncu-rep`` files without changing the generated code.
* C++ standard: torch <= 2.12 builds extensions as C++17, torch >= 2.13 as C++20 (pytorch
  RELEASE.md).  cpp_extension only appends its own ``-std=`` when none is given, so we pick the
  one matching the installed torch; pinning C++17 on a C++20 torch fails to parse its headers.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

HERE = Path(__file__).resolve().parent  # <root>/python
ROOT = HERE.parent
# setuptools resolves package paths relative to the CWD, not to setup.py; make both agree so
# ``python python/setup.py ...`` from the repo root behaves like ``cd python && python setup.py``.
os.chdir(HERE)


def _torch_version() -> tuple[int, int]:
    import torch

    major, minor = torch.__version__.split("+")[0].split(".")[:2]
    return int(major), int(minor)


def kernel_sources() -> list[str]:
    """``.cu`` files from ``<root>/sources.txt`` (fallback ``src/*.cu``), relative to python/."""
    listing = ROOT / "sources.txt"
    if listing.exists():
        rel = [
            line.strip()
            for line in listing.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    else:
        rel = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "src").glob("*.cu"))
        print(f"[fused_attn setup] {listing} absent; falling back to src/*.cu", file=sys.stderr)
    missing = [r for r in rel if not (ROOT / r).exists()]
    if missing:
        raise FileNotFoundError(f"sources.txt lists files that do not exist: {missing}")
    return [os.path.relpath(ROOT / r, HERE) for r in rel]


def main() -> None:
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        print(
            "[fused_attn setup] TORCH_CUDA_ARCH_LIST is unset: torch will probe the local GPU "
            '(or build for all archs). On Colab T4 use TORCH_CUDA_ARCH_LIST="7.5".',
            file=sys.stderr,
        )
    std = "-std=c++20" if _torch_version() >= (2, 13) else "-std=c++17"
    sources = ["fused_attn/ext.cpp"] + kernel_sources()
    ext = CUDAExtension(
        name="fused_attn._C",
        sources=sources,
        include_dirs=[str(ROOT / "include")],
        extra_compile_args={
            "cxx": ["-O3", std],
            "nvcc": ["-O3", "-lineinfo", "--expt-relaxed-constexpr", std],
        },
    )
    setup(
        name="fused_attn",
        version="0.1.0",
        description="FlashAttention-style fused attention CUDA kernels as a PyTorch custom op",
        packages=["fused_attn"],
        package_dir={"": "."},
        ext_modules=[ext],
        cmdclass={"build_ext": BuildExtension},
        python_requires=">=3.10",
        zip_safe=False,
    )


if __name__ == "__main__":
    main()
