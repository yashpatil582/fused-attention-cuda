# `experiments/wmma_gemm` — S4 tensor-core de-risking microkernel

Standalone CUDA program (no torch, no cuBLAS unless `-DUSE_CUBLAS`) that runs **one
iteration of the Stage-4 fused-attention inner loop on `nvcuda::wmma` tensor cores** for a
single 64-row query tile × one 64-key K/V block, in isolation, and checks it against an fp64
CPU oracle. It exists so that every WMMA-specific failure mode (opaque fragment layout, `ldm`
alignment, 256-bit pointer alignment, the 48 KB static / 64 KB T4 shared-memory caps, bf16
being sm_80+ only) is hit and fixed here, before `src/stage4_wmma.cu` is written on the
critical path (12:00 Sep 4 cut line in `.research/PLAN.md`).

## What it proves (each is a printed line, never a hard-coded number)

| Step in the kernel | What it de-risks for `stage4_wmma.cu` |
|---|---|
| `S = A·Bᵀ` with `A [M×K]` and `B [N×K]` both row-major fp16, B loaded as `matrix_b col_major` | **Kᵀ is free**: `matrix_b` is K×N and col-major means element (k, n) sits at `ptr[n·ldm + k]`, which is exactly where row-major `K[n][k]` already is. No transpose kernel, no transposed smem tile. |
| 64×64 block tile, 4 warps in a 2×2 grid, each warp a 32×32 sub-tile = 2×2 `16×16×16` fragments, K loop in steps of 16 | The fragment shapes and the warp tiling S4 will use for d = 64. |
| Operand tiles in smem with row stride `K + 8` halves (144 B for K = 64) | `ldm` legality (multiple of 8 halves = 16 B), 256-bit fragment pointers, 16-B vector loads staying aligned, and the bank pattern (below). |
| `store_matrix_sync` of the fp32 accumulators into a `64 × (64+4)` fp32 smem tile | **The opaque-layout escape hatch** — the only portable way to get (row, col) of a score. |
| Row max and row sum with `__shfl_xor_sync` butterflies over smem rows (warp w owns rows 16w…16w+15, each lane 2 adjacent columns) | The S5b reduction pattern, already in place. |
| `P = exp2((S − rowmax)·scale·log2e)` written as packed `__half2` into the **dead B tile**, reloaded as `matrix_a row_major` for `O = P·V` (V `[64×64]` row-major as `matrix_b row_major`) | The second tensor-core GEMM from a 16-bit P produced by the softmax, and the aliasing that makes the S4 smem budget fit (Q+K+V+S = 45 568 B for d = 64, under 48 KB — P has no buffer of its own). |
| `O / l` through a second `store_matrix_sync` round trip | The row-wise divide is position-dependent too; S4's online-softmax rescale `O ← α·O + P·V` has the same problem (see "Consequences for S4"). |
| A per-block checksum `atomicAdd` sink | Timed runs write neither S nor O, and still cannot be dead-code-eliminated. |
| `fp64` CPU oracle: `max|S_gpu − S_ref| ≤ 5e-2` and the FA-suite ratio rule `err_mine ≤ 2·err_ref + 1e-6`, where `err_ref` is the error of the same fp64 math with P rounded to fp16 before P·V | Correctness gate; the ratio rule is the one `docs/NUMERICS.md` uses for the real kernels. |
| `cudaEvent` timing, median of `--iters` (default 50) after 5 warm-ups, `2·M·N·K + 2·M·N·D` FLOP → TFLOPS | Whether the tensor pipe is actually exercised (cross-check with `ncu --metrics sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active,sm__inst_executed_pipe_tensor.sum ./wmma_gemm`). |
| `--bf16`: `__nv_bfloat16` fragments under `#if __CUDA_ARCH__ >= 800`, host refuses on sm < 80 **and** when the binary's `binaryVersion < 80` | The Track-B bf16 row compiles in CI and never mis-launches on the T4. |
| `-DUSE_CUBLAS`: `cublasGemmEx` on the same S GEMM (fp16 in, fp32 out, `CUBLAS_COMPUTE_32F`) | A vendor tensor-core throughput reference point for the same shape. |

## Run

```bash
cd experiments/wmma_gemm
./run.sh                                  # nvcc -O3 -std=c++17 -lineinfo -arch=sm_75 -Xptxas -v; fp16; 2048 2048 64
./run.sh 2048 2048 128                    # K=128: 61 952 B smem → dynamic opt-in path (fits T4's 64 KB, 1 block/SM)
ARCH=sm_80 ./run.sh --bf16                # Track-B box only
NVCC_EXTRA="-DUSE_CUBLAS -lcublas" ./run.sh
./run.sh --dump build/out.bin && python3 check_vs_torch.py --compare build/out.bin   # vs torch.matmul
build/wmma_gemm --ref-only                # no CUDA call: prints the fp64 oracle stats (any box)
python3 check_vs_torch.py                 # Mac, CPU torch: the same stats from torch — must match
```

`run.sh` builds into `build/` (gitignored) and tees the ptxas report to `build/ptxas_<arch>.txt`
(registers, spill stores/loads — the S4 gate is zero spills). The program prints the smem carve-up, the occupancy from
`cudaOccupancyMaxActiveBlocksPerMultiprocessor`, both checks, the timing, and `RESULT: PASS|FAIL`;
exit code 0/1/2/3 = pass / S check failed / ratio rule failed / usage-unsupported-CUDA error.
Constraints: `M % 64 == 0`, `N % 64 == 0`, `K % 16 == 0`, D fixed at 64. `--ref-only` stops after
the `[ref]` lines without touching CUDA, so the C++ oracle (LCG, rounding chain, softmax math) can
be diffed against `check_vs_torch.py` on a GPU-less machine.

Measured results (T4, first Colab run): **TBD** — paste the printed `[check]`, `[time]`, ptxas and
occupancy lines here verbatim, with the commit SHA, driver and nvcc version from
`bench/results/t4/env.txt`.

### Verified on the Mac without a GPU (2026-09-02)

Compile-only, in the arm64 `nvidia/cuda:12.8.1-devel-ubuntu22.04` container (nvcc 12.8.93, the
PLAN's PROBE-L loop: `docker run --rm --platform linux/arm64 -v "$PWD:/w" -w /w
nvidia/cuda:12.8.1-devel-ubuntu22.04 nvcc -O3 -std=c++17 -lineinfo -arch=sm_75 -Xptxas -v
wmma_gemm.cu`), with `-Xcompiler -Wall,-Wextra` clean:

| build | `fa_wmma_f16_kernel` | `fa_wmma_bf16_kernel` |
|---|---|---|
| `-arch=sm_75` | 80 registers, 0 spill stores/loads, 0 B static smem, 1 barrier | 4 registers (trap stub, body not compiled) |
| `-arch=sm_80` (+ `-DUSE_CUBLAS -lcublas`) | 64 registers, 0 spills | 78 registers, 0 spills |
| `-arch=sm_90` | 72 registers, 0 spills | 80 registers, 0 spills |

The fp64 oracle inside the binary (`--ref-only`, no CUDA call) and `check_vs_torch.py` print
identical statistics to every displayed digit for fp16 and bf16 at 128×128×64 and for fp16 at
2048×2048×64 and 2048×2048×128 — i.e. the LCG, the double→float→half rounding chain, the
block softmax and the P-rounding reference agree between C++ and `torch.matmul`. The GPU
`[check]`/`[time]` lines above remain **TBD** until the first T4 run.

## The WMMA facts this code is built on (from `.research/tech-tensor-core-programming-constraints-by-n.json`)

1. **Fragment layout is opaque.** CUDA C++ Programming Guide, *Warp Matrix Functions → Description*:
   "the way matrix elements map to fragment internal storage is unspecified"; the fragment exposes
   only "a public member array `x[]` containing `num_elements` entries".
   NVIDIA blog *Programming Tensor Cores in CUDA 9*: "The mapping between individual matrix
   parameters to their fragments is opaque, so your program should not make assumptions about it."
   — https://docs.nvidia.com/cuda/archive/10.2/cuda-c-programming-guide/index.html#wmma-description ,
   https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/
2. **Element-wise work on fragments is still legal** (same blog): "Although the mapping from matrix
   coordinates to threads isn't defined, element-wise operations don't have to know this mapping
   so they can still be performed using fragments." What attention needs is *position-dependent*
   (row max/sum, masks, feeding P as the next A operand), which is exactly what WMMA cannot express.
3. **`store_matrix_sync` is the escape hatch**: it writes the accumulator "with a specified leading
   dimension and memory layout (row or column major), ensuring results are stored in a predictable
   format" (research bundle, NVIDIA guidance). Cost: one smem round trip + barriers per tile per
   inner iteration, plus `BLOCK_M × BLOCK_N` fp32 of smem for S — "which on Turing, where you only
   have 64 KB, is exactly the budget you cannot spare".
4. **`ldm` and pointer rules** (Programming Guide, `load_matrix_sync`/`store_matrix_sync`;
   paraphrased): `ldm` must be a multiple of 8 for `__half` and of 4 for `float` — 16 bytes either
   way — and `mptr` must be 256-bit aligned. Hence **pad by +8 halves / +4 floats, never +1**
   (`judge_plan.md` 4.3), and every fragment sub-pointer in this file is a multiple of 32 B from a
   128-B-aligned base (16-row × 144-B row blocks, 16-column × 2-B column blocks).
5. **Why +8 also fixes banks** (derivation, not a measurement): 32 banks × 4 B = 128 B per sweep.
   A 64-half row is 128 B, so with no pad every row starts in bank 0 and one column of 8 rows is an
   8-way conflict. With a 144-B stride (36 words) row `r` starts at bank `4r mod 32`; 8 consecutive
   rows × a 16-B chunk each cover all 32 banks exactly once. The fp32 tile uses stride 68 floats
   (272 B = 68 words → the same shift of 4 banks per row). `cudaTensorCoreGemm`'s `SKEW_HALF`
   is the same idea. S5a measures it with
   `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`.
6. **WMMA shapes for `__half` on sm_75**: 16×16×16, 32×8×16, 8×32×16 only (Programming Guide,
   *Element Types and Matrix Sizes*). None is 16×8×16 — the `mma.sync` shape set is a disjoint
   vocabulary, so a kernel does not mix the two APIs.
7. **Dtype support by arch**: Turing tensor cores take FP16 (fp16 or fp32 accumulate) and
   INT8/4/1 only; the Turing Tuning Guide makes no mention of BF16/TF32
   (https://docs.nvidia.com/cuda/turing-tuning-guide/index.html). bf16/tf32 WMMA are gated at
   compute capability 8.0 in the Programming Guide's *Alternate Floating Point* subsection. So
   `fa_wmma_bf16_kernel` has a real body only for `__CUDA_ARCH__ >= 800`, and the host checks both
   the device's `major >= 8` and the loaded kernel's `cudaFuncAttributes::binaryVersion >= 80`.
8. **Shared-memory caps**: static allocations are limited to 48 KB on every arch; above that the
   buffer must be `extern __shared__` plus
   `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes)`, whose
   return value must be checked because it can fail on a smaller device. T4/sm_75 allows a single
   block to address 64 KB (not 96 KB — that figure is Volta). sm_80 163 KB, sm_86/8.9 99 KB, sm_90
   227 KB ("CUDA reserves 1 KB of shared memory per thread block").
9. **Occupancy trap on Turing**: 32 warps/SM maximum (vs 64 on sm_80, 48 on sm_86). A 4-warp
   block that needs > 32 KB smem gets 1 block/SM = 4 warps = 12.5 % of the ceiling. The program
   prints the real number; it is an S4 tuning input (Bc = 32 vs 64), not a surprise.
10. **Why FlashAttention-2 uses `mma.sync` instead**: PTX ISA 9.7.15.5 publishes the lane→element
    map (`groupID = laneid >> 2`, `threadID_in_group = laneid % 4`; accumulator rows `groupID`,
    `groupID+8`, cols `threadID_in_group·2 + i`), and `flash_fwd_kernel.h` / `mask.h` index rows
    with `(tidx % 32) / 4` and `row_idx_base + i·8` — a documented layout lets FA2 keep S and P in
    registers with no smem round trip. sm_75 only has `m16n8k8` f16 (`SM75_16x8x8_F32F16F16F32_TN`),
    sm_80+ `m16n8k16`; `ldmatrix` is sm_75+, `cp.async` is sm_80+. Out of scope in-window; listed as
    W6 in the plan.

## How this becomes `src/stage4_wmma.cu`

Map of this file → the fused kernel (`fa_s4_wmma`, contract kernel name):

- `A` tile → the block's resident **Q tile** (loaded once, `[Br × (D+8)]` halves).
- `B` tile → the streamed **K tile** for KV block `j`; `V` tile → the streamed **V tile**.
  The `for (bn …)` grid dimension here becomes the `for j in KV blocks` loop inside one block
  (FA2 loop order: block owns Q, iterates K/V).
- Step (2)–(3) unchanged: `S_j = Q·K_jᵀ` → `store_matrix_sync` → fp32 `Ss`.
- Step (4) becomes the **online** softmax: `m_new = max(m, rowmax(S_j))`,
  `P = exp2((S_j − m_new)·c)`, `α = exp2((m − m_new)·c)`, `l = α·l + rowsum(P)`, with the `-inf`
  row guard from `include/fa/softmax_math.cuh` once causal masking is applied on the diagonal tile.
  P still goes into the dead K tile.
- Step (5) unchanged: `P·V_j` on tensor cores.
- Step (6) is the part that changes: the O accumulator **cannot** stay in an opaque fragment
  across KV blocks because `O ← α·O + P·V_j` is row-wise. Two portable options, both a 64×64 fp32
  smem round trip per KV block:
  (a) keep O in `Ss`-style smem in a known layout: `store_matrix_sync` the fresh `P·V_j` product
      to a scratch tile, then `O[r][d] = α[r]·O[r][d] + PV[r][d]` with plain loads/stores; or
  (b) keep O as an accumulator fragment, `store_matrix_sync` it, rescale rows in smem,
      `load_matrix_sync` it back as `accumulator` (`mem_row_major`), then `mma_sync(O, P, V, O)`.
  Option (a) needs one extra `64 × 68` fp32 tile (17 408 B) → 62 976 B total for d = 64, still under
  the T4's 64 KB but 1 block/SM; option (b) reuses `Ss` after P is written and keeps the 45 568 B
  budget. S4 starts with (b) and records the choice in `docs/STAGES.md`.
- Epilogue: `O / l` and `LSE = m·scale + log(l)` per row, written once (`[B,H,N,D]`, `[B,H,N]`).
- Everything printed by this program (ptxas registers/spills, occupancy, TFLOPS, the two checks)
  becomes the "before" column of the S4 entry in `docs/STAGES.md`; the S4 ncu evidence is
  `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active` going from 0 (S3) to
  non-zero and `sm__inst_executed_pipe_tensor.sum > 0`.

## Limitations of this experiment (deliberate)

- One KV block, no online rescale, no causal mask, no LSE — those are S4 itself.
- No L2 flush in the timing loop: the inputs (0.75 MB at the default size) are meant to sit in L2
  so the number reflects the tensor-core inner loop, not HBM. `bench/` follows the contract
  protocol (256 MB flush, ≥100 iterations).
- `D` is a compile-time 64; `K` (the Q/K head dim) is a runtime argument so K = 128 exercises the
  dynamic-smem opt-in path on the same binary.
- The global S/O writes in check mode are for verification only (16 MB each at 2048²) and are
  skipped in the timed runs.
- Not compiled on the Mac (no nvcc); first compile happens in `scripts/local_compile.sh` or on
  Colab. Host-only code paths were written to build with a plain C++17 compiler.
