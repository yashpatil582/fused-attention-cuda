# Numerics

Correctness comes before speed at every stage: a benchmark row is only written by a process that
passed these checks on the same shapes (`docs/BENCHMARK_PROTOCOL.md` rule 3). This file states the
oracle, the tolerance rule and why the obvious alternative is wrong, the fp16 guards, the causal
convention, determinism, and the planned error study.

## 1. Oracle

`F.scaled_dot_product_attention(q64, k64, v64, is_causal=causal)` in **float64** under
`sdpa_kernel([SDPBackend.MATH])`; only the math backend accepts float64. Default scale
`1 / sqrt(D)` (the SDPA default is `1 / math.sqrt(query.size(-1))`); an explicit `scale=0.123` case
guards against applying the scale twice or after the max subtraction. The CPU reference in
`python/fused_attn/reference.py` (`math_reference`, fp64) is the same computation without torch's
SDPA machinery and is what the Mac-side tests use.

## 2. Tolerance policy: the ratio rule

For each case, with `ref64` the oracle and `torch_same_dtype` the unfused
`softmax(q k^T * s) @ v` computed by torch in the kernel's own dtype:

```
err_mine  = max |out_kernel.double() - ref64|
err_torch = max |torch_same_dtype.double() - ref64|
assert err_mine <= 2 * err_torch + 1e-6
```

plus `torch.isfinite(out).all()` and sanity caps `atol 1e-4` (fp32), `atol 1e-2` (fp16). This is the
flash-attention test-suite pattern (`tests/test_flash_attn.py`: `(out - out_ref).abs().max() <= 2 *
(out_pt - out_ref).abs().max()`, with 3x for gradients), and it is dtype- and shape-agnostic.

**Why not `torch.testing.assert_close` defaults.** Its defaults (PyTorch 2.14 docs) are
fp16 `rtol=1e-3, atol=1e-5`; bf16 `rtol=1.6e-2, atol=1e-5`; fp32 `rtol=1.3e-6, atol=1e-5`;
check `|a - b| <= atol + rtol * |b|`. Those are round-off-level bounds for two implementations that
sum in the same order. A fused kernel sums the softmax denominator and the `P V` products in tile
order with per-tile rescaling, so its rounding differs from an unfused reference by design, and the
difference grows with the number of tiles (N / B_c), not with a fixed constant. A fixed fp32
`rtol 1.3e-6` therefore fails legitimately correct kernels at large N and passes broken ones at small
N. Comparing against fp64 and bounding by torch's *own* same-dtype error measures what matters: the
kernel is no worse than the framework's unfused path. `assert_close` defaults are kept only for
small-N smoke tests. `torch.autograd.gradcheck` is not used (forward only; its defaults "are designed
for input of double precision").

## 3. fp16 overflow guards

Half has max 65504 and min normal ~6.1e-5. The rule, enforced in every kernel from S4 on and true of
S1-S3 by construction: Q, K, V and the P operand fed to the tensor cores are half; **S (scores), m
(running max), l (running sum), the O accumulator and LSE are always fp32** (`wmma::fragment<accumulator, ..., float>`;
`fa::Dtype` comment in `include/fa/attention.cuh`). Tests: `test_half_precision_dynamic_range`
(`tests/test_kernels_gpu.py`) scales q by `1e2` and by `1e-3` and requires finite output within the
ratio rule.

Guards, as implemented in `include/fa/softmax_math.cuh` (and mirrored by the CPU models):
- running max still `-inf` (first K/V tile, or an all-masked prefix): `exp2(-inf - (-inf))` is
  NaN, so `rescale_alpha` returns `alpha = 0` and `p_from_score` returns `P = 0`. Exact, because
  `acc` and `l` are still 0 whenever `m == -inf`; no separate first-tile branch exists;
- empty denominator at the end: `finalize_inv_l` returns `inv_l = 0` when `l == 0` (so `O = 0`,
  not NaN) and `finalize_lse` returns `LSE = +inf` (the non-split FlashAttention convention).

FlashAttention's `csrc/flash_attn/src/softmax.h` is the reference for the recurrence; its
`m_safe = (m == -INFINITY) ? 0 : m` and `inv = (l == 0 || l != l) ? 1 : 1 / l` variant is
equivalent (it multiplies zeros where this code multiplies by zero), but it is not what this
repo's code does, and the tests (`test_fully_masked_row_guards_never_nan` on the CPU model) pin the
repo's form.

exp2 with a folded scale: `P = exp2((S - m_new) * scale * log2(e))` and `alpha = exp2((m_prev -
m_new) * scale * log2(e))` with the constant folded once; m is kept in raw `Q K^T` units and
`LSE = m * scale + log(l)`. The error cost of `exp2f` versus `expf` and of the folding is
**measured, not assumed**: the ratio rule is the gate and the S3 STAGES.md entry records the observed
`err_mine / err_torch`. `--use_fast_math` is off for S1-S4 and only ever enabled as its own measured
S5 commit.

## 4. Causal alignment convention

**Top-left**, i.e. query row i attends to key columns `j <= i`, matching PyTorch SDPA `is_causal`
("The attention masking has the form of the upper left causal bias due to the alignment when the mask
is a non-square matrix"). In-window `Nq == Nkv`, where top-left and bottom-right agree; the convention
is still pinned as the module constant `python/fused_attn/reference.py::CAUSAL_ALIGN = "top_left"`
and in `_keep_mask`'s docstring because flash-attn >= 2.1 aligns "to the bottom right corner of the
attention matrix" for `L != S`, which is the classic false "my kernel is wrong" fire drill. Tests:
`test_causal_equals_explicit_tril_mask` (both `tests/test_reference_cpu.py` and
`tests/test_kernels_gpu.py`) compares the causal path against an explicit
`torch.ones(N, N).tril()` mask (diagonal 0, the L == S case) and the oracle. Kernels mask with
`>` on the diagonal tile only (S3/S4) and, once S6 lands, skip fully masked tiles before any load with
`if (bn * Bc > bm * Br + Br - 1) continue;` (three regimes: skipped / diagonal masked / interior
unmasked).

## 5. Determinism and tile invariance

- The forward pass has no atomics and a fixed reduction order, so it is bitwise deterministic:
  `test_determinism_50_runs_bitwise_equal` runs the kernel 50 times on the same input and asserts
  identical bytes.
- `test_tile_invariance` forces `FA_FORCE_BC in {N, N/2, N/4}` (and `FA_FORCE_BR`) through
  `fa::set_debug_overrides` and checks all outputs against each other and the oracle with the ratio
  rule; one tile means no rescale ever happens, so this is the test that catches a missing or
  half-applied `acc *= alpha` before any GPU time is spent. The CPU model in `reference.py`
  (`tiled_online_softmax_reference(block_m, block_n)`) runs the same test on the Mac in seconds.
- `N % 32 == 0` and `Nq == Nkv` are enforced with a clear error in-window; ragged-N tests exist as
  `xfail` for the week after.

## 6. Planned error-vs-N study (week after, W4)

Sweep N in {512, ..., 4096} (and beyond on Track B) and plot max-abs and RMS error versus the fp64
oracle for: fp32 in / fp32 accumulate; fp16 in / fp32 accumulate; bf16 in / fp32 accumulate (sm_80+
only, marked pending on T4); and torch's own fp16 unfused path as the control line. Then fix N and
vary `B_c` to show that error tracks the number of online-softmax rescales (N / B_c), not N itself.
Framing reference: "Is Flash Attention Stable?" (arXiv 2405.02803) reports roughly an order of
magnitude more numeric deviation for Flash Attention than baseline attention at bf16, while its
Wasserstein-distance proxy is 2-5x less significant than low-precision training. All plots go to
`docs/img/` with the generating script in `scripts/`; no number appears here until then.
