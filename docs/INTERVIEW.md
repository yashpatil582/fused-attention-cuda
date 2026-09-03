# Interview notes

What I would say at a whiteboard about this repo, in the order an interviewer is likely to ask.
Every hardware number is from an NVIDIA tuning guide, the PTX ISA, the Nsight docs or a paper (URLs
in `docs/RESEARCH.md`); every measurement is a `TBD` until it exists in a committed CSV or profile.

## 1. Online softmax: the identity and the rescale factor

Softmax of a row `x` is `exp(x_i - m) / l` with `m = max_i x_i` and `l = sum_i exp(x_i - m)`. If I
only ever see the row in pieces, `x = [x1, x2]`, then `m = max(m1, m2)` and
`l = exp(m1 - m) * l1 + exp(m2 - m) * l2` (Milakov & Gimelshein 2018; FA1 Sec 3.1). The same factors
rescale the unnormalised numerator. So per K/V tile `j`:

```
m_new = max(m, rowmax(S_j))
alpha = exp(m - m_new)                 # in (0, 1]; a multiplier, never an inverse
P     = exp(S_j - m_new)
l     = alpha * l + rowsum(P)
acc   = alpha * acc + P @ V_j
...at the end:  O = acc / l ;  LSE = m + log(l)
```

Deferred normalisation is the FA2 form: I divide once, not every tile, and store one statistic
`LSE = m + log l` instead of `m` and `l`. In the shipped FlashAttention this is
`csrc/flash_attn/src/softmax.h::softmax_rescale_o`,
`scores_scale = exp2f((scores_max_prev - scores_max_cur) * softmax_scale_log2)`. I use `exp2f`
with `scale * log2(e)` folded into one constant because `ex2.approx.f32` is a single SFU instruction;
`m` stays in raw `Q K^T` units and the true `LSE = m * scale + log(l)` is reassembled at the end.
Guards, as written in `include/fa/softmax_math.cuh`: a row whose running max is still `-inf` (the
first K/V tile, or an all-masked prefix) would give `exp2(-inf - (-inf)) = NaN`, so
`rescale_alpha` returns `alpha = 0` when `m_old == -inf`; that is exact because `acc` and `l` are
still zero at that point, so there is nothing to rescale (no `Is_first` branch is needed).
`p_from_score` returns 0 for the same case. At the end an empty row (`l == 0`) gets
`inv_l = 0` from `finalize_inv_l`, so `O = 0` rather than NaN, and `finalize_lse` reports
`LSE = +inf` (the non-split FlashAttention convention: `exp(s - lse) = 0` for every key). The
shipped FlashAttention (`softmax.h`) uses the equivalent `m_safe = 0` / `inv = 1` variant, which
also multiplies zeros; the guards there are the citation, not the description. The
tile-invariance test (block size N, N/2, N/4 must agree) is what catches a missing `acc *= alpha`,
because with one tile the rescale never fires.

## 2. Why the HBM bytes collapse (FA1 Theorem 2)

Standard attention materialises `S = Q K^T` and `P = softmax(S)`, two `N x N` objects, and reads them
back: `Theta(N d + N^2)` HBM accesses. With `M` elements of on-chip SRAM and tiles
`B_c = ceil(M / 4d)`, `B_r = min(ceil(M / 4d), d)`, FlashAttention streams K and V once
(`Theta(N d)`) and touches Q/O once per K/V tile, `T_c = ceil(N / B_c) ~ 4 N d / M` times, so the
total is `Theta(N^2 d^2 / M)`. Against the `N^2` term that is a factor of `~ M / d^2` with `M` in
*elements*. On T4 that factor is small: the 64 KB per-block SRAM is `M = 16K` fp32 elements, so
`M / d^2 = 16384 / 4096 = 4` at `d = 64` and `16384 / 16384 = 1` at `d = 128`; by the theorem
alone the fused kernel promises a 4x reduction at `d = 64` and none at `d = 128` (H100's 227 KB,
`M = 58K` elements, gives 14x and 3.5x). More FLOPs (recompute), fewer bytes, and attention is memory
bound, so it pays where `M / d^2 > 1`. FA1 also proves a matching lower bound for `M in [d, N d]`.

What actually makes `dram__bytes.sum` collapse on T4, then, is not the theorem but the cache: with
FA2's loop order every block streams the same K/V, and the per-Q-tile re-reads are served from L2
as long as a head's K/V stays resident (at C128 in fp32 that is `2 x 2048 x 128 x 4 B = 2 MB` per
head; whether it stays resident is measured, not assumed). That is why
`lts__t_bytes.sum` is recorded next to `dram__bytes.sum`: the L2 number carries the re-reads the
theorem counts as HBM traffic, the DRAM number is what the theorem-plus-cache actually costs. What I
expect to see: `dram__bytes.sum` for S1/S2 grows like `N^2` (S and P written and read); for S3, Q is
read once, O and LSE written once, and `dram__bytes.sum` approaches the compulsory
`B H (4 N D sizeof + 4 N)` while `lts__t_bytes.sum` stays ~`T_c` times larger. The S3 entry reports
both. Measured on the T4 at 3add37c (`profiles/t4/stage{1,2,3}_*.raw.csv`, C64 = B2 H8 N2048 D64 fp32,
C128 = D128): S1's three kernels move 365.7 + 620.0 + 560.8 = **1,546 MB** through DRAM at C64
(1,609 MB at C128); S2's tiled kernels still materialise S and P and move **1,499 MB** — tiling
fixed coalescing and reuse, not the byte count; S3 moves **38.7 MB** at C64 and **100.3 MB** at
C128. That is 40x / 16x fewer DRAM bytes. The compulsory `B H (4 N D sizeof + 4 N)` is 33.7 MB /
67.2 MB, so S3 sits at 1.15x / 1.49x of compulsory, and `lts__t_bytes.sum` = 650 MB / 1,283 MB
(17x / 13x the DRAM bytes) carries the K/V re-reads the theorem counts as HBM traffic: at D=128 a
head's K/V no longer stays fully L2-resident across the 32 Q blocks, which is the 1.49x. The fp16
stages read the same 1.13-1.14x of compulsory (19.1 MB at C64, 38.4 MB at C128).

## 3. FA1 -> FA2: loop order and split-Q warp partitioning

FA1 loops over K/V tiles outside and Q tiles inside, so every Q tile's `O, m, l` bounce through HBM
each outer step. FA2 gives one thread block one Q tile for its whole life and loops over K/V inside,
so the accumulator and statistics stay in registers and are written once; the grid becomes
`(ceil(N / B_r), batch, heads)` so long-context small-batch runs still fill the SMs. Inside the block
FA1 split K/V across the four warps ("split-K"): each warp held a partial sum of the *same* output
rows, so they had to write to shared memory, `__syncthreads`, and add up. FA2 splits Q across warps:
each warp owns whole rows, so the row max and row sum are warp-local. With `mma.sync` the accumulator
puts one row across 4 lanes, so the reduction is a 4-lane butterfly (`quad_allreduce_` in
`softmax.h`); that is the entire cross-thread cost of softmax. FA2 also drops the per-tile
`diag(l)^-1` because on A100 non-matmul fp32 (19.5 TFLOPs/s) is ~16x more expensive than fp16
matmul (312 TFLOPs/s). My S3 is the FA2 loop order; my S4 uses WMMA so the reduction goes through
shared memory instead of a quad butterfly, and section 5 says why.

## 4. Shared-memory bank conflicts

32 banks, 4 bytes each, `bank(word) = word mod 32`; a warp is served in one wavefront when its lanes
hit distinct banks (same-address broadcast excepted), and "the hardware splits a memory request that
has bank conflicts into as many separate conflict-free requests as necessary" (Best Practices
10.2.3.1). The trap in attention is reading K column-wise from a row-major tile for `Q K^T`:

```
Ks[r][c], fp32, row stride 64 words (D = 64):   word = 64 r + c  ->  bank = c mod 32  (r drops out)

        bank:   0    1    2   ...  31  |  0    1   ...  31
  row 0:       K00  K01  K02  ...      | K0,32 ...
  row 1:       K10  K11  K12  ...      | K1,32 ...
  row 2:       K20  K21  K22  ...      |
                     ^ 32 lanes, one row each, all reading column 1 -> all in bank 1
                       32-way conflict, 32 wavefronts

Ks[r][64 + 1]:  word = 65 r + c  ->  bank = (r + c) mod 32
        bank:   0    1    2    3  ...
  row 0:       K00  K01  K02  K03 ...
  row 1:       pad  K10  K11  K12 ...      each row shifts one bank right
  row 2:       ..   pad  K20  K21 ...      column 1 now spans 32 distinct banks
```

For scalar column reads `+1` is the textbook fix. For 16-byte vector loads and WMMA the leading
dimension must stay a multiple of 16 bytes ("Programming Tensor Cores in CUDA 9": fp16 dimensions in
multiples of 8 elements), so S5a pads fp32 tiles by +4 floats and fp16 tiles by +8 halves, never +1
for half; cuda-samples' `cudaTensorCoreGemm` does the same with `SKEW_HALF`. The XOR swizzle
(`Swizzle<3,3,3>`, CuTe `swizzle.hpp`) is the zero-smem alternative, pure index arithmetic, portable
to sm_75. I do not argue conflict-freedom from memory for the vector case: the arbiter is
`l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`. Values at C64 / C128 (T4, 3add37c
and d835758): S2 `fa_s2_qk_tiled` with the deliberately unpadded `Ks[r][BK]` = **520 M / 1,040 M**
load conflicts (the textbook column-read case, kept as the "before"), `fa_s2_pv_tiled` 16.8 M /
33.6 M; S3 (`Qs[BR][D+1]`, thread-per-row) 0 / 0 load conflicts and 55 K / 2.7 M store conflicts;
S4 with +8-half padding 4.9 M / 8.1 M load and 4.9 M / 11.2 M store conflicts out of 47.7 M /
84.1 M warp instructions; S5 unchanged (4.9 M / 8.1 M and 5.4 M / 12.0 M). The padding took the
S2-style 32-way conflict off the table; what remains is the fragment store/reload traffic, and the
S4 stall breakdown ranked `long_scoreboard` (DRAM latency) and `mio_throttle` above it, so the S5
budget went to the K/V prefetch and the block height. The XOR swizzle stays the next step and would
be judged by this counter.

## 5. WMMA opacity versus the documented `mma.sync` lane map

The Programming Guide says of `nvcuda::wmma` fragments that "the way matrix elements map to fragment
internal storage is unspecified"; NVIDIA's blog adds that element-wise operations "don't have to know
this mapping". So `frag.x[i] *= alpha` is legal, but a row max, a causal mask, or feeding `P` straight
into the second GEMM as an A-operand is not: all three are position-dependent. The portable way out
is `store_matrix_sync` to shared memory in a known layout, reduce and mask there, reload as
`matrix_a`. That costs one smem round trip per tile per iteration, `B_r x B_c` fp32 of shared memory
and two barriers, and on T4's 64 KB it is exactly the budget that forces `B_c` down. `mma.sync`
avoids it because the PTX ISA publishes the lane map: `groupID = laneid >> 2`,
`threadID_in_group = laneid % 4`; for the m16n8k16 f32 accumulator `row = groupID` and `groupID + 8`,
`col = threadID_in_group * 2 + i`. FlashAttention's `mask.h` computes `(lane_id % 4) * 2` and
`row_idx_base + i * 8` from exactly that, and its row reductions are 4-lane shuffles. On T4 the shape
is m16n8k8 (`SM75_16x8x8_F32F16F16F32_TN` in `kernel_traits.h`), twice as many `mma` per k = 16, and
`utils.h` notes the accumulator-to-A-operand reshape is the identity there. S4 is WMMA on purpose:
correctness first on the opaque API with padding; the register-resident `mma.sync` + `ldmatrix` path
is the stated next step, with a committed SASS listing when it lands.

## 6. Ampere features and the T4 caps

T4 is sm_75, Turing. From the Turing Tuning Guide: 96 KB unified L1/shared per SM with 64/32 or
32/64 carveouts, "Turing allows a single thread block to address the full 64 KB of shared memory",
static allocations capped at 48 KB, 64K 32-bit registers per SM, 255 per thread, and "the maximum
number of concurrent warps per SM is 32 on Turing (versus 64 on Volta)". Tensor cores take fp16 with
fp16/fp32 accumulate, WMMA 16x16x16 / 32x8x16 / 8x32x16. The Ampere Tuning Guide lists as *new*:
bf16 tensor cores, TF32, and the asynchronous global-to-shared copy that "avoid[s] using extra
registers for memory copies and can also bypass the L1 cache" (PTX `cp.async`, "Requires sm_80 or
higher"). `mma.sync` m16n8k16 f16 is also sm_80+ (CUTLASS `functionality.md`). Consequences I state
rather than hide: on T4 my K/V prefetch is a register-staged `ld.global -> st.shared` double buffer,
which costs the registers `cp.async` would have saved; bf16 and `cp.async` live under
`#if __CUDA_ARCH__ >= 800`, compiled by CI, run only on Track B. Occupancy from the tile budget is
a compile-time constant in `include/fa/tile_config.h`, not a typed number — and the profile
corrected it. The header's S3 estimate (`TileCfg<750, 64>::kSmemF32 == 30720`, 2 blocks x 4 warps
= 8 warps/SM at `d = 64`) assumed 128-thread blocks and a 30 KB tile; the kernel as written is
one thread per row (BR = 64 -> 64 threads = 2 warps) with a 33,024 B tile (padded `Qs[BR][D+1]`
plus the S strip), so `64 KB / 33 KB = 1` block and the achieved
`sm__warps_active.avg.pct_of_peak_sustained_active` is **6.26 %** = 2 warps at `d = 64` and 6.25 %
at `d = 128` (49,408 B, > 48 KB, launched through the opt-in). S4: 12.52 % (one 4-warp block,
45,056 B) at `d = 64`, 6.25 % (one 2-warp block, 43,008 B) at `d = 128`. S5 keeps 12.49 % at
`d = 64` and raises `d = 128` to a 4-warp block from N = 2048 up (BR = 64, BC = 16, 59,904 B,
opt-in; S4's 2-warp tile below that, where the sweep showed the tall tile losing) — the achieved
value of that tile is in `docs/STAGES.md` S5. Registers never bind before shared memory
on any of these (`launch__occupancy_limit_registers` 3-6 blocks vs `_shared_mem` 1).

## 7. Why FA3 is Hopper-only

FA3 overlaps softmax with an in-flight GEMM. That needs `wgmma.mma_async`, an asynchronous
warpgroup-wide tensor-core op that sources operands from shared memory (compiled with
`arch=compute_sm90a,code=sm_90a`); TMA (`cp.async.bulk.tensor`, sm_90+), a copy engine that needs no
registers; `setmaxnreg`, so producer warpgroups hand their registers to consumers; and clusters with
distributed shared memory. Producer/consumer specialisation plus two-warpgroup pingpong scheduling
exists because the H100 does ~989 TFLOPS of fp16 matmul but ~3.9 TFLOPS of special-function work, a
256x gap, so `exp` would otherwise take a large fraction of the cycles. Pre-Hopper `mma.sync` is
synchronous with register operands: there is nothing to overlap softmax against, which is why I do
not cargo-cult a warp-specialised structure onto Turing. The ablation in the paper (661 -> 582 -> 570
TFLOPs/s removing pipelining then warp specialisation) says what each trick is worth *on H100*; on my
hardware the analogous knobs are prefetch depth and barrier count, and they are measured, not assumed.

## 8. Reading an ncu profile: SOL first, then stalls

Speed Of Light "reports the achieved percentage of utilization with respect to the theoretical
maximum" per unit (Crovella, Analysis-Driven Optimization). I read SM % and Memory % first. Both low
means latency-bound: go to `WarpStateStats` for the top stall reasons and `Occupancy` for why there
are too few warps to hide them. Memory high means memory-bound: `MemoryWorkloadAnalysis`, hit rates,
sectors per request, `dram__bytes.sum` against the compulsory byte count. SM high means compute-bound:
`ComputeWorkloadAnalysis`, which pipe is saturated (`sm__pipe_tensor_op_hmma_cycles_active` for S4).
The stall names, in ncu's vocabulary: Long Scoreboard = waiting on global/L2 (K/V loads); Short
Scoreboard = waiting on shared memory or MIO results; MIO Throttle = the shared-memory/shuffle queue
is full; Barrier = `__syncthreads`; Math Pipe Throttle = a pipe is the limiter; Not Selected = fine,
enough eligible warps. What I expect across the ladder: S1/S2 memory-bound and coalescing-limited;
S3 shifting to `mio_throttle` and `barrier`; S4 lighting up the tensor pipe and pushing the
bottleneck back to the smem round trip; S5a/b attacking bank conflicts and barriers. Each expectation
is written down before the profile and then checked; STAGES.md keeps the ones that were wrong.

## 9. Benchmark hygiene, line by line

At least 10 warmup and 100 timed iterations, one `cudaEvent` pair per iteration, one synchronize after
the loop, median with p10/p90 (a bare host timer measures launch overhead, Best Practices 9.1.1). A
256 MB int32 buffer zeroed before every timed iteration, Triton's `do_bench` convention, so small-N
inputs do not sit in L2; four rotating input buffers for the same reason. The output checksummed into
a volatile sink so the compiler cannot delete the kernel, and a bench row refused unless the same
process passed the correctness check. Clocks locked to base on the VM and recorded per row; on Colab
sampled and flagged `clock=unlocked` if locking is refused. FLOPs `4 B H N^2 D`, halved for causal
only once the kernel actually skips blocks, using `(n + 1) / (2n)`, and stated on the table. Bytes as
actually moved, with `oom_by_design` rows instead of silently dropping the shapes the naive kernels
cannot run. Baseline named: SDPA `EFFICIENT_ATTENTION` on T4 (FLASH refuses `sm < 80`), `MATH` fp32
labelled oracle-only because it upcasts half inputs, unfused torch as the floor. Every row carries the
commit, GPU, driver, nvcc, torch, ncu and clock; README tables regenerate from CSVs and a test fails
if they are stale. ncu durations never appear in a speedup.

## 10. Causal FLOP accounting

With square tiles and `n = ceil(N / block)` tile rows, the causal kernel computes the tiles
`j <= i`: `n (n + 1) / 2` of `n^2`, a fraction of `(n + 1) / (2n)`, which is 1/2 only as `n -> inf`
(at element granularity it is `(N + 1) / (2N)`). Three regimes in the K/V loop: fully masked tiles
skipped *before any load* (`bn * Bc > bm * Br + Br - 1`), the diagonal tile masked element-wise, the
interior unmasked with no mask instructions at all. FA2 reports "around 1.7-1.8x" from skipping.
Skipping by loop bound (`kv_end = causal ? min(i0 + BR, N) : N`) has been in every fused stage
since S3, and the T4 sweep measures it: causal / non-causal time at fixed shape is **1.73-1.87x**
over N in {1024, 2048, 4096}, D in {64, 128}, B x H in {8, 256} (`bench/results/t4/stage5.csv`
@ d835758, fp16; e.g. B1_H8_N2048_D64 1.108 vs 1.956 ms = 1.77x, B8_H32_N4096_D64 112.45 vs
209.33 ms = 1.86x; S4 gives the same 1.73-1.87x). It is below 2x for exactly the `(n + 1) / (2n)`
reason plus the masked diagonal tile. Accounting follows CONTRACT §5: a causal row of a
tile-skipping kernel is credited half of `4 B H N^2 D` (`flops=causal_half`, the FlashAttention
benchmark convention) so its TFLOP/s is comparable with the dense row; rows benchmarked before that
convention landed (S1-S4 sweeps at 3add37c) still say `flops=full` in their notes, and a
materialising kernel (S1, S2, MATH oracle) always keeps the full count.

## 11. Production context: TensorRT-LLM and cuDNN

TensorRT-LLM's `gpt_attention` takes packed QKV (`[num_tokens, 3 * hidden]`) and runs two different
kernels: a FlashAttention-style fused kernel in the context (prefill) phase, selected by
`context_fmha_type` with FP8 on Ada/Hopper, and a "masked multi-head attention" kernel in the
generation phase that fuses QKV bias, RoPE and KV-cache quantisation and uses `multi_block_mode`
(split-KV across thread blocks, flash-decoding's idea) when batch x heads cannot fill the SMs. KV
cache is contiguous or paged (`KVCacheManager`), INT8 or FP8 with a per-tensor scale; XQA is the
GQA/MQA decode kernel. My kernel is the context-phase shape; split-KV decode and paged KV are the
stated next steps, and LSE is emitted from day one because the decode combine needs it. cuDNN 9's
runtime fusion engine ships "Fused Flash Attention fprop/bprop" patterns behind the cudnn-frontend
`sdpa()` node, head_dim a multiple of 8 up to 128 on Ampere and 256 on Hopper, FP8 on Hopper only,
and "requires SM80 (Ampere) or newer". On a T4 neither cuDNN's fused SDPA nor PyTorch's FLASH backend
nor upstream flash-attention exists, which is why writing one by hand is not a toy on that card.

## 12. Per-stage summaries (filled from STAGES.md)

- S0: see `docs/STAGES.md` S0, 5-line summary and Q1/Q2.
- **S1 — naive three kernels, fp32** (measured values in `docs/STAGES.md`):
  1. Three kernels, one thread per element, everything through global memory — the way a first port
     looks, so every later stage is a one-variable change against it.
  2. The K loads are strided by D on purpose: adjacent threads own adjacent keys, so a warp touches 32
     sectors and uses 4 bytes of each; that is what "uncoalesced" means and it shows up as
     bytes-per-sector, not as an error.
  3. The N×N score and probability matrices round-trip HBM, so `dram__bytes.sum` grows quadratically in
     the sequence length while the tensors themselves are linear in it — the number FlashAttention's
     IO argument is about.
  4. Correctness is a differential test against an fp64 oracle with the FlashAttention ratio rule,
     plus a CPU emulation of the index math, so the T4 run only had to catch layout and sync bugs.
  5. It is a baseline, not a product: its job is to make the S2 and S3 deltas measurable.
- **S2 — shared-memory tiled GEMMs + coalesced loads** (measured values in `docs/STAGES.md`):
  1. Same math and the same N×N intermediates as S1; only the way bytes reach the SM changed.
  2. Each block loads a 64×16 slice of Q and K into shared memory with 16-byte vector loads where
     consecutive lanes are consecutive addresses, so a warp's request is four full 128-byte lines.
  3. Every value that reaches a register feeds four FMAs (4×4 register tile), so the inner loop is
     8 shared loads per 32 FMAs instead of 2 loads per FMA.
  4. The unroll factor was chosen by the ptxas register report, not by taste: unroll 2 keeps 61
     registers and four resident blocks; full unroll spills under a launch bound.
  5. The bank conflicts on the K tile are left in deliberately — they are the measured "before" for the
     S5a padding change.
- **S3 — fused kernel, online softmax, FA2 loop order** (measured values in `docs/STAGES.md`):
  1. The block owns a 64-row Q tile for its whole life and walks the keys in tiles; the score tile
     lives in registers, so nothing of size N×N ever leaves the SM.
  2. The softmax is computed online: keep a running max and a running sum per row; when a new tile
     raises the max, multiply the old sum and the old accumulator by exp(old − new) — a factor in (0, 1]
     — and add the new tile's contributions.
  3. Normalisation is deferred to one divide per row at the end (FA2), which also means the `l = 0` guard
     exists in exactly one place.
  4. The measured effect is in DRAM bytes, not time: fp32 thread-per-row is register-heavy and
     low-occupancy, and the profile says so.
  5. Everything the kernel does is mirrored by a ten-line Python model with a tile-invariance test, so
     the recurrence is defended on a whiteboard before it is defended in SASS.
- **S4 — fp16 WMMA, fp32 accumulate** (measured values in `docs/STAGES.md`):
  1. Both matmuls run on the tensor cores as 16×16×16 fp16 fragment products with fp32 accumulation;
     Q fragments are hoisted out of the K/V loop and Kᵀ is free by declaring K column-major.
  2. WMMA's fragment layout is opaque, so the softmax cannot index the accumulator; the score tile is
     stored to shared memory, reduced there, and the probabilities go back in as an fp16 operand.
  3. Each warp owns 16 rows end to end, which keeps the running max and sum warp-local and removes
     every cross-warp barrier from the softmax.
  4. The output accumulator must be rescaled by alpha per row, which is another round trip — skipped
     by a warp vote once the max stops changing.
  5. The tensor pipe metric is the proof this stage is real; the barrier and MIO stalls it introduces
     are what S5 is for.
- **S5 — register-staged K/V prefetch + block height** (measured values in `docs/STAGES.md`):
  1. The S4 profile said DRAM latency (`long_scoreboard`) and a single resident block — not the
     tensor pipe — were the next costs; S5 attacks exactly those two, each as a template knob so
     every variant has its own ptxas line and its own profile.
  2. Turing has no `cp.async`, so the double buffer is hand-rolled through registers: tile t+1's
     loads are issued before tile t's math and stored to shared memory after its last reader passed;
     the stall fell from 11–17 % to under 1 %.
  3. Registers are the price (150 → 168, and 168 → 254 on S4's D=128 tile) and 0 spills is the
     gate; the A/B, not the reasoning, chose the defaults.
  4. Taller blocks lose at D=64 because they force smaller K/V tiles (more barriers per row) and
     win at D=128 where S4's block was only 2 warps — but only once the grid is full (−14.5 % at
     N=2048, +39 % at N=128), so the D=128 tile is chosen by N with one sweep-derived threshold.
  5. What is left is `mio_throttle` — the WMMA round trips — which is the register-resident
     `mma.sync` design FlashAttention-2 uses and the stated next step.
- **S6 — torch op dispatch, GPT block, causal accounting**:
  1. One op, one schema, dispatch by capability: the highest stage that supports (dtype, D, sm)
     runs, so the ladder's best kernel is what PyTorch users get without naming a stage.
  2. `register_fake` + `opcheck` + a `fullgraph=True` compile test are the difference between "it
     runs" and "it composes with torch.compile"; all three pass on the T4.
  3. The GPT-block number is measured with our binding-boundary copies inside the window, so it is
     the cost a user would pay, not the kernel's best case.
  4. Causal skipping is a loop bound, not a branch per element; it gives 1.73–1.87× and is credited
     half the FLOPs only from the commit where the convention was written down.
  5. The block still runs faster on SDPA than on our kernel on this GPU; the number is committed
     because the docs promise measurements, not wins.
