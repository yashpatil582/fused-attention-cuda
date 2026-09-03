// Online-softmax arithmetic shared by every fused stage (S3 fp32, S4/S5 WMMA) and by the
// bench's self-check. One definition so the kernels, the CPU model (python/fused_attn/
// reference.py) and src/reference/cpu_reference.cpp implement the SAME recurrence:
//
//   m_new = max(m, rowmax(S))                       m in RAW score units (before `scale`)
//   P     = exp2((S - m_new) * scale * log2(e))     one FFMA + one EX2 per element
//   alpha = exp2((m - m_new) * scale * log2(e))
//   l     = alpha * l + rowsum(P)
//   acc   = alpha * acc + P @ V                     multiply by alpha, never divide (FA2 softmax.h)
//   o     = acc * (1 / l)                           single divide per row at the end
//   lse   = m * scale + log(l)                      natural log, torch.logsumexp units
//
// WHY exp2 with a folded constant: exp(x*scale) == exp2(x*scale*log2e); pre-multiplying
// scale*log2e once per launch turns every exp into FFMA + EX2 (the hardware unit computes
// 2^x), which is what flash-attention does (softmax.h `scale_apply_exp2`).
#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <limits>

namespace fa {

constexpr float kLog2e = 1.4426950408889634f;
constexpr float kNegInf = -std::numeric_limits<float>::infinity();
constexpr float kPosInf = std::numeric_limits<float>::infinity();

// scale * log2(e), computed once per launch on the host and passed to the kernel.
__host__ __device__ __forceinline__ float scale_log2(float scale) {
  return scale * kLog2e;
}

// P element from a raw score: exp2(s*c - m*c). `m_c` is the row max ALREADY multiplied by c
// (hoisted per row by the caller). No guard: only call when m is finite, i.e. at least one
// unmasked key in the row so far. That is the common path and must stay one FFMA + one EX2.
__host__ __device__ __forceinline__ float exp2_scaled(float s, float c, float m_c) {
  return exp2f(fmaf(s, c, -m_c));
}

// Guarded P: a fully masked row so far has m == -inf and every s == -inf, and
// exp2(-inf - (-inf)) is NaN. The softmax of an all-masked prefix is defined as 0 here
// (contributes nothing to acc and l); the final guards below turn the empty row into
// o = 0, lse = +inf. Cannot happen for top-left causal with Nq == Nkv (row i always sees
// key 0) but kernels use it on the diagonal tile anyway so the invariant is local.
__host__ __device__ __forceinline__ float p_from_score(float s, float c, float m_raw) {
  return (m_raw == kNegInf) ? 0.0f : exp2f(fmaf(s, c, -(m_raw * c)));
}

// Rescale factor for the running sums when the row max moves from m_old to m_new.
// Guard: m_old == -inf (first tile of the row, or an all-masked prefix) must give 0, not
// NaN: (-inf) - (-inf) is NaN when m_new is also -inf, and exp2(-inf) is 0 otherwise. acc
// and l are 0 at that point, so alpha = 0 is exact.
__host__ __device__ __forceinline__ float rescale_alpha(float m_old, float m_new, float c) {
  return (m_old == kNegInf) ? 0.0f : exp2f((m_old - m_new) * c);
}

// 1 / l for the deferred normalisation. Guard: l == 0 only for a row that saw no unmasked
// key; returning 0 makes o = acc * 0 = 0 instead of NaN (flash-attention softmax.h
// `normalize_softmax_lse` does the same).
__host__ __device__ __forceinline__ float finalize_inv_l(float l) {
  return (l > 0.0f) ? 1.0f / l : 0.0f;
}

// logsumexp of the row in natural-log units: m*scale + ln(l), with m in raw score units.
// Guard: an empty row (l == 0 or m == -inf) reports +inf, flash-attention's non-split
// convention (exp(s - lse) is then 0 for every key, consistent with o = 0). The split-KV
// combine (week-after) will need the -inf sentinel instead; that is that op's concern.
__host__ __device__ __forceinline__ float finalize_lse(float m_raw, float l, float scale) {
  return (l > 0.0f && m_raw != kNegInf) ? fmaf(m_raw, scale, logf(l)) : kPosInf;
}

}  // namespace fa
