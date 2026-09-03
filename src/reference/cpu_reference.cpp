// See cpu_reference.h. Everything is fp64 so the reference is ~1e-12 accurate and can serve
// as the oracle for both fp32 and fp16 kernels (the tolerance policy lives in
// docs/NUMERICS.md and tests/, not here).
#include "cpu_reference.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace fa {
namespace ref {

namespace {

constexpr double kNegInf = -std::numeric_limits<double>::infinity();
constexpr double kPosInf = std::numeric_limits<double>::infinity();
constexpr double kLog2e = 1.4426950408889634;

inline std::size_t idx(const Shape& s, int heads, int b, int h, int n) {
  return ((static_cast<std::size_t>(b) * heads + h) * s.N + n) * s.D;
}

inline int kv_head(const Shape& s, int h) {
  return h / (s.H / s.Hkv);
}

inline double dot(const double* a, const double* b, int d) {
  double acc = 0.0;
  for (int i = 0; i < d; ++i)
    acc += a[i] * b[i];
  return acc;
}

}  // namespace

std::size_t qo_elems(const Shape& s) {
  return static_cast<std::size_t>(s.B) * s.H * s.N * s.D;
}
std::size_t kv_elems(const Shape& s) {
  return static_cast<std::size_t>(s.B) * s.Hkv * s.N * s.D;
}
std::size_t lse_elems(const Shape& s) {
  return static_cast<std::size_t>(s.B) * s.H * s.N;
}

void attention_rows(const double* q, const double* k, const double* v, const Shape& s, int b, int h,
                    int row0, int rows, double* o, double* lse) {
  const int hk = kv_head(s, h);
  std::vector<double> scores(static_cast<std::size_t>(s.N));
  for (int r = 0; r < rows; ++r) {
    const int i = row0 + r;
    const double* qi = q + idx(s, s.H, b, h, i);
    // Scores in SCALED units here (direct form); lse = m + log(l) is then already
    // m_raw*scale + log(l), the same quantity the tiled form reports.
    double m = kNegInf;
    const int jmax = s.causal ? i : s.N - 1;  // top-left causal: key j visible iff j <= i
    for (int j = 0; j < s.N; ++j) {
      const double sc = (j <= jmax) ? s.scale * dot(qi, k + idx(s, s.Hkv, b, hk, j), s.D) : kNegInf;
      scores[static_cast<std::size_t>(j)] = sc;
      m = std::max(m, sc);
    }
    double* oi = o + static_cast<std::size_t>(r) * s.D;
    std::fill(oi, oi + s.D, 0.0);
    if (m == kNegInf) {  // fully masked row: o = 0, lse = +inf (fa/softmax_math.cuh guards)
      if (lse) lse[r] = kPosInf;
      continue;
    }
    double l = 0.0;
    for (int j = 0; j <= jmax; ++j) {
      const double p = std::exp(scores[static_cast<std::size_t>(j)] - m);
      l += p;
      const double* vj = v + idx(s, s.Hkv, b, hk, j);
      for (int d = 0; d < s.D; ++d)
        oi[d] += p * vj[d];
    }
    for (int d = 0; d < s.D; ++d)
      oi[d] /= l;
    if (lse) lse[r] = m + std::log(l);
  }
}

void attention(const double* q, const double* k, const double* v, const Shape& s, double* o,
               double* lse) {
  for (int b = 0; b < s.B; ++b) {
    for (int h = 0; h < s.H; ++h) {
      double* ob = o + idx(s, s.H, b, h, 0);
      double* lb = lse ? lse + (static_cast<std::size_t>(b) * s.H + h) * s.N : nullptr;
      attention_rows(q, k, v, s, b, h, 0, s.N, ob, lb);
    }
  }
}

void attention_tiled(const double* q, const double* k, const double* v, const Shape& s, int block_m,
                     int block_n, double* o, double* lse) {
  const double c = s.scale * kLog2e;  // folded constant, as in the kernels
  std::vector<double> m(static_cast<std::size_t>(block_m));
  std::vector<double> l(static_cast<std::size_t>(block_m));
  std::vector<double> acc(static_cast<std::size_t>(block_m) * s.D);
  std::vector<double> sc(static_cast<std::size_t>(block_n));
  for (int b = 0; b < s.B; ++b) {
    for (int h = 0; h < s.H; ++h) {
      const int hk = kv_head(s, h);
      for (int bm = 0; bm < s.N; bm += block_m) {
        const int rows = std::min(block_m, s.N - bm);
        std::fill(m.begin(), m.end(), kNegInf);
        std::fill(l.begin(), l.end(), 0.0);
        std::fill(acc.begin(), acc.end(), 0.0);
        for (int bn = 0; bn < s.N; bn += block_n) {
          // Regime 1 of the causal loop: the tile's first key is past the block's last query,
          // so nothing in it is visible. Skipped before any "load", as the kernel will.
          if (s.causal && bn > bm + rows - 1) break;
          const int cols = std::min(block_n, s.N - bn);
          for (int r = 0; r < rows; ++r) {
            const int i = bm + r;
            const double* qi = q + idx(s, s.H, b, h, i);
            double tile_max = kNegInf;
            for (int cidx = 0; cidx < cols; ++cidx) {
              const int j = bn + cidx;
              const double raw = (!s.causal || j <= i)
                                     ? dot(qi, k + idx(s, s.Hkv, b, hk, j), s.D)
                                     : kNegInf;  // RAW score units; scale folded into c
              sc[static_cast<std::size_t>(cidx)] = raw;
              tile_max = std::max(tile_max, raw);
            }
            const double m_new = std::max(m[static_cast<std::size_t>(r)], tile_max);
            if (m_new == kNegInf) continue;  // row saw no key yet: nothing to rescale
            // alpha guard: m_old == -inf -> 0 (acc and l are 0, and -inf - -inf would be NaN)
            const double m_old = m[static_cast<std::size_t>(r)];
            const double alpha = (m_old == kNegInf) ? 0.0 : std::exp2((m_old - m_new) * c);
            double rowsum = 0.0;
            double* accr = acc.data() + static_cast<std::size_t>(r) * s.D;
            for (int d = 0; d < s.D; ++d)
              accr[d] *= alpha;
            for (int cidx = 0; cidx < cols; ++cidx) {
              const double raw = sc[static_cast<std::size_t>(cidx)];
              const double p = (raw == kNegInf) ? 0.0 : std::exp2((raw - m_new) * c);
              rowsum += p;
              const double* vj = v + idx(s, s.Hkv, b, hk, bn + cidx);
              for (int d = 0; d < s.D; ++d)
                accr[d] += p * vj[d];
            }
            l[static_cast<std::size_t>(r)] = alpha * l[static_cast<std::size_t>(r)] + rowsum;
            m[static_cast<std::size_t>(r)] = m_new;
          }
        }
        // Deferred normalisation: one multiply by 1/l per row, with the l == 0 guard.
        for (int r = 0; r < rows; ++r) {
          const int i = bm + r;
          const double lr = l[static_cast<std::size_t>(r)];
          const double inv = (lr > 0.0) ? 1.0 / lr : 0.0;
          double* oi = o + idx(s, s.H, b, h, i);
          const double* accr = acc.data() + static_cast<std::size_t>(r) * s.D;
          for (int d = 0; d < s.D; ++d)
            oi[d] = accr[d] * inv;
          if (lse) {
            const double mr = m[static_cast<std::size_t>(r)];
            lse[(static_cast<std::size_t>(b) * s.H + h) * s.N + i] =
                (lr > 0.0 && mr != kNegInf) ? mr * s.scale + std::log(lr) : kPosInf;
          }
        }
      }
    }
  }
}

double max_abs_diff(const double* a, const double* b, std::size_t n) {
  double worst = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double d = std::fabs(a[i] - b[i]);
    if (d != d) return std::numeric_limits<double>::infinity();  // NaN counts as a failure
    worst = std::max(worst, d);
  }
  return worst;
}

}  // namespace ref
}  // namespace fa
