// ctest for the fp64 CPU reference (runs on the Mac, no CUDA).
//   1. A 2x2 case whose expected values are closed-form (hand-checkable on a whiteboard).
//   2. Tiled online softmax == direct softmax for several tile shapes, causal and not,
//      partial tiles, GQA head mapping. This is the property the S3 kernel's
//      FA_FORCE_BC tile-invariance tests rely on.
//   3. attention_rows() equals the matching slice of attention().
//   4. Scale folding: (2q, scale/2) == (q, scale).
//   5. lse is a real logsumexp: sum_j exp(s_ij - lse_i) == 1.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "cpu_reference.h"

namespace {

int g_failures = 0;

#define CHECK(cond, ...)                               \
  do {                                                 \
    if (!(cond)) {                                     \
      ++g_failures;                                    \
      std::printf("FAIL %s:%d: ", __FILE__, __LINE__); \
      std::printf(__VA_ARGS__);                        \
      std::printf("\n");                               \
    }                                                  \
  } while (0)

// Deterministic fill in [-1, 1): a 64-bit LCG, so every run of the test sees the same data.
void fill(std::vector<double>& x, std::uint64_t seed) {
  std::uint64_t st = seed * 6364136223846793005ULL + 1442695040888963407ULL;
  for (double& e : x) {
    st = st * 6364136223846793005ULL + 1442695040888963407ULL;
    e = static_cast<double>(st >> 11) * (1.0 / 9007199254740992.0) * 2.0 - 1.0;
  }
}

void test_hand_checkable() {
  // q = k = I2, v = [[1,2],[3,4]], scale = 1 -> S = I2.
  fa::ref::Shape s;
  s.B = 1, s.H = 1, s.Hkv = 1, s.N = 2, s.D = 2, s.scale = 1.0;
  const std::vector<double> q = {1, 0, 0, 1};
  const std::vector<double> v = {1, 2, 3, 4};
  std::vector<double> o(4), lse(2);
  const double e = std::exp(1.0);

  s.causal = false;
  fa::ref::attention(q.data(), q.data(), v.data(), s, o.data(), lse.data());
  // row0: softmax([1,0]) = [e, 1]/(1+e);  row1: softmax([0,1]) = [1, e]/(1+e)
  const double exp_o[4] = {(e * 1 + 3) / (1 + e), (e * 2 + 4) / (1 + e), (1 + 3 * e) / (1 + e),
                           (2 + 4 * e) / (1 + e)};
  for (int i = 0; i < 4; ++i)
    CHECK(std::fabs(o[i] - exp_o[i]) < 1e-14, "non-causal o[%d]=%.17g expected %.17g", i, o[i],
          exp_o[i]);
  for (int i = 0; i < 2; ++i)
    CHECK(std::fabs(lse[i] - std::log(1 + e)) < 1e-14, "non-causal lse[%d]=%.17g", i, lse[i]);

  s.causal = true;
  fa::ref::attention(q.data(), q.data(), v.data(), s, o.data(), lse.data());
  // row0 sees key 0 only -> o0 = v0, lse0 = s00 = 1; row1 unchanged.
  CHECK(o[0] == 1.0 && o[1] == 2.0, "causal row0 = %.17g %.17g", o[0], o[1]);
  CHECK(std::fabs(lse[0] - 1.0) < 1e-15, "causal lse0=%.17g", lse[0]);
  CHECK(std::fabs(o[2] - exp_o[2]) < 1e-14 && std::fabs(o[3] - exp_o[3]) < 1e-14,
        "causal row1 = %.17g %.17g", o[2], o[3]);
  CHECK(std::fabs(lse[1] - std::log(1 + e)) < 1e-14, "causal lse1=%.17g", lse[1]);
}

void test_tiled_equals_direct(int N, int D, int H, int Hkv) {
  fa::ref::Shape s;
  s.B = 2, s.H = H, s.Hkv = Hkv, s.N = N, s.D = D, s.scale = 1.0 / std::sqrt(double(D));
  std::vector<double> q(fa::ref::qo_elems(s)), k(fa::ref::kv_elems(s)), v(fa::ref::kv_elems(s));
  fill(q, 1), fill(k, 2), fill(v, 3);
  std::vector<double> o_ref(q.size()), lse_ref(fa::ref::lse_elems(s));
  std::vector<double> o_t(q.size()), lse_t(lse_ref.size());
  const int tiles[][2] = {{N, N}, {32, 16}, {16, 8}, {8, 64}, {64, 8}, {1, 1}, {N, 7}};
  for (int causal = 0; causal < 2; ++causal) {
    s.causal = causal != 0;
    fa::ref::attention(q.data(), k.data(), v.data(), s, o_ref.data(), lse_ref.data());
    for (const auto& t : tiles) {
      fa::ref::attention_tiled(q.data(), k.data(), v.data(), s, t[0], t[1], o_t.data(),
                               lse_t.data());
      const double eo = fa::ref::max_abs_diff(o_ref.data(), o_t.data(), o_t.size());
      const double el = fa::ref::max_abs_diff(lse_ref.data(), lse_t.data(), lse_t.size());
      CHECK(eo < 1e-12, "N=%d D=%d H=%d Hkv=%d causal=%d tile=%dx%d: max|o-o_ref|=%.3g", N, D, H,
            Hkv, causal, t[0], t[1], eo);
      CHECK(el < 1e-11, "N=%d D=%d causal=%d tile=%dx%d: max|lse-lse_ref|=%.3g", N, D, causal, t[0],
            t[1], el);
    }
  }
}

void test_rows_slice_and_scale_folding() {
  fa::ref::Shape s;
  s.B = 2, s.H = 4, s.Hkv = 2, s.N = 40, s.D = 16, s.scale = 0.25, s.causal = true;
  std::vector<double> q(fa::ref::qo_elems(s)), k(fa::ref::kv_elems(s)), v(fa::ref::kv_elems(s));
  fill(q, 7), fill(k, 8), fill(v, 9);
  std::vector<double> o(q.size()), lse(fa::ref::lse_elems(s));
  fa::ref::attention(q.data(), k.data(), v.data(), s, o.data(), lse.data());

  const int b = 1, h = 3, row0 = 5, rows = 7;
  std::vector<double> o_rows(static_cast<std::size_t>(rows) * s.D), lse_rows(rows);
  fa::ref::attention_rows(q.data(), k.data(), v.data(), s, b, h, row0, rows, o_rows.data(),
                          lse_rows.data());
  const std::size_t off = ((static_cast<std::size_t>(b) * s.H + h) * s.N + row0);
  CHECK(fa::ref::max_abs_diff(o_rows.data(), o.data() + off * s.D, o_rows.size()) == 0.0,
        "attention_rows differs from the attention() slice");
  CHECK(fa::ref::max_abs_diff(lse_rows.data(), lse.data() + off, lse_rows.size()) == 0.0,
        "attention_rows lse differs from the attention() slice");

  // Scale folding: doubling q and halving scale is the same function.
  std::vector<double> q2(q);
  for (double& e : q2)
    e *= 2.0;
  fa::ref::Shape s2 = s;
  s2.scale = s.scale / 2.0;
  std::vector<double> o2(q.size()), lse2(lse.size());
  fa::ref::attention(q2.data(), k.data(), v.data(), s2, o2.data(), lse2.data());
  CHECK(fa::ref::max_abs_diff(o.data(), o2.data(), o.size()) < 1e-13, "scale folding (o)");
  CHECK(fa::ref::max_abs_diff(lse.data(), lse2.data(), lse.size()) < 1e-12, "scale folding (lse)");

  // lse is a real logsumexp: sum_j exp(scale*q_i.k_j - lse_i) == 1 over visible keys.
  for (int i = 0; i < s.N; ++i) {
    const std::size_t qi = ((static_cast<std::size_t>(b) * s.H + h) * s.N + i);
    const int hk = h / (s.H / s.Hkv);
    double sum = 0.0;
    for (int j = 0; j <= i; ++j) {
      const std::size_t kj = ((static_cast<std::size_t>(b) * s.Hkv + hk) * s.N + j);
      double dot = 0.0;
      for (int d = 0; d < s.D; ++d)
        dot += q[qi * s.D + d] * k[kj * s.D + d];
      sum += std::exp(s.scale * dot - lse[qi]);
    }
    CHECK(std::fabs(sum - 1.0) < 1e-12, "row %d: sum exp(s - lse) = %.17g", i, sum);
  }
}

}  // namespace

int main() {
  test_hand_checkable();
  test_tiled_equals_direct(64, 64, 4, 2);   // canonical tile sizes divide N
  test_tiled_equals_direct(96, 32, 2, 2);   // partial tiles: 96 = 64 + 32, 96 % 7 != 0
  test_tiled_equals_direct(40, 128, 4, 1);  // MQA mapping, D = 128
  test_rows_slice_and_scale_folding();
  if (g_failures == 0) {
    std::printf("test_cpu_reference: PASS\n");
    return 0;
  }
  std::printf("test_cpu_reference: %d FAILURE(S)\n", g_failures);
  return 1;
}
