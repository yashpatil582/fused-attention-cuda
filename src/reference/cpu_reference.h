// fp64 CPU attention reference. Plain C++17, no CUDA: builds on the Mac and is the ctest
// oracle for small shapes and the in-process self-check inside fa_bench (torch-free).
//
// Two implementations of the same function on purpose:
//   attention()/attention_rows()  direct: materialise the row's scores, exact softmax.
//   attention_tiled()             the tiled online-softmax recurrence the fused kernels
//                                 implement (block_m x block_n tiles, running max/sum,
//                                 deferred normalisation, same guards). Its equivalence to
//                                 the direct form for ANY tile size is what makes the S3
//                                 kernel's tile-invariance tests meaningful.
// Layouts match CONTRACT §1: q/o [B, H, N, D], k/v [B, Hkv, N, D], lse [B, H, N], all
// contiguous, D fastest. Causal mask is top-left aligned (key j visible to query i iff
// j <= i). Head h reads kv head h / (H / Hkv) (GQA/MQA mapping).
#pragma once

#include <cstddef>

namespace fa {
namespace ref {

struct Shape {
  int B = 1;
  int H = 1;
  int Hkv = 1;
  int N = 1;
  int D = 1;
  double scale = 1.0;
  bool causal = false;
};

std::size_t qo_elems(const Shape& s);   // B*H*N*D
std::size_t kv_elems(const Shape& s);   // B*Hkv*N*D
std::size_t lse_elems(const Shape& s);  // B*H*N

// Direct fp64 attention for query rows [row0, row0+rows) of head (b, h).
// o: rows*D doubles (row-major), lse: rows doubles or nullptr.
void attention_rows(const double* q, const double* k, const double* v, const Shape& s, int b, int h,
                    int row0, int rows, double* o, double* lse);

// Direct fp64 attention for the whole tensor. o: qo_elems, lse: lse_elems or nullptr.
void attention(const double* q, const double* k, const double* v, const Shape& s, double* o,
               double* lse);

// Tiled online-softmax model (the algorithm of S3/S4/S5) in fp64. block_m/block_n are the
// Q-row / K-row tile sizes; partial tiles at the end are handled; fully masked K tiles are
// skipped before any load exactly as the causal kernel will (W2). Must equal attention()
// up to fp64 summation-order noise for every block_m, block_n >= 1.
void attention_tiled(const double* q, const double* k, const double* v, const Shape& s, int block_m,
                     int block_n, double* o, double* lse);

double max_abs_diff(const double* a, const double* b, std::size_t n);

}  // namespace ref
}  // namespace fa
