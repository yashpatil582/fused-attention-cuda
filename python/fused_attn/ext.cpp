// fused_attn._C -- PyTorch custom-op binding for fused-attention-cuda.
//
// WHY TORCH_LIBRARY instead of a plain pybind11 function: a pybind11 call is opaque to
// torch.compile (a graph break on every call).  Ops registered through the dispatcher show up as
// torch.ops.fused_attn.*, are traceable, get their fake (meta) kernels from
// python/fused_attn/ops.py, and are verified end to end by torch.library.opcheck in
// tests/test_op_gpu.py.
//
// Schemas FROZEN in CONTRACT.md section 4:
//   fused_attn::forward(Tensor q, Tensor k, Tensor v, bool is_causal, float scale) -> Tensor
//   fused_attn::forward_stage(Tensor q, Tensor k, Tensor v, bool is_causal, float scale,
//                             int stage) -> (Tensor, Tensor)
// plus three scalar-only helper ops (not part of the frozen contract, used by the tests):
//   set_debug_overrides(int block_m, int block_n) -> ()   TEST-ONLY tile-size override
//   impl_supports(int stage, str dtype, int D) -> bool     "fp32" | "fp16" | "bf16"
//   impl_name(int stage) -> str                            "naive" | ... | "tuned"
//
// This file is host C++ only (compiled by the host compiler, not nvcc): every kernel launch
// happens behind fa::forward() in src/dispatch.cu, so the binding never needs device code.

#include <Python.h>  // first, per the CPython docs (it sets feature macros)
// (this comment line ends the include block, so clang-format cannot sort Python.h below ATen)

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/library.h>

#include <cstdint>
#include <limits>
#include <string>
#include <tuple>

#include "fa/attention.cuh"

namespace {

constexpr int64_t kMinStage = 1;
constexpr int64_t kMaxStage = 5;
// Stages whose kernels write the logsumexp.  Naive/Tiled (1, 2) may pass lse == nullptr
// (attention.cuh); CONTRACT.md 4 says forward_stage then returns an empty lse tensor.
constexpr int64_t kFirstLseStage = 3;

fa::Dtype to_fa_dtype(at::ScalarType t) {
  if (t == at::kFloat) return fa::Dtype::F32;
  if (t == at::kHalf) return fa::Dtype::F16;
  TORCH_CHECK(t == at::kBFloat16, "fused_attn: unsupported dtype ", t,
              " (supported: float32, float16, bfloat16)");
  return fa::Dtype::BF16;
}

fa::Dtype dtype_from_name(const std::string& name) {
  if (name == "fp32") return fa::Dtype::F32;
  if (name == "fp16") return fa::Dtype::F16;
  TORCH_CHECK(name == "bf16", "fused_attn: unknown dtype name '", name,
              "' (expected fp32, fp16 or bf16)");
  return fa::Dtype::BF16;
}

// Compute capability of the CURRENT device as major*10+minor (75 for T4), the encoding
// fa::impl_supports() expects.  Callers set the device with a CUDAGuard first.
int current_sm() {
  const cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
  return prop->major * 10 + prop->minor;
}

void check_stage(int64_t stage) {
  TORCH_CHECK(stage >= kMinStage && stage <= kMaxStage, "fused_attn: stage must be in 1..5, got ",
              stage);
}

int checked_int(int64_t v, const char* what) {
  TORCH_CHECK(v >= 0 && v <= std::numeric_limits<int>::max(), "fused_attn: ", what, "=", v,
              " does not fit fa::Params (int)");
  return static_cast<int>(v);
}

// Validates the frozen input contract (CONTRACT.md 1 and 4) and builds fa::Params.  Every check
// is a TORCH_CHECK so a bad call raises a RuntimeError with a message, never a CUDA fault.
fa::Params make_params(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                       bool is_causal, double scale) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(),
              "fused_attn: q, k, v must be CUDA tensors");
  TORCH_CHECK(q.device() == k.device() && q.device() == v.device(),
              "fused_attn: q, k, v must live on the same device");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
              "fused_attn: q, k, v must be [B, H, N, D]; got ", q.dim(), "-D, ", k.dim(), "-D, ",
              v.dim(), "-D");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
              "fused_attn: q, k, v must be contiguous [B, H, N, D] (call .contiguous())");
  // S2+ kernels use 16-byte vector loads (float4 / uint2) and only check row strides, so the
  // tensor bases must be 16-byte aligned. Fresh allocations always are; a contiguous offset
  // view such as t[2:].view(...) is not, and would fault inside the kernel instead of here.
  for (const at::Tensor* t : {&q, &k, &v}) {
    TORCH_CHECK(reinterpret_cast<std::uintptr_t>(t->data_ptr()) % 16 == 0,
                "fused_attn: q/k/v must be 16-byte aligned (use a fresh contiguous tensor)");
  }
  TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type(),
              "fused_attn: q, k, v must share one dtype; got ", q.scalar_type(), ", ",
              k.scalar_type(), ", ", v.scalar_type());
  (void)to_fa_dtype(q.scalar_type());  // rejects anything but fp32/fp16/bf16

  const int64_t B = q.size(0), H = q.size(1), N = q.size(2), D = q.size(3);
  const int64_t Hkv = k.size(1);
  TORCH_CHECK(k.size(0) == B && v.size(0) == B, "fused_attn: batch mismatch between q, k, v");
  TORCH_CHECK(v.size(1) == Hkv, "fused_attn: k and v must have the same number of heads");
  TORCH_CHECK(k.size(2) == N && v.size(2) == N,
              "fused_attn: Nq == Nkv is required in-window (got Nq=", N, ", Nkv=", k.size(2), ")");
  TORCH_CHECK(k.size(3) == D && v.size(3) == D, "fused_attn: head-dim mismatch between q, k, v");
  TORCH_CHECK(D == 64 || D == 128, "fused_attn: D must be 64 or 128, got ", D);
  TORCH_CHECK(N % 32 == 0, "fused_attn: N must be a multiple of 32, got ", N,
              " (ragged sequence lengths are a stated in-window limitation)");
  TORCH_CHECK(Hkv >= 1 && H % Hkv == 0, "fused_attn: H=", H, " must be a multiple of Hkv=", Hkv);

  fa::Params p;
  p.B = checked_int(B, "B");
  p.H = checked_int(H, "H");
  p.Hkv = checked_int(Hkv, "Hkv");
  p.N = checked_int(N, "N");
  p.D = checked_int(D, "D");
  p.scale = static_cast<float>(scale);
  p.causal = is_causal;
  return p;
}

// Runs one explicit stage.  Returns (o, lse); lse is an empty fp32 tensor for stages < 3.
std::tuple<at::Tensor, at::Tensor> run_stage(const at::Tensor& q, const at::Tensor& k,
                                             const at::Tensor& v, bool is_causal, double scale,
                                             int64_t stage) {
  check_stage(stage);
  const fa::Params p = make_params(q, k, v, is_causal, scale);
  // Make q's device current: getCurrentCUDAStream / device properties / at::empty all follow it.
  const c10::cuda::CUDAGuard guard(q.device());
  const auto impl = static_cast<fa::Impl>(stage);
  const fa::Dtype dtype = to_fa_dtype(q.scalar_type());
  const int sm = current_sm();
  TORCH_CHECK(fa::impl_supports(impl, dtype, p.D, sm), "fused_attn: stage ", stage, " (",
              fa::impl_name(impl), ") is not implemented for ", fa::dtype_name(dtype), " D=", p.D,
              " on sm_", sm);

  at::Tensor o = at::empty_like(q);
  const bool has_lse = stage >= kFirstLseStage;
  at::Tensor lse = has_lse ? at::empty({p.B, p.H, p.N}, q.options().dtype(at::kFloat))
                           : at::empty({0}, q.options().dtype(at::kFloat));

  // Scratch for the materializing stages (fp32 S and P for naive/tiled; 0 for fused+).  It goes
  // out of scope right after the asynchronous launch, which is safe: the caching allocator frees
  // stream-ordered, so the block cannot be reused before this kernel finishes on `stream`.
  const std::size_t ws_bytes = fa::workspace_bytes(impl, dtype, p);
  at::Tensor workspace;
  void* ws_ptr = nullptr;
  if (ws_bytes > 0) {
    workspace = at::empty({static_cast<int64_t>(ws_bytes)}, q.options().dtype(at::kByte));
    ws_ptr = workspace.data_ptr();
  }

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(q.device().index()).stream();
  const cudaError_t err =
      fa::forward(impl, dtype, q.data_ptr(), k.data_ptr(), v.data_ptr(), o.data_ptr(),
                  has_lse ? lse.data_ptr<float>() : nullptr, ws_ptr, p, stream);
  // fa::forward never aborts; it reports launch/configuration errors (cudaErrorNotSupported,
  // cudaErrorInvalidValue, ...).  Asynchronous faults surface at the next synchronization, as
  // for any CUDA op.
  TORCH_CHECK(err == cudaSuccess, "fused_attn: stage ", stage, " (", fa::impl_name(impl),
              ") failed: ", cudaGetErrorString(err));
  return std::make_tuple(std::move(o), std::move(lse));
}

// forward: dispatch to the BEST implemented stage.  "Best" = highest stage number that
// impl_supports() accepts for (dtype, D, sm) -- the ladder is monotone by construction (each stage
// keeps the previous one's correctness and is only merged when its bench row improves).
at::Tensor forward_cuda(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                        bool is_causal, double scale) {
  const fa::Params p = make_params(q, k, v, is_causal, scale);  // clear errors before dispatch
  const c10::cuda::CUDAGuard guard(q.device());
  const fa::Dtype dtype = to_fa_dtype(q.scalar_type());
  const int sm = current_sm();
  for (int64_t stage = kMaxStage; stage >= kMinStage; --stage) {
    if (fa::impl_supports(static_cast<fa::Impl>(stage), dtype, p.D, sm)) {
      return std::get<0>(run_stage(q, k, v, is_causal, scale, stage));
    }
  }
  TORCH_CHECK(false, "fused_attn: no stage is implemented for ", fa::dtype_name(dtype), " D=", p.D,
              " on sm_", sm);
  return at::Tensor();  // unreachable; keeps -Wreturn-type quiet
}

std::tuple<at::Tensor, at::Tensor> forward_stage_cuda(const at::Tensor& q, const at::Tensor& k,
                                                      const at::Tensor& v, bool is_causal,
                                                      double scale, int64_t stage) {
  return run_stage(q, k, v, is_causal, scale, stage);
}

// TEST-ONLY.  Forces kernel tile sizes (0 = kernel default) so tests/test_kernels_gpu.py can
// prove the result does not depend on how the KV axis is tiled (the online-softmax rescale is
// exercised only when a row spans several tiles).  Process-global; tests reset it to (0, 0).
void set_debug_overrides_op(int64_t block_m, int64_t block_n) {
  fa::DebugOverrides o;
  o.block_m = checked_int(block_m, "block_m");
  o.block_n = checked_int(block_n, "block_n");
  fa::set_debug_overrides(o);
}

bool impl_supports_op(int64_t stage, std::string dtype, int64_t D) {
  check_stage(stage);
  if (c10::cuda::device_count() == 0) return false;
  const fa::Dtype dt = dtype_from_name(dtype);
  return fa::impl_supports(static_cast<fa::Impl>(stage), dt, checked_int(D, "D"), current_sm());
}

std::string impl_name_op(int64_t stage) {
  check_stage(stage);
  return std::string(fa::impl_name(static_cast<fa::Impl>(stage)));
}

}  // namespace

TORCH_LIBRARY(fused_attn, m) {
  m.def("forward(Tensor q, Tensor k, Tensor v, bool is_causal, float scale) -> Tensor");
  m.def(
      "forward_stage(Tensor q, Tensor k, Tensor v, bool is_causal, float scale, int stage)"
      " -> (Tensor, Tensor)");
  // Scalar-only ops carry no dispatch key (no Tensor argument), so their kernels are attached
  // here as catch-alls rather than through TORCH_LIBRARY_IMPL.
  m.def("set_debug_overrides(int block_m, int block_n) -> ()", &set_debug_overrides_op);
  m.def("impl_supports(int stage, str dtype, int D) -> bool", &impl_supports_op);
  m.def("impl_name(int stage) -> str", &impl_name_op);
}

TORCH_LIBRARY_IMPL(fused_attn, CUDA, m) {
  m.impl("forward", &forward_cuda);
  m.impl("forward_stage", &forward_stage_cuda);
}

// Minimal module object so `import fused_attn._C` succeeds.  Importing the .so is what runs the
// TORCH_LIBRARY static initializers above; the module itself exports nothing (the official
// cpp_custom_ops tutorial pattern -- no pybind11 needed, no Python-version-specific ABI).
extern "C" PyObject* PyInit__C(void) {
  static struct PyModuleDef module_def = {
      PyModuleDef_HEAD_INIT,
      "_C",
      "fused-attention-cuda extension. The ops live in torch.ops.fused_attn.*",
      -1,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
  };
  return PyModule_Create(&module_def);
}
