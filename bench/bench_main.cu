// bench/bench_main.cu — torch-free benchmark + smoke harness for fa::forward (fa_bench).
//
// Protocol (CONTRACT §3, judge_plan §4.4; bench/bench.py implements the same one in torch):
//   * 4 rotating input sets, so no iteration re-reads the operands of the previous one;
//   * a 256 MB int32 buffer is memset before EVERY timed iteration to evict L2 (Triton
//     do_bench convention); it is enqueued before the start event, so it is never timed;
//   * one cudaEvent pair per iteration, ONE cudaDeviceSynchronize after the loop;
//   * median / p10 / p90 over the timed iterations (linear interpolation, numpy default);
//   * the output is checksummed into a volatile sink, so no work can be elided;
//   * a row is written only if this process passed the fp64 self-check (--check-rows query
//     rows split between the first and the last Q tile of heads (0,0) and (B-1,H-1), vs
//     src/reference/cpu_reference.cpp) and every output is finite;
//   * iteration count: >= --iters (100) or >= 500 ms of timed work, sized from three launches
//     AFTER the warmup (never from the first launch, which carries lazy module loading);
//     notes carry `iters=<n>;warmup=<n>` so the sanity test can enforce rule 1;
//   * SM clock sampled through NVML (dlopen) at four points of the timed loop: clock_mhz =
//     median, `clock=<min>-<max>` and `noisy_clock` (>10% spread) in notes; `--clock <mhz>`
//     (the value nvidia-smi -lgc was given) is verified against the samples;
//   * FLOPs = 4*B*H*N^2*D (full count for causal until block skipping: notes=flops=full);
//   * bytes = modelled minimum: Q+K+V+O (+LSE for the fused stages) (+ S,P write+read for
//     naive/tiled); inputs are uniform in [-1,1) (bench.py: randn), stated in the protocol.
// Provenance: commit = --commit, else $FA_COMMIT, else `git rev-parse --short HEAD` with
// `-dirty` when a tracked source file differs (bench/results and profiles excluded); ncu =
// `ncu --version`; torch is blank by design (torch-free binary, BENCHMARK_PROTOCOL rule 8).
// Exit codes: 0 ok; 1 CUDA error / check failure; 2 usage; 3 stage not implemented in this
// build (S0: every stage); 4 inputs do not fit in device memory.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <sys/stat.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "cpu_reference.h"
#include "fa/attention.cuh"
#include "fa/cuda_check.cuh"

namespace fa {
namespace smoke {  // defined in src/smoke.cu (no header on purpose: S0-only kernel)
cudaError_t forward(const float* v, float* o, float scale, std::size_t n, cudaStream_t stream);
}  // namespace smoke
}  // namespace fa

namespace {

constexpr std::size_t kFlushBytes = 256ull << 20;  // 256 MB, larger than any L2 in scope
constexpr int kMaxSets = 4;
constexpr double kMinTimedMs = 500.0;  // protocol: >= 100 iters OR >= 500 ms timed work
constexpr int kMaxIters = 20000;       // cap for the 500 ms floor on very fast kernels
constexpr double kNoiseRatio = 1.15;   // p90/p10 >= this -> notes=noisy (= fa_common.NOISE_RATIO)
constexpr double kClockSpread =
    0.10;  // (max-min)/median of clock samples above this -> noisy_clock
constexpr double kLockTolerance = 0.02;  // |median - locked| <= 2% of locked -> lock verified
enum ExitCode { kOk = 0, kError = 1, kUsage = 2, kNotImplemented = 3, kNoMemory = 4 };

#define FA_STR2(x) #x
#define FA_STR(x) FA_STR2(x)
#if defined(__CUDACC_VER_MAJOR__)
const char* const kNvcc =
    FA_STR(__CUDACC_VER_MAJOR__) "." FA_STR(__CUDACC_VER_MINOR__) "." FA_STR(__CUDACC_VER_BUILD__);
#else
const char* const kNvcc = "unknown";
#endif
#if defined(FA_CUDA_ARCHS)
const char* const kArchs = FA_CUDA_ARCHS;
#else
const char* const kArchs = "unknown";
#endif
#if defined(FA_BUILD_TYPE)
const char* const kBuildType = FA_BUILD_TYPE;
#else
const char* const kBuildType = "unknown";
#endif

const char* const kCsvHeader =
    "commit,gpu,driver,nvcc,torch,ncu,clock_mhz,impl,stage,dtype,B,H,N,D,causal,"
    "ms_median,ms_p10,ms_p90,tflops,gbps,notes";

struct Options {
  int stage = 0;
  std::string cfg = "C64";
  std::string dtype = "fp32";
  int causal = 0;
  int iters = 100;
  int warmup = 10;
  bool iters_given = false;
  bool warmup_given = false;
  std::string out;
  std::string commit;
  std::string clock = "unlocked";
  bool smoke = false;
  bool list = false;
  int check_rows = 64;
  int hkv = 0;
  float scale = 0.0f;
};

void usage(FILE* f) {
  std::fprintf(
      f,
      "fa_bench --stage N [--cfg B2_H8_N2048_D64|C64|C128] [--dtype fp32|fp16|bf16]\n"
      "         [--causal 0|1] [--iters 100] [--warmup 10] [--out path.csv] [--commit sha]\n"
      "         [--clock <mhz>|unlocked] [--check-rows 64] [--hkv H] [--scale s]\n"
      "  --clock <mhz>: the value nvidia-smi -lgc was given; verified against NVML samples\n"
      "fa_bench --smoke [--stage N] [--out path.csv]   # fa_smoke round trip (+ one stage run)\n"
      "fa_bench --list                                  # impl x dtype x D support on this GPU\n");
}

bool parse_args(int argc, char** argv, Options* o) {
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--smoke") {
      o->smoke = true;
    } else if (a == "--list") {
      o->list = true;
    } else if (a == "--help" || a == "-h") {
      usage(stdout);
      std::exit(kOk);
    } else {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", a.c_str());
        return false;
      }
      const std::string v = argv[++i];
      if (a == "--stage") {
        o->stage = std::atoi(v.c_str());
      } else if (a == "--cfg") {
        o->cfg = v;
      } else if (a == "--dtype") {
        o->dtype = v;
      } else if (a == "--causal") {
        o->causal = std::atoi(v.c_str());
      } else if (a == "--iters") {
        o->iters = std::atoi(v.c_str()), o->iters_given = true;
      } else if (a == "--warmup") {
        o->warmup = std::atoi(v.c_str()), o->warmup_given = true;
      } else if (a == "--out") {
        o->out = v;
      } else if (a == "--commit") {
        o->commit = v;
      } else if (a == "--clock") {
        o->clock = v;
      } else if (a == "--check-rows") {
        o->check_rows = std::atoi(v.c_str());
      } else if (a == "--hkv") {
        o->hkv = std::atoi(v.c_str());
      } else if (a == "--scale") {
        o->scale = static_cast<float>(std::atof(v.c_str()));
      } else {
        std::fprintf(stderr, "unknown option %s\n", a.c_str());
        return false;
      }
    }
  }
  if (o->iters < 1 || o->warmup < 0) {
    std::fprintf(stderr, "--iters must be >= 1 and --warmup >= 0\n");
    return false;
  }
  return true;
}

// ---- device-side input fill: counter-based hash, no curand dependency ------------------
__device__ __forceinline__ uint32_t hash32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}
template <typename T>
__device__ __forceinline__ T from_float(float f);
template <>
__device__ __forceinline__ float from_float<float>(float f) {
  return f;
}
template <>
__device__ __forceinline__ __half from_float<__half>(float f) {
  return __float2half(f);
}
template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float f) {
  return __float2bfloat16(f);
}

// Uniform in [-1, 1): the softmax then sees O(1) scores at scale 1/sqrt(D), a regime where
// fp16 inputs neither overflow nor flush to zero (fp16 range tests live in tests/).
template <typename T>
__global__ void fill_uniform(T* p, std::size_t n, uint32_t seed) {
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += stride) {
    const uint32_t r =
        hash32(static_cast<uint32_t>(i) + hash32(static_cast<uint32_t>(i >> 32) ^ seed));
    p[i] = from_float<T>(static_cast<float>(r >> 8) * (1.0f / 16777216.0f) * 2.0f - 1.0f);
  }
}

void fill(void* p, std::size_t n, fa::Dtype dt, uint32_t seed, cudaStream_t s) {
  switch (dt) {
    case fa::Dtype::F32:
      fill_uniform<float><<<2048, 256, 0, s>>>(static_cast<float*>(p), n, seed);
      break;
    case fa::Dtype::F16:
      fill_uniform<__half><<<2048, 256, 0, s>>>(static_cast<__half*>(p), n, seed);
      break;
    case fa::Dtype::BF16:
      fill_uniform<__nv_bfloat16><<<2048, 256, 0, s>>>(static_cast<__nv_bfloat16*>(p), n, seed);
      break;
  }
  CUDA_CHECK_LAST();
}

double to_double(const void* p, std::size_t i, fa::Dtype dt) {
  switch (dt) {
    case fa::Dtype::F32:
      return static_cast<const float*>(p)[i];
    case fa::Dtype::F16:
      return __half2float(static_cast<const __half*>(p)[i]);
    case fa::Dtype::BF16:
      return __bfloat162float(static_cast<const __nv_bfloat16*>(p)[i]);
  }
  return 0.0;
}

bool parse_dtype(const std::string& s, fa::Dtype* dt) {
  if (s == "fp32") return *dt = fa::Dtype::F32, true;
  if (s == "fp16") return *dt = fa::Dtype::F16, true;
  if (s == "bf16") return *dt = fa::Dtype::BF16, true;
  return false;
}

// C64 / C128 are the canonical profiling configs (CONTRACT §5).
bool parse_cfg(const std::string& s, fa::Params* p) {
  if (s == "C64" || s == "c64") return p->B = 2, p->H = 8, p->N = 2048, p->D = 64, true;
  if (s == "C128" || s == "c128") return p->B = 2, p->H = 8, p->N = 2048, p->D = 128, true;
  return std::sscanf(s.c_str(), "B%d_H%d_N%d_D%d", &p->B, &p->H, &p->N, &p->D) == 4 && p->B > 0 &&
         p->H > 0 && p->N > 0 && p->D > 0;
}

// ---- device / build info -------------------------------------------------------------
struct DeviceInfo {
  std::string name;
  int sm = 0;
  std::string driver;
  std::string runtime;
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
};

// NVML through dlopen: keeps fa_bench free of a link-time dependency on libnvidia-ml (absent
// on CI's compile-only runners). Gives nvidia-smi's driver_version and, during the timed loop,
// SM clock samples (nvmlDeviceGetClockInfo, microseconds per call: cheap enough to sample
// in-process, unlike an nvidia-smi subprocess). Device index 0 = nvidia-smi's first GPU.
struct Nvml {
  void* h = nullptr;
  void* dev = nullptr;
  int (*get_clock)(void*, int, unsigned*) = nullptr;
  int (*shutdown)() = nullptr;
  std::string driver;

  Nvml() {
    h = dlopen("libnvidia-ml.so.1", RTLD_NOW);
    if (h == nullptr) return;
    auto init = reinterpret_cast<int (*)()>(dlsym(h, "nvmlInit_v2"));
    auto ver = reinterpret_cast<int (*)(char*, unsigned)>(dlsym(h, "nvmlSystemGetDriverVersion"));
    auto by_index =
        reinterpret_cast<int (*)(unsigned, void**)>(dlsym(h, "nvmlDeviceGetHandleByIndex_v2"));
    get_clock =
        reinterpret_cast<int (*)(void*, int, unsigned*)>(dlsym(h, "nvmlDeviceGetClockInfo"));
    shutdown = reinterpret_cast<int (*)()>(dlsym(h, "nvmlShutdown"));
    if (init == nullptr || init() != 0) {
      shutdown = nullptr;
      return;
    }
    char buf[96] = {0};
    if (ver != nullptr && ver(buf, sizeof(buf)) == 0) driver = buf;
    if (by_index == nullptr || by_index(0, &dev) != 0) dev = nullptr;
  }
  // Current SM clock in MHz, 0 when NVML is unavailable.
  int sm_clock_mhz() const {
    unsigned c = 0;
    if (dev != nullptr && get_clock != nullptr && get_clock(dev, /*NVML_CLOCK_SM=*/1, &c) == 0)
      return static_cast<int>(c);
    return 0;
  }
  ~Nvml() {
    if (shutdown != nullptr) shutdown();
    if (h != nullptr) dlclose(h);
  }
};

const Nvml& nvml() {
  static const Nvml n;
  return n;
}

// stdout of a shell command, trailing newlines stripped; empty on failure.
std::string shell(const char* cmd) {
  std::string out;
  if (FILE* p = popen(cmd, "r")) {
    char buf[256];
    while (fgets(buf, sizeof(buf), p) != nullptr)
      out += buf;
    pclose(p);
  }
  while (!out.empty() && (out.back() == '\n' || out.back() == '\r'))
    out.pop_back();
  return out;
}

// Same rule as scripts/fa_common.py::git_commit: short SHA, `-dirty` when a TRACKED source
// file differs from HEAD; bench/results and profiles are excluded because a run appends there.
std::string git_commit() {
  const std::string sha = shell("git rev-parse --short HEAD 2>/dev/null");
  if (sha.empty()) return "unknown";
  const std::string dirty = shell(
      "git status --porcelain --untracked-files=no -- . ':(exclude)bench/results' "
      "':(exclude)profiles' 2>/dev/null");
  return dirty.empty() ? sha : sha + "-dirty";
}

// `Version 2025.2.0.0` from `ncu --version`; empty when ncu is absent.
std::string ncu_version() {
  const std::string out = shell("ncu --version 2>/dev/null");
  const std::size_t i = out.find("Version ");
  if (i == std::string::npos) return "";
  const std::size_t b = i + 8, e = out.find_first_of(" \n\r\t", b);
  return out.substr(b, e == std::string::npos ? std::string::npos : e - b);
}

DeviceInfo query_device() {
  DeviceInfo d;
  int dev = 0, drv = 0, rt = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  d.name = prop.name;
  d.sm = prop.major * 10 + prop.minor;
  CUDA_CHECK(cudaDriverGetVersion(&drv));
  CUDA_CHECK(cudaRuntimeGetVersion(&rt));
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%d.%d", rt / 1000, (rt % 1000) / 10);
  d.runtime = buf;
  d.driver = nvml().driver;
  if (d.driver.empty()) {  // fall back to the driver's CUDA API version, labelled as such
    std::snprintf(buf, sizeof(buf), "cuda%d.%d", drv / 1000, (drv % 1000) / 10);
    d.driver = buf;
  }
  CUDA_CHECK(cudaMemGetInfo(&d.free_bytes, &d.total_bytes));
  return d;
}

// ---- statistics, checksum, CSV -------------------------------------------------------
struct Stats {
  double median = 0, p10 = 0, p90 = 0, min = 0, max = 0;
};

double quantile(const std::vector<float>& sorted, double q) {
  const double pos = q * static_cast<double>(sorted.size() - 1);
  const std::size_t lo = static_cast<std::size_t>(pos);
  const std::size_t hi = std::min(lo + 1, sorted.size() - 1);
  const double w = pos - static_cast<double>(lo);
  return sorted[lo] * (1.0 - w) + sorted[hi] * w;
}

Stats stats_of(std::vector<float> ms) {
  std::sort(ms.begin(), ms.end());
  return {quantile(ms, 0.5), quantile(ms, 0.1), quantile(ms, 0.9), ms.front(), ms.back()};
}

volatile uint64_t g_sink = 0;  // the checksum lands here so the output is observably used

uint64_t fnv1a(const void* data, std::size_t n) {
  const auto* p = static_cast<const uint8_t*>(data);
  uint64_t h = 1469598103934665603ull;
  for (std::size_t i = 0; i < n; ++i)
    h = (h ^ p[i]) * 1099511628211ull;
  return h;
}

std::string fmt(const char* f, double v) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), f, v);
  return buf;
}

std::string csv_field(const std::string& s) {
  if (s.find_first_of(",\"\n") == std::string::npos) return s;
  std::string q = "\"";
  for (char c : s)
    q += (c == '"') ? "\"\"" : std::string(1, c);
  return q + "\"";
}

std::string join(const std::vector<std::string>& v, const char* sep) {
  std::string s;
  for (std::size_t i = 0; i < v.size(); ++i)
    s += (i ? sep : "") + v[i];
  return s;
}

struct Row {
  std::string commit, gpu, driver, nvcc, ncu, clock_mhz, impl, dtype;
  int stage = 0, B = 0, H = 0, N = 0, D = 0, causal = 0;
  std::string ms_median, ms_p10, ms_p90, tflops, gbps, notes;
};

std::string to_csv(const Row& r) {
  const std::vector<std::string> f = {csv_field(r.commit),
                                      csv_field(r.gpu),
                                      csv_field(r.driver),
                                      r.nvcc,
                                      "" /* torch: torch-free binary (protocol rule 8) */,
                                      r.ncu,
                                      r.clock_mhz,
                                      r.impl,
                                      std::to_string(r.stage),
                                      r.dtype,
                                      std::to_string(r.B),
                                      std::to_string(r.H),
                                      std::to_string(r.N),
                                      std::to_string(r.D),
                                      std::to_string(r.causal),
                                      r.ms_median,
                                      r.ms_p10,
                                      r.ms_p90,
                                      r.tflops,
                                      r.gbps,
                                      csv_field(r.notes)};
  return join(f, ",");
}

void mkdir_parents(const std::string& path) {
  for (std::size_t i = 1; i < path.size(); ++i) {
    if (path[i] == '/') mkdir(path.substr(0, i).c_str(), 0755);
  }
}

// Prints the row (with header) to stdout and appends it to --out, writing the header when the
// file is new or empty.
bool emit(const Row& row, const Options& opt) {
  const std::string line = to_csv(row);
  std::printf("%s\n%s\n", kCsvHeader, line.c_str());
  if (opt.out.empty()) return true;
  mkdir_parents(opt.out);
  long size = 0;
  if (FILE* probe = std::fopen(opt.out.c_str(), "r")) {
    std::fseek(probe, 0, SEEK_END), size = std::ftell(probe), std::fclose(probe);
  }
  FILE* f = std::fopen(opt.out.c_str(), "a");
  if (f == nullptr) {
    std::fprintf(stderr, "cannot open %s for append\n", opt.out.c_str());
    return false;
  }
  if (size <= 0) std::fprintf(f, "%s\n", kCsvHeader);
  std::fprintf(f, "%s\n", line.c_str());
  std::fclose(f);
  std::printf("appended to %s\n", opt.out.c_str());
  return true;
}

Row base_row(const Options& opt, const DeviceInfo& dev) {
  Row r;
  const char* env_commit = std::getenv("FA_COMMIT");
  r.commit = !opt.commit.empty()                              ? opt.commit
             : (env_commit != nullptr && *env_commit != '\0') ? env_commit
                                                              : git_commit();
  r.gpu = dev.name, r.driver = dev.driver, r.nvcc = kNvcc, r.ncu = ncu_version();
  if (opt.clock != "unlocked") r.clock_mhz = opt.clock;
  return r;
}

// clock_mhz column + notes from the NVML samples of one timed loop (mirrors bench.py
// clock_notes): median, `clock=<min>-<max>` when they differ, `noisy_clock` above
// kClockSpread; `--clock <mhz>` is verified (median within kLockTolerance, spread within
// kClockSpread) and otherwise reported as clock=unlocked;clock_lock_failed=<mhz>.
void apply_clock_notes(const std::vector<int>& samples, const std::string& requested, Row* row,
                       std::vector<std::string>* notes) {
  const bool locked = requested != "unlocked";
  const int want = locked ? std::atoi(requested.c_str()) : 0;
  if (samples.empty()) {
    if (!locked) notes->push_back("clock=unlocked");
    return;  // clock_mhz stays at the requested value (or empty)
  }
  std::vector<int> s = samples;
  std::sort(s.begin(), s.end());
  const int med = s[s.size() / 2], lo = s.front(), hi = s.back();
  const double spread = med > 0 ? static_cast<double>(hi - lo) / med : 0.0;
  row->clock_mhz = std::to_string(med);
  if (!locked) {
    notes->push_back("clock=unlocked");
  } else if (want > 0 && std::abs(med - want) <= kLockTolerance * want && spread <= kClockSpread) {
    notes->push_back("clock_lock=" + std::to_string(want));
  } else if (want > 0) {
    notes->push_back("clock=unlocked");
    notes->push_back("clock_lock_failed=" + std::to_string(want));
  }
  if (lo != hi) notes->push_back("clock=" + std::to_string(lo) + "-" + std::to_string(hi));
  if (spread > kClockSpread) notes->push_back("noisy_clock");
}

// ---- the timing loop (shared by smoke and stage runs) ----------------------------------
// *iters is the minimum; with `autosize` it is raised so the timed work lasts >= kMinTimedMs,
// estimated from three launches AFTER the warmup (bench.py does the same; the first launch
// carries lazy module loading and would under-size the loop). SM clock samples are taken at
// the 1/4, 1/2, 3/4 host positions and after the last enqueue: the launch queue is bounded,
// so at each of those points the GPU is executing the loop, not idling after it.
template <typename Launch>
cudaError_t time_loop(Launch launch, int nsets, int warmup, int* iters, bool autosize, void* flush,
                      cudaStream_t stream, std::vector<float>* ms, std::vector<int>* clocks) {
  for (int w = 0; w < warmup; ++w) {
    CUDA_CHECK(cudaMemsetAsync(flush, 0, kFlushBytes, stream));
    FA_CUDA_TRY(launch(w % nsets));
  }
  if (autosize) {
    cudaEvent_t e0 = nullptr, e1 = nullptr;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventRecord(e0, stream));
    for (int i = 0; i < 3; ++i)
      FA_CUDA_TRY(launch(i % nsets));
    CUDA_CHECK(cudaEventRecord(e1, stream));
    FA_CUDA_TRY(cudaDeviceSynchronize());
    float est_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&est_ms, e0, e1));
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
    const double per_launch = std::max(static_cast<double>(est_ms) / 3.0, 1e-3);
    const double want = std::ceil(kMinTimedMs / per_launch);
    *iters = std::max(*iters, std::min(kMaxIters, static_cast<int>(want)));
  }
  const int n = *iters;
  std::vector<cudaEvent_t> ev(2 * static_cast<std::size_t>(n));
  for (cudaEvent_t& e : ev)
    CUDA_CHECK(cudaEventCreate(&e));
  clocks->clear();
  auto sample = [&]() {
    const int c = nvml().sm_clock_mhz();
    if (c > 0) clocks->push_back(c);
  };
  for (int i = 0; i < n; ++i) {
    if (n >= 4 && (i == n / 4 || i == n / 2 || i == 3 * n / 4)) sample();
    CUDA_CHECK(cudaMemsetAsync(flush, 0, kFlushBytes, stream));  // L2 eviction, untimed
    CUDA_CHECK(cudaEventRecord(ev[2 * i], stream));
    FA_CUDA_TRY(launch(i % nsets));
    CUDA_CHECK(cudaEventRecord(ev[2 * i + 1], stream));
  }
  sample();
  FA_CUDA_TRY(cudaDeviceSynchronize());  // the protocol's single synchronisation
  FA_CUDA_TRY(cudaGetLastError());
  ms->resize(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i)
    CUDA_CHECK(cudaEventElapsedTime(&(*ms)[i], ev[2 * i], ev[2 * i + 1]));
  for (cudaEvent_t e : ev)
    CUDA_CHECK(cudaEventDestroy(e));
  return cudaSuccess;
}

// Modelled MINIMUM bytes of one forward (the gbps model, CONTRACT §3 / protocol rule 6): the
// measured DRAM traffic is the profile's dram__bytes.sum.
double bytes_moved(fa::Impl impl, const fa::Params& p, std::size_t es) {
  const double q = static_cast<double>(p.B) * p.H * p.N * p.D * es;
  const double kv = 2.0 * p.B * p.Hkv * p.N * p.D * es;
  double bytes = q + kv + q;  // Q, K, V, O
  if (impl == fa::Impl::Naive || impl == fa::Impl::Tiled) {
    bytes += 4.0 * p.B * p.H * static_cast<double>(p.N) * p.N * sizeof(float);  // S w+r, P w+r
  } else {
    bytes += static_cast<double>(p.B) * p.H * p.N * sizeof(float);  // LSE, fused stages only
  }
  return bytes;
}

// fp64 self-check on a strided subset that spans the tensor: query rows [0, rows/2) and
// [N - rows/2, N) — the first and the last Q tile — of head (0,0) AND of head (B-1, H-1). The
// last rows see every K/V tile (a broken online-softmax rescale on later tiles, or a causal
// mask that is only right on the first tile, cannot pass), and the last head catches a wrong
// batch/head stride. Cost: <= 2 x rows x N fp64 dot products on the host, milliseconds.
// Sanity caps only (judge_plan §4.2): the ratio rule against torch lives in
// tests/test_kernels_gpu.py.
bool self_check(const fa::Params& p, fa::Dtype dt, const void* dq, const void* dk, const void* dv,
                const void* dO, const float* dlse, int rows, bool check_lse, int* rows_checked) {
  const std::size_t es = fa::dtype_size(dt);
  const std::size_t nd = static_cast<std::size_t>(p.N) * p.D;
  const int half = std::max(1, std::min(rows, p.N) / 2);
  std::vector<std::pair<int, int>> ranges;  // (row0, nrows)
  if (p.N <= rows) {
    ranges.emplace_back(0, p.N);
  } else {
    ranges.emplace_back(0, half);
    ranges.emplace_back(p.N - half, half);
  }
  std::vector<std::pair<int, int>> heads = {{0, 0}};
  if (p.B > 1 || p.H > 1) heads.emplace_back(p.B - 1, p.H - 1);
  const double cap_o = (dt == fa::Dtype::F32) ? 1e-4 : (dt == fa::Dtype::F16) ? 1e-2 : 5e-2;
  double err_o = 0.0, err_l = 0.0;
  *rows_checked = 0;
  fa::ref::Shape s1;  // one head at a time: the slices below ARE the whole tensor to it
  s1.B = 1, s1.H = 1, s1.Hkv = 1, s1.N = p.N, s1.D = p.D, s1.scale = p.scale, s1.causal = p.causal;
  std::vector<uint8_t> hk(nd * es), hv(nd * es);
  std::vector<double> k(nd), v(nd);
  for (const auto& bh : heads) {
    const int b = bh.first, h = bh.second, hkv = h / (p.H / p.Hkv);
    const std::size_t kv_off = (static_cast<std::size_t>(b) * p.Hkv + hkv) * nd * es;
    CUDA_CHECK(cudaMemcpy(hk.data(), static_cast<const uint8_t*>(dk) + kv_off, hk.size(),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hv.data(), static_cast<const uint8_t*>(dv) + kv_off, hv.size(),
                          cudaMemcpyDeviceToHost));
    for (std::size_t i = 0; i < nd; ++i)
      k[i] = to_double(hk.data(), i, dt), v[i] = to_double(hv.data(), i, dt);
    for (const auto& rg : ranges) {
      const int row0 = rg.first, nrows = rg.second;
      const std::size_t rd = static_cast<std::size_t>(nrows) * p.D;
      const std::size_t q_off = ((static_cast<std::size_t>(b) * p.H + h) * p.N + row0) * p.D * es;
      const std::size_t l_off = (static_cast<std::size_t>(b) * p.H + h) * p.N + row0;
      std::vector<uint8_t> hq(rd * es), ho(rd * es);
      std::vector<float> hl(static_cast<std::size_t>(nrows));
      CUDA_CHECK(cudaMemcpy(hq.data(), static_cast<const uint8_t*>(dq) + q_off, hq.size(),
                            cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(ho.data(), static_cast<const uint8_t*>(dO) + q_off, ho.size(),
                            cudaMemcpyDeviceToHost));
      if (check_lse)
        CUDA_CHECK(cudaMemcpy(hl.data(), dlse + l_off, hl.size() * 4, cudaMemcpyDeviceToHost));
      // attention_rows reads q rows [row0, row0+nrows) of the (single) head it is given, so
      // the q slice must start at row 0 of that head: hand it the rows shifted by -row0.
      std::vector<double> q(static_cast<std::size_t>(p.N) * p.D, 0.0), o(rd), o_ref(rd),
          lse(hl.size()), lse_ref(hl.size());
      for (std::size_t i = 0; i < rd; ++i) {
        q[static_cast<std::size_t>(row0) * p.D + i] = to_double(hq.data(), i, dt);
        o[i] = to_double(ho.data(), i, dt);
      }
      for (std::size_t i = 0; i < hl.size(); ++i)
        lse[i] = hl[i];
      fa::ref::attention_rows(q.data(), k.data(), v.data(), s1, 0, 0, row0, nrows, o_ref.data(),
                              lse_ref.data());
      err_o = std::max(err_o, fa::ref::max_abs_diff(o.data(), o_ref.data(), rd));
      if (check_lse)
        err_l = std::max(err_l, fa::ref::max_abs_diff(lse.data(), lse_ref.data(), hl.size()));
      *rows_checked += nrows;
    }
  }
  const bool ok = err_o <= cap_o && err_l <= 10.0 * cap_o;
  std::printf("%s rows=%d heads=%zu max_abs_err_o=%.3g (cap %.0e) max_abs_err_lse=%.3g%s\n",
              ok ? "CHECK_OK" : "CHECK_FAIL", *rows_checked, heads.size(), err_o, cap_o, err_l,
              check_lse ? "" : " (lse not checked for this stage)");
  return ok;
}

// ---- stage run ---------------------------------------------------------------------------
int run_stage(const Options& opt, const DeviceInfo& dev, cudaStream_t stream, bool smoke_mode) {
  fa::Params p{};
  fa::Dtype dtype = fa::Dtype::F32;
  if (!parse_cfg(opt.cfg, &p) || !parse_dtype(opt.dtype, &dtype)) {
    std::fprintf(stderr, "bad --cfg '%s' or --dtype '%s'\n", opt.cfg.c_str(), opt.dtype.c_str());
    return kUsage;
  }
  if (opt.stage < 1 || opt.stage > 5) {
    std::fprintf(stderr, "--stage must be 1..5 (0 is the smoke kernel: use --smoke)\n");
    return kUsage;
  }
  p.Hkv = opt.hkv > 0 ? opt.hkv : p.H;
  p.causal = opt.causal != 0;
  p.scale = opt.scale > 0.0f ? opt.scale : 1.0f / std::sqrt(static_cast<float>(p.D));
  const fa::Impl impl = static_cast<fa::Impl>(opt.stage);
  const std::size_t es = fa::dtype_size(dtype);
  const int iters_base = (smoke_mode && !opt.iters_given) ? 3 : opt.iters;
  const int warmup = (smoke_mode && !opt.warmup_given) ? 1 : opt.warmup;

  Row row = base_row(opt, dev);
  row.impl = fa::impl_name(impl), row.stage = opt.stage, row.dtype = fa::dtype_name(dtype);
  row.B = p.B, row.H = p.H, row.N = p.N, row.D = p.D, row.causal = p.causal ? 1 : 0;
  std::vector<std::string> notes;
  if (smoke_mode) notes.push_back("smoke");
  // Protocol rule 5: fused stages (Fused/Wmma/Tuned) skip the fully masked K/V tiles, so their
  // causal rows are accounted at half the FLOPs (FlashAttention convention); Naive/Tiled still
  // compute the whole N x N score matrix and keep the full count. The note names the rule.
  const bool causal_half = p.causal && impl != fa::Impl::Naive && impl != fa::Impl::Tiled;
  if (p.causal) notes.push_back(causal_half ? "flops=causal_half" : "flops=full");
  if (p.Hkv != p.H) notes.push_back("hkv=" + std::to_string(p.Hkv));

  // OOM by design: fp32 NxN intermediates above 25% of TOTAL device memory are not attempted
  // (protocol rule 6, cudaDeviceProp.totalGlobalMem; the same rule as bench.py, so both
  // harnesses classify a shape identically regardless of what else is resident).
  const std::size_t ws_bytes = fa::workspace_bytes(impl, dtype, p);
  if (ws_bytes > dev.total_bytes / 4) {
    notes.push_back("oom_by_design");
    if (opt.clock == "unlocked") notes.push_back("clock=unlocked");
    row.notes = join(notes, ";");
    std::printf("OOM_BY_DESIGN workspace=%.2f GB > 25%% of total %.2f GB (free %.2f GB)\n",
                ws_bytes / 1e9, dev.total_bytes / 1e9, dev.free_bytes / 1e9);
    return emit(row, opt) ? kOk : kError;
  }
  const std::size_t q_bytes = static_cast<std::size_t>(p.B) * p.H * p.N * p.D * es;
  const std::size_t kv_bytes = static_cast<std::size_t>(p.B) * p.Hkv * p.N * p.D * es;
  const std::size_t lse_bytes = static_cast<std::size_t>(p.B) * p.H * p.N * sizeof(float);
  const std::size_t set_bytes = q_bytes + 2 * kv_bytes;
  const std::size_t fixed = q_bytes + lse_bytes + ws_bytes + kFlushBytes;
  const std::size_t budget = dev.free_bytes - dev.free_bytes / 10;  // 10% head-room
  int nsets = kMaxSets;
  while (nsets > 1 && fixed + static_cast<std::size_t>(nsets) * set_bytes > budget)
    --nsets;
  if (fixed + set_bytes > budget) {
    std::fprintf(stderr, "inputs do not fit: need %.2f GB, budget %.2f GB\n",
                 (fixed + set_bytes) / 1e9, budget / 1e9);
    return kNoMemory;
  }
  if (nsets < kMaxSets) notes.push_back("sets=" + std::to_string(nsets));

  std::vector<void*> q(nsets), k(nsets), v(nsets);
  for (int s = 0; s < nsets; ++s) {
    CUDA_CHECK(cudaMalloc(&q[s], q_bytes));
    CUDA_CHECK(cudaMalloc(&k[s], kv_bytes));
    CUDA_CHECK(cudaMalloc(&v[s], kv_bytes));
    fill(q[s], q_bytes / es, dtype, 0x1000u * (s + 1) + 1, stream);
    fill(k[s], kv_bytes / es, dtype, 0x1000u * (s + 1) + 2, stream);
    fill(v[s], kv_bytes / es, dtype, 0x1000u * (s + 1) + 3, stream);
  }
  void* o = nullptr;
  float* lse = nullptr;
  void* ws = nullptr;
  void* flush = nullptr;
  CUDA_CHECK(cudaMalloc(&o, q_bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&lse), lse_bytes));
  if (ws_bytes > 0) CUDA_CHECK(cudaMalloc(&ws, ws_bytes));
  CUDA_CHECK(cudaMalloc(&flush, kFlushBytes));
  CUDA_CHECK(cudaMemsetAsync(o, 0, q_bytes, stream));
  CUDA_CHECK(cudaMemsetAsync(lse, 0, lse_bytes, stream));

  auto launch = [&](int s) {
    return fa::forward(impl, dtype, q[s], k[s], v[s], o, lse, ws, p, stream);
  };

  // Pre-flight on set 0: is this (impl, dtype, D) implemented on this device at all? Its
  // output feeds the correctness gate; the iteration count is sized later, after the warmup.
  cudaError_t err = launch(0);
  if (err == cudaErrorNotSupported) {
    std::printf("stage %d not implemented (%s/%s D=%d on sm_%d, build archs=%s)\n", opt.stage,
                fa::impl_name(impl), fa::dtype_name(dtype), p.D, dev.sm, kArchs);
    return kNotImplemented;
  }
  if (err == cudaSuccess) err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::fprintf(stderr, "fa::forward failed: %s\n", cudaGetErrorString(err));
    return kError;
  }
  int iters = iters_base;
  // docs/BENCHMARK_PROTOCOL.md rule 1: >= 100 iterations or >= 500 ms, whichever is larger;
  // time_loop sizes it from three post-warmup launches unless --iters was given.
  const bool autosize = !smoke_mode && !opt.iters_given;

  // Correctness gate on the pre-flight output (set 0), then finiteness of the whole tensor.
  const bool has_lse = impl != fa::Impl::Naive && impl != fa::Impl::Tiled;
  if (opt.check_rows > 0) {
    int checked = 0;
    if (!self_check(p, dtype, q[0], k[0], v[0], o, lse, opt.check_rows, has_lse, &checked))
      return kError;
    notes.push_back("check=rows" + std::to_string(checked));
  }
  std::vector<uint8_t> host_o(q_bytes);
  CUDA_CHECK(cudaMemcpy(host_o.data(), o, q_bytes, cudaMemcpyDeviceToHost));
  std::size_t nonfinite = 0;
  for (std::size_t i = 0; i < q_bytes / es; ++i)
    nonfinite += std::isfinite(to_double(host_o.data(), i, dtype)) ? 0 : 1;
  if (nonfinite > 0) {
    std::fprintf(stderr, "CHECK_FAIL nonfinite=%zu output elements\n", nonfinite);
    return kError;
  }
  g_sink = fnv1a(host_o.data(), host_o.size());
  std::printf("checksum=%016llx\n", static_cast<unsigned long long>(g_sink));

  std::vector<float> ms;
  std::vector<int> clocks;
  err = time_loop(launch, nsets, warmup, &iters, autosize, flush, stream, &ms, &clocks);
  if (err != cudaSuccess) {
    std::fprintf(stderr, "timed loop failed: %s\n", cudaGetErrorString(err));
    return kError;
  }
  const Stats st = stats_of(ms);
  apply_clock_notes(clocks, opt.clock, &row, &notes);
  notes.push_back("iters=" + std::to_string(iters));
  notes.push_back("warmup=" + std::to_string(warmup));
  if (st.p10 > 0.0 && st.p90 / st.p10 >= kNoiseRatio) notes.push_back("noisy");
  const double flops =
      4.0 * p.B * p.H * static_cast<double>(p.N) * p.N * p.D * (causal_half ? 0.5 : 1.0);
  const double bytes = bytes_moved(impl, p, es);
  row.ms_median = fmt("%.4f", st.median), row.ms_p10 = fmt("%.4f", st.p10),
  row.ms_p90 = fmt("%.4f", st.p90);
  row.tflops = fmt("%.3f", flops / (st.median * 1e-3) / 1e12);
  row.gbps = fmt("%.1f", bytes / (st.median * 1e-3) / 1e9);
  row.notes = join(notes, ";");
  std::printf(
      "stage %d %s %s %s causal=%d: median %.4f ms (p10 %.4f, p90 %.4f, min %.4f, max %.4f) "
      "over %d iters/%d warmup/%d sets -> %s TFLOP/s, %s GB/s\n",
      opt.stage, fa::impl_name(impl), opt.cfg.c_str(), fa::dtype_name(dtype), row.causal, st.median,
      st.p10, st.p90, st.min, st.max, iters, warmup, nsets, row.tflops.c_str(), row.gbps.c_str());
  const bool ok = smoke_mode ? true : emit(row, opt);  // a smoke run is not a benchmark row

  for (int s = 0; s < nsets; ++s) {
    CUDA_CHECK(cudaFree(q[s]));
    CUDA_CHECK(cudaFree(k[s]));
    CUDA_CHECK(cudaFree(v[s]));
  }
  CUDA_CHECK(cudaFree(o));
  CUDA_CHECK(cudaFree(lse));
  CUDA_CHECK(cudaFree(flush));
  if (ws != nullptr) CUDA_CHECK(cudaFree(ws));
  return ok ? kOk : kError;
}

// ---- smoke run: fa_smoke on C64 (o = v * scale), exact host comparison, CSV row ----------
int run_smoke(const Options& opt, const DeviceInfo& dev, cudaStream_t stream) {
  fa::Params p{};
  if (!parse_cfg(opt.cfg, &p)) return usage(stderr), kUsage;
  const std::size_t n = static_cast<std::size_t>(p.B) * p.H * p.N * p.D;
  const float scale = 1.0f / std::sqrt(static_cast<float>(p.D));
  const int iters = opt.iters_given ? opt.iters : 3;  // small: this also runs under the sanitizers
  const int warmup = opt.warmup_given ? opt.warmup : 1;
  float *v = nullptr, *o = nullptr;
  void* flush = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&v), n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&o), n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&flush, kFlushBytes));
  fill(v, n, fa::Dtype::F32, 7u, stream);
  cudaError_t err = fa::smoke::forward(v, o, scale, n, stream);
  if (err == cudaSuccess) err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::fprintf(stderr, "SMOKE_FAIL launch: %s\n", cudaGetErrorString(err));
    return kError;
  }
  std::vector<float> hv(n), ho(n);
  CUDA_CHECK(cudaMemcpy(hv.data(), v, n * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(ho.data(), o, n * sizeof(float), cudaMemcpyDeviceToHost));
  std::size_t mismatches = 0, first = 0;
  for (std::size_t i = 0; i < n; ++i) {
    if (ho[i] != hv[i] * scale) {  // one IEEE fp32 multiply on both sides: bit-exact
      if (mismatches++ == 0) first = i;
    }
  }
  if (mismatches > 0) {
    std::fprintf(stderr, "SMOKE_FAIL mismatches=%zu first@%zu got %g want %g\n", mismatches, first,
                 ho[first], hv[first] * scale);
    return kError;
  }
  g_sink = fnv1a(ho.data(), n * sizeof(float));
  std::printf("SMOKE_OK checksum=%016llx\n", static_cast<unsigned long long>(g_sink));

  std::vector<float> ms;
  std::vector<int> clocks;
  auto launch = [&](int) {
    return fa::smoke::forward(v, o, scale, n, stream);
  };
  int smoke_iters = iters;
  err = time_loop(launch, 1, warmup, &smoke_iters, false, flush, stream, &ms, &clocks);
  if (err != cudaSuccess) {
    std::fprintf(stderr, "SMOKE_FAIL timed loop: %s\n", cudaGetErrorString(err));
    return kError;
  }
  const Stats st = stats_of(ms);
  Row row = base_row(opt, dev);
  row.impl = "smoke", row.stage = 0, row.dtype = "fp32";
  row.B = p.B, row.H = p.H, row.N = p.N, row.D = p.D, row.causal = 0;
  row.ms_median = fmt("%.4f", st.median), row.ms_p10 = fmt("%.4f", st.p10),
  row.ms_p90 = fmt("%.4f", st.p90);
  row.gbps = fmt("%.1f", 2.0 * n * sizeof(float) / (st.median * 1e-3) / 1e9);  // v read + o write
  std::vector<std::string> notes = {"smoke"};
  apply_clock_notes(clocks, opt.clock, &row, &notes);
  notes.push_back("iters=" + std::to_string(smoke_iters));
  notes.push_back("warmup=" + std::to_string(warmup));
  if (st.p10 > 0.0 && st.p90 / st.p10 >= kNoiseRatio) notes.push_back("noisy");
  row.notes = join(notes, ";");
  const bool ok = emit(row, opt);
  CUDA_CHECK(cudaFree(v));
  CUDA_CHECK(cudaFree(o));
  CUDA_CHECK(cudaFree(flush));
  if (!ok) return kError;
  // With --stage N the real kernel runs once under the same process (sanitizer coverage).
  if (opt.stage > 0) {
    const int rc = run_stage(opt, dev, stream, /*smoke_mode=*/true);
    if (rc != kOk) return rc;
    std::printf("SMOKE_STAGE_OK stage=%d\n", opt.stage);
  }
  return kOk;
}

int run_list(const DeviceInfo& dev) {
  const fa::Dtype dts[] = {fa::Dtype::F32, fa::Dtype::F16, fa::Dtype::BF16};
  std::printf("%-6s %-5s", "impl", "stage");
  for (fa::Dtype dt : dts)
    for (int D : {64, 128})
      std::printf(" %s/%-3d", fa::dtype_name(dt), D);
  std::printf("\n");
  for (int st = 1; st <= 5; ++st) {
    const auto impl = static_cast<fa::Impl>(st);
    std::printf("%-6s %-5d", fa::impl_name(impl), st);
    for (fa::Dtype dt : dts)
      for (int D : {64, 128})
        std::printf(" %-8s", fa::impl_supports(impl, dt, D, dev.sm) ? "yes" : "no");
    std::printf("\n");
  }
  return kOk;
}

}  // namespace

int main(int argc, char** argv) {
  Options opt;
  if (!parse_args(argc, argv, &opt)) return usage(stderr), kUsage;
  const DeviceInfo dev = query_device();
  std::printf(
      "fa_bench: %s (sm_%d) driver %s runtime %s nvcc %s archs=%s build=%s free %.2f/%.2f GB\n",
      dev.name.c_str(), dev.sm, dev.driver.c_str(), dev.runtime.c_str(), kNvcc, kArchs, kBuildType,
      dev.free_bytes / 1e9, dev.total_bytes / 1e9);
  if (opt.list) return run_list(dev);
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreate(&stream));
  const int rc = opt.smoke ? run_smoke(opt, dev, stream) : run_stage(opt, dev, stream, false);
  CUDA_CHECK(cudaStreamDestroy(stream));
  return rc;
}
