#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <span>
#include <string>
#include <vector>

#include "program/program_object/object_reader.h"
#include "runtime/hip_runtime/hip_runtime.h"
#include "runtime/model_runtime/model_runtime.h"
#include "tests/test_utils/hipcc_cache_test_utils.h"
#include "test_matrix_profile.h"

namespace gpu_model {
namespace {

bool HasHipHostToolchain() {
  return std::system("command -v hipcc >/dev/null 2>&1") == 0 &&
         std::system("command -v clang-offload-bundler >/dev/null 2>&1") == 0 &&
         std::system("command -v llvm-objcopy >/dev/null 2>&1") == 0 &&
         std::system("command -v llvm-objdump >/dev/null 2>&1") == 0 &&
         std::system("command -v readelf >/dev/null 2>&1") == 0 &&
         std::system("command -v llvm-readelf >/dev/null 2>&1") == 0;
}

std::filesystem::path MakeUniqueTempDir(const std::string& stem) {
  const auto suffix =
      std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
  const auto path = std::filesystem::temp_directory_path() / (stem + "_" + suffix);
  std::filesystem::remove_all(path);
  std::filesystem::create_directories(path);
  return path;
}

enum class ArtifactKind {
  Common,
  Mfma,
};

enum class KernelKind {
  VecAdd,
  FmaLoop,
  BiasChain,
  ByValueAggregate,
  AtomicCount,
  SharedReverse,
  DynamicSharedSum,
  BlockReduceSum,
  Softmax,
  Mfma,
};

struct HipCtsCase {
  std::string name;
  ArtifactKind artifact = ArtifactKind::Common;
  KernelKind kernel = KernelKind::VecAdd;
  uint32_t grid_x = 1;
  uint32_t block_x = 1;
  uint32_t n = 1;
  uint32_t iters = 1;
  float f0 = 0.0f;
  float f1 = 0.0f;
  float f2 = 0.0f;
  uint32_t pattern = 0;
};

struct BuiltArtifact {
  std::filesystem::path temp_dir;
  std::filesystem::path exe_path;
};

const BuiltArtifact& CommonArtifact() {
  static std::once_flag once;
  static BuiltArtifact artifact;
  std::call_once(once, []() {
    artifact.temp_dir = MakeUniqueTempDir("gpu_model_hip_cts_common");
    const auto src_path = artifact.temp_dir / "hip_cts_common.cpp";
    artifact.exe_path = artifact.temp_dir / "hip_cts_common.out";
    {
      std::ofstream out(src_path);
      out << R"(
#include <hip/hip_runtime.h>
#include <math.h>
extern "C" __global__ void vecadd(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}
extern "C" __global__ void fma_loop(const float* a, const float* b, float* c, int n, int iters) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x = a[i];
  float y = b[i];
  float acc = 0.0f;
  for (int k = 0; k < iters; ++k) acc = acc * x + y;
  c[i] = acc;
}
extern "C" __global__ void bias_chain(const float* a, const float* b, float* c, int n, float b0, float b1, float b2) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i] + b0 + b1 + b2;
}
struct Payload {
  int x;
  int y;
  int z;
};
extern "C" __global__ void by_value_aggregate(int* out, Payload payload) {
  if (threadIdx.x == 0) {
    out[0] = payload.x + payload.y + payload.z;
  }
}
extern "C" __global__ void atomic_count(int* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) atomicAdd(out, 1);
}
extern "C" __global__ void shared_reverse(const int* in, int* out, int n) {
  __shared__ int scratch[64];
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + tid;
  if (idx < n) scratch[tid] = in[idx];
  __syncthreads();
  if (idx < n) out[idx] = scratch[blockDim.x - 1 - tid];
}
extern "C" __global__ void dynamic_shared_sum(int* out) {
  extern __shared__ int scratch[];
  int tid = threadIdx.x;
  scratch[tid] = tid + 1;
  __syncthreads();
  if (tid == 0) {
    int acc = 0;
    for (int i = 0; i < blockDim.x; ++i) acc += scratch[i];
    out[blockIdx.x] = acc;
  }
}
extern "C" __global__ void block_reduce_sum(const float* in, float* out, int n) {
  __shared__ float scratch[256];
  int tid = threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int idx = blockIdx.x * blockDim.x + tid;
  float sum = 0.0f;
  for (int i = idx; i < n; i += stride) {
    sum += in[i];
  }
  scratch[tid] = sum;
  __syncthreads();
  for (int step = blockDim.x / 2; step > 0; step >>= 1) {
    if (tid < step) scratch[tid] += scratch[tid + step];
    __syncthreads();
  }
  if (tid == 0) out[blockIdx.x] = scratch[0];
}
extern "C" __global__ void softmax_row(const float* in, float* out, int n) {
  __shared__ float scratch[64];
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + tid;
  float x = idx < n ? in[idx] : -1.0e20f;
  scratch[tid] = x;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
    __syncthreads();
  }
  float m = scratch[0];
  float e = idx < n ? expf(x - m) : 0.0f;
  scratch[tid] = e;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) scratch[tid] += scratch[tid + stride];
    __syncthreads();
  }
  if (idx < n) out[idx] = e / scratch[0];
}
int main() { return 0; }
)";
    }
    const std::string command =
        test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + artifact.exe_path.string();
    if (std::system(command.c_str()) != 0) {
      throw std::runtime_error("failed to build common HIP CTS artifact");
    }
  });
  return artifact;
}

const BuiltArtifact& MfmaArtifact() {
  static std::once_flag once;
  static BuiltArtifact artifact;
  std::call_once(once, []() {
    artifact.temp_dir = MakeUniqueTempDir("gpu_model_hip_cts_mfma");
    const auto src_path = artifact.temp_dir / "hip_cts_mfma.cpp";
    artifact.exe_path = artifact.temp_dir / "hip_cts_mfma.out";
    {
      std::ofstream out(src_path);
      out << R"(
#include <hip/hip_runtime.h>
typedef float v4f __attribute__((ext_vector_type(4)));
extern "C" __global__ void mfma_probe(float* out) {
#if defined(__AMDGCN__)
  v4f acc = {0.0f, 0.0f, 0.0f, 0.0f};
  acc = __builtin_amdgcn_mfma_f32_16x16x4f32(1.0f, 1.0f, acc, 0, 0, 0);
  if (threadIdx.x == 0) out[0] = acc[0];
#else
  if (threadIdx.x == 0) out[0] = 0.0f;
#endif
}
int main() { return 0; }
)";
    }
    const std::string command =
        test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + artifact.exe_path.string();
    if (std::system(command.c_str()) != 0) {
      throw std::runtime_error("failed to build mfma HIP CTS artifact");
    }
  });
  return artifact;
}

const std::filesystem::path& ArtifactPath(ArtifactKind kind) {
  return kind == ArtifactKind::Common ? CommonArtifact().exe_path : MfmaArtifact().exe_path;
}

const char* KernelName(KernelKind kernel) {
  switch (kernel) {
    case KernelKind::VecAdd:
      return "vecadd";
    case KernelKind::FmaLoop:
      return "fma_loop";
    case KernelKind::BiasChain:
      return "bias_chain";
    case KernelKind::ByValueAggregate:
      return "by_value_aggregate";
    case KernelKind::AtomicCount:
      return "atomic_count";
    case KernelKind::SharedReverse:
      return "shared_reverse";
    case KernelKind::DynamicSharedSum:
      return "dynamic_shared_sum";
    case KernelKind::BlockReduceSum:
      return "block_reduce_sum";
    case KernelKind::Softmax:
      return "softmax_row";
    case KernelKind::Mfma:
      return "mfma_probe";
  }
  return "unknown";
}

void FillFloatPattern(std::vector<float>& a, std::vector<float>& b, uint32_t pattern) {
  for (uint32_t i = 0; i < a.size(); ++i) {
    switch (pattern % 4u) {
      case 0:
        a[i] = 0.5f * static_cast<float>(i);
        b[i] = 0.25f * static_cast<float>(100 + i);
        break;
      case 1:
        a[i] = 1.0f + 0.001f * static_cast<float>(i);
        b[i] = 2.0f + 0.002f * static_cast<float>(i);
        break;
      case 2:
        a[i] = static_cast<float>((i % 17u) - 8);
        b[i] = static_cast<float>((i % 9u) - 4) * 0.5f;
        break;
      default:
        a[i] = static_cast<float>(i % 13u) * 1.25f;
        b[i] = static_cast<float>((i * 3u) % 11u) * -0.75f;
        break;
    }
  }
}

std::vector<float> ExpectedVecAdd(uint32_t n, uint32_t pattern) {
  std::vector<float> a(n), b(n), out(n);
  FillFloatPattern(a, b, pattern);
  for (uint32_t i = 0; i < n; ++i) {
    out[i] = a[i] + b[i];
  }
  return out;
}

std::vector<float> ExpectedFma(uint32_t n, uint32_t iters, uint32_t pattern) {
  std::vector<float> a(n), b(n), out(n);
  FillFloatPattern(a, b, pattern);
  for (uint32_t i = 0; i < n; ++i) {
    float acc = 0.0f;
    for (uint32_t k = 0; k < iters; ++k) {
      acc = acc * a[i] + b[i];
    }
    out[i] = acc;
  }
  return out;
}

std::vector<float> ExpectedBias(uint32_t n, uint32_t pattern, float b0, float b1, float b2) {
  std::vector<float> a(n), b(n), out(n);
  FillFloatPattern(a, b, pattern);
  for (uint32_t i = 0; i < n; ++i) {
    out[i] = a[i] + b[i] + b0 + b1 + b2;
  }
  return out;
}

std::vector<int32_t> ExpectedSharedReverse(uint32_t n, uint32_t pattern) {
  std::vector<int32_t> in(n), out(n, -1);
  for (uint32_t i = 0; i < n; ++i) {
    in[i] = static_cast<int32_t>(i + 1 + pattern * 7u);
  }
  for (uint32_t block = 0; block < n / 64u; ++block) {
    const uint32_t base = block * 64u;
    for (uint32_t lane = 0; lane < 64u; ++lane) {
      out[base + lane] = in[base + (63u - lane)];
    }
  }
  return out;
}

std::vector<float> ExpectedSoftmax(uint32_t n) {
  return std::vector<float>(n, 1.0f / 64.0f);
}

std::vector<int32_t> ExpectedDynamicSharedSum(uint32_t grid_x, uint32_t block_x) {
  return std::vector<int32_t>(
      grid_x, static_cast<int32_t>(block_x * (block_x + 1u) / 2u));
}

int32_t ExpectedAtomicCount(uint32_t n) { return static_cast<int32_t>(n); }

int32_t ExpectedByValueAggregate() { return 5 + 9 + 17; }

std::vector<float> ExpectedBlockReduceSum(uint32_t n,
                                          uint32_t pattern,
                                          uint32_t grid_x,
                                          uint32_t block_x) {
  std::vector<float> in(n);
  FillFloatPattern(in, in, pattern);
  std::vector<float> out(grid_x, 0.0f);
  const uint32_t stride = grid_x * block_x;
  for (uint32_t block = 0; block < grid_x; ++block) {
    float block_sum = 0.0f;
    for (uint32_t tid = 0; tid < block_x; ++tid) {
      for (uint32_t i = block * block_x + tid; i < n; i += stride) {
        block_sum += in[i];
      }
    }
    out[block] = block_sum;
  }
  return out;
}

std::vector<HipCtsCase> MakeModelRuntimeCasesFull() {
  std::vector<HipCtsCase> cases;
  uint32_t id = 0;
  const auto add = [&](HipCtsCase c) {
    c.name = "rh_" + std::to_string(id++) + "_" + KernelName(c.kernel);
    cases.push_back(std::move(c));
  };

  for (const auto& [n, block] :
       std::vector<std::pair<uint32_t, uint32_t>>{{1, 1},   {32, 32}, {60, 60},  {64, 64},
                                                  {65, 65}, {96, 96}, {127, 64}, {128, 128},
                                                  {129, 128}, {191, 64}, {192, 64}, {255, 128},
                                                  {256, 128}, {257, 128}, {511, 256}, {512, 256},
                                                  {777, 256}, {1024, 1024}, {2048, 512}, {30720, 1024}}) {
    add(HipCtsCase{
        .name = {},
        .artifact = ArtifactKind::Common,
        .kernel = KernelKind::VecAdd,
        .grid_x = (n + block - 1) / block,
        .block_x = block,
        .n = n,
        .pattern = id % 4u,
    });
  }

  for (const auto& [n, iters, block] :
       std::vector<std::tuple<uint32_t, uint32_t, uint32_t>>{
           {64, 1, 64},    {64, 2, 64},    {64, 4, 64},   {128, 3, 64},
           {128, 7, 128},  {192, 5, 64},   {257, 7, 128}, {257, 3, 64},
           {512, 2, 128},  {512, 8, 128},  {768, 4, 256}, {1024, 1, 256},
           {1024, 6, 128}, {1536, 3, 256}, {2048, 5, 256}}) {
    add(HipCtsCase{
        .name = {},
        .artifact = ArtifactKind::Common,
        .kernel = KernelKind::FmaLoop,
        .grid_x = (n + block - 1) / block,
        .block_x = block,
        .n = n,
        .iters = iters,
        .pattern = id % 4u,
    });
  }

  for (const auto& [n, block, b0, b1, b2] :
       std::vector<std::tuple<uint32_t, uint32_t, float, float, float>>{
           {64, 64, 1.5f, -2.0f, 3.25f},   {96, 96, 0.5f, 1.0f, -0.5f},
           {128, 64, -1.0f, 2.0f, 4.0f},   {129, 64, 3.5f, -1.0f, 0.25f},
           {192, 64, 0.0f, 0.0f, 1.0f},    {256, 128, 1.0f, 2.0f, 3.0f},
           {257, 128, -2.5f, 1.5f, 0.75f}, {512, 128, 4.0f, -1.0f, -3.0f},
           {1024, 256, 0.25f, 0.5f, 0.75f}, {2048, 256, -1.5f, -0.5f, 2.0f}}) {
    add(HipCtsCase{
        .name = {},
        .artifact = ArtifactKind::Common,
        .kernel = KernelKind::BiasChain,
        .grid_x = (n + block - 1) / block,
        .block_x = block,
        .n = n,
        .f0 = b0,
        .f1 = b1,
        .f2 = b2,
        .pattern = id % 4u,
    });
  }

  add(HipCtsCase{
      .name = {},
      .artifact = ArtifactKind::Common,
      .kernel = KernelKind::ByValueAggregate,
      .grid_x = 1,
      .block_x = 64,
      .n = 1,
  });

  for (const auto& [n, block] :
       std::vector<std::pair<uint32_t, uint32_t>>{{1, 1}, {64, 64}, {257, 128}, {1024, 256}}) {
    add(HipCtsCase{
        .name = {},
        .artifact = ArtifactKind::Common,
        .kernel = KernelKind::AtomicCount,
        .grid_x = (n + block - 1) / block,
        .block_x = block,
        .n = n,
    });
  }

  for (const auto blocks : std::vector<uint32_t>{1, 2, 4, 8, 16, 3, 5, 6, 7, 10}) {
    add(HipCtsCase{
        .name = {},
        .artifact = ArtifactKind::Common,
        .kernel = KernelKind::SharedReverse,
        .grid_x = blocks,
        .block_x = 64,
        .n = blocks * 64u,
        .pattern = id % 3u,
    });
  }

  for (const auto blocks : std::vector<uint32_t>{1, 2, 4, 8, 16}) {
    add(HipCtsCase{
        .name = {},
        .artifact = ArtifactKind::Common,
        .kernel = KernelKind::DynamicSharedSum,
        .grid_x = blocks,
        .block_x = 64,
        .n = blocks,
    });
  }

  for (const auto& [n, grid, block] :
       std::vector<std::tuple<uint32_t, uint32_t, uint32_t>>{
           {64, 1, 64}, {257, 2, 128}, {1024, 4, 256}, {4097, 8, 256}}) {
    add(HipCtsCase{
        .name = {},
        .artifact = ArtifactKind::Common,
        .kernel = KernelKind::BlockReduceSum,
        .grid_x = grid,
        .block_x = block,
        .n = n,
        .pattern = id % 4u,
    });
  }

  for (const auto blocks : std::vector<uint32_t>{1, 2, 4, 8, 16, 3, 5, 6, 7, 10}) {
    add(HipCtsCase{
        .name = {},
        .artifact = ArtifactKind::Common,
        .kernel = KernelKind::Softmax,
        .grid_x = blocks,
        .block_x = 64,
        .n = blocks * 64u,
        .pattern = id % 4u,
    });
  }

  for (int i = 0; i < 15; ++i) {
    add(HipCtsCase{
        .name = {},
        .artifact = ArtifactKind::Mfma,
        .kernel = KernelKind::Mfma,
        .grid_x = 1,
        .block_x = 64,
        .n = 1,
    });
  }

  EXPECT_EQ(cases.size(), 94u);
  return cases;
}

std::vector<HipCtsCase> MakeModelRuntimeCasesQuick() {
  return test::SelectIndexedCases(MakeModelRuntimeCasesFull(),
                                  {0, 4, 10, 19, 20, 24, 34, 35, 39, 40, 49, 53, 63, 67, 68, 77});
}

std::vector<HipCtsCase> MakeModelRuntimeCases() {
  if (test::Phase1CompatibilityAliasGateEnabled()) {
    return {};
  }
  return test::FullTestMatrixEnabled() ? MakeModelRuntimeCasesFull() : MakeModelRuntimeCasesQuick();
}

std::vector<HipCtsCase> MakeHipRuntimeAbiCasesFull() {
  std::vector<HipCtsCase> cases;
  uint32_t id = 0;
  const auto add = [&](HipCtsCase c) {
    c.name = "is_" + std::to_string(id++) + "_" + KernelName(c.kernel);
    cases.push_back(std::move(c));
  };

  for (const auto& [n, block] :
       std::vector<std::pair<uint32_t, uint32_t>>{{64, 64}, {128, 128}, {129, 64},
                                                  {257, 128}, {1024, 256}, {30720, 1024}}) {
    add(HipCtsCase{
        .name = {},
        .artifact = ArtifactKind::Common,
        .kernel = KernelKind::VecAdd,
        .grid_x = (n + block - 1) / block,
        .block_x = block,
        .n = n,
        .pattern = id % 4u,
    });
  }
  for (const auto& [n, iters, block] :
       std::vector<std::tuple<uint32_t, uint32_t, uint32_t>>{{64, 2, 64}, {128, 7, 128},
                                                              {257, 7, 128}, {1024, 4, 256}}) {
    add(HipCtsCase{
        .name = {},
        .artifact = ArtifactKind::Common,
        .kernel = KernelKind::FmaLoop,
        .grid_x = (n + block - 1) / block,
        .block_x = block,
        .n = n,
        .iters = iters,
        .pattern = id % 4u,
    });
  }
  for (const auto& [n, block, b0, b1, b2] :
       std::vector<std::tuple<uint32_t, uint32_t, float, float, float>>{
           {129, 64, 1.5f, -2.0f, 3.25f},
           {257, 128, -2.5f, 1.5f, 0.75f},
           {1024, 256, 0.25f, 0.5f, 0.75f}}) {
    add(HipCtsCase{
        .name = {},
        .artifact = ArtifactKind::Common,
        .kernel = KernelKind::BiasChain,
        .grid_x = (n + block - 1) / block,
        .block_x = block,
        .n = n,
        .f0 = b0,
        .f1 = b1,
        .f2 = b2,
        .pattern = id % 4u,
    });
  }
  add(HipCtsCase{
      .name = {},
      .artifact = ArtifactKind::Common,
      .kernel = KernelKind::ByValueAggregate,
      .grid_x = 1,
      .block_x = 64,
      .n = 1,
  });
  for (const auto& [n, block] :
       std::vector<std::pair<uint32_t, uint32_t>>{{1, 1}, {257, 128}}) {
    add(HipCtsCase{
        .name = {},
        .artifact = ArtifactKind::Common,
        .kernel = KernelKind::AtomicCount,
        .grid_x = (n + block - 1) / block,
        .block_x = block,
        .n = n,
    });
  }
  for (const auto blocks : std::vector<uint32_t>{2, 4, 8}) {
    add(HipCtsCase{
        .name = {},
        .artifact = ArtifactKind::Common,
        .kernel = KernelKind::SharedReverse,
        .grid_x = blocks,
        .block_x = 64,
        .n = blocks * 64u,
        .pattern = id % 3u,
    });
  }
  for (const auto blocks : std::vector<uint32_t>{1, 4}) {
    add(HipCtsCase{
        .name = {},
        .artifact = ArtifactKind::Common,
        .kernel = KernelKind::DynamicSharedSum,
        .grid_x = blocks,
        .block_x = 64,
        .n = blocks,
    });
  }
  for (const auto& [n, grid, block] :
       std::vector<std::tuple<uint32_t, uint32_t, uint32_t>>{{64, 1, 64}, {1024, 4, 256}}) {
    add(HipCtsCase{
        .name = {},
        .artifact = ArtifactKind::Common,
        .kernel = KernelKind::BlockReduceSum,
        .grid_x = grid,
        .block_x = block,
        .n = n,
        .pattern = id % 4u,
    });
  }
  for (const auto blocks : std::vector<uint32_t>{1, 4}) {
    add(HipCtsCase{
        .name = {},
        .artifact = ArtifactKind::Common,
        .kernel = KernelKind::Softmax,
        .grid_x = blocks,
        .block_x = 64,
        .n = blocks * 64u,
        .pattern = id % 4u,
    });
  }
  for (int i = 0; i < 2; ++i) {
    add(HipCtsCase{
        .name = {},
        .artifact = ArtifactKind::Mfma,
        .kernel = KernelKind::Mfma,
        .grid_x = 1,
        .block_x = 64,
        .n = 1,
    });
  }

  EXPECT_EQ(cases.size(), 27u);
  return cases;
}

std::vector<HipCtsCase> MakeHipRuntimeAbiCasesQuick() {
  return test::SelectIndexedCases(MakeHipRuntimeAbiCasesFull(), {0, 3, 6, 9, 11, 12, 16, 18, 20, 22, 25});
}

std::vector<HipCtsCase> MakeHipRuntimeAbiCases() {
  if (test::Phase1CompatibilityAliasGateEnabled()) {
    return {};
  }
  return test::FullTestMatrixEnabled() ? MakeHipRuntimeAbiCasesFull() : MakeHipRuntimeAbiCasesQuick();
}

class HipCtsModelRuntimeTest : public ::testing::TestWithParam<HipCtsCase> {};
class HipCtsRuntimeAbiTest : public ::testing::TestWithParam<HipCtsCase> {};

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(HipCtsModelRuntimeTest);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(HipCtsRuntimeAbiTest);

std::string CaseName(const ::testing::TestParamInfo<HipCtsCase>& info) { return info.param.name; }

void ExpectNearVector(const std::vector<float>& actual,
                      const std::vector<float>& expected,
                      float tol = 1.0e-4f) {
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < actual.size(); ++i) {
    if (std::isnan(actual[i]) || std::isnan(expected[i])) {
      EXPECT_TRUE(std::isnan(actual[i]) && std::isnan(expected[i])) << "idx=" << i;
    } else if (std::isinf(actual[i]) || std::isinf(expected[i])) {
      EXPECT_TRUE(std::isinf(actual[i]) && std::isinf(expected[i])) << "idx=" << i;
      EXPECT_EQ(std::signbit(actual[i]), std::signbit(expected[i])) << "idx=" << i;
    } else {
      EXPECT_NEAR(actual[i], expected[i], tol) << "idx=" << i;
    }
  }
}

TEST_P(HipCtsModelRuntimeTest, ExecutesHipOutAndValidatesResults) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const HipCtsCase& c = GetParam();
  ModelRuntime hooks;

  switch (c.kernel) {
    case KernelKind::VecAdd: {
      std::vector<float> a(c.n), b(c.n), out(c.n, -1.0f);
      FillFloatPattern(a, b, c.pattern);
      const auto expected = ExpectedVecAdd(c.n, c.pattern);
      const uint64_t a_addr = hooks.Malloc(c.n * sizeof(float));
      const uint64_t b_addr = hooks.Malloc(c.n * sizeof(float));
      const uint64_t out_addr = hooks.Malloc(c.n * sizeof(float));
      hooks.MemcpyHtoD<float>(a_addr, std::span<const float>(a));
      hooks.MemcpyHtoD<float>(b_addr, std::span<const float>(b));
      hooks.MemcpyHtoD<float>(out_addr, std::span<const float>(out));
      KernelArgPack args;
      args.PushU64(a_addr);
      args.PushU64(b_addr);
      args.PushU64(out_addr);
      args.PushU32(c.n);
      const auto result = hooks.LaunchProgramObject(
          ObjectReader{}.LoadProgramObject(ArtifactPath(c.artifact), KernelName(c.kernel)),
          LaunchConfig{.grid_dim_x = c.grid_x, .block_dim_x = c.block_x},
          std::move(args),
          ExecutionMode::Functional,
          "mac500",
          nullptr);
      ASSERT_TRUE(result.ok) << result.error_message;
      hooks.MemcpyDtoH<float>(out_addr, std::span<float>(out));
      ExpectNearVector(out, expected);
      return;
    }
    case KernelKind::FmaLoop: {
      std::vector<float> a(c.n), b(c.n), out(c.n, -1.0f);
      FillFloatPattern(a, b, c.pattern);
      const auto expected = ExpectedFma(c.n, c.iters, c.pattern);
      const uint64_t a_addr = hooks.Malloc(c.n * sizeof(float));
      const uint64_t b_addr = hooks.Malloc(c.n * sizeof(float));
      const uint64_t out_addr = hooks.Malloc(c.n * sizeof(float));
      hooks.MemcpyHtoD<float>(a_addr, std::span<const float>(a));
      hooks.MemcpyHtoD<float>(b_addr, std::span<const float>(b));
      hooks.MemcpyHtoD<float>(out_addr, std::span<const float>(out));
      KernelArgPack args;
      args.PushU64(a_addr);
      args.PushU64(b_addr);
      args.PushU64(out_addr);
      args.PushU32(c.n);
      args.PushU32(c.iters);
      const auto result = hooks.LaunchProgramObject(
          ObjectReader{}.LoadProgramObject(ArtifactPath(c.artifact), KernelName(c.kernel)),
          LaunchConfig{.grid_dim_x = c.grid_x, .block_dim_x = c.block_x},
          std::move(args),
          ExecutionMode::Functional,
          "mac500",
          nullptr);
      ASSERT_TRUE(result.ok) << result.error_message;
      hooks.MemcpyDtoH<float>(out_addr, std::span<float>(out));
      ExpectNearVector(out, expected, 1.0e-3f);
      return;
    }
    case KernelKind::BiasChain: {
      std::vector<float> a(c.n), b(c.n), out(c.n, -1.0f);
      FillFloatPattern(a, b, c.pattern);
      const auto expected = ExpectedBias(c.n, c.pattern, c.f0, c.f1, c.f2);
      const uint64_t a_addr = hooks.Malloc(c.n * sizeof(float));
      const uint64_t b_addr = hooks.Malloc(c.n * sizeof(float));
      const uint64_t out_addr = hooks.Malloc(c.n * sizeof(float));
      hooks.MemcpyHtoD<float>(a_addr, std::span<const float>(a));
      hooks.MemcpyHtoD<float>(b_addr, std::span<const float>(b));
      hooks.MemcpyHtoD<float>(out_addr, std::span<const float>(out));
      KernelArgPack args;
      args.PushU64(a_addr);
      args.PushU64(b_addr);
      args.PushU64(out_addr);
      args.PushU32(c.n);
      args.PushF32(c.f0);
      args.PushF32(c.f1);
      args.PushF32(c.f2);
      const auto result = hooks.LaunchProgramObject(
          ObjectReader{}.LoadProgramObject(ArtifactPath(c.artifact), KernelName(c.kernel)),
          LaunchConfig{.grid_dim_x = c.grid_x, .block_dim_x = c.block_x},
          std::move(args),
          ExecutionMode::Functional,
          "mac500",
          nullptr);
      ASSERT_TRUE(result.ok) << result.error_message;
      hooks.MemcpyDtoH<float>(out_addr, std::span<float>(out));
      ExpectNearVector(out, expected, 1.0e-4f);
      return;
    }
    case KernelKind::ByValueAggregate: {
      int32_t out = 0;
      const int32_t expected = ExpectedByValueAggregate();
      const uint64_t out_addr = hooks.Malloc(sizeof(int32_t));
      hooks.MemcpyHtoD<int32_t>(out_addr, std::span<const int32_t>(&out, 1));
      struct PayloadHost {
        int32_t x;
        int32_t y;
        int32_t z;
      } payload{5, 9, 17};
      KernelArgPack args;
      args.PushU64(out_addr);
      args.PushBytes(&payload, sizeof(payload));
      const auto result = hooks.LaunchProgramObject(
          ObjectReader{}.LoadProgramObject(ArtifactPath(c.artifact), KernelName(c.kernel)),
          LaunchConfig{.grid_dim_x = 1, .block_dim_x = 64},
          std::move(args),
          ExecutionMode::Functional,
          "mac500",
          nullptr);
      ASSERT_TRUE(result.ok) << result.error_message;
      hooks.MemcpyDtoH<int32_t>(out_addr, std::span<int32_t>(&out, 1));
      EXPECT_EQ(out, expected);
      return;
    }
    case KernelKind::AtomicCount: {
      int32_t out = 0;
      const int32_t expected = ExpectedAtomicCount(c.n);
      const uint64_t out_addr = hooks.Malloc(sizeof(int32_t));
      hooks.MemcpyHtoD<int32_t>(out_addr, std::span<const int32_t>(&out, 1));
      KernelArgPack args;
      args.PushU64(out_addr);
      args.PushU32(c.n);
      const auto result = hooks.LaunchProgramObject(
          ObjectReader{}.LoadProgramObject(ArtifactPath(c.artifact), KernelName(c.kernel)),
          LaunchConfig{.grid_dim_x = c.grid_x, .block_dim_x = c.block_x},
          std::move(args),
          ExecutionMode::Functional,
          "mac500",
          nullptr);
      ASSERT_TRUE(result.ok) << result.error_message;
      hooks.MemcpyDtoH<int32_t>(out_addr, std::span<int32_t>(&out, 1));
      EXPECT_EQ(out, expected);
      return;
    }
    case KernelKind::SharedReverse: {
      std::vector<int32_t> out(c.n, -1);
      const auto expected = ExpectedSharedReverse(c.n, c.pattern);
      std::vector<int32_t> input = expected;
      for (uint32_t block = 0; block < c.n / 64u; ++block) {
        const uint32_t base = block * 64u;
        for (uint32_t lane = 0; lane < 64u; ++lane) {
          input[base + lane] = expected[base + (63u - lane)];
        }
      }
      const uint64_t in_addr = hooks.Malloc(c.n * sizeof(int32_t));
      const uint64_t out_addr = hooks.Malloc(c.n * sizeof(int32_t));
      hooks.MemcpyHtoD<int32_t>(in_addr, std::span<const int32_t>(input));
      hooks.MemcpyHtoD<int32_t>(out_addr, std::span<const int32_t>(out));
      KernelArgPack args;
      args.PushU64(in_addr);
      args.PushU64(out_addr);
      args.PushU32(c.n);
      const auto result = hooks.LaunchProgramObject(
          ObjectReader{}.LoadProgramObject(ArtifactPath(c.artifact), KernelName(c.kernel)),
          LaunchConfig{.grid_dim_x = c.grid_x, .block_dim_x = c.block_x},
          std::move(args),
          ExecutionMode::Functional,
          "mac500",
          nullptr);
      ASSERT_TRUE(result.ok) << result.error_message;
      hooks.MemcpyDtoH<int32_t>(out_addr, std::span<int32_t>(out));
      EXPECT_EQ(out, expected);
      return;
    }
    case KernelKind::DynamicSharedSum: {
      std::vector<int32_t> out(c.grid_x, 0);
      const auto expected = ExpectedDynamicSharedSum(c.grid_x, c.block_x);
      const uint64_t out_addr = hooks.Malloc(c.grid_x * sizeof(int32_t));
      hooks.MemcpyHtoD<int32_t>(out_addr, std::span<const int32_t>(out));
      KernelArgPack args;
      args.PushU64(out_addr);
      const auto result = hooks.LaunchProgramObject(
          ObjectReader{}.LoadProgramObject(ArtifactPath(c.artifact), KernelName(c.kernel)),
          LaunchConfig{
              .grid_dim_x = c.grid_x,
              .block_dim_x = c.block_x,
              .shared_memory_bytes = static_cast<uint32_t>(c.block_x * sizeof(int32_t)),
          },
          std::move(args),
          ExecutionMode::Functional,
          "mac500",
          nullptr);
      ASSERT_TRUE(result.ok) << result.error_message;
      hooks.MemcpyDtoH<int32_t>(out_addr, std::span<int32_t>(out));
      EXPECT_EQ(out, expected);
      return;
    }
    case KernelKind::BlockReduceSum: {
      std::vector<float> in(c.n), out(c.grid_x, -1.0f);
      FillFloatPattern(in, in, c.pattern);
      const auto expected = ExpectedBlockReduceSum(c.n, c.pattern, c.grid_x, c.block_x);
      const uint64_t in_addr = hooks.Malloc(c.n * sizeof(float));
      const uint64_t out_addr = hooks.Malloc(c.grid_x * sizeof(float));
      hooks.MemcpyHtoD<float>(in_addr, std::span<const float>(in));
      hooks.MemcpyHtoD<float>(out_addr, std::span<const float>(out));
      KernelArgPack args;
      args.PushU64(in_addr);
      args.PushU64(out_addr);
      args.PushU32(c.n);
      const auto result = hooks.LaunchProgramObject(
          ObjectReader{}.LoadProgramObject(ArtifactPath(c.artifact), KernelName(c.kernel)),
          LaunchConfig{.grid_dim_x = c.grid_x, .block_dim_x = c.block_x},
          std::move(args),
          ExecutionMode::Functional,
          "mac500",
          nullptr);
      ASSERT_TRUE(result.ok) << result.error_message;
      hooks.MemcpyDtoH<float>(out_addr, std::span<float>(out));
      ExpectNearVector(out, expected, 1.0e-2f);
      return;
    }
    case KernelKind::Softmax: {
      std::vector<float> input(c.n, static_cast<float>(c.pattern + 1u));
      std::vector<float> out(c.n, 0.0f);
      const auto expected = ExpectedSoftmax(c.n);
      const uint64_t in_addr = hooks.Malloc(c.n * sizeof(float));
      const uint64_t out_addr = hooks.Malloc(c.n * sizeof(float));
      hooks.MemcpyHtoD<float>(in_addr, std::span<const float>(input));
      hooks.MemcpyHtoD<float>(out_addr, std::span<const float>(out));
      KernelArgPack args;
      args.PushU64(in_addr);
      args.PushU64(out_addr);
      args.PushU32(c.n);
      const auto result = hooks.LaunchProgramObject(
          ObjectReader{}.LoadProgramObject(ArtifactPath(c.artifact), KernelName(c.kernel)),
          LaunchConfig{.grid_dim_x = c.grid_x, .block_dim_x = c.block_x},
          std::move(args),
          ExecutionMode::Functional,
          "mac500",
          nullptr);
      ASSERT_TRUE(result.ok) << result.error_message;
      hooks.MemcpyDtoH<float>(out_addr, std::span<float>(out));
      ExpectNearVector(out, expected, 1.0e-4f);
      return;
    }
    case KernelKind::Mfma: {
      float out = 0.0f;
      const uint64_t out_addr = hooks.Malloc(sizeof(float));
      hooks.MemcpyHtoD<float>(out_addr, std::span<const float>(&out, 1));
      KernelArgPack args;
      args.PushU64(out_addr);
      const auto result = hooks.LaunchProgramObject(
          ObjectReader{}.LoadProgramObject(ArtifactPath(c.artifact), KernelName(c.kernel)),
          LaunchConfig{.grid_dim_x = 1, .block_dim_x = 64},
          std::move(args),
          ExecutionMode::Functional,
          "mac500",
          nullptr);
      ASSERT_TRUE(result.ok) << result.error_message;
      hooks.MemcpyDtoH<float>(out_addr, std::span<float>(&out, 1));
      EXPECT_NEAR(out, 4.0f, 1.0e-5f);
      return;
    }
  }
}

TEST(ModelRuntimeCtsTest, FullCaseCountMatchesExpectation) {
  EXPECT_EQ(MakeModelRuntimeCasesFull().size() + MakeHipRuntimeAbiCasesFull().size(), 121u);
}

TEST_P(HipCtsRuntimeAbiTest, ExecutesHipOutThroughRegisteredHostFunctionAndValidatesResults) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const HipCtsCase& c = GetParam();
  HipRuntime state;
  state.ResetCompatibilityState();
  static int host_vecadd = 0;
  static int host_fma = 0;
  static int host_bias = 0;
  static int host_by_value = 0;
  static int host_atomic = 0;
  static int host_shared = 0;
  static int host_dynamic_shared = 0;
  static int host_block_reduce = 0;
  static int host_softmax = 0;
  static int host_mfma = 0;
  const void* host_symbol = nullptr;
  switch (c.kernel) {
    case KernelKind::VecAdd:
      host_symbol = &host_vecadd;
      break;
    case KernelKind::FmaLoop:
      host_symbol = &host_fma;
      break;
    case KernelKind::BiasChain:
      host_symbol = &host_bias;
      break;
    case KernelKind::ByValueAggregate:
      host_symbol = &host_by_value;
      break;
    case KernelKind::AtomicCount:
      host_symbol = &host_atomic;
      break;
    case KernelKind::SharedReverse:
      host_symbol = &host_shared;
      break;
    case KernelKind::DynamicSharedSum:
      host_symbol = &host_dynamic_shared;
      break;
    case KernelKind::BlockReduceSum:
      host_symbol = &host_block_reduce;
      break;
    case KernelKind::Softmax:
      host_symbol = &host_softmax;
      break;
    case KernelKind::Mfma:
      host_symbol = &host_mfma;
      break;
  }
  state.RegisterFunction(host_symbol, KernelName(c.kernel));

  switch (c.kernel) {
    case KernelKind::VecAdd: {
      std::vector<float> a(c.n), b(c.n), out(c.n, -1.0f);
      FillFloatPattern(a, b, c.pattern);
      const auto expected = ExpectedVecAdd(c.n, c.pattern);
      void* a_dev = state.AllocateDevice(c.n * sizeof(float));
      void* b_dev = state.AllocateDevice(c.n * sizeof(float));
      void* out_dev = state.AllocateDevice(c.n * sizeof(float));
      state.MemcpyHostToDevice(a_dev, a.data(), c.n * sizeof(float));
      state.MemcpyHostToDevice(b_dev, b.data(), c.n * sizeof(float));
      state.MemcpyHostToDevice(out_dev, out.data(), c.n * sizeof(float));
      uint32_t n_arg = c.n;
      void* args[] = {&a_dev, &b_dev, &out_dev, &n_arg};
      const auto result = state.LaunchExecutableKernel(
          ArtifactPath(c.artifact), host_symbol,
          LaunchConfig{.grid_dim_x = c.grid_x, .block_dim_x = c.block_x}, args);
      ASSERT_TRUE(result.ok) << result.error_message;
      state.MemcpyDeviceToHost(out.data(), out_dev, c.n * sizeof(float));
      ExpectNearVector(out, expected);
      return;
    }
    case KernelKind::FmaLoop: {
      std::vector<float> a(c.n), b(c.n), out(c.n, -1.0f);
      FillFloatPattern(a, b, c.pattern);
      const auto expected = ExpectedFma(c.n, c.iters, c.pattern);
      void* a_dev = state.AllocateDevice(c.n * sizeof(float));
      void* b_dev = state.AllocateDevice(c.n * sizeof(float));
      void* out_dev = state.AllocateDevice(c.n * sizeof(float));
      state.MemcpyHostToDevice(a_dev, a.data(), c.n * sizeof(float));
      state.MemcpyHostToDevice(b_dev, b.data(), c.n * sizeof(float));
      state.MemcpyHostToDevice(out_dev, out.data(), c.n * sizeof(float));
      uint32_t n_arg = c.n;
      uint32_t iters_arg = c.iters;
      void* args[] = {&a_dev, &b_dev, &out_dev, &n_arg, &iters_arg};
      const auto result = state.LaunchExecutableKernel(
          ArtifactPath(c.artifact), host_symbol,
          LaunchConfig{.grid_dim_x = c.grid_x, .block_dim_x = c.block_x}, args);
      ASSERT_TRUE(result.ok) << result.error_message;
      state.MemcpyDeviceToHost(out.data(), out_dev, c.n * sizeof(float));
      ExpectNearVector(out, expected, 1.0e-3f);
      return;
    }
    case KernelKind::BiasChain: {
      std::vector<float> a(c.n), b(c.n), out(c.n, -1.0f);
      FillFloatPattern(a, b, c.pattern);
      const auto expected = ExpectedBias(c.n, c.pattern, c.f0, c.f1, c.f2);
      void* a_dev = state.AllocateDevice(c.n * sizeof(float));
      void* b_dev = state.AllocateDevice(c.n * sizeof(float));
      void* out_dev = state.AllocateDevice(c.n * sizeof(float));
      state.MemcpyHostToDevice(a_dev, a.data(), c.n * sizeof(float));
      state.MemcpyHostToDevice(b_dev, b.data(), c.n * sizeof(float));
      state.MemcpyHostToDevice(out_dev, out.data(), c.n * sizeof(float));
      uint32_t n_arg = c.n;
      float b0 = c.f0, b1 = c.f1, b2 = c.f2;
      void* args[] = {&a_dev, &b_dev, &out_dev, &n_arg, &b0, &b1, &b2};
      const auto result = state.LaunchExecutableKernel(
          ArtifactPath(c.artifact), host_symbol,
          LaunchConfig{.grid_dim_x = c.grid_x, .block_dim_x = c.block_x}, args);
      ASSERT_TRUE(result.ok) << result.error_message;
      state.MemcpyDeviceToHost(out.data(), out_dev, c.n * sizeof(float));
      ExpectNearVector(out, expected, 1.0e-4f);
      return;
    }
    case KernelKind::ByValueAggregate: {
      int32_t out = 0;
      const int32_t expected = ExpectedByValueAggregate();
      void* out_dev = state.AllocateDevice(sizeof(int32_t));
      state.MemcpyHostToDevice(out_dev, &out, sizeof(out));
      struct PayloadHost {
        int32_t x;
        int32_t y;
        int32_t z;
      } payload{5, 9, 17};
      void* args[] = {&out_dev, &payload};
      const auto result = state.LaunchExecutableKernel(
          ArtifactPath(c.artifact),
          host_symbol,
          LaunchConfig{.grid_dim_x = 1, .block_dim_x = 64},
          args);
      ASSERT_TRUE(result.ok) << result.error_message;
      state.MemcpyDeviceToHost(&out, out_dev, sizeof(out));
      EXPECT_EQ(out, expected);
      return;
    }
    case KernelKind::AtomicCount: {
      int32_t out = 0;
      const int32_t expected = ExpectedAtomicCount(c.n);
      void* out_dev = state.AllocateDevice(sizeof(int32_t));
      state.MemcpyHostToDevice(out_dev, &out, sizeof(out));
      uint32_t n_arg = c.n;
      void* args[] = {&out_dev, &n_arg};
      const auto result = state.LaunchExecutableKernel(
          ArtifactPath(c.artifact),
          host_symbol,
          LaunchConfig{.grid_dim_x = c.grid_x, .block_dim_x = c.block_x},
          args);
      ASSERT_TRUE(result.ok) << result.error_message;
      state.MemcpyDeviceToHost(&out, out_dev, sizeof(out));
      EXPECT_EQ(out, expected);
      return;
    }
    case KernelKind::SharedReverse: {
      std::vector<int32_t> out(c.n, -1);
      const auto expected = ExpectedSharedReverse(c.n, c.pattern);
      std::vector<int32_t> input = expected;
      for (uint32_t block = 0; block < c.n / 64u; ++block) {
        const uint32_t base = block * 64u;
        for (uint32_t lane = 0; lane < 64u; ++lane) {
          input[base + lane] = expected[base + (63u - lane)];
        }
      }
      void* in_dev = state.AllocateDevice(c.n * sizeof(int32_t));
      void* out_dev = state.AllocateDevice(c.n * sizeof(int32_t));
      state.MemcpyHostToDevice(in_dev, input.data(), c.n * sizeof(int32_t));
      state.MemcpyHostToDevice(out_dev, out.data(), c.n * sizeof(int32_t));
      uint32_t n_arg = c.n;
      void* args[] = {&in_dev, &out_dev, &n_arg};
      const auto result = state.LaunchExecutableKernel(
          ArtifactPath(c.artifact), host_symbol,
          LaunchConfig{.grid_dim_x = c.grid_x, .block_dim_x = c.block_x}, args);
      ASSERT_TRUE(result.ok) << result.error_message;
      state.MemcpyDeviceToHost(out.data(), out_dev, c.n * sizeof(int32_t));
      EXPECT_EQ(out, expected);
      return;
    }
    case KernelKind::DynamicSharedSum: {
      std::vector<int32_t> out(c.grid_x, 0);
      const auto expected = ExpectedDynamicSharedSum(c.grid_x, c.block_x);
      void* out_dev = state.AllocateDevice(c.grid_x * sizeof(int32_t));
      state.MemcpyHostToDevice(out_dev, out.data(), c.grid_x * sizeof(int32_t));
      void* args[] = {&out_dev};
      const auto result = state.LaunchExecutableKernel(
          ArtifactPath(c.artifact),
          host_symbol,
          LaunchConfig{
              .grid_dim_x = c.grid_x,
              .block_dim_x = c.block_x,
              .shared_memory_bytes = static_cast<uint32_t>(c.block_x * sizeof(int32_t)),
          },
          args);
      ASSERT_TRUE(result.ok) << result.error_message;
      state.MemcpyDeviceToHost(out.data(), out_dev, c.grid_x * sizeof(int32_t));
      EXPECT_EQ(out, expected);
      return;
    }
    case KernelKind::BlockReduceSum: {
      std::vector<float> in(c.n), out(c.grid_x, -1.0f);
      FillFloatPattern(in, in, c.pattern);
      const auto expected = ExpectedBlockReduceSum(c.n, c.pattern, c.grid_x, c.block_x);
      void* in_dev = state.AllocateDevice(c.n * sizeof(float));
      void* out_dev = state.AllocateDevice(c.grid_x * sizeof(float));
      state.MemcpyHostToDevice(in_dev, in.data(), c.n * sizeof(float));
      state.MemcpyHostToDevice(out_dev, out.data(), c.grid_x * sizeof(float));
      uint32_t n_arg = c.n;
      void* args[] = {&in_dev, &out_dev, &n_arg};
      const auto result = state.LaunchExecutableKernel(
          ArtifactPath(c.artifact),
          host_symbol,
          LaunchConfig{.grid_dim_x = c.grid_x, .block_dim_x = c.block_x},
          args);
      ASSERT_TRUE(result.ok) << result.error_message;
      state.MemcpyDeviceToHost(out.data(), out_dev, c.grid_x * sizeof(float));
      ExpectNearVector(out, expected, 1.0e-2f);
      return;
    }
    case KernelKind::Softmax: {
      std::vector<float> input(c.n, static_cast<float>(c.pattern + 1u));
      std::vector<float> out(c.n, 0.0f);
      const auto expected = ExpectedSoftmax(c.n);
      void* in_dev = state.AllocateDevice(c.n * sizeof(float));
      void* out_dev = state.AllocateDevice(c.n * sizeof(float));
      state.MemcpyHostToDevice(in_dev, input.data(), c.n * sizeof(float));
      state.MemcpyHostToDevice(out_dev, out.data(), c.n * sizeof(float));
      uint32_t n_arg = c.n;
      void* args[] = {&in_dev, &out_dev, &n_arg};
      const auto result = state.LaunchExecutableKernel(
          ArtifactPath(c.artifact), host_symbol,
          LaunchConfig{.grid_dim_x = c.grid_x, .block_dim_x = c.block_x}, args);
      ASSERT_TRUE(result.ok) << result.error_message;
      state.MemcpyDeviceToHost(out.data(), out_dev, c.n * sizeof(float));
      ExpectNearVector(out, expected, 1.0e-4f);
      return;
    }
    case KernelKind::Mfma: {
      float out = 0.0f;
      void* out_dev = state.AllocateDevice(sizeof(float));
      state.MemcpyHostToDevice(out_dev, &out, sizeof(float));
      void* args[] = {&out_dev};
      const auto result = state.LaunchExecutableKernel(
          ArtifactPath(c.artifact), host_symbol,
          LaunchConfig{.grid_dim_x = 1, .block_dim_x = 64}, args);
      ASSERT_TRUE(result.ok) << result.error_message;
      state.MemcpyDeviceToHost(&out, out_dev, sizeof(float));
      EXPECT_NEAR(out, 4.0f, 1.0e-5f);
      return;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(ModelRuntimeCTS, HipCtsModelRuntimeTest,
                         ::testing::ValuesIn(MakeModelRuntimeCases()), CaseName);
INSTANTIATE_TEST_SUITE_P(HipRuntimeAbiCTS, HipCtsRuntimeAbiTest,
                         ::testing::ValuesIn(MakeHipRuntimeAbiCases()), CaseName);

TEST(ModelRuntimeCtsTest, CtsCaseCountMatchesExpectation) {
  EXPECT_EQ(MakeModelRuntimeCasesFull().size() + MakeHipRuntimeAbiCasesFull().size(), 121u);
}

}  // namespace
}  // namespace gpu_model
