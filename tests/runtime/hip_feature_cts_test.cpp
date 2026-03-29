#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <span>
#include <string>
#include <tuple>
#include <vector>

#include "gpu_model/runtime/hip_interposer_state.h"
#include "gpu_model/runtime/runtime_hooks.h"
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

enum class FeatureKernelKind {
  SaxpyAffine,
  GridStrideAffine,
  ClampRelu,
  Stencil1D,
  BlockReduceSum,
};

struct HipFeatureCase {
  std::string name;
  FeatureKernelKind kernel = FeatureKernelKind::SaxpyAffine;
  uint32_t grid_x = 1;
  uint32_t block_x = 1;
  uint32_t n = 1;
  uint32_t pattern = 0;
  float f0 = 0.0f;
  float f1 = 0.0f;
};

struct BuiltArtifact {
  std::filesystem::path temp_dir;
  std::filesystem::path exe_path;
};

const BuiltArtifact& FeatureArtifact() {
  static std::once_flag once;
  static BuiltArtifact artifact;
  std::call_once(once, []() {
    artifact.temp_dir = MakeUniqueTempDir("gpu_model_hip_feature_cts");
    const auto src_path = artifact.temp_dir / "hip_feature_cts.cpp";
    artifact.exe_path = artifact.temp_dir / "hip_feature_cts.out";
    {
      std::ofstream out(src_path);
      out << R"(
#include <hip/hip_runtime.h>

extern "C" __global__ void saxpy_affine(const float* in, float* out, float alpha, float beta, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = alpha * in[i] + beta;
}

extern "C" __global__ void grid_stride_affine(const float* in, float* out, float alpha, float beta, int n) {
  int stride = blockDim.x * gridDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    out[i] = alpha * in[i] + beta;
  }
}

extern "C" __global__ void clamp_relu(const float* in, float* out, float bias, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float x = in[i] + bias;
    out[i] = x > 0.0f ? x : 0.0f;
  }
}

extern "C" __global__ void stencil1d(const float* in, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float left = i > 0 ? in[i - 1] : 0.0f;
    float center = in[i];
    float right = i + 1 < n ? in[i + 1] : 0.0f;
    out[i] = left + 2.0f * center + right;
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

int main() { return 0; }
)";
    }
    const std::string command = "hipcc " + src_path.string() + " -o " + artifact.exe_path.string();
    if (std::system(command.c_str()) != 0) {
      throw std::runtime_error("failed to build HIP feature CTS artifact");
    }
  });
  return artifact;
}

const char* KernelName(FeatureKernelKind kernel) {
  switch (kernel) {
    case FeatureKernelKind::SaxpyAffine:
      return "saxpy_affine";
    case FeatureKernelKind::GridStrideAffine:
      return "grid_stride_affine";
    case FeatureKernelKind::ClampRelu:
      return "clamp_relu";
    case FeatureKernelKind::Stencil1D:
      return "stencil1d";
    case FeatureKernelKind::BlockReduceSum:
      return "block_reduce_sum";
  }
  return "unknown";
}

void FillInputPattern(std::vector<float>& values, uint32_t pattern) {
  for (uint32_t i = 0; i < values.size(); ++i) {
    switch (pattern % 5u) {
      case 0:
        values[i] = 0.5f * static_cast<float>(i);
        break;
      case 1:
        values[i] = -4.0f + 0.25f * static_cast<float>(i % 37u);
        break;
      case 2:
        values[i] = static_cast<float>((i % 11u) - 5);
        break;
      case 3:
        values[i] = 1.0f + 0.001f * static_cast<float>(i);
        break;
      default:
        values[i] = static_cast<float>((i * 7u) % 23u) - 9.0f;
        break;
    }
  }
}

std::vector<float> ExpectedSaxpy(uint32_t n, uint32_t pattern, float alpha, float beta) {
  std::vector<float> in(n), out(n);
  FillInputPattern(in, pattern);
  for (uint32_t i = 0; i < n; ++i) {
    out[i] = alpha * in[i] + beta;
  }
  return out;
}

std::vector<float> ExpectedClampRelu(uint32_t n, uint32_t pattern, float bias) {
  std::vector<float> in(n), out(n);
  FillInputPattern(in, pattern);
  for (uint32_t i = 0; i < n; ++i) {
    const float x = in[i] + bias;
    out[i] = x > 0.0f ? x : 0.0f;
  }
  return out;
}

std::vector<float> ExpectedStencil1D(uint32_t n, uint32_t pattern) {
  std::vector<float> in(n), out(n);
  FillInputPattern(in, pattern);
  for (uint32_t i = 0; i < n; ++i) {
    const float left = i > 0 ? in[i - 1] : 0.0f;
    const float center = in[i];
    const float right = i + 1 < n ? in[i + 1] : 0.0f;
    out[i] = left + 2.0f * center + right;
  }
  return out;
}

std::vector<float> ExpectedBlockReduceSum(uint32_t n,
                                          uint32_t pattern,
                                          uint32_t grid_x,
                                          uint32_t block_x) {
  std::vector<float> in(n);
  FillInputPattern(in, pattern);
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

std::vector<HipFeatureCase> MakeRuntimeHooksFeatureCasesFull() {
  std::vector<HipFeatureCase> cases;
  uint32_t id = 0;
  const auto add = [&](HipFeatureCase c) {
    c.name = "frh_" + std::to_string(id++) + "_" + KernelName(c.kernel);
    cases.push_back(std::move(c));
  };

  for (const auto& [n, block, alpha, beta] :
       std::vector<std::tuple<uint32_t, uint32_t, float, float>>{
           {1, 1, 1.0f, 0.0f},       {31, 32, 2.0f, -1.0f},    {64, 64, 0.5f, 3.0f},
           {65, 64, -1.25f, 0.5f},   {96, 96, 4.0f, -2.0f},    {127, 64, 1.5f, 1.0f},
           {128, 128, 2.5f, -3.5f},  {129, 128, -0.75f, 0.25f}, {191, 64, 3.0f, 2.0f},
           {192, 64, -2.0f, 5.0f},   {255, 128, 0.25f, -4.0f}, {256, 128, 1.75f, 0.75f},
           {257, 128, -1.5f, -0.5f}, {511, 256, 0.875f, 1.125f}, {1024, 256, -0.5f, 2.5f},
           {30720, 1024, 1.25f, -1.75f}}) {
    add(HipFeatureCase{
        .name = {},
        .kernel = FeatureKernelKind::SaxpyAffine,
        .grid_x = (n + block - 1) / block,
        .block_x = block,
        .n = n,
        .pattern = id % 5u,
        .f0 = alpha,
        .f1 = beta,
    });
  }

  for (const auto& [n, grid, block, alpha, beta] :
       std::vector<std::tuple<uint32_t, uint32_t, uint32_t, float, float>>{
           {1, 1, 1, 1.0f, 0.0f},      {33, 1, 32, 2.0f, 1.0f},      {64, 1, 64, -1.0f, 0.5f},
           {65, 2, 64, 1.5f, -0.5f},   {127, 2, 64, 0.25f, 3.0f},    {128, 2, 64, -2.0f, 4.0f},
           {129, 2, 128, 1.0f, -1.0f}, {255, 3, 128, 2.5f, 2.0f},    {256, 4, 64, -0.75f, 0.25f},
           {257, 4, 64, 0.875f, -3.0f}, {511, 4, 128, 1.25f, 1.5f},  {512, 8, 64, -1.25f, 0.75f},
           {777, 6, 128, 0.5f, -2.5f}, {1024, 8, 128, 1.75f, 2.25f}, {2048, 8, 256, -0.5f, 0.0f},
           {30720, 30, 256, 0.625f, -1.125f}}) {
    add(HipFeatureCase{
        .name = {},
        .kernel = FeatureKernelKind::GridStrideAffine,
        .grid_x = grid,
        .block_x = block,
        .n = n,
        .pattern = id % 5u,
        .f0 = alpha,
        .f1 = beta,
    });
  }

  for (const auto& [n, block, bias] :
       std::vector<std::tuple<uint32_t, uint32_t, float>>{
           {1, 1, 0.0f},        {32, 32, 1.0f},      {60, 60, -1.0f},    {64, 64, 2.5f},
           {65, 65, -2.5f},     {96, 96, 0.5f},      {127, 64, -0.75f},  {128, 128, 3.0f},
           {129, 128, -3.0f},   {191, 64, 1.25f},    {192, 64, -1.25f},  {255, 128, 4.0f},
           {256, 128, -4.0f},   {257, 128, 0.25f},   {1024, 256, -0.5f}, {30720, 1024, 1.5f}}) {
    add(HipFeatureCase{
        .name = {},
        .kernel = FeatureKernelKind::ClampRelu,
        .grid_x = (n + block - 1) / block,
        .block_x = block,
        .n = n,
        .pattern = id % 5u,
        .f0 = bias,
    });
  }

  for (const auto& [n, block] :
       std::vector<std::pair<uint32_t, uint32_t>>{{1, 1},      {32, 32},    {60, 60},    {64, 64},
                                                  {65, 65},    {96, 96},    {127, 64},   {128, 128},
                                                  {129, 128},  {191, 64},   {192, 64},   {255, 128},
                                                  {256, 128},  {257, 128},  {1024, 256}, {30720, 1024}}) {
    add(HipFeatureCase{
        .name = {},
        .kernel = FeatureKernelKind::Stencil1D,
        .grid_x = (n + block - 1) / block,
        .block_x = block,
        .n = n,
        .pattern = id % 5u,
    });
  }

  for (const auto& [n, grid, block] :
       std::vector<std::tuple<uint32_t, uint32_t, uint32_t>>{
           {64, 1, 64},     {65, 1, 64},     {127, 2, 64},   {128, 2, 64},
           {129, 2, 64},    {255, 2, 128},   {256, 2, 128},  {257, 2, 128},
           {511, 4, 128},   {512, 4, 128},   {777, 3, 256},  {1024, 4, 256},
           {1536, 6, 256},  {2048, 8, 256},  {4097, 8, 256}, {30720, 16, 256}}) {
    add(HipFeatureCase{
        .name = {},
        .kernel = FeatureKernelKind::BlockReduceSum,
        .grid_x = grid,
        .block_x = block,
        .n = n,
        .pattern = id % 5u,
    });
  }

  EXPECT_EQ(cases.size(), 80u);
  return cases;
}

std::vector<HipFeatureCase> MakeRuntimeHooksFeatureCasesQuick() {
  return test::SelectIndexedCases(MakeRuntimeHooksFeatureCasesFull(),
                                  {0, 8, 15, 16, 24, 31, 32, 40, 47, 48, 56, 63, 64, 72, 79});
}

std::vector<HipFeatureCase> MakeRuntimeHooksFeatureCases() {
  return test::FullTestMatrixEnabled() ? MakeRuntimeHooksFeatureCasesFull()
                                       : MakeRuntimeHooksFeatureCasesQuick();
}

std::vector<HipFeatureCase> MakeInterposerFeatureCasesFull() {
  std::vector<HipFeatureCase> cases;
  uint32_t id = 0;
  const auto add = [&](HipFeatureCase c) {
    c.name = "fis_" + std::to_string(id++) + "_" + KernelName(c.kernel);
    cases.push_back(std::move(c));
  };

  add(HipFeatureCase{.name = {},
                     .kernel = FeatureKernelKind::SaxpyAffine,
                     .grid_x = 3,
                     .block_x = 128,
                     .n = 257,
                     .pattern = 1,
                     .f0 = 1.25f,
                     .f1 = -0.5f});
  add(HipFeatureCase{.name = {},
                     .kernel = FeatureKernelKind::SaxpyAffine,
                     .grid_x = 30,
                     .block_x = 1024,
                     .n = 30720,
                     .pattern = 3,
                     .f0 = -0.75f,
                     .f1 = 2.0f});
  add(HipFeatureCase{.name = {},
                     .kernel = FeatureKernelKind::SaxpyAffine,
                     .grid_x = 1,
                     .block_x = 64,
                     .n = 64,
                     .pattern = 4,
                     .f0 = 0.5f,
                     .f1 = 1.5f});
  add(HipFeatureCase{.name = {},
                     .kernel = FeatureKernelKind::SaxpyAffine,
                     .grid_x = 2,
                     .block_x = 64,
                     .n = 65,
                     .pattern = 2,
                     .f0 = 2.0f,
                     .f1 = -1.0f});

  add(HipFeatureCase{.name = {},
                     .kernel = FeatureKernelKind::GridStrideAffine,
                     .grid_x = 1,
                     .block_x = 64,
                     .n = 257,
                     .pattern = 0,
                     .f0 = 1.0f,
                     .f1 = 0.25f});
  add(HipFeatureCase{.name = {},
                     .kernel = FeatureKernelKind::GridStrideAffine,
                     .grid_x = 4,
                     .block_x = 128,
                     .n = 1024,
                     .pattern = 1,
                     .f0 = -1.25f,
                     .f1 = 2.0f});
  add(HipFeatureCase{.name = {},
                     .kernel = FeatureKernelKind::GridStrideAffine,
                     .grid_x = 8,
                     .block_x = 256,
                     .n = 2048,
                     .pattern = 2,
                     .f0 = 0.75f,
                     .f1 = -3.0f});
  add(HipFeatureCase{.name = {},
                     .kernel = FeatureKernelKind::GridStrideAffine,
                     .grid_x = 30,
                     .block_x = 256,
                     .n = 30720,
                     .pattern = 3,
                     .f0 = 1.5f,
                     .f1 = -0.5f});

  add(HipFeatureCase{.name = {},
                     .kernel = FeatureKernelKind::ClampRelu,
                     .grid_x = 1,
                     .block_x = 64,
                     .n = 64,
                     .pattern = 0,
                     .f0 = -1.0f});
  add(HipFeatureCase{.name = {},
                     .kernel = FeatureKernelKind::ClampRelu,
                     .grid_x = 3,
                     .block_x = 128,
                     .n = 257,
                     .pattern = 1,
                     .f0 = 1.5f});
  add(HipFeatureCase{.name = {},
                     .kernel = FeatureKernelKind::ClampRelu,
                     .grid_x = 4,
                     .block_x = 256,
                     .n = 1024,
                     .pattern = 2,
                     .f0 = -2.5f});
  add(HipFeatureCase{.name = {},
                     .kernel = FeatureKernelKind::ClampRelu,
                     .grid_x = 30,
                     .block_x = 1024,
                     .n = 30720,
                     .pattern = 4,
                     .f0 = 0.5f});

  add(HipFeatureCase{.name = {},
                     .kernel = FeatureKernelKind::Stencil1D,
                     .grid_x = 1,
                     .block_x = 64,
                     .n = 64,
                     .pattern = 0});
  add(HipFeatureCase{.name = {},
                     .kernel = FeatureKernelKind::Stencil1D,
                     .grid_x = 3,
                     .block_x = 128,
                     .n = 257,
                     .pattern = 1});
  add(HipFeatureCase{.name = {},
                     .kernel = FeatureKernelKind::Stencil1D,
                     .grid_x = 4,
                     .block_x = 256,
                     .n = 1024,
                     .pattern = 3});
  add(HipFeatureCase{.name = {},
                     .kernel = FeatureKernelKind::Stencil1D,
                     .grid_x = 30,
                     .block_x = 1024,
                     .n = 30720,
                     .pattern = 4});

  add(HipFeatureCase{.name = {},
                     .kernel = FeatureKernelKind::BlockReduceSum,
                     .grid_x = 1,
                     .block_x = 64,
                     .n = 64,
                     .pattern = 0});
  add(HipFeatureCase{.name = {},
                     .kernel = FeatureKernelKind::BlockReduceSum,
                     .grid_x = 2,
                     .block_x = 128,
                     .n = 257,
                     .pattern = 1});
  add(HipFeatureCase{.name = {},
                     .kernel = FeatureKernelKind::BlockReduceSum,
                     .grid_x = 4,
                     .block_x = 256,
                     .n = 1024,
                     .pattern = 2});
  add(HipFeatureCase{.name = {},
                     .kernel = FeatureKernelKind::BlockReduceSum,
                     .grid_x = 16,
                     .block_x = 256,
                     .n = 30720,
                     .pattern = 3});

  EXPECT_EQ(cases.size(), 20u);
  return cases;
}

std::vector<HipFeatureCase> MakeInterposerFeatureCasesQuick() {
  return test::SelectIndexedCases(MakeInterposerFeatureCasesFull(),
                                  {0, 1, 4, 7, 8, 11, 12, 15, 16, 19});
}

std::vector<HipFeatureCase> MakeInterposerFeatureCases() {
  return test::FullTestMatrixEnabled() ? MakeInterposerFeatureCasesFull()
                                       : MakeInterposerFeatureCasesQuick();
}

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

class HipFeatureRuntimeHooksTest : public ::testing::TestWithParam<HipFeatureCase> {};
class HipFeatureInterposerStateTest : public ::testing::TestWithParam<HipFeatureCase> {};

std::string FeatureCaseName(const ::testing::TestParamInfo<HipFeatureCase>& info) {
  return info.param.name;
}

TEST(RuntimeHooksTest, FeatureCtsCaseCountIsOneHundred) {
  EXPECT_EQ(MakeRuntimeHooksFeatureCasesFull().size() + MakeInterposerFeatureCasesFull().size(),
            100u);
}

TEST_P(HipFeatureRuntimeHooksTest, ExecutesFeatureKernelAndValidatesResults) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const HipFeatureCase& c = GetParam();
  RuntimeHooks hooks;

  switch (c.kernel) {
    case FeatureKernelKind::SaxpyAffine:
    case FeatureKernelKind::GridStrideAffine:
    case FeatureKernelKind::ClampRelu:
    case FeatureKernelKind::Stencil1D: {
      std::vector<float> in(c.n), out(c.n, -1.0f), expected;
      FillInputPattern(in, c.pattern);
      if (c.kernel == FeatureKernelKind::SaxpyAffine ||
          c.kernel == FeatureKernelKind::GridStrideAffine) {
        expected = ExpectedSaxpy(c.n, c.pattern, c.f0, c.f1);
      } else if (c.kernel == FeatureKernelKind::ClampRelu) {
        expected = ExpectedClampRelu(c.n, c.pattern, c.f0);
      } else {
        expected = ExpectedStencil1D(c.n, c.pattern);
      }
      const uint64_t in_addr = hooks.Malloc(c.n * sizeof(float));
      const uint64_t out_addr = hooks.Malloc(c.n * sizeof(float));
      hooks.MemcpyHtoD<float>(in_addr, std::span<const float>(in));
      hooks.MemcpyHtoD<float>(out_addr, std::span<const float>(out));

      KernelArgPack args;
      args.PushU64(in_addr);
      args.PushU64(out_addr);
      if (c.kernel == FeatureKernelKind::SaxpyAffine ||
          c.kernel == FeatureKernelKind::GridStrideAffine) {
        args.PushF32(c.f0);
        args.PushF32(c.f1);
      } else if (c.kernel == FeatureKernelKind::ClampRelu) {
        args.PushF32(c.f0);
      }
      args.PushU32(c.n);

      const auto result = hooks.LaunchAmdgpuObject(
          FeatureArtifact().exe_path,
          LaunchConfig{.grid_dim_x = c.grid_x, .block_dim_x = c.block_x},
          std::move(args), ExecutionMode::Functional, "c500", nullptr, KernelName(c.kernel));
      ASSERT_TRUE(result.ok) << result.error_message;
      hooks.MemcpyDtoH<float>(out_addr, std::span<float>(out));
      ExpectNearVector(out, expected, 1.0e-4f);
      return;
    }
    case FeatureKernelKind::BlockReduceSum: {
      std::vector<float> in(c.n);
      FillInputPattern(in, c.pattern);
      std::vector<float> out(c.grid_x, -1.0f);
      const auto expected = ExpectedBlockReduceSum(c.n, c.pattern, c.grid_x, c.block_x);

      const uint64_t in_addr = hooks.Malloc(c.n * sizeof(float));
      const uint64_t out_addr = hooks.Malloc(c.grid_x * sizeof(float));
      hooks.MemcpyHtoD<float>(in_addr, std::span<const float>(in));
      hooks.MemcpyHtoD<float>(out_addr, std::span<const float>(out));

      KernelArgPack args;
      args.PushU64(in_addr);
      args.PushU64(out_addr);
      args.PushU32(c.n);

      const auto result = hooks.LaunchAmdgpuObject(
          FeatureArtifact().exe_path,
          LaunchConfig{.grid_dim_x = c.grid_x, .block_dim_x = c.block_x},
          std::move(args), ExecutionMode::Functional, "c500", nullptr, KernelName(c.kernel));
      ASSERT_TRUE(result.ok) << result.error_message;
      hooks.MemcpyDtoH<float>(out_addr, std::span<float>(out));
      ExpectNearVector(out, expected, 1.0e-2f);
      return;
    }
  }
}

TEST_P(HipFeatureInterposerStateTest, ExecutesFeatureKernelThroughRegisteredHostFunction) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const HipFeatureCase& c = GetParam();
  auto& state = HipInterposerState::Instance();
  state.ResetForTest();

  static int host_saxpy = 0;
  static int host_grid_stride = 0;
  static int host_clamp = 0;
  static int host_stencil = 0;
  static int host_reduce = 0;
  const void* host_symbol = nullptr;
  switch (c.kernel) {
    case FeatureKernelKind::SaxpyAffine:
      host_symbol = &host_saxpy;
      break;
    case FeatureKernelKind::GridStrideAffine:
      host_symbol = &host_grid_stride;
      break;
    case FeatureKernelKind::ClampRelu:
      host_symbol = &host_clamp;
      break;
    case FeatureKernelKind::Stencil1D:
      host_symbol = &host_stencil;
      break;
    case FeatureKernelKind::BlockReduceSum:
      host_symbol = &host_reduce;
      break;
  }
  state.RegisterFunction(host_symbol, KernelName(c.kernel));

  switch (c.kernel) {
    case FeatureKernelKind::SaxpyAffine:
    case FeatureKernelKind::GridStrideAffine:
    case FeatureKernelKind::ClampRelu:
    case FeatureKernelKind::Stencil1D: {
      std::vector<float> in(c.n), out(c.n, -1.0f), expected;
      FillInputPattern(in, c.pattern);
      if (c.kernel == FeatureKernelKind::SaxpyAffine ||
          c.kernel == FeatureKernelKind::GridStrideAffine) {
        expected = ExpectedSaxpy(c.n, c.pattern, c.f0, c.f1);
      } else if (c.kernel == FeatureKernelKind::ClampRelu) {
        expected = ExpectedClampRelu(c.n, c.pattern, c.f0);
      } else {
        expected = ExpectedStencil1D(c.n, c.pattern);
      }

      void* in_dev = state.AllocateDevice(c.n * sizeof(float));
      void* out_dev = state.AllocateDevice(c.n * sizeof(float));
      state.MemcpyHostToDevice(in_dev, in.data(), c.n * sizeof(float));
      state.MemcpyHostToDevice(out_dev, out.data(), c.n * sizeof(float));

      uint32_t n_arg = c.n;
      float f0 = c.f0;
      float f1 = c.f1;
      void* args[5] = {&in_dev, &out_dev, &f0, &f1, &n_arg};
      if (c.kernel == FeatureKernelKind::ClampRelu) {
        void* clamp_args[] = {&in_dev, &out_dev, &f0, &n_arg};
        const auto result = state.LaunchExecutableKernel(
            FeatureArtifact().exe_path, host_symbol,
            LaunchConfig{.grid_dim_x = c.grid_x, .block_dim_x = c.block_x}, clamp_args);
        ASSERT_TRUE(result.ok) << result.error_message;
      } else if (c.kernel == FeatureKernelKind::Stencil1D) {
        void* stencil_args[] = {&in_dev, &out_dev, &n_arg};
        const auto result = state.LaunchExecutableKernel(
            FeatureArtifact().exe_path, host_symbol,
            LaunchConfig{.grid_dim_x = c.grid_x, .block_dim_x = c.block_x}, stencil_args);
        ASSERT_TRUE(result.ok) << result.error_message;
      } else {
        const auto result = state.LaunchExecutableKernel(
            FeatureArtifact().exe_path, host_symbol,
            LaunchConfig{.grid_dim_x = c.grid_x, .block_dim_x = c.block_x}, args);
        ASSERT_TRUE(result.ok) << result.error_message;
      }

      state.MemcpyDeviceToHost(out.data(), out_dev, c.n * sizeof(float));
      ExpectNearVector(out, expected, 1.0e-4f);
      return;
    }
    case FeatureKernelKind::BlockReduceSum: {
      std::vector<float> in(c.n);
      FillInputPattern(in, c.pattern);
      std::vector<float> out(c.grid_x, -1.0f);
      const auto expected = ExpectedBlockReduceSum(c.n, c.pattern, c.grid_x, c.block_x);

      void* in_dev = state.AllocateDevice(c.n * sizeof(float));
      void* out_dev = state.AllocateDevice(c.grid_x * sizeof(float));
      state.MemcpyHostToDevice(in_dev, in.data(), c.n * sizeof(float));
      state.MemcpyHostToDevice(out_dev, out.data(), c.grid_x * sizeof(float));

      uint32_t n_arg = c.n;
      void* args[] = {&in_dev, &out_dev, &n_arg};
      const auto result = state.LaunchExecutableKernel(
          FeatureArtifact().exe_path, host_symbol,
          LaunchConfig{.grid_dim_x = c.grid_x, .block_dim_x = c.block_x}, args);
      ASSERT_TRUE(result.ok) << result.error_message;

      state.MemcpyDeviceToHost(out.data(), out_dev, c.grid_x * sizeof(float));
      ExpectNearVector(out, expected, 1.0e-2f);
      return;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(RuntimeHooksFeatureCTS,
                         HipFeatureRuntimeHooksTest,
                         ::testing::ValuesIn(MakeRuntimeHooksFeatureCases()),
                         FeatureCaseName);

INSTANTIATE_TEST_SUITE_P(InterposerFeatureCTS,
                         HipFeatureInterposerStateTest,
                         ::testing::ValuesIn(MakeInterposerFeatureCases()),
                         FeatureCaseName);

}  // namespace
}  // namespace gpu_model
