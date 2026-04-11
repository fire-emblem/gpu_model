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

#include "gpu_model/program/object_reader.h"
#include "gpu_model/runtime/model_runtime.h"
#include "tests/test_utils/hipcc_cache_test_utils.h"

namespace gpu_model {
namespace {

bool HasHipHostToolchain() {
  return std::system("command -v hipcc >/dev/null 2>&1") == 0 &&
         std::system("command -v clang-offload-bundler >/dev/null 2>&1") == 0 &&
         std::system("command -v llvm-objcopy >/dev/null 2>&1") == 0 &&
         std::system("command -v llvm-objdump >/dev/null 2>&1") == 0 &&
         std::system("command -v readelf >/dev/null 2>&1") == 0;
}

std::filesystem::path MakeUniqueTempDir(const std::string& stem) {
  const auto suffix =
      std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
  const auto path = std::filesystem::temp_directory_path() / (stem + "_" + suffix);
  std::filesystem::remove_all(path);
  std::filesystem::create_directories(path);
  return path;
}

// =============================================================================
// HIP Kernel Source Code
// =============================================================================

const char* kCommonHipKernels = R"(
#include <hip/hip_runtime.h>
#include <math.h>

// Simple vector addition: c[i] = a[i] + b[i]
extern "C" __global__ void vecadd(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

// Vector multiplication: c[i] = a[i] * b[i]
extern "C" __global__ void vecmul(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] * b[i];
}

// FMA loop: c[i] = iterative FMA for iters iterations
extern "C" __global__ void fma_loop(const float* a, const float* b, float* c, int n, int iters) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x = a[i];
  float y = b[i];
  float acc = 0.0f;
  for (int k = 0; k < iters; ++k) acc = acc * x + y;
  c[i] = acc;
}

// Scalar add: c[i] = a[i] + scalar
extern "C" __global__ void scalar_add(const float* a, float* c, int n, float scalar) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + scalar;
}

// Shared memory reverse: reverse elements within each block
extern "C" __global__ void shared_reverse(const int* in, int* out, int n) {
  __shared__ int scratch[64];
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + tid;
  if (idx < n) scratch[tid] = in[idx];
  __syncthreads();
  if (idx < n) out[idx] = scratch[blockDim.x - 1 - tid];
}

// Shared memory sum reduction
extern "C" __global__ void shared_sum(const float* in, float* out, int n) {
  __shared__ float scratch[256];
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + tid;

  float sum = 0.0f;
  for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
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

// Dynamic shared memory
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

// Atomic counter
extern "C" __global__ void atomic_count(int* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) atomicAdd(out, 1);
}

// Matrix transpose using shared memory
extern "C" __global__ void transpose_shared(const float* in, float* out, int width, int height) {
  __shared__ float block[16][16];
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    block[threadIdx.y][threadIdx.x] = in[y * width + x];
  }
  __syncthreads();

  x = blockIdx.y * blockDim.y + threadIdx.x;
  y = blockIdx.x * blockDim.x + threadIdx.y;
  if (x < height && y < width) {
    out[y * height + x] = block[threadIdx.x][threadIdx.y];
  }
}

// Softmax row
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

struct BuiltArtifact {
  std::filesystem::path temp_dir;
  std::filesystem::path exe_path;
};

const BuiltArtifact& GetCommonArtifact() {
  static std::once_flag once;
  static BuiltArtifact artifact;
  std::call_once(once, []() {
    artifact.temp_dir = MakeUniqueTempDir("gpu_model_hip_cycle_validation");
    const auto src_path = artifact.temp_dir / "hip_cycle_validation.cpp";
    artifact.exe_path = artifact.temp_dir / "hip_cycle_validation.out";
    {
      std::ofstream out(src_path);
      out << kCommonHipKernels;
    }
    const std::string command =
        test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + artifact.exe_path.string();
    if (std::system(command.c_str()) != 0) {
      throw std::runtime_error("failed to build HIP cycle validation artifact");
    }
  });
  return artifact;
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

void ExpectEqVector(const std::vector<int32_t>& actual,
                    const std::vector<int32_t>& expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < actual.size(); ++i) {
    EXPECT_EQ(actual[i], expected[i]) << "idx=" << i;
  }
}

// =============================================================================
// VecAdd Tests
// =============================================================================

TEST(HipCycleValidationTest, VecAddFunctionalMt) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  ModelRuntime runtime;
  const uint32_t n = 256;

  std::vector<float> a(n), b(n), expected(n);
  for (uint32_t i = 0; i < n; ++i) {
    a[i] = static_cast<float>(i) * 0.5f;
    b[i] = static_cast<float>(100 + i) * 0.25f;
    expected[i] = a[i] + b[i];
  }

  const uint64_t a_addr = runtime.Malloc(n * sizeof(float));
  const uint64_t b_addr = runtime.Malloc(n * sizeof(float));
  const uint64_t out_addr = runtime.Malloc(n * sizeof(float));

  runtime.MemcpyHtoD<float>(a_addr, std::span<const float>(a));
  runtime.MemcpyHtoD<float>(b_addr, std::span<const float>(b));

  KernelArgPack args;
  args.PushU64(a_addr);
  args.PushU64(b_addr);
  args.PushU64(out_addr);
  args.PushU32(n);

  auto result = runtime.LaunchProgramObject(
      ObjectReader{}.LoadProgramObject(GetCommonArtifact().exe_path, "vecadd"),
      LaunchConfig{.grid_dim_x = 4, .block_dim_x = 64},
      std::move(args),
      ExecutionMode::Functional,
      "c500",
      nullptr);
  ASSERT_TRUE(result.ok) << result.error_message;

  std::vector<float> actual(n);
  runtime.MemcpyDtoH<float>(out_addr, std::span<float>(actual));
  ExpectNearVector(actual, expected);
}

TEST(HipCycleValidationTest, VecAddCycle) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  ModelRuntime runtime;
  const uint32_t n = 256;

  std::vector<float> a(n), b(n), expected(n);
  for (uint32_t i = 0; i < n; ++i) {
    a[i] = static_cast<float>(i) * 0.5f;
    b[i] = static_cast<float>(100 + i) * 0.25f;
    expected[i] = a[i] + b[i];
  }

  const uint64_t a_addr = runtime.Malloc(n * sizeof(float));
  const uint64_t b_addr = runtime.Malloc(n * sizeof(float));
  const uint64_t out_addr = runtime.Malloc(n * sizeof(float));

  runtime.MemcpyHtoD<float>(a_addr, std::span<const float>(a));
  runtime.MemcpyHtoD<float>(b_addr, std::span<const float>(b));

  KernelArgPack args;
  args.PushU64(a_addr);
  args.PushU64(b_addr);
  args.PushU64(out_addr);
  args.PushU32(n);

  auto result = runtime.LaunchProgramObject(
      ObjectReader{}.LoadProgramObject(GetCommonArtifact().exe_path, "vecadd"),
      LaunchConfig{.grid_dim_x = 4, .block_dim_x = 64},
      std::move(args),
      ExecutionMode::Cycle,
      "c500",
      nullptr);
  ASSERT_TRUE(result.ok) << result.error_message;

  std::vector<float> actual(n);
  runtime.MemcpyDtoH<float>(out_addr, std::span<float>(actual));
  ExpectNearVector(actual, expected);
}

// =============================================================================
// FMA Loop Tests
// =============================================================================

TEST(HipCycleValidationTest, FmaLoopFunctionalMt) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  ModelRuntime runtime;
  const uint32_t n = 128;
  const uint32_t iters = 5;

  std::vector<float> a(n), b(n), expected(n);
  for (uint32_t i = 0; i < n; ++i) {
    a[i] = 1.0f + 0.001f * static_cast<float>(i);
    b[i] = 2.0f + 0.002f * static_cast<float>(i);
    float acc = 0.0f;
    for (uint32_t k = 0; k < iters; ++k) {
      acc = acc * a[i] + b[i];
    }
    expected[i] = acc;
  }

  const uint64_t a_addr = runtime.Malloc(n * sizeof(float));
  const uint64_t b_addr = runtime.Malloc(n * sizeof(float));
  const uint64_t out_addr = runtime.Malloc(n * sizeof(float));

  runtime.MemcpyHtoD<float>(a_addr, std::span<const float>(a));
  runtime.MemcpyHtoD<float>(b_addr, std::span<const float>(b));

  KernelArgPack args;
  args.PushU64(a_addr);
  args.PushU64(b_addr);
  args.PushU64(out_addr);
  args.PushU32(n);
  args.PushU32(iters);

  auto result = runtime.LaunchProgramObject(
      ObjectReader{}.LoadProgramObject(GetCommonArtifact().exe_path, "fma_loop"),
      LaunchConfig{.grid_dim_x = 2, .block_dim_x = 64},
      std::move(args),
      ExecutionMode::Functional,
      "c500",
      nullptr);
  ASSERT_TRUE(result.ok) << result.error_message;

  std::vector<float> actual(n);
  runtime.MemcpyDtoH<float>(out_addr, std::span<float>(actual));
  ExpectNearVector(actual, expected, 1.0e-3f);
}

TEST(HipCycleValidationTest, FmaLoopCycle) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  ModelRuntime runtime;
  const uint32_t n = 128;
  const uint32_t iters = 5;

  std::vector<float> a(n), b(n), expected(n);
  for (uint32_t i = 0; i < n; ++i) {
    a[i] = 1.0f + 0.001f * static_cast<float>(i);
    b[i] = 2.0f + 0.002f * static_cast<float>(i);
    float acc = 0.0f;
    for (uint32_t k = 0; k < iters; ++k) {
      acc = acc * a[i] + b[i];
    }
    expected[i] = acc;
  }

  const uint64_t a_addr = runtime.Malloc(n * sizeof(float));
  const uint64_t b_addr = runtime.Malloc(n * sizeof(float));
  const uint64_t out_addr = runtime.Malloc(n * sizeof(float));

  runtime.MemcpyHtoD<float>(a_addr, std::span<const float>(a));
  runtime.MemcpyHtoD<float>(b_addr, std::span<const float>(b));

  KernelArgPack args;
  args.PushU64(a_addr);
  args.PushU64(b_addr);
  args.PushU64(out_addr);
  args.PushU32(n);
  args.PushU32(iters);

  auto result = runtime.LaunchProgramObject(
      ObjectReader{}.LoadProgramObject(GetCommonArtifact().exe_path, "fma_loop"),
      LaunchConfig{.grid_dim_x = 2, .block_dim_x = 64},
      std::move(args),
      ExecutionMode::Cycle,
      "c500",
      nullptr);
  ASSERT_TRUE(result.ok) << result.error_message;

  std::vector<float> actual(n);
  runtime.MemcpyDtoH<float>(out_addr, std::span<float>(actual));
  ExpectNearVector(actual, expected, 1.0e-3f);
}

// =============================================================================
// Shared Memory Tests
// =============================================================================

TEST(HipCycleValidationTest, SharedReverseFunctionalMt) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  ModelRuntime runtime;
  const uint32_t n = 128;  // 2 blocks of 64

  std::vector<int32_t> in(n), expected(n);
  for (uint32_t i = 0; i < n; ++i) {
    in[i] = static_cast<int32_t>(i + 1);
  }
  // Reverse within each block
  for (uint32_t block = 0; block < 2; ++block) {
    for (uint32_t lane = 0; lane < 64; ++lane) {
      expected[block * 64 + lane] = in[block * 64 + (63 - lane)];
    }
  }

  const uint64_t in_addr = runtime.Malloc(n * sizeof(int32_t));
  const uint64_t out_addr = runtime.Malloc(n * sizeof(int32_t));

  runtime.MemcpyHtoD<int32_t>(in_addr, std::span<const int32_t>(in));

  KernelArgPack args;
  args.PushU64(in_addr);
  args.PushU64(out_addr);
  args.PushU32(n);

  auto result = runtime.LaunchProgramObject(
      ObjectReader{}.LoadProgramObject(GetCommonArtifact().exe_path, "shared_reverse"),
      LaunchConfig{.grid_dim_x = 2, .block_dim_x = 64},
      std::move(args),
      ExecutionMode::Functional,
      "c500",
      nullptr);
  ASSERT_TRUE(result.ok) << result.error_message;

  std::vector<int32_t> actual(n);
  runtime.MemcpyDtoH<int32_t>(out_addr, std::span<int32_t>(actual));
  ExpectEqVector(actual, expected);
}

TEST(HipCycleValidationTest, SharedReverseCycle) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  ModelRuntime runtime;
  const uint32_t n = 128;  // 2 blocks of 64

  std::vector<int32_t> in(n), expected(n);
  for (uint32_t i = 0; i < n; ++i) {
    in[i] = static_cast<int32_t>(i + 1);
  }
  // Reverse within each block
  for (uint32_t block = 0; block < 2; ++block) {
    for (uint32_t lane = 0; lane < 64; ++lane) {
      expected[block * 64 + lane] = in[block * 64 + (63 - lane)];
    }
  }

  const uint64_t in_addr = runtime.Malloc(n * sizeof(int32_t));
  const uint64_t out_addr = runtime.Malloc(n * sizeof(int32_t));

  runtime.MemcpyHtoD<int32_t>(in_addr, std::span<const int32_t>(in));

  KernelArgPack args;
  args.PushU64(in_addr);
  args.PushU64(out_addr);
  args.PushU32(n);

  auto result = runtime.LaunchProgramObject(
      ObjectReader{}.LoadProgramObject(GetCommonArtifact().exe_path, "shared_reverse"),
      LaunchConfig{.grid_dim_x = 2, .block_dim_x = 64},
      std::move(args),
      ExecutionMode::Cycle,
      "c500",
      nullptr);
  ASSERT_TRUE(result.ok) << result.error_message;

  std::vector<int32_t> actual(n);
  runtime.MemcpyDtoH<int32_t>(out_addr, std::span<int32_t>(actual));
  ExpectEqVector(actual, expected);
}

// =============================================================================
// Dynamic Shared Memory Tests
// =============================================================================

TEST(HipCycleValidationTest, DynamicSharedSumFunctionalMt) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  ModelRuntime runtime;
  const uint32_t grid_x = 4;
  const uint32_t block_x = 64;

  // Expected: each block computes sum(1..64) = 2080
  std::vector<int32_t> expected(grid_x, 64 * 65 / 2);

  const uint64_t out_addr = runtime.Malloc(grid_x * sizeof(int32_t));

  KernelArgPack args;
  args.PushU64(out_addr);

  auto result = runtime.LaunchProgramObject(
      ObjectReader{}.LoadProgramObject(GetCommonArtifact().exe_path, "dynamic_shared_sum"),
      LaunchConfig{.grid_dim_x = grid_x, .block_dim_x = block_x, .shared_memory_bytes = block_x * sizeof(int)},
      std::move(args),
      ExecutionMode::Functional,
      "c500",
      nullptr);
  ASSERT_TRUE(result.ok) << result.error_message;

  std::vector<int32_t> actual(grid_x);
  runtime.MemcpyDtoH<int32_t>(out_addr, std::span<int32_t>(actual));
  ExpectEqVector(actual, expected);
}

TEST(HipCycleValidationTest, DynamicSharedSumCycle) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  ModelRuntime runtime;
  const uint32_t grid_x = 4;
  const uint32_t block_x = 64;

  // Expected: each block computes sum(1..64) = 2080
  std::vector<int32_t> expected(grid_x, 64 * 65 / 2);

  const uint64_t out_addr = runtime.Malloc(grid_x * sizeof(int32_t));

  KernelArgPack args;
  args.PushU64(out_addr);

  auto result = runtime.LaunchProgramObject(
      ObjectReader{}.LoadProgramObject(GetCommonArtifact().exe_path, "dynamic_shared_sum"),
      LaunchConfig{.grid_dim_x = grid_x, .block_dim_x = block_x, .shared_memory_bytes = block_x * sizeof(int)},
      std::move(args),
      ExecutionMode::Cycle,
      "c500",
      nullptr);
  ASSERT_TRUE(result.ok) << result.error_message;

  std::vector<int32_t> actual(grid_x);
  runtime.MemcpyDtoH<int32_t>(out_addr, std::span<int32_t>(actual));
  ExpectEqVector(actual, expected);
}

// =============================================================================
// Atomic Tests
// =============================================================================

TEST(HipCycleValidationTest, AtomicCountFunctionalMt) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  ModelRuntime runtime;
  const uint32_t n = 256;

  const int32_t expected = static_cast<int32_t>(n);

  const uint64_t out_addr = runtime.Malloc(sizeof(int32_t));
  int32_t zero = 0;
  runtime.MemcpyHtoD<int32_t>(out_addr, std::span<const int32_t>(&zero, 1));

  KernelArgPack args;
  args.PushU64(out_addr);
  args.PushU32(n);

  auto result = runtime.LaunchProgramObject(
      ObjectReader{}.LoadProgramObject(GetCommonArtifact().exe_path, "atomic_count"),
      LaunchConfig{.grid_dim_x = 4, .block_dim_x = 64},
      std::move(args),
      ExecutionMode::Functional,
      "c500",
      nullptr);
  ASSERT_TRUE(result.ok) << result.error_message;

  int32_t actual = 0;
  runtime.MemcpyDtoH<int32_t>(out_addr, std::span<int32_t>(&actual, 1));
  EXPECT_EQ(actual, expected);
}

TEST(HipCycleValidationTest, AtomicCountCycle) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  ModelRuntime runtime;
  const uint32_t n = 256;

  const int32_t expected = static_cast<int32_t>(n);

  const uint64_t out_addr = runtime.Malloc(sizeof(int32_t));
  int32_t zero = 0;
  runtime.MemcpyHtoD<int32_t>(out_addr, std::span<const int32_t>(&zero, 1));

  KernelArgPack args;
  args.PushU64(out_addr);
  args.PushU32(n);

  auto result = runtime.LaunchProgramObject(
      ObjectReader{}.LoadProgramObject(GetCommonArtifact().exe_path, "atomic_count"),
      LaunchConfig{.grid_dim_x = 4, .block_dim_x = 64},
      std::move(args),
      ExecutionMode::Cycle,
      "c500",
      nullptr);
  ASSERT_TRUE(result.ok) << result.error_message;

  int32_t actual = 0;
  runtime.MemcpyDtoH<int32_t>(out_addr, std::span<int32_t>(&actual, 1));
  EXPECT_EQ(actual, expected);
}

// =============================================================================
// Multi-Launch Tests with HIP Kernels
// =============================================================================

TEST(HipCycleValidationTest, MultiLaunchVecAddFunctionalMt) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  ModelRuntime runtime;
  const uint32_t n = 128;

  std::vector<float> a(n), b(n), expected(n);
  for (uint32_t i = 0; i < n; ++i) {
    a[i] = static_cast<float>(i);
    b[i] = static_cast<float>(i * 2);
    expected[i] = a[i] + b[i];
  }

  const uint64_t a_addr = runtime.Malloc(n * sizeof(float));
  const uint64_t b_addr = runtime.Malloc(n * sizeof(float));
  const uint64_t out_addr = runtime.Malloc(n * sizeof(float));

  runtime.MemcpyHtoD<float>(a_addr, std::span<const float>(a));
  runtime.MemcpyHtoD<float>(b_addr, std::span<const float>(b));

  // Launch twice
  for (int launch = 0; launch < 2; ++launch) {
    KernelArgPack args;
    args.PushU64(a_addr);
    args.PushU64(b_addr);
    args.PushU64(out_addr);
    args.PushU32(n);

    auto result = runtime.LaunchProgramObject(
        ObjectReader{}.LoadProgramObject(GetCommonArtifact().exe_path, "vecadd"),
        LaunchConfig{.grid_dim_x = 2, .block_dim_x = 64},
        std::move(args),
        ExecutionMode::Functional,
        "c500",
        nullptr);
    ASSERT_TRUE(result.ok) << result.error_message;

    std::vector<float> actual(n);
    runtime.MemcpyDtoH<float>(out_addr, std::span<float>(actual));
    ExpectNearVector(actual, expected);
  }
}

TEST(HipCycleValidationTest, MultiLaunchVecAddCycle) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  ModelRuntime runtime;
  const uint32_t n = 128;

  std::vector<float> a(n), b(n), expected(n);
  for (uint32_t i = 0; i < n; ++i) {
    a[i] = static_cast<float>(i);
    b[i] = static_cast<float>(i * 2);
    expected[i] = a[i] + b[i];
  }

  const uint64_t a_addr = runtime.Malloc(n * sizeof(float));
  const uint64_t b_addr = runtime.Malloc(n * sizeof(float));
  const uint64_t out_addr = runtime.Malloc(n * sizeof(float));

  runtime.MemcpyHtoD<float>(a_addr, std::span<const float>(a));
  runtime.MemcpyHtoD<float>(b_addr, std::span<const float>(b));

  // Launch twice
  for (int launch = 0; launch < 2; ++launch) {
    KernelArgPack args;
    args.PushU64(a_addr);
    args.PushU64(b_addr);
    args.PushU64(out_addr);
    args.PushU32(n);

    auto result = runtime.LaunchProgramObject(
        ObjectReader{}.LoadProgramObject(GetCommonArtifact().exe_path, "vecadd"),
        LaunchConfig{.grid_dim_x = 2, .block_dim_x = 64},
        std::move(args),
        ExecutionMode::Cycle,
        "c500",
        nullptr);
    ASSERT_TRUE(result.ok) << result.error_message;

    std::vector<float> actual(n);
    runtime.MemcpyDtoH<float>(out_addr, std::span<float>(actual));
    ExpectNearVector(actual, expected);
  }
}

// Chained launches: vecadd -> vecmul
TEST(HipCycleValidationTest, ChainedLaunchVecAddThenMulFunctionalMt) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  ModelRuntime runtime;
  const uint32_t n = 128;

  std::vector<float> a(n), b(n), c(n);
  for (uint32_t i = 0; i < n; ++i) {
    a[i] = static_cast<float>(i);
    b[i] = static_cast<float>(i * 2);
    c[i] = static_cast<float>(i + 1);
  }

  const uint64_t a_addr = runtime.Malloc(n * sizeof(float));
  const uint64_t b_addr = runtime.Malloc(n * sizeof(float));
  const uint64_t mid_addr = runtime.Malloc(n * sizeof(float));
  const uint64_t c_addr = runtime.Malloc(n * sizeof(float));
  const uint64_t out_addr = runtime.Malloc(n * sizeof(float));

  runtime.MemcpyHtoD<float>(a_addr, std::span<const float>(a));
  runtime.MemcpyHtoD<float>(b_addr, std::span<const float>(b));
  runtime.MemcpyHtoD<float>(c_addr, std::span<const float>(c));

  // First: mid = a + b
  {
    KernelArgPack args;
    args.PushU64(a_addr);
    args.PushU64(b_addr);
    args.PushU64(mid_addr);
    args.PushU32(n);

    auto result = runtime.LaunchProgramObject(
        ObjectReader{}.LoadProgramObject(GetCommonArtifact().exe_path, "vecadd"),
        LaunchConfig{.grid_dim_x = 2, .block_dim_x = 64},
        std::move(args),
        ExecutionMode::Functional,
        "c500",
        nullptr);
    ASSERT_TRUE(result.ok) << result.error_message;
  }

  // Second: out = mid * c
  {
    KernelArgPack args;
    args.PushU64(mid_addr);
    args.PushU64(c_addr);
    args.PushU64(out_addr);
    args.PushU32(n);

    auto result = runtime.LaunchProgramObject(
        ObjectReader{}.LoadProgramObject(GetCommonArtifact().exe_path, "vecmul"),
        LaunchConfig{.grid_dim_x = 2, .block_dim_x = 64},
        std::move(args),
        ExecutionMode::Functional,
        "c500",
        nullptr);
    ASSERT_TRUE(result.ok) << result.error_message;
  }

  // Verify: out[i] = (a[i] + b[i]) * c[i]
  std::vector<float> actual(n), expected(n);
  for (uint32_t i = 0; i < n; ++i) {
    expected[i] = (a[i] + b[i]) * c[i];
  }
  runtime.MemcpyDtoH<float>(out_addr, std::span<float>(actual));
  ExpectNearVector(actual, expected);
}

TEST(HipCycleValidationTest, ChainedLaunchVecAddThenMulCycle) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  ModelRuntime runtime;
  const uint32_t n = 128;

  std::vector<float> a(n), b(n), c(n);
  for (uint32_t i = 0; i < n; ++i) {
    a[i] = static_cast<float>(i);
    b[i] = static_cast<float>(i * 2);
    c[i] = static_cast<float>(i + 1);
  }

  const uint64_t a_addr = runtime.Malloc(n * sizeof(float));
  const uint64_t b_addr = runtime.Malloc(n * sizeof(float));
  const uint64_t mid_addr = runtime.Malloc(n * sizeof(float));
  const uint64_t c_addr = runtime.Malloc(n * sizeof(float));
  const uint64_t out_addr = runtime.Malloc(n * sizeof(float));

  runtime.MemcpyHtoD<float>(a_addr, std::span<const float>(a));
  runtime.MemcpyHtoD<float>(b_addr, std::span<const float>(b));
  runtime.MemcpyHtoD<float>(c_addr, std::span<const float>(c));

  // First: mid = a + b
  {
    KernelArgPack args;
    args.PushU64(a_addr);
    args.PushU64(b_addr);
    args.PushU64(mid_addr);
    args.PushU32(n);

    auto result = runtime.LaunchProgramObject(
        ObjectReader{}.LoadProgramObject(GetCommonArtifact().exe_path, "vecadd"),
        LaunchConfig{.grid_dim_x = 2, .block_dim_x = 64},
        std::move(args),
        ExecutionMode::Cycle,
        "c500",
        nullptr);
    ASSERT_TRUE(result.ok) << result.error_message;
  }

  // Second: out = mid * c
  {
    KernelArgPack args;
    args.PushU64(mid_addr);
    args.PushU64(c_addr);
    args.PushU64(out_addr);
    args.PushU32(n);

    auto result = runtime.LaunchProgramObject(
        ObjectReader{}.LoadProgramObject(GetCommonArtifact().exe_path, "vecmul"),
        LaunchConfig{.grid_dim_x = 2, .block_dim_x = 64},
        std::move(args),
        ExecutionMode::Cycle,
        "c500",
        nullptr);
    ASSERT_TRUE(result.ok) << result.error_message;
  }

  // Verify: out[i] = (a[i] + b[i]) * c[i]
  std::vector<float> actual(n), expected(n);
  for (uint32_t i = 0; i < n; ++i) {
    expected[i] = (a[i] + b[i]) * c[i];
  }
  runtime.MemcpyDtoH<float>(out_addr, std::span<float>(actual));
  ExpectNearVector(actual, expected);
}

// =============================================================================
// Cross-Mode Consistency Tests
// =============================================================================

TEST(HipCycleValidationTest, VecAddMtAndCycleProduceIdenticalResults) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const uint32_t n = 256;

  std::vector<float> a(n), b(n);
  for (uint32_t i = 0; i < n; ++i) {
    a[i] = static_cast<float>(i) * 0.5f;
    b[i] = static_cast<float>(100 + i) * 0.25f;
  }

  // MT mode
  ModelRuntime mt_runtime;
  const uint64_t mt_a = mt_runtime.Malloc(n * sizeof(float));
  const uint64_t mt_b = mt_runtime.Malloc(n * sizeof(float));
  const uint64_t mt_out = mt_runtime.Malloc(n * sizeof(float));
  mt_runtime.MemcpyHtoD<float>(mt_a, std::span<const float>(a));
  mt_runtime.MemcpyHtoD<float>(mt_b, std::span<const float>(b));

  {
    KernelArgPack args;
    args.PushU64(mt_a);
    args.PushU64(mt_b);
    args.PushU64(mt_out);
    args.PushU32(n);
    auto result = mt_runtime.LaunchProgramObject(
        ObjectReader{}.LoadProgramObject(GetCommonArtifact().exe_path, "vecadd"),
        LaunchConfig{.grid_dim_x = 4, .block_dim_x = 64},
        std::move(args),
        ExecutionMode::Functional,
        "c500",
        nullptr);
    ASSERT_TRUE(result.ok) << result.error_message;
  }

  // Cycle mode
  ModelRuntime cycle_runtime;
  const uint64_t cycle_a = cycle_runtime.Malloc(n * sizeof(float));
  const uint64_t cycle_b = cycle_runtime.Malloc(n * sizeof(float));
  const uint64_t cycle_out = cycle_runtime.Malloc(n * sizeof(float));
  cycle_runtime.MemcpyHtoD<float>(cycle_a, std::span<const float>(a));
  cycle_runtime.MemcpyHtoD<float>(cycle_b, std::span<const float>(b));

  {
    KernelArgPack args;
    args.PushU64(cycle_a);
    args.PushU64(cycle_b);
    args.PushU64(cycle_out);
    args.PushU32(n);
    auto result = cycle_runtime.LaunchProgramObject(
        ObjectReader{}.LoadProgramObject(GetCommonArtifact().exe_path, "vecadd"),
        LaunchConfig{.grid_dim_x = 4, .block_dim_x = 64},
        std::move(args),
        ExecutionMode::Cycle,
        "c500",
        nullptr);
    ASSERT_TRUE(result.ok) << result.error_message;
  }

  // Compare results
  std::vector<float> mt_result(n), cycle_result(n);
  mt_runtime.MemcpyDtoH<float>(mt_out, std::span<float>(mt_result));
  cycle_runtime.MemcpyDtoH<float>(cycle_out, std::span<float>(cycle_result));
  ExpectNearVector(mt_result, cycle_result);
}

}  // namespace
}  // namespace gpu_model
