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

#include "debug/trace/sink.h"
#include "program/program_object/object_reader.h"
#include "runtime/model_runtime/core/model_runtime.h"
#include "tests/test_utils/hipcc_cache_test_utils.h"

namespace gpu_model {
namespace {

bool HasHipHostToolchain() {
  return test_utils::HasHipHostToolchain();
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

// Global atomic max
extern "C" __global__ void global_atomic_max(int* out, int n, int val) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) atomicMax(out, val + i);
}

// Global atomic min
extern "C" __global__ void global_atomic_min(int* out, int n, int val) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) atomicMin(out, val - i);
}

// Global atomic exchange
extern "C" __global__ void global_atomic_exch(int* out, int n, int val) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) atomicExch(out + i, val);
}

// Shared memory atomic add
extern "C" __global__ void shared_atomic_add(int* out, int n) {
  __shared__ int scratch[64];
  int tid = threadIdx.x;

  if (tid == 0) scratch[0] = 0;
  __syncthreads();

  if (tid < n) atomicAdd(&scratch[0], tid + 1);
  __syncthreads();

  if (tid == 0) out[blockIdx.x] = scratch[0];
}

// Shared memory atomic exchange
extern "C" __global__ void shared_atomic_exch(int* out, int n) {
  __shared__ int scratch[64];
  int tid = threadIdx.x;

  // Initialize
  if (tid < n) scratch[tid] = -1;
  __syncthreads();

  // Each thread writes to its slot
  if (tid < n) atomicExch(&scratch[tid], tid * 10);
  __syncthreads();

  // Read back and store
  if (tid < n) out[tid] = scratch[tid];
}

// Private memory test - each thread uses private array
extern "C" __global__ void private_array_sum(const int* in, int* out, int n) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + tid;

  // Private array (compiler will put in registers or scratch)
  int priv[8];
  for (int i = 0; i < 8; ++i) {
    priv[i] = idx * 8 + i;
  }

  int sum = 0;
  for (int i = 0; i < 8; ++i) {
    sum += priv[i];
  }

  if (idx < n) out[idx] = sum;
}

// Private memory with load/store pattern
extern "C" __global__ void private_load_store(const float* in, float* out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;

  // Load to private
  float priv = in[idx];

  // Compute in private
  priv = priv * 2.0f + 1.0f;

  // Store from private
  out[idx] = priv;
}

// Histogram using shared atomics then global write
extern "C" __global__ void histogram_shared(const int* data, int* hist, int n, int bins) {
  __shared__ int local_hist[64];
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + tid;

  // Initialize local histogram
  if (tid < bins) local_hist[tid] = 0;
  __syncthreads();

  // Count in shared memory
  if (idx < n) {
    int bin = data[idx] % bins;
    atomicAdd(&local_hist[bin], 1);
  }
  __syncthreads();

  // Write to global
  if (tid < bins) {
    atomicAdd(&hist[tid], local_hist[tid]);
  }
}

// Histogram probe that exposes both per-block shared accumulation and the
// final cross-block global accumulation in one launch.
extern "C" __global__ void histogram_shared_observe(const int* data,
                                                    int* observe,
                                                    int n,
                                                    int bins) {
  __shared__ int local_hist[64];
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + tid;
  int* block_hist = observe;
  int* hist = observe + gridDim.x * bins;

  if (tid < bins) local_hist[tid] = 0;
  __syncthreads();

  if (idx < n) {
    int bin = data[idx] % bins;
    atomicAdd(&local_hist[bin], 1);
  }
  __syncthreads();

  if (tid < bins) {
    const int count = local_hist[tid];
    // Each block owns a disjoint block_hist slice, so this write is race-free.
    block_hist[blockIdx.x * bins + tid] = count;
    // All blocks merge into the same global histogram, so the add must stay atomic.
    atomicAdd(&hist[tid], count);
  }
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
      "mac500",
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
      "mac500",
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
      "mac500",
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
      "mac500",
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
      "mac500",
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
      "mac500",
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
      "mac500",
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
      "mac500",
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
      "mac500",
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
      "mac500",
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
        "mac500",
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
        "mac500",
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
        "mac500",
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
        "mac500",
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
        "mac500",
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
        "mac500",
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
        "mac500",
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
        "mac500",
        nullptr);
    ASSERT_TRUE(result.ok) << result.error_message;
  }

  // Compare results
  std::vector<float> mt_result(n), cycle_result(n);
  mt_runtime.MemcpyDtoH<float>(mt_out, std::span<float>(mt_result));
  cycle_runtime.MemcpyDtoH<float>(cycle_out, std::span<float>(cycle_result));
  ExpectNearVector(mt_result, cycle_result);
}

// =============================================================================
// Global Atomic Max/Min/Exchange Tests
// =============================================================================

TEST(HipCycleValidationTest, GlobalAtomicMaxFunctionalMt) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  ModelRuntime runtime;
  const uint32_t n = 64;
  const int base_val = 100;

  const uint64_t out_addr = runtime.Malloc(sizeof(int));
  int initial = 0;
  runtime.MemcpyHtoD<int>(out_addr, std::span<const int>(&initial, 1));

  KernelArgPack args;
  args.PushU64(out_addr);
  args.PushU32(n);
  args.PushI32(base_val);

  auto result = runtime.LaunchProgramObject(
      ObjectReader{}.LoadProgramObject(GetCommonArtifact().exe_path, "global_atomic_max"),
      LaunchConfig{.grid_dim_x = 1, .block_dim_x = 64},
      std::move(args),
      ExecutionMode::Functional,
      "mac500",
      nullptr);
  ASSERT_TRUE(result.ok) << result.error_message;

  int actual = 0;
  runtime.MemcpyDtoH<int>(out_addr, std::span<int>(&actual, 1));
  // Max of (100 + i) for i in 0..63 = 163
  EXPECT_EQ(actual, base_val + static_cast<int>(n) - 1);
}

TEST(HipCycleValidationTest, GlobalAtomicMaxCycle) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  ModelRuntime runtime;
  const uint32_t n = 64;
  const int base_val = 100;

  const uint64_t out_addr = runtime.Malloc(sizeof(int));
  int initial = 0;
  runtime.MemcpyHtoD<int>(out_addr, std::span<const int>(&initial, 1));

  KernelArgPack args;
  args.PushU64(out_addr);
  args.PushU32(n);
  args.PushI32(base_val);

  auto result = runtime.LaunchProgramObject(
      ObjectReader{}.LoadProgramObject(GetCommonArtifact().exe_path, "global_atomic_max"),
      LaunchConfig{.grid_dim_x = 1, .block_dim_x = 64},
      std::move(args),
      ExecutionMode::Cycle,
      "mac500",
      nullptr);
  ASSERT_TRUE(result.ok) << result.error_message;

  int actual = 0;
  runtime.MemcpyDtoH<int>(out_addr, std::span<int>(&actual, 1));
  EXPECT_EQ(actual, base_val + static_cast<int>(n) - 1);
}

TEST(HipCycleValidationTest, GlobalAtomicMinFunctionalMt) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  ModelRuntime runtime;
  const uint32_t n = 64;
  const int base_val = 1000;

  const uint64_t out_addr = runtime.Malloc(sizeof(int));
  int initial = 9999;
  runtime.MemcpyHtoD<int>(out_addr, std::span<const int>(&initial, 1));

  KernelArgPack args;
  args.PushU64(out_addr);
  args.PushU32(n);
  args.PushI32(base_val);

  auto result = runtime.LaunchProgramObject(
      ObjectReader{}.LoadProgramObject(GetCommonArtifact().exe_path, "global_atomic_min"),
      LaunchConfig{.grid_dim_x = 1, .block_dim_x = 64},
      std::move(args),
      ExecutionMode::Functional,
      "mac500",
      nullptr);
  ASSERT_TRUE(result.ok) << result.error_message;

  int actual = 0;
  runtime.MemcpyDtoH<int>(out_addr, std::span<int>(&actual, 1));
  // Min of (1000 - i) for i in 0..63 = 936
  EXPECT_EQ(actual, base_val - static_cast<int>(n) + 1);
}

TEST(HipCycleValidationTest, GlobalAtomicMinCycle) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  ModelRuntime runtime;
  const uint32_t n = 64;
  const int base_val = 1000;

  const uint64_t out_addr = runtime.Malloc(sizeof(int));
  int initial = 9999;
  runtime.MemcpyHtoD<int>(out_addr, std::span<const int>(&initial, 1));

  KernelArgPack args;
  args.PushU64(out_addr);
  args.PushU32(n);
  args.PushI32(base_val);

  auto result = runtime.LaunchProgramObject(
      ObjectReader{}.LoadProgramObject(GetCommonArtifact().exe_path, "global_atomic_min"),
      LaunchConfig{.grid_dim_x = 1, .block_dim_x = 64},
      std::move(args),
      ExecutionMode::Cycle,
      "mac500",
      nullptr);
  ASSERT_TRUE(result.ok) << result.error_message;

  int actual = 0;
  runtime.MemcpyDtoH<int>(out_addr, std::span<int>(&actual, 1));
  EXPECT_EQ(actual, base_val - static_cast<int>(n) + 1);
}

TEST(HipCycleValidationTest, GlobalAtomicExchFunctionalMt) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  ModelRuntime runtime;
  const uint32_t n = 64;
  const int val = 42;

  const uint64_t out_addr = runtime.Malloc(n * sizeof(int));
  std::vector<int> initial(n, -1);
  runtime.MemcpyHtoD<int>(out_addr, std::span<const int>(initial));

  KernelArgPack args;
  args.PushU64(out_addr);
  args.PushU32(n);
  args.PushI32(val);

  auto result = runtime.LaunchProgramObject(
      ObjectReader{}.LoadProgramObject(GetCommonArtifact().exe_path, "global_atomic_exch"),
      LaunchConfig{.grid_dim_x = 1, .block_dim_x = 64},
      std::move(args),
      ExecutionMode::Functional,
      "mac500",
      nullptr);
  ASSERT_TRUE(result.ok) << result.error_message;

  std::vector<int> actual(n);
  runtime.MemcpyDtoH<int>(out_addr, std::span<int>(actual));
  for (uint32_t i = 0; i < n; ++i) {
    EXPECT_EQ(actual[i], val);
  }
}

TEST(HipCycleValidationTest, GlobalAtomicExchCycle) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  ModelRuntime runtime;
  const uint32_t n = 64;
  const int val = 42;

  const uint64_t out_addr = runtime.Malloc(n * sizeof(int));
  std::vector<int> initial(n, -1);
  runtime.MemcpyHtoD<int>(out_addr, std::span<const int>(initial));

  KernelArgPack args;
  args.PushU64(out_addr);
  args.PushU32(n);
  args.PushI32(val);

  auto result = runtime.LaunchProgramObject(
      ObjectReader{}.LoadProgramObject(GetCommonArtifact().exe_path, "global_atomic_exch"),
      LaunchConfig{.grid_dim_x = 1, .block_dim_x = 64},
      std::move(args),
      ExecutionMode::Cycle,
      "mac500",
      nullptr);
  ASSERT_TRUE(result.ok) << result.error_message;

  std::vector<int> actual(n);
  runtime.MemcpyDtoH<int>(out_addr, std::span<int>(actual));
  for (uint32_t i = 0; i < n; ++i) {
    EXPECT_EQ(actual[i], val);
  }
}

// =============================================================================
// Shared Memory Atomic Tests
// =============================================================================

TEST(HipCycleValidationTest, SharedAtomicAddFunctionalMt) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  ModelRuntime runtime;
  const uint32_t n = 64;
  const uint32_t grid_dim_x = 4;
  const uint64_t out_addr = runtime.Malloc(grid_dim_x * sizeof(int));
  std::vector<int> out_init(grid_dim_x, 0);
  runtime.MemcpyHtoD<int>(out_addr, std::span<const int>(out_init));

  KernelArgPack args;
  args.PushU64(out_addr);
  args.PushU32(n);

  auto result = runtime.LaunchProgramObject(
      ObjectReader{}.LoadProgramObject(GetCommonArtifact().exe_path, "shared_atomic_add"),
      LaunchConfig{.grid_dim_x = grid_dim_x, .block_dim_x = 64, .shared_memory_bytes = 256},
      std::move(args),
      ExecutionMode::Functional,
      "mac500",
      nullptr);
  ASSERT_TRUE(result.ok) << result.error_message;

  std::vector<int> actual(grid_dim_x);
  runtime.MemcpyDtoH<int>(out_addr, std::span<int>(actual));
  // Sum of (tid + 1) for tid in 0..63 = 64*65/2 = 2080
  for (uint32_t block = 0; block < grid_dim_x; ++block) {
    EXPECT_EQ(actual[block], 64 * 65 / 2) << "block=" << block;
  }
}

TEST(HipCycleValidationTest, SharedAtomicAddCycle) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  ModelRuntime runtime;
  const uint32_t n = 64;
  const uint32_t grid_dim_x = 4;
  const uint64_t out_addr = runtime.Malloc(grid_dim_x * sizeof(int));
  std::vector<int> out_init(grid_dim_x, 0);
  runtime.MemcpyHtoD<int>(out_addr, std::span<const int>(out_init));

  KernelArgPack args;
  args.PushU64(out_addr);
  args.PushU32(n);

  auto result = runtime.LaunchProgramObject(
      ObjectReader{}.LoadProgramObject(GetCommonArtifact().exe_path, "shared_atomic_add"),
      LaunchConfig{.grid_dim_x = grid_dim_x, .block_dim_x = 64, .shared_memory_bytes = 256},
      std::move(args),
      ExecutionMode::Cycle,
      "mac500",
      nullptr);
  ASSERT_TRUE(result.ok) << result.error_message;

  std::vector<int> actual(grid_dim_x);
  runtime.MemcpyDtoH<int>(out_addr, std::span<int>(actual));
  for (uint32_t block = 0; block < grid_dim_x; ++block) {
    EXPECT_EQ(actual[block], 64 * 65 / 2) << "block=" << block;
  }
}

TEST(HipCycleValidationTest, SharedAtomicExchFunctionalMt) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  ModelRuntime runtime;
  const uint32_t n = 64;

  const uint64_t out_addr = runtime.Malloc(n * sizeof(int));

  KernelArgPack args;
  args.PushU64(out_addr);
  args.PushU32(n);

  auto result = runtime.LaunchProgramObject(
      ObjectReader{}.LoadProgramObject(GetCommonArtifact().exe_path, "shared_atomic_exch"),
      LaunchConfig{.grid_dim_x = 1, .block_dim_x = 64, .shared_memory_bytes = 256},
      std::move(args),
      ExecutionMode::Functional,
      "mac500",
      nullptr);
  ASSERT_TRUE(result.ok) << result.error_message;

  std::vector<int> actual(n);
  runtime.MemcpyDtoH<int>(out_addr, std::span<int>(actual));
  for (uint32_t i = 0; i < n; ++i) {
    EXPECT_EQ(actual[i], static_cast<int>(i) * 10);
  }
}

TEST(HipCycleValidationTest, SharedAtomicExchCycle) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  ModelRuntime runtime;
  const uint32_t n = 64;

  const uint64_t out_addr = runtime.Malloc(n * sizeof(int));

  KernelArgPack args;
  args.PushU64(out_addr);
  args.PushU32(n);

  auto result = runtime.LaunchProgramObject(
      ObjectReader{}.LoadProgramObject(GetCommonArtifact().exe_path, "shared_atomic_exch"),
      LaunchConfig{.grid_dim_x = 1, .block_dim_x = 64, .shared_memory_bytes = 256},
      std::move(args),
      ExecutionMode::Cycle,
      "mac500",
      nullptr);
  ASSERT_TRUE(result.ok) << result.error_message;

  std::vector<int> actual(n);
  runtime.MemcpyDtoH<int>(out_addr, std::span<int>(actual));
  for (uint32_t i = 0; i < n; ++i) {
    EXPECT_EQ(actual[i], static_cast<int>(i) * 10);
  }
}

// =============================================================================
// Private Memory Tests
// =============================================================================

TEST(HipCycleValidationTest, PrivateArraySumFunctionalMt) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  ModelRuntime runtime;
  const uint32_t n = 64;

  const uint64_t in_addr = runtime.Malloc(n * sizeof(int));
  const uint64_t out_addr = runtime.Malloc(n * sizeof(int));

  KernelArgPack args;
  args.PushU64(in_addr);
  args.PushU64(out_addr);
  args.PushU32(n);

  auto result = runtime.LaunchProgramObject(
      ObjectReader{}.LoadProgramObject(GetCommonArtifact().exe_path, "private_array_sum"),
      LaunchConfig{.grid_dim_x = 1, .block_dim_x = 64},
      std::move(args),
      ExecutionMode::Functional,
      "mac500",
      nullptr);
  ASSERT_TRUE(result.ok) << result.error_message;

  std::vector<int> actual(n);
  runtime.MemcpyDtoH<int>(out_addr, std::span<int>(actual));
  // Each thread computes sum of idx*8 + i for i in 0..7
  for (uint32_t idx = 0; idx < n; ++idx) {
    int expected = 0;
    for (int i = 0; i < 8; ++i) {
      expected += static_cast<int>(idx) * 8 + i;
    }
    EXPECT_EQ(actual[idx], expected);
  }
}

TEST(HipCycleValidationTest, PrivateArraySumCycle) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  ModelRuntime runtime;
  const uint32_t n = 64;

  const uint64_t in_addr = runtime.Malloc(n * sizeof(int));
  const uint64_t out_addr = runtime.Malloc(n * sizeof(int));

  KernelArgPack args;
  args.PushU64(in_addr);
  args.PushU64(out_addr);
  args.PushU32(n);

  auto result = runtime.LaunchProgramObject(
      ObjectReader{}.LoadProgramObject(GetCommonArtifact().exe_path, "private_array_sum"),
      LaunchConfig{.grid_dim_x = 1, .block_dim_x = 64},
      std::move(args),
      ExecutionMode::Cycle,
      "mac500",
      nullptr);
  ASSERT_TRUE(result.ok) << result.error_message;

  std::vector<int> actual(n);
  runtime.MemcpyDtoH<int>(out_addr, std::span<int>(actual));
  for (uint32_t idx = 0; idx < n; ++idx) {
    int expected = 0;
    for (int i = 0; i < 8; ++i) {
      expected += static_cast<int>(idx) * 8 + i;
    }
    EXPECT_EQ(actual[idx], expected);
  }
}

TEST(HipCycleValidationTest, PrivateLoadStoreFunctionalMt) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  ModelRuntime runtime;
  const uint32_t n = 64;

  std::vector<float> input(n);
  for (uint32_t i = 0; i < n; ++i) {
    input[i] = static_cast<float>(i);
  }

  const uint64_t in_addr = runtime.Malloc(n * sizeof(float));
  const uint64_t out_addr = runtime.Malloc(n * sizeof(float));
  runtime.MemcpyHtoD<float>(in_addr, std::span<const float>(input));

  KernelArgPack args;
  args.PushU64(in_addr);
  args.PushU64(out_addr);
  args.PushU32(n);

  auto result = runtime.LaunchProgramObject(
      ObjectReader{}.LoadProgramObject(GetCommonArtifact().exe_path, "private_load_store"),
      LaunchConfig{.grid_dim_x = 1, .block_dim_x = 64},
      std::move(args),
      ExecutionMode::Functional,
      "mac500",
      nullptr);
  ASSERT_TRUE(result.ok) << result.error_message;

  std::vector<float> actual(n);
  runtime.MemcpyDtoH<float>(out_addr, std::span<float>(actual));
  for (uint32_t i = 0; i < n; ++i) {
    float expected = input[i] * 2.0f + 1.0f;
    EXPECT_NEAR(actual[i], expected, 1e-4f);
  }
}

TEST(HipCycleValidationTest, PrivateLoadStoreCycle) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  ModelRuntime runtime;
  const uint32_t n = 64;

  std::vector<float> input(n);
  for (uint32_t i = 0; i < n; ++i) {
    input[i] = static_cast<float>(i);
  }

  const uint64_t in_addr = runtime.Malloc(n * sizeof(float));
  const uint64_t out_addr = runtime.Malloc(n * sizeof(float));
  runtime.MemcpyHtoD<float>(in_addr, std::span<const float>(input));

  KernelArgPack args;
  args.PushU64(in_addr);
  args.PushU64(out_addr);
  args.PushU32(n);

  auto result = runtime.LaunchProgramObject(
      ObjectReader{}.LoadProgramObject(GetCommonArtifact().exe_path, "private_load_store"),
      LaunchConfig{.grid_dim_x = 1, .block_dim_x = 64},
      std::move(args),
      ExecutionMode::Cycle,
      "mac500",
      nullptr);
  ASSERT_TRUE(result.ok) << result.error_message;

  std::vector<float> actual(n);
  runtime.MemcpyDtoH<float>(out_addr, std::span<float>(actual));
  for (uint32_t i = 0; i < n; ++i) {
    float expected = input[i] * 2.0f + 1.0f;
    EXPECT_NEAR(actual[i], expected, 1e-4f);
  }
}

// =============================================================================
// Histogram with Shared Atomics Tests
// =============================================================================

TEST(HipCycleValidationTest, HistogramSharedFunctionalMt) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  ModelRuntime runtime;
  const uint32_t n = 256;
  const uint32_t bins = 8;
  const uint32_t grid_dim_x = 4;
  const uint32_t block_dim_x = 64;
  const uint32_t shared_memory_bytes = 256;

  std::vector<int> data(n);
  for (uint32_t i = 0; i < n; ++i) {
    data[i] = static_cast<int>(i % bins);
  }

  const uint64_t data_addr = runtime.Malloc(n * sizeof(int));
  const uint32_t observe_count = grid_dim_x * bins + bins;
  const uint64_t observe_addr = runtime.Malloc(observe_count * sizeof(int));
  runtime.MemcpyHtoD<int>(data_addr, std::span<const int>(data));

  std::vector<int> observe_init(observe_count, 0);
  const auto run_histogram_observe = [&](uint32_t launch_n) {
    runtime.MemcpyHtoD<int>(observe_addr, std::span<const int>(observe_init));

    KernelArgPack args;
    args.PushU64(data_addr);
    args.PushU64(observe_addr);
    args.PushU32(launch_n);
    args.PushU32(bins);

    auto result = runtime.LaunchProgramObject(
        ObjectReader{}.LoadProgramObject(GetCommonArtifact().exe_path, "histogram_shared_observe"),
        LaunchConfig{
            .grid_dim_x = grid_dim_x,
            .block_dim_x = block_dim_x,
            .shared_memory_bytes = shared_memory_bytes,
        },
        std::move(args),
        ExecutionMode::Functional,
        "mac500",
        nullptr);

    std::vector<int> observe(observe_count);
    runtime.MemcpyDtoH<int>(observe_addr, std::span<int>(observe));
    std::vector<int> actual_block_hist(observe.begin(), observe.begin() + grid_dim_x * bins);
    std::vector<int> actual_hist(observe.begin() + grid_dim_x * bins, observe.end());
    return std::tuple{std::move(result), std::move(actual_block_hist), std::move(actual_hist)};
  };

  auto [result, actual_block_hist, actual_hist] = run_histogram_observe(/*launch_n=*/n);
  ASSERT_TRUE(result.ok) << result.error_message;

  {
    SCOPED_TRACE("per_block_shared_atomic_accumulation");
    for (uint32_t block = 0; block < grid_dim_x; ++block) {
      for (uint32_t b = 0; b < bins; ++b) {
        EXPECT_EQ(actual_block_hist[block * bins + b], static_cast<int>(block_dim_x / bins))
            << "block=" << block << " bin=" << b;
      }
    }
  }

  {
    SCOPED_TRACE("multi_block_global_aggregation");
    for (uint32_t b = 0; b < bins; ++b) {
      EXPECT_EQ(actual_hist[b], static_cast<int>(n / bins));
    }
  }
}

TEST(HipCycleValidationTest, HistogramSharedCycle) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  ModelRuntime runtime;
  const uint32_t n = 256;
  const uint32_t bins = 8;
  const uint32_t grid_dim_x = 4;
  const uint32_t block_dim_x = 64;
  const uint32_t shared_memory_bytes = 256;

  std::vector<int> data(n);
  for (uint32_t i = 0; i < n; ++i) {
    data[i] = static_cast<int>(i % bins);
  }

  const uint64_t data_addr = runtime.Malloc(n * sizeof(int));
  const uint32_t observe_count = grid_dim_x * bins + bins;
  const uint64_t observe_addr = runtime.Malloc(observe_count * sizeof(int));
  runtime.MemcpyHtoD<int>(data_addr, std::span<const int>(data));

  std::vector<int> observe_init(observe_count, 0);
  const auto run_histogram_observe = [&](uint32_t launch_n) {
    runtime.MemcpyHtoD<int>(observe_addr, std::span<const int>(observe_init));

    KernelArgPack args;
    args.PushU64(data_addr);
    args.PushU64(observe_addr);
    args.PushU32(launch_n);
    args.PushU32(bins);

    auto result = runtime.LaunchProgramObject(
        ObjectReader{}.LoadProgramObject(GetCommonArtifact().exe_path, "histogram_shared_observe"),
        LaunchConfig{
            .grid_dim_x = grid_dim_x,
            .block_dim_x = block_dim_x,
            .shared_memory_bytes = shared_memory_bytes,
        },
        std::move(args),
        ExecutionMode::Cycle,
        "mac500",
        nullptr);

    std::vector<int> observe(observe_count);
    runtime.MemcpyDtoH<int>(observe_addr, std::span<int>(observe));
    std::vector<int> actual_block_hist(observe.begin(), observe.begin() + grid_dim_x * bins);
    std::vector<int> actual_hist(observe.begin() + grid_dim_x * bins, observe.end());
    return std::tuple{std::move(result), std::move(actual_block_hist), std::move(actual_hist)};
  };

  auto [result, actual_block_hist, actual_hist] = run_histogram_observe(/*launch_n=*/n);
  ASSERT_TRUE(result.ok) << result.error_message;

  {
    SCOPED_TRACE("per_block_shared_atomic_accumulation");
    for (uint32_t block = 0; block < grid_dim_x; ++block) {
      for (uint32_t b = 0; b < bins; ++b) {
        EXPECT_EQ(actual_block_hist[block * bins + b], static_cast<int>(block_dim_x / bins))
            << "block=" << block << " bin=" << b;
      }
    }
  }

  {
    SCOPED_TRACE("multi_block_global_aggregation");
    for (uint32_t b = 0; b < bins; ++b) {
      EXPECT_EQ(actual_hist[b], static_cast<int>(n / bins));
    }
  }
}

TEST(HipCycleValidationTest, HistogramSharedObserveTraceIncludesMemoryAddressesAndValues) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto run_and_collect = [](ExecutionMode mode) {
    ModelRuntime runtime;
    constexpr uint32_t n = 64;
    constexpr uint32_t bins = 8;
    constexpr uint32_t grid_dim_x = 1;
    constexpr uint32_t block_dim_x = 64;
    constexpr uint32_t shared_memory_bytes = 256;

    std::vector<int> data(n);
    for (uint32_t i = 0; i < n; ++i) {
      data[i] = static_cast<int>(i % bins);
    }

    const uint64_t data_addr = runtime.Malloc(n * sizeof(int));
    const uint32_t observe_count = grid_dim_x * bins + bins;
    const uint64_t observe_addr = runtime.Malloc(observe_count * sizeof(int));
    runtime.MemcpyHtoD<int>(data_addr, std::span<const int>(data));

    std::vector<int> observe_init(observe_count, 0);
    runtime.MemcpyHtoD<int>(observe_addr, std::span<const int>(observe_init));

    CollectingTraceSink trace;
    KernelArgPack args;
    args.PushU64(data_addr);
    args.PushU64(observe_addr);
    args.PushU32(n);
    args.PushU32(bins);

    const auto result = runtime.LaunchProgramObject(
        ObjectReader{}.LoadProgramObject(GetCommonArtifact().exe_path, "histogram_shared_observe"),
        LaunchConfig{
            .grid_dim_x = grid_dim_x,
            .block_dim_x = block_dim_x,
            .shared_memory_bytes = shared_memory_bytes,
        },
        std::move(args),
        mode,
        "mac500",
        &trace);
    return std::pair{result, trace.events()};
  };

  const auto assert_mem_summary = [](const std::vector<TraceEvent>& events,
                                     std::string_view mnemonic,
                                     std::initializer_list<std::string_view> required_tokens) {
    const auto it = std::find_if(events.begin(), events.end(), [&](const TraceEvent& event) {
      return event.kind == TraceEventKind::WaveStep &&
             !event.display_name.empty() &&
             event.display_name.find(mnemonic) != std::string::npos;
    });
    ASSERT_NE(it, events.end()) << mnemonic;
    ASSERT_TRUE(it->step_detail.has_value()) << mnemonic;
    EXPECT_NE(it->step_detail->mem_summary, "none") << mnemonic;
    for (const auto token : required_tokens) {
      EXPECT_NE(it->step_detail->mem_summary.find(token), std::string::npos)
          << mnemonic << " mem_summary=" << it->step_detail->mem_summary;
    }
  };

  for (const auto mode : {ExecutionMode::Functional, ExecutionMode::Cycle}) {
    const auto [result, events] = run_and_collect(mode);
    ASSERT_TRUE(result.ok) << result.error_message;

    assert_mem_summary(events, "global_load_dword", {"space=global", "kind=load", "addr=", "read="});
    assert_mem_summary(events, "ds_add_u32", {"space=shared", "kind=atomic", "addr=", "read=", "write="});
    assert_mem_summary(events, "ds_read_b32", {"space=shared", "kind=load", "addr=", "read="});
    assert_mem_summary(events, "global_store_dword", {"space=global", "kind=store", "addr=", "write="});
    assert_mem_summary(events, "global_atomic_add", {"space=global", "kind=atomic", "addr=", "read=", "write="});
  }
}

}  // namespace
}  // namespace gpu_model
