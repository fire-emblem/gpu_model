#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <vector>

#include "gpu_model/runtime/hip_interposer_state.h"

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
  const auto suffix = std::to_string(
      std::chrono::steady_clock::now().time_since_epoch().count());
  const auto path = std::filesystem::temp_directory_path() / (stem + "_" + suffix);
  std::filesystem::remove_all(path);
  std::filesystem::create_directories(path);
  return path;
}

TEST(HipInterposerStateTest, LaunchesHipVecAddExecutableThroughRegisteredHostFunction) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hip_interposer_vecadd");
  const auto src_path = temp_dir / "hip_vecadd.cpp";
  const auto exe_path = temp_dir / "hip_vecadd.out";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << "#include <hip/hip_runtime.h>\n\n"
           "extern \"C\" __global__ void vecadd(const float* a, const float* b, float* c, int n) {\n"
           "  int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
           "  if (i < n) c[i] = a[i] + b[i];\n"
           "}\n\n"
           "int main() { return 0; }\n";
  }

  const std::string command = "hipcc " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  auto& state = HipInterposerState::Instance();
  state.ResetForTest();
  static int host_symbol = 0;
  state.RegisterFunction(&host_symbol, "vecadd");

  constexpr uint32_t n = 129;
  std::vector<float> a(n), b(n), c(n, -1.0f);
  for (uint32_t i = 0; i < n; ++i) {
    a[i] = static_cast<float>(i) * 0.5f;
    b[i] = static_cast<float>(50 + i) * 0.25f;
  }

  void* a_dev = state.AllocateDevice(n * sizeof(float));
  void* b_dev = state.AllocateDevice(n * sizeof(float));
  void* c_dev = state.AllocateDevice(n * sizeof(float));
  state.MemcpyHostToDevice(a_dev, a.data(), n * sizeof(float));
  state.MemcpyHostToDevice(b_dev, b.data(), n * sizeof(float));
  state.MemcpyHostToDevice(c_dev, c.data(), n * sizeof(float));

  void* args[] = {&a_dev, &b_dev, &c_dev, const_cast<uint32_t*>(&n)};
  const auto result = state.LaunchExecutableKernel(
      exe_path, &host_symbol, LaunchConfig{.grid_dim_x = 3, .block_dim_x = 64}, args);
  ASSERT_TRUE(result.ok) << result.error_message;

  state.MemcpyDeviceToHost(c.data(), c_dev, n * sizeof(float));
  for (uint32_t i = 0; i < n; ++i) {
    EXPECT_FLOAT_EQ(c[i], a[i] + b[i]);
  }

  std::filesystem::remove_all(temp_dir);
}

TEST(HipInterposerStateTest, BuildsExecutableLoadPlanThroughRegisteredHostFunction) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hip_interposer_load_plan");
  const auto src_path = temp_dir / "hip_shared_reverse.cpp";
  const auto exe_path = temp_dir / "hip_shared_reverse.out";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << "#include <hip/hip_runtime.h>\n"
           "extern \"C\" __global__ void shared_reverse(const int* in, int* out, int n) {\n"
           "  __shared__ int scratch[64];\n"
           "  int tid = threadIdx.x;\n"
           "  int idx = blockIdx.x * blockDim.x + tid;\n"
           "  if (idx < n) scratch[tid] = in[idx];\n"
           "  __syncthreads();\n"
           "  if (idx < n) out[idx] = scratch[blockDim.x - 1 - tid];\n"
           "}\n"
           "int main() { return 0; }\n";
  }

  const std::string command = "hipcc " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  auto& state = HipInterposerState::Instance();
  state.ResetForTest();
  static int host_symbol = 0;
  state.RegisterFunction(&host_symbol, "shared_reverse");

  const auto plan = state.BuildExecutableLoadPlan(exe_path, &host_symbol);
  ASSERT_EQ(plan.segments.size(), 1u);
  EXPECT_EQ(plan.segments[0].pool, MemoryPoolKind::Code);
  EXPECT_GT(plan.segments[0].required_bytes, 0u);
  EXPECT_EQ(plan.required_shared_bytes, 256u);
  EXPECT_EQ(plan.preferred_kernarg_bytes, 20u);

  std::filesystem::remove_all(temp_dir);
}

TEST(HipInterposerStateTest, LaunchesHipFmaLoopExecutableThroughRegisteredHostFunction) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hip_interposer_fma_loop");
  const auto src_path = temp_dir / "hip_fma_loop.cpp";
  const auto exe_path = temp_dir / "hip_fma_loop.out";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << "#include <hip/hip_runtime.h>\n\n"
           "extern \"C\" __global__ void fma_loop(const float* a, const float* b, float* c, int n, int iters) {\n"
           "  int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
           "  if (i >= n) return;\n"
           "  float x = a[i];\n"
           "  float y = b[i];\n"
           "  float acc = 0.0f;\n"
           "  for (int k = 0; k < iters; ++k) {\n"
           "    acc = acc * x + y;\n"
           "  }\n"
           "  c[i] = acc;\n"
           "}\n\n"
           "int main() { return 0; }\n";
  }

  const std::string command = "hipcc " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  auto& state = HipInterposerState::Instance();
  state.ResetForTest();
  static int host_symbol = 0;
  state.RegisterFunction(&host_symbol, "fma_loop");

  constexpr uint32_t n = 257;
  constexpr uint32_t iters = 7;
  std::vector<float> a(n), b(n), c(n, -1.0f), expect(n);
  for (uint32_t i = 0; i < n; ++i) {
    a[i] = 1.0f + 0.001f * static_cast<float>(i);
    b[i] = 2.0f + 0.002f * static_cast<float>(i);
    float acc = 0.0f;
    for (uint32_t k = 0; k < iters; ++k) {
      acc = acc * a[i] + b[i];
    }
    expect[i] = acc;
  }

  void* a_dev = state.AllocateDevice(n * sizeof(float));
  void* b_dev = state.AllocateDevice(n * sizeof(float));
  void* c_dev = state.AllocateDevice(n * sizeof(float));
  state.MemcpyHostToDevice(a_dev, a.data(), n * sizeof(float));
  state.MemcpyHostToDevice(b_dev, b.data(), n * sizeof(float));
  state.MemcpyHostToDevice(c_dev, c.data(), n * sizeof(float));

  uint32_t n_arg = n;
  uint32_t iters_arg = iters;
  void* args[] = {&a_dev, &b_dev, &c_dev, &n_arg, &iters_arg};
  const auto result = state.LaunchExecutableKernel(
      exe_path, &host_symbol, LaunchConfig{.grid_dim_x = 3, .block_dim_x = 128}, args);
  ASSERT_TRUE(result.ok) << result.error_message;

  state.MemcpyDeviceToHost(c.data(), c_dev, n * sizeof(float));
  for (uint32_t i = 0; i < n; ++i) {
    EXPECT_FLOAT_EQ(c[i], expect[i]);
  }

  std::filesystem::remove_all(temp_dir);
}

TEST(HipInterposerStateTest, LaunchesHipBiasChainExecutableThroughRegisteredHostFunction) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hip_interposer_bias_chain");
  const auto src_path = temp_dir / "hip_bias_chain.cpp";
  const auto exe_path = temp_dir / "hip_bias_chain.out";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << "#include <hip/hip_runtime.h>\n\n"
           "extern \"C\" __global__ void bias_chain(const float* a, const float* b, float* c, int n, float b0, float b1, float b2) {\n"
           "  int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
           "  if (i < n) {\n"
           "    c[i] = a[i] + b[i] + b0 + b1 + b2;\n"
           "  }\n"
           "}\n\n"
           "int main() { return 0; }\n";
  }

  const std::string command = "hipcc " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  auto& state = HipInterposerState::Instance();
  state.ResetForTest();
  static int host_symbol = 0;
  state.RegisterFunction(&host_symbol, "bias_chain");

  constexpr uint32_t n = 129;
  constexpr float b0 = 1.5f;
  constexpr float b1 = -2.0f;
  constexpr float b2 = 3.25f;
  std::vector<float> a(n), b(n), c(n, -1.0f), expect(n);
  for (uint32_t i = 0; i < n; ++i) {
    a[i] = static_cast<float>(i) * 0.5f;
    b[i] = static_cast<float>(100 + i) * 0.25f;
    expect[i] = a[i] + b[i] + b0 + b1 + b2;
  }

  void* a_dev = state.AllocateDevice(n * sizeof(float));
  void* b_dev = state.AllocateDevice(n * sizeof(float));
  void* c_dev = state.AllocateDevice(n * sizeof(float));
  state.MemcpyHostToDevice(a_dev, a.data(), n * sizeof(float));
  state.MemcpyHostToDevice(b_dev, b.data(), n * sizeof(float));
  state.MemcpyHostToDevice(c_dev, c.data(), n * sizeof(float));

  float b0_arg = b0;
  float b1_arg = b1;
  float b2_arg = b2;
  uint32_t n_arg = n;
  void* args[] = {&a_dev, &b_dev, &c_dev, &n_arg, &b0_arg, &b1_arg, &b2_arg};
  const auto result = state.LaunchExecutableKernel(
      exe_path, &host_symbol, LaunchConfig{.grid_dim_x = 3, .block_dim_x = 64}, args);
  ASSERT_TRUE(result.ok) << result.error_message;

  state.MemcpyDeviceToHost(c.data(), c_dev, n * sizeof(float));
  for (uint32_t i = 0; i < n; ++i) {
    EXPECT_FLOAT_EQ(c[i], expect[i]);
  }

  std::filesystem::remove_all(temp_dir);
}

TEST(HipInterposerStateTest, LaunchesHipVecAddExecutableThroughRegisteredHostFunctionAtLargeScale) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hip_interposer_vecadd_large");
  const auto src_path = temp_dir / "hip_vecadd_large.cpp";
  const auto exe_path = temp_dir / "hip_vecadd_large.out";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << "#include <hip/hip_runtime.h>\n\n"
           "extern \"C\" __global__ void vecadd(const float* a, const float* b, float* c, int n) {\n"
           "  int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
           "  if (i < n) c[i] = a[i] + b[i];\n"
           "}\n\n"
           "int main() { return 0; }\n";
  }

  const std::string command = "hipcc " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  auto& state = HipInterposerState::Instance();
  state.ResetForTest();
  static int host_symbol = 0;
  state.RegisterFunction(&host_symbol, "vecadd");

  constexpr uint32_t n = 30u * 1024u;
  std::vector<float> a(n), b(n), c(n, -1.0f);
  for (uint32_t i = 0; i < n; ++i) {
    a[i] = static_cast<float>(i) * 0.5f;
    b[i] = static_cast<float>(100 + i) * 0.25f;
  }

  void* a_dev = state.AllocateDevice(n * sizeof(float));
  void* b_dev = state.AllocateDevice(n * sizeof(float));
  void* c_dev = state.AllocateDevice(n * sizeof(float));
  state.MemcpyHostToDevice(a_dev, a.data(), n * sizeof(float));
  state.MemcpyHostToDevice(b_dev, b.data(), n * sizeof(float));
  state.MemcpyHostToDevice(c_dev, c.data(), n * sizeof(float));

  uint32_t n_arg = n;
  void* args[] = {&a_dev, &b_dev, &c_dev, &n_arg};
  const auto result = state.LaunchExecutableKernel(
      exe_path, &host_symbol, LaunchConfig{.grid_dim_x = 30, .block_dim_x = 1024}, args);
  ASSERT_TRUE(result.ok) << result.error_message;

  state.MemcpyDeviceToHost(c.data(), c_dev, n * sizeof(float));
  for (uint32_t i = 0; i < n; ++i) {
    EXPECT_FLOAT_EQ(c[i], a[i] + b[i]);
  }

  std::filesystem::remove_all(temp_dir);
}

TEST(HipInterposerStateTest, LaunchesHipVecAddExecutableThroughManagedAllocations) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hip_interposer_managed_vecadd");
  const auto src_path = temp_dir / "hip_managed_vecadd.cpp";
  const auto exe_path = temp_dir / "hip_managed_vecadd.out";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << "#include <hip/hip_runtime.h>\n\n"
           "extern \"C\" __global__ void vecadd(const float* a, const float* b, float* c, int n) {\n"
           "  int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
           "  if (i < n) c[i] = a[i] + b[i];\n"
           "}\n\n"
           "int main() { return 0; }\n";
  }

  const std::string command = "hipcc " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  auto& state = HipInterposerState::Instance();
  state.ResetForTest();
  static int host_symbol = 0;
  state.RegisterFunction(&host_symbol, "vecadd");

  constexpr uint32_t n = 257;
  std::vector<float> a(n), b(n), c(n, -1.0f);
  for (uint32_t i = 0; i < n; ++i) {
    a[i] = 0.5f * static_cast<float>(i);
    b[i] = 0.25f * static_cast<float>(100 + i);
  }

  void* a_dev = state.AllocateManaged(n * sizeof(float));
  void* b_dev = state.AllocateManaged(n * sizeof(float));
  void* c_dev = state.AllocateManaged(n * sizeof(float));
  EXPECT_EQ(state.hooks().runtime().memory().pool_memory_size(MemoryPoolKind::Managed),
            3u * n * sizeof(float));
  state.MemcpyHostToDevice(a_dev, a.data(), n * sizeof(float));
  state.MemcpyHostToDevice(b_dev, b.data(), n * sizeof(float));
  state.MemcpyHostToDevice(c_dev, c.data(), n * sizeof(float));

  uint32_t n_arg = n;
  void* args[] = {&a_dev, &b_dev, &c_dev, &n_arg};
  const auto result = state.LaunchExecutableKernel(
      exe_path, &host_symbol, LaunchConfig{.grid_dim_x = 3, .block_dim_x = 128}, args);
  ASSERT_TRUE(result.ok) << result.error_message;

  state.MemcpyDeviceToHost(c.data(), c_dev, n * sizeof(float));
  for (uint32_t i = 0; i < n; ++i) {
    EXPECT_FLOAT_EQ(c[i], a[i] + b[i]);
  }

  std::filesystem::remove_all(temp_dir);
}

TEST(HipInterposerStateTest, LaunchesHipSharedReverseExecutableThroughRegisteredHostFunction) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hip_interposer_shared_reverse");
  const auto src_path = temp_dir / "hip_shared_reverse.cpp";
  const auto exe_path = temp_dir / "hip_shared_reverse.out";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << "#include <hip/hip_runtime.h>\n\n"
           "extern \"C\" __global__ void shared_reverse(const int* in, int* out, int n) {\n"
           "  __shared__ int scratch[64];\n"
           "  int tid = threadIdx.x;\n"
           "  int idx = blockIdx.x * blockDim.x + tid;\n"
           "  if (idx < n) scratch[tid] = in[idx];\n"
           "  __syncthreads();\n"
           "  if (idx < n) out[idx] = scratch[blockDim.x - 1 - tid];\n"
           "}\n\n"
           "int main() { return 0; }\n";
  }

  const std::string command = "hipcc " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  auto& state = HipInterposerState::Instance();
  state.ResetForTest();
  static int host_symbol = 0;
  state.RegisterFunction(&host_symbol, "shared_reverse");

  constexpr uint32_t n = 128;
  std::vector<int32_t> in(n), out(n, -1), expect(n, -1);
  for (uint32_t i = 0; i < n; ++i) {
    in[i] = static_cast<int32_t>(i + 1);
  }
  for (uint32_t block = 0; block < 2; ++block) {
    const uint32_t base = block * 64;
    for (uint32_t lane = 0; lane < 64; ++lane) {
      expect[base + lane] = in[base + (63 - lane)];
    }
  }

  void* in_dev = state.AllocateDevice(n * sizeof(int32_t));
  void* out_dev = state.AllocateDevice(n * sizeof(int32_t));
  state.MemcpyHostToDevice(in_dev, in.data(), n * sizeof(int32_t));
  state.MemcpyHostToDevice(out_dev, out.data(), n * sizeof(int32_t));

  uint32_t n_arg = n;
  void* args[] = {&in_dev, &out_dev, &n_arg};
  const auto result = state.LaunchExecutableKernel(
      exe_path, &host_symbol, LaunchConfig{.grid_dim_x = 2, .block_dim_x = 64}, args);
  ASSERT_TRUE(result.ok) << result.error_message;

  state.MemcpyDeviceToHost(out.data(), out_dev, n * sizeof(int32_t));
  for (uint32_t i = 0; i < n; ++i) {
    EXPECT_EQ(out[i], expect[i]);
  }

  std::filesystem::remove_all(temp_dir);
}

TEST(HipInterposerStateTest, LaunchesHipSoftmaxExecutableThroughRegisteredHostFunction) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hip_interposer_softmax");
  const auto src_path = temp_dir / "hip_softmax.cpp";
  const auto exe_path = temp_dir / "hip_softmax.out";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << "#include <hip/hip_runtime.h>\n"
           "#include <math.h>\n"
           "extern \"C\" __global__ void softmax_row(const float* in, float* out, int n) {\n"
           "  __shared__ float scratch[64];\n"
           "  int tid = threadIdx.x;\n"
           "  int idx = blockIdx.x * blockDim.x + tid;\n"
           "  float x = idx < n ? in[idx] : -1.0e20f;\n"
           "  scratch[tid] = x;\n"
           "  __syncthreads();\n"
           "  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {\n"
           "    if (tid < stride) scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);\n"
           "    __syncthreads();\n"
           "  }\n"
           "  float m = scratch[0];\n"
           "  float e = idx < n ? expf(x - m) : 0.0f;\n"
           "  scratch[tid] = e;\n"
           "  __syncthreads();\n"
           "  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {\n"
           "    if (tid < stride) scratch[tid] += scratch[tid + stride];\n"
           "    __syncthreads();\n"
           "  }\n"
           "  if (idx < n) out[idx] = e / scratch[0];\n"
           "}\n"
           "int main() { return 0; }\n";
  }

  const std::string command = "hipcc " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  auto& state = HipInterposerState::Instance();
  state.ResetForTest();
  static int host_symbol = 0;
  state.RegisterFunction(&host_symbol, "softmax_row");

  constexpr uint32_t n = 64;
  std::vector<float> input(n, 1.0f), output(n, 0.0f);
  void* in_dev = state.AllocateDevice(n * sizeof(float));
  void* out_dev = state.AllocateDevice(n * sizeof(float));
  state.MemcpyHostToDevice(in_dev, input.data(), n * sizeof(float));
  state.MemcpyHostToDevice(out_dev, output.data(), n * sizeof(float));

  uint32_t n_arg = n;
  void* args[] = {&in_dev, &out_dev, &n_arg};
  const auto result = state.LaunchExecutableKernel(
      exe_path, &host_symbol, LaunchConfig{.grid_dim_x = 1, .block_dim_x = 64}, args);
  ASSERT_TRUE(result.ok) << result.error_message;

  state.MemcpyDeviceToHost(output.data(), out_dev, n * sizeof(float));
  constexpr float expected = 1.0f / 64.0f;
  for (uint32_t i = 0; i < n; ++i) {
    EXPECT_NEAR(output[i], expected, 1.0e-4f);
  }

  std::filesystem::remove_all(temp_dir);
}

TEST(HipInterposerStateTest, LaunchesHipMfmaExecutableThroughRegisteredHostFunction) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hip_interposer_mfma");
  const auto src_path = temp_dir / "hip_mfma.cpp";
  const auto exe_path = temp_dir / "hip_mfma.out";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << "#include <hip/hip_runtime.h>\n"
           "typedef float v4f __attribute__((ext_vector_type(4)));\n"
           "extern \"C\" __global__ void mfma_probe(float* out) {\n"
           "#if defined(__AMDGCN__)\n"
           "  v4f acc = {0.0f, 0.0f, 0.0f, 0.0f};\n"
           "  acc = __builtin_amdgcn_mfma_f32_16x16x4f32(1.0f, 1.0f, acc, 0, 0, 0);\n"
           "  if (threadIdx.x == 0) out[0] = acc[0];\n"
           "#else\n"
           "  if (threadIdx.x == 0) out[0] = 0.0f;\n"
           "#endif\n"
           "}\n"
           "int main() { return 0; }\n";
  }

  const std::string command =
      "hipcc --offload-arch=gfx90a " + src_path.string() + " -o " + exe_path.string();
  if (std::system(command.c_str()) != 0) {
    GTEST_SKIP() << "gfx90a mfma compilation not available";
  }

  auto& state = HipInterposerState::Instance();
  state.ResetForTest();
  static int host_symbol = 0;
  state.RegisterFunction(&host_symbol, "mfma_probe");

  float output = 0.0f;
  void* out_dev = state.AllocateDevice(sizeof(float));
  state.MemcpyHostToDevice(out_dev, &output, sizeof(float));

  void* args[] = {&out_dev};
  const auto result = state.LaunchExecutableKernel(
      exe_path, &host_symbol, LaunchConfig{.grid_dim_x = 1, .block_dim_x = 64}, args);
  ASSERT_TRUE(result.ok) << result.error_message;

  state.MemcpyDeviceToHost(&output, out_dev, sizeof(float));
  EXPECT_NEAR(output, 4.0f, 1.0e-5f);

  std::filesystem::remove_all(temp_dir);
}

}  // namespace
}  // namespace gpu_model
