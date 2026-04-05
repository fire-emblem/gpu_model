#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

#include "gpu_model/program/object_reader.h"
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

std::filesystem::path CurrentTestBinaryPath() {
  std::array<char, 4096> buffer{};
  const ssize_t length = ::readlink("/proc/self/exe", buffer.data(), buffer.size() - 1);
  if (length < 0) {
    throw std::runtime_error("failed to resolve /proc/self/exe for test binary");
  }
  buffer[static_cast<size_t>(length)] = '\0';
  return std::filesystem::path(buffer.data());
}

std::filesystem::path BuildDirPath() {
  return CurrentTestBinaryPath().parent_path().parent_path();
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

TEST(HipInterposerStateTest, LaunchesHipVecAddExecutableInCycleModeThroughRegisteredHostFunction) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hip_interposer_vecadd_cycle");
  const auto src_path = temp_dir / "hip_vecadd_cycle.cpp";
  const auto exe_path = temp_dir / "hip_vecadd_cycle.out";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << "#include <hip/hip_runtime.h>\n\n"
           "extern \"C\" __global__ void vecadd_cycle(const float* a, const float* b, float* c, int n) {\n"
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
  state.RegisterFunction(&host_symbol, "vecadd_cycle");

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
      exe_path,
      &host_symbol,
      LaunchConfig{.grid_dim_x = 3, .block_dim_x = 64},
      args,
      ExecutionMode::Cycle);
  ASSERT_TRUE(result.ok) << result.error_message;
  EXPECT_GT(result.total_cycles, 0u);
  ASSERT_TRUE(result.program_cycle_stats.has_value());
  EXPECT_EQ(result.total_cycles, result.program_cycle_stats->total_cycles);
  EXPECT_GT(result.program_cycle_stats->total_issued_work_cycles, 0u);

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
  ASSERT_EQ(plan.segments.size(), 2u);
  EXPECT_EQ(plan.segments[0].pool, MemoryPoolKind::Code);
  EXPECT_EQ(plan.segments[1].pool, MemoryPoolKind::Kernarg);
  EXPECT_GT(plan.segments[0].required_bytes, 0u);
  EXPECT_EQ(plan.required_shared_bytes, 256u);
  EXPECT_EQ(plan.preferred_kernarg_bytes, 280u);

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

TEST(HipInterposerStateTest, LaunchesHipByValueAggregateExecutableThroughRegisteredHostFunction) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hip_interposer_by_value_aggregate");
  const auto src_path = temp_dir / "hip_by_value_aggregate.cpp";
  const auto exe_path = temp_dir / "hip_by_value_aggregate.out";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << "#include <hip/hip_runtime.h>\n\n"
           "struct Payload {\n"
           "  int x;\n"
           "  int y;\n"
           "  int z;\n"
           "};\n\n"
           "extern \"C\" __global__ void by_value_aggregate(int* out, Payload payload) {\n"
           "  if (threadIdx.x == 0) {\n"
           "    out[0] = payload.x + payload.y + payload.z;\n"
           "  }\n"
           "}\n\n"
           "int main() { return 0; }\n";
  }

  const std::string command = "hipcc " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  auto& state = HipInterposerState::Instance();
  state.ResetForTest();
  static int host_symbol = 0;
  state.RegisterFunction(&host_symbol, "by_value_aggregate");

  const auto image = ObjectReader{}.LoadEncodedObject(exe_path, "by_value_aggregate");
  ASSERT_TRUE(image.metadata.values.contains("arg_layout"));
  EXPECT_NE(image.metadata.values.at("arg_layout").find("by_value:"), std::string::npos);

  void* out_dev = state.AllocateDevice(sizeof(int32_t));
  int32_t zero = 0;
  state.MemcpyHostToDevice(out_dev, &zero, sizeof(zero));

  struct Payload {
    int32_t x;
    int32_t y;
    int32_t z;
  } payload{5, 9, 17};

  void* args[] = {&out_dev, &payload};
  const auto result = state.LaunchExecutableKernel(
      exe_path, &host_symbol, LaunchConfig{.grid_dim_x = 1, .block_dim_x = 64}, args);
  ASSERT_TRUE(result.ok) << result.error_message;

  int32_t output = 0;
  state.MemcpyDeviceToHost(&output, out_dev, sizeof(output));
  EXPECT_EQ(output, payload.x + payload.y + payload.z);

  std::filesystem::remove_all(temp_dir);
}

TEST(HipInterposerStateTest, LaunchesHipThreeDimensionalHiddenArgsExecutableThroughRegisteredHostFunction) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hip_interposer_hidden_args_3d");
  const auto src_path = temp_dir / "hip_three_dimensional_hidden_args.cpp";
  const auto exe_path = temp_dir / "hip_three_dimensional_hidden_args.out";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << "#include <hip/hip_runtime.h>\n\n"
           "extern \"C\" __global__ void three_dimensional_hidden_args(int* out) {\n"
           "  out[0] = static_cast<int>(gridDim.z) + static_cast<int>(blockDim.z);\n"
           "}\n\n"
           "int main() { return 0; }\n";
  }

  const std::string command = "hipcc " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  auto& state = HipInterposerState::Instance();
  state.ResetForTest();
  static int host_symbol = 0;
  state.RegisterFunction(&host_symbol, "three_dimensional_hidden_args");

  void* out_dev = state.AllocateDevice(sizeof(int32_t));
  int32_t zero = 0;
  state.MemcpyHostToDevice(out_dev, &zero, sizeof(zero));

  void* args[] = {&out_dev};
  const auto result = state.LaunchExecutableKernel(
      exe_path,
      &host_symbol,
      LaunchConfig{
          .grid_dim_x = 1,
          .grid_dim_y = 1,
          .grid_dim_z = 4,
          .block_dim_x = 8,
          .block_dim_y = 1,
          .block_dim_z = 32,
      },
      args);
  ASSERT_TRUE(result.ok) << result.error_message;

  int32_t output = 0;
  state.MemcpyDeviceToHost(&output, out_dev, sizeof(output));
  EXPECT_EQ(output, 4 + 32);

  std::filesystem::remove_all(temp_dir);
}

TEST(HipInterposerStateTest, LaunchesHipThreeDimensionalBuiltinIdsExecutableThroughRegisteredHostFunction) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hip_interposer_builtin_ids_3d");
  const auto src_path = temp_dir / "hip_three_dimensional_builtin_ids.cpp";
  const auto exe_path = temp_dir / "hip_three_dimensional_builtin_ids.out";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << "#include <hip/hip_runtime.h>\n\n"
           "extern \"C\" __global__ void three_dimensional_builtin_ids(int* out) {\n"
           "  int z = threadIdx.z;\n"
           "  out[z] = static_cast<int>(blockIdx.z) + z;\n"
           "}\n\n"
           "int main() { return 0; }\n";
  }

  const std::string command = "hipcc " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  auto& state = HipInterposerState::Instance();
  state.ResetForTest();
  static int host_symbol = 0;
  state.RegisterFunction(&host_symbol, "three_dimensional_builtin_ids");

  constexpr uint32_t depth = 64;
  std::vector<int32_t> out(depth, -1);
  void* out_dev = state.AllocateDevice(depth * sizeof(int32_t));
  state.MemcpyHostToDevice(out_dev, out.data(), depth * sizeof(int32_t));

  void* args[] = {&out_dev};
  const auto result = state.LaunchExecutableKernel(
      exe_path,
      &host_symbol,
      LaunchConfig{
          .grid_dim_x = 1,
          .grid_dim_y = 1,
          .grid_dim_z = 1,
          .block_dim_x = 1,
          .block_dim_y = 1,
          .block_dim_z = depth,
      },
      args);
  ASSERT_TRUE(result.ok) << result.error_message;

  state.MemcpyDeviceToHost(out.data(), out_dev, depth * sizeof(int32_t));
  for (uint32_t i = 0; i < depth; ++i) {
    EXPECT_EQ(out[i], static_cast<int32_t>(i));
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

TEST(HipInterposerStateTest, RunsHipHostExecutableThroughLdPreloadInterposer) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto build_dir = BuildDirPath();
  const auto interposer_path = build_dir / "libgpu_model_hip_interposer.so";
  if (!std::filesystem::exists(interposer_path)) {
    GTEST_SKIP() << "missing interposer library: " << interposer_path;
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hip_ld_preload_host");
  const auto src_path = temp_dir / "hip_ld_preload_host.cpp";
  const auto exe_path = temp_dir / "hip_ld_preload_host.out";
  const auto stdout_path = temp_dir / "stdout.txt";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << R"(
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>

extern "C" __global__ void affine2x_bias(const float* in, float* out, float bias, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[i] * 2.0f + bias;
}

int main() {
  constexpr int n = 257;
  constexpr float bias = 3.5f;
  int device_count = 0;
  int device = -1;
  if (hipGetDeviceCount(&device_count) != hipSuccess || device_count != 1) return 10;
  if (hipSetDevice(0) != hipSuccess) return 11;
  if (hipGetDevice(&device) != hipSuccess || device != 0) return 12;

  hipStream_t stream = nullptr;
  if (hipStreamCreate(&stream) != hipSuccess || stream == nullptr) return 13;

  float* in = nullptr;
  float* out = nullptr;
  if (hipMallocManaged(&in, n * sizeof(float)) != hipSuccess) return 14;
  if (hipMalloc(&out, n * sizeof(float)) != hipSuccess) return 15;
  if (hipMemset(out, 0, n * sizeof(float)) != hipSuccess) return 16;

  for (int i = 0; i < n; ++i) {
    in[i] = 0.5f * static_cast<float>(i);
  }

  hipLaunchKernelGGL(affine2x_bias, dim3(3), dim3(128), 0, stream, in, out, bias, n);
  if (hipPeekAtLastError() != hipSuccess) return 17;
  if (hipStreamSynchronize(stream) != hipSuccess) return 18;

  std::vector<float> host(n, -1.0f);
  if (hipMemcpyAsync(host.data(), out, n * sizeof(float), hipMemcpyDeviceToHost, stream) != hipSuccess) return 19;
  if (hipStreamSynchronize(stream) != hipSuccess) return 20;
  if (hipGetLastError() != hipSuccess) return 21;
  if (hipGetLastError() != hipSuccess) return 22;

  for (int i = 0; i < n; ++i) {
    const float expected = in[i] * 2.0f + bias;
    if (host[i] != expected) {
      std::printf("mismatch %d got=%f expected=%f\n", i, host[i], expected);
      return 30;
    }
  }

  if (hipStreamDestroy(stream) != hipSuccess) return 31;
  if (hipFree(out) != hipSuccess) return 32;
  if (hipFree(in) != hipSuccess) return 33;
  std::puts("ld_preload host path ok");
  return 0;
}
)";
  }

  const std::string compile_command = "env -u LD_PRELOAD hipcc " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(compile_command.c_str()), 0);

  const std::string run_command =
      "env LD_PRELOAD=" + interposer_path.string() +
      " GPU_MODEL_HIP_INTERPOSER_DEBUG=1 " + exe_path.string() + " > " + stdout_path.string() + " 2>&1";
  ASSERT_EQ(std::system(run_command.c_str()), 0);

  std::ifstream in(stdout_path);
  ASSERT_TRUE(static_cast<bool>(in));
  const std::string output((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  EXPECT_NE(output.find("ld_preload host path ok"), std::string::npos);

  std::filesystem::remove_all(temp_dir);
}

TEST(HipInterposerStateTest, RunsHipHostExecutableWithEventsThroughLdPreloadInterposer) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto build_dir = BuildDirPath();
  const auto interposer_path = build_dir / "libgpu_model_hip_interposer.so";
  if (!std::filesystem::exists(interposer_path)) {
    GTEST_SKIP() << "missing interposer library: " << interposer_path;
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hip_ld_preload_events");
  const auto src_path = temp_dir / "hip_ld_preload_events.cpp";
  const auto exe_path = temp_dir / "hip_ld_preload_events.out";
  const auto stdout_path = temp_dir / "stdout.txt";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << R"(
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>

extern "C" __global__ void vecadd(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

int main() {
  constexpr int n = 257;
  std::vector<float> a(n), b(n), c(n, -1.0f);
  for (int i = 0; i < n; ++i) {
    a[i] = 0.5f * i;
    b[i] = 0.25f * (100 + i);
  }

  hipStream_t stream = nullptr;
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  if (hipStreamCreate(&stream) != hipSuccess) return 10;
  if (hipEventCreate(&start) != hipSuccess) return 11;
  if (hipEventCreateWithFlags(&stop, 0) != hipSuccess) return 12;

  float *da = nullptr, *db = nullptr, *dc = nullptr;
  if (hipMalloc(&da, n * sizeof(float)) != hipSuccess) return 13;
  if (hipMalloc(&db, n * sizeof(float)) != hipSuccess) return 14;
  if (hipMalloc(&dc, n * sizeof(float)) != hipSuccess) return 15;
  if (hipMemcpyAsync(da, a.data(), n * sizeof(float), hipMemcpyHostToDevice, stream) != hipSuccess) return 16;
  if (hipMemcpyAsync(db, b.data(), n * sizeof(float), hipMemcpyHostToDevice, stream) != hipSuccess) return 17;
  if (hipEventRecord(start, stream) != hipSuccess) return 18;
  hipLaunchKernelGGL(vecadd, dim3(3), dim3(128), 0, stream, da, db, dc, n);
  if (hipEventRecord(stop, stream) != hipSuccess) return 19;
  if (hipStreamWaitEvent(stream, stop, 0) != hipSuccess) return 20;
  if (hipEventSynchronize(stop) != hipSuccess) return 21;
  float ms = -1.0f;
  if (hipEventElapsedTime(&ms, start, stop) != hipSuccess) return 22;
  if (ms < 0.0f) return 23;
  if (hipMemcpyAsync(c.data(), dc, n * sizeof(float), hipMemcpyDeviceToHost, stream) != hipSuccess) return 24;
  if (hipStreamSynchronize(stream) != hipSuccess) return 25;

  for (int i = 0; i < n; ++i) {
    const float expected = a[i] + b[i];
    if (c[i] != expected) {
      std::printf("mismatch %d got=%f expected=%f\n", i, c[i], expected);
      return 30;
    }
  }

  if (hipEventDestroy(start) != hipSuccess) return 31;
  if (hipEventDestroy(stop) != hipSuccess) return 32;
  if (hipStreamDestroy(stream) != hipSuccess) return 33;
  if (hipFree(da) != hipSuccess) return 34;
  if (hipFree(db) != hipSuccess) return 35;
  if (hipFree(dc) != hipSuccess) return 36;
  std::puts("ld_preload events path ok");
  return 0;
}
)";
  }

  const std::string compile_command =
      "env -u LD_PRELOAD hipcc " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(compile_command.c_str()), 0);

  const std::string run_command =
      "env LD_PRELOAD=" + interposer_path.string() +
      " GPU_MODEL_HIP_INTERPOSER_DEBUG=1 " + exe_path.string() + " > " + stdout_path.string() +
      " 2>&1";
  ASSERT_EQ(std::system(run_command.c_str()), 0);

  std::ifstream in(stdout_path);
  ASSERT_TRUE(static_cast<bool>(in));
  const std::string output((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  EXPECT_NE(output.find("ld_preload events path ok"), std::string::npos);

  std::filesystem::remove_all(temp_dir);
}

TEST(HipInterposerStateTest, RunsHipHostExecutableQueryingDevicePropertiesThroughLdPreloadInterposer) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto build_dir = BuildDirPath();
  const auto interposer_path = build_dir / "libgpu_model_hip_interposer.so";
  if (!std::filesystem::exists(interposer_path)) {
    GTEST_SKIP() << "missing interposer library: " << interposer_path;
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hip_ld_preload_props");
  const auto src_path = temp_dir / "hip_ld_preload_props.cpp";
  const auto exe_path = temp_dir / "hip_ld_preload_props.out";
  const auto stdout_path = temp_dir / "stdout.txt";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << R"(
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>

int main() {
  int count = 0;
  int device = -1;
  int warp_size = -1;
  int max_threads = -1;
  hipDeviceProp_t props{};

  if (hipGetDeviceCount(&count) != hipSuccess || count != 1) return 10;
  if (hipSetDevice(0) != hipSuccess) return 11;
  if (hipGetDevice(&device) != hipSuccess || device != 0) return 12;
  if (hipGetDeviceProperties(&props, 0) != hipSuccess) return 13;
  if (std::strcmp(props.name, "c500") != 0) return 14;
  if (props.warpSize != 64) return 15;
  if (props.maxThreadsPerBlock != 1024) return 16;
  if (props.multiProcessorCount != 104) return 17;
  if (hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0) != hipSuccess || warp_size != 64) return 18;
  if (hipDeviceGetAttribute(&max_threads, hipDeviceAttributeMaxThreadsPerBlock, 0) != hipSuccess || max_threads != 1024) return 19;

  std::puts("ld_preload property path ok");
  return 0;
}
)";
  }

  const std::string compile_command =
      "env -u LD_PRELOAD hipcc " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(compile_command.c_str()), 0);

  const std::string run_command =
      "env LD_PRELOAD=" + interposer_path.string() +
      " GPU_MODEL_HIP_INTERPOSER_DEBUG=1 " + exe_path.string() + " > " + stdout_path.string() +
      " 2>&1";
  ASSERT_EQ(std::system(run_command.c_str()), 0);

  std::ifstream in(stdout_path);
  ASSERT_TRUE(static_cast<bool>(in));
  const std::string output((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  EXPECT_NE(output.find("ld_preload property path ok"), std::string::npos);

  std::filesystem::remove_all(temp_dir);
}

TEST(HipInterposerStateTest, EnforcesSingleStreamBoundaryThroughLdPreloadInterposer) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto build_dir = BuildDirPath();
  const auto interposer_path = build_dir / "libgpu_model_hip_interposer.so";
  if (!std::filesystem::exists(interposer_path)) {
    GTEST_SKIP() << "missing interposer library: " << interposer_path;
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hip_ld_preload_stream_boundary");
  const auto src_path = temp_dir / "hip_ld_preload_stream_boundary.cpp";
  const auto exe_path = temp_dir / "hip_ld_preload_stream_boundary.out";
  const auto stdout_path = temp_dir / "stdout.txt";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << R"(
#include <hip/hip_runtime.h>
#include <cstdio>

int main() {
  hipStream_t stream0 = nullptr;
  hipStream_t stream1 = nullptr;

  if (hipStreamCreate(&stream0) != hipSuccess || stream0 == nullptr) return 10;
  if (hipStreamCreate(&stream1) == hipSuccess) return 11;
  if (hipStreamSynchronize(stream0) != hipSuccess) return 12;
  if (hipStreamDestroy(stream0) != hipSuccess) return 13;
  if (hipStreamSynchronize(stream0) == hipSuccess) return 14;
  if (hipStreamDestroy(stream0) == hipSuccess) return 15;

  std::puts("ld_preload single stream boundary ok");
  return 0;
}
)";
  }

  const std::string compile_command =
      "env -u LD_PRELOAD hipcc " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(compile_command.c_str()), 0);

  const std::string run_command =
      "env LD_PRELOAD=" + interposer_path.string() +
      " GPU_MODEL_HIP_INTERPOSER_DEBUG=1 " + exe_path.string() + " > " + stdout_path.string() +
      " 2>&1";
  ASSERT_EQ(std::system(run_command.c_str()), 0);

  std::ifstream in(stdout_path);
  ASSERT_TRUE(static_cast<bool>(in));
  const std::string output((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  EXPECT_NE(output.find("ld_preload single stream boundary ok"), std::string::npos);

  std::filesystem::remove_all(temp_dir);
}

TEST(HipInterposerStateTest, RunsHipHostExecutableWithMemsetAsyncThroughLdPreloadInterposer) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto build_dir = BuildDirPath();
  const auto interposer_path = build_dir / "libgpu_model_hip_interposer.so";
  if (!std::filesystem::exists(interposer_path)) {
    GTEST_SKIP() << "missing interposer library: " << interposer_path;
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hip_ld_preload_memset_async");
  const auto src_path = temp_dir / "hip_ld_preload_memset_async.cpp";
  const auto exe_path = temp_dir / "hip_ld_preload_memset_async.out";
  const auto stdout_path = temp_dir / "stdout.txt";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << R"(
#include <hip/hip_runtime.h>
#include <array>
#include <cstdio>

int main() {
  hipStream_t stream = nullptr;
  void* ptr = nullptr;
  std::array<unsigned char, 64> host{};

  if (hipStreamCreate(&stream) != hipSuccess) return 10;
  if (hipMalloc(&ptr, host.size()) != hipSuccess) return 11;
  if (hipMemsetAsync(ptr, 0x5a, host.size(), stream) != hipSuccess) return 12;
  if (hipStreamSynchronize(stream) != hipSuccess) return 13;
  if (hipMemcpy(host.data(), ptr, host.size(), hipMemcpyDeviceToHost) != hipSuccess) return 14;
  for (unsigned char value : host) {
    if (value != static_cast<unsigned char>(0x5a)) return 15;
  }
  if (hipFree(ptr) != hipSuccess) return 16;
  if (hipStreamDestroy(stream) != hipSuccess) return 17;

  std::puts("ld_preload memset async ok");
  return 0;
}
)";
  }

  const std::string compile_command =
      "env -u LD_PRELOAD hipcc " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(compile_command.c_str()), 0);

  const std::string run_command =
      "env LD_PRELOAD=" + interposer_path.string() +
      " GPU_MODEL_HIP_INTERPOSER_DEBUG=1 " + exe_path.string() + " > " + stdout_path.string() +
      " 2>&1";
  ASSERT_EQ(std::system(run_command.c_str()), 0);

  std::ifstream in(stdout_path);
  ASSERT_TRUE(static_cast<bool>(in));
  const std::string output((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  EXPECT_NE(output.find("ld_preload memset async ok"), std::string::npos);

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

TEST(HipInterposerStateTest, LaunchesHipDynamicSharedExecutableThroughRegisteredHostFunction) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hip_interposer_dynamic_shared");
  const auto src_path = temp_dir / "hip_dynamic_shared.cpp";
  const auto exe_path = temp_dir / "hip_dynamic_shared.out";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << "#include <hip/hip_runtime.h>\n\n"
           "extern \"C\" __global__ void dynamic_shared_sum(int* out) {\n"
           "  extern __shared__ int scratch[];\n"
           "  int tid = threadIdx.x;\n"
           "  scratch[tid] = tid + 1;\n"
           "  __syncthreads();\n"
           "  if (tid == 0) {\n"
           "    int acc = 0;\n"
           "    for (int i = 0; i < blockDim.x; ++i) acc += scratch[i];\n"
           "    out[0] = acc;\n"
           "  }\n"
           "}\n\n"
           "int main() { return 0; }\n";
  }

  const std::string command = "hipcc " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  auto& state = HipInterposerState::Instance();
  state.ResetForTest();
  static int host_symbol = 0;
  state.RegisterFunction(&host_symbol, "dynamic_shared_sum");

  int32_t output = 0;
  void* out_dev = state.AllocateDevice(sizeof(int32_t));
  state.MemcpyHostToDevice(out_dev, &output, sizeof(output));

  void* args[] = {&out_dev};
  constexpr uint32_t block_dim = 64;
  const auto result = state.LaunchExecutableKernel(
      exe_path,
      &host_symbol,
      LaunchConfig{
          .grid_dim_x = 1,
          .block_dim_x = block_dim,
          .shared_memory_bytes = block_dim * sizeof(int32_t),
      },
      args);
  ASSERT_TRUE(result.ok) << result.error_message;

  state.MemcpyDeviceToHost(&output, out_dev, sizeof(output));
  EXPECT_EQ(output, static_cast<int32_t>(block_dim * (block_dim + 1) / 2));

  std::filesystem::remove_all(temp_dir);
}

TEST(HipInterposerStateTest, LaunchesHipAtomicCountExecutableThroughRegisteredHostFunction) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hip_interposer_atomic_count");
  const auto src_path = temp_dir / "hip_atomic_count.cpp";
  const auto exe_path = temp_dir / "hip_atomic_count.out";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << "#include <hip/hip_runtime.h>\n\n"
           "extern \"C\" __global__ void atomic_count(int* out, int n) {\n"
           "  int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
           "  if (i < n) atomicAdd(out, 1);\n"
           "}\n\n"
           "int main() { return 0; }\n";
  }

  const std::string command = "hipcc " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  auto& state = HipInterposerState::Instance();
  state.ResetForTest();
  static int host_symbol = 0;
  state.RegisterFunction(&host_symbol, "atomic_count");

  struct AtomicCase {
    const char* name = nullptr;
    uint32_t grid_dim_x = 1;
    uint32_t block_dim_x = 1;
    uint32_t n = 1;
  };
  const std::vector<AtomicCase> cases = {
      {.name = "single", .grid_dim_x = 1, .block_dim_x = 1, .n = 1},
      {.name = "wave", .grid_dim_x = 1, .block_dim_x = 64, .n = 64},
      {.name = "multi_block", .grid_dim_x = 3, .block_dim_x = 128, .n = 257},
  };

  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.name);
    int32_t zero = 0;
    void* out_dev = state.AllocateDevice(sizeof(int32_t));
    state.MemcpyHostToDevice(out_dev, &zero, sizeof(zero));
    uint32_t n_arg = test_case.n;
    void* args[] = {&out_dev, &n_arg};
    const auto result = state.LaunchExecutableKernel(
        exe_path, &host_symbol,
        LaunchConfig{.grid_dim_x = test_case.grid_dim_x, .block_dim_x = test_case.block_dim_x},
        args);
    ASSERT_TRUE(result.ok) << result.error_message;

    int32_t value = -1;
    state.MemcpyDeviceToHost(&value, out_dev, sizeof(value));
    EXPECT_EQ(value, static_cast<int32_t>(test_case.n));
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

TEST(HipInterposerStateTest, LaunchesHipBlockReduceExecutableThroughRegisteredHostFunction) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hip_interposer_block_reduce");
  const auto src_path = temp_dir / "hip_block_reduce.cpp";
  const auto exe_path = temp_dir / "hip_block_reduce.out";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << "#include <hip/hip_runtime.h>\n"
           "extern \"C\" __global__ void block_reduce_sum(const float* in, float* out, int n) {\n"
           "  __shared__ float scratch[256];\n"
           "  int tid = threadIdx.x;\n"
           "  int stride = blockDim.x * gridDim.x;\n"
           "  int idx = blockIdx.x * blockDim.x + tid;\n"
           "  float sum = 0.0f;\n"
           "  for (int i = idx; i < n; i += stride) {\n"
           "    sum += in[i];\n"
           "  }\n"
           "  scratch[tid] = sum;\n"
           "  __syncthreads();\n"
           "  for (int step = blockDim.x / 2; step > 0; step >>= 1) {\n"
           "    if (tid < step) scratch[tid] += scratch[tid + step];\n"
           "    __syncthreads();\n"
           "  }\n"
           "  if (tid == 0) out[blockIdx.x] = scratch[0];\n"
           "}\n"
           "int main() { return 0; }\n";
  }

  const std::string command = "hipcc " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  auto& state = HipInterposerState::Instance();
  state.ResetForTest();
  static int host_symbol = 0;
  state.RegisterFunction(&host_symbol, "block_reduce_sum");

  constexpr uint32_t n = 1024;
  constexpr uint32_t grid_dim = 4;
  constexpr uint32_t block_dim = 256;
  std::vector<float> input(n, 1.0f);
  std::vector<float> output(grid_dim, -1.0f);
  std::vector<float> expect(grid_dim, static_cast<float>(n / grid_dim));
  void* in_dev = state.AllocateDevice(n * sizeof(float));
  void* out_dev = state.AllocateDevice(grid_dim * sizeof(float));
  state.MemcpyHostToDevice(in_dev, input.data(), n * sizeof(float));
  state.MemcpyHostToDevice(out_dev, output.data(), grid_dim * sizeof(float));

  uint32_t n_arg = n;
  void* args[] = {&in_dev, &out_dev, &n_arg};
  const auto result = state.LaunchExecutableKernel(
      exe_path, &host_symbol, LaunchConfig{.grid_dim_x = grid_dim, .block_dim_x = block_dim}, args);
  ASSERT_TRUE(result.ok) << result.error_message;

  state.MemcpyDeviceToHost(output.data(), out_dev, grid_dim * sizeof(float));
  for (uint32_t i = 0; i < grid_dim; ++i) {
    EXPECT_NEAR(output[i], expect[i], 1.0e-4f);
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

TEST(HipInterposerStateTest, BuildsExecutableLoadPlanForHipMfmaWithTypedTensorAbi) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hip_interposer_mfma_plan");
  const auto src_path = temp_dir / "hip_mfma_plan.cpp";
  const auto exe_path = temp_dir / "hip_mfma_plan.out";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << "#include <hip/hip_runtime.h>\n"
           "typedef float v4f __attribute__((ext_vector_type(4)));\n"
           "extern \"C\" __global__ void mfma_plan_probe(float* out) {\n"
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
  state.RegisterFunction(&host_symbol, "mfma_plan_probe");

  const auto plan = state.BuildExecutableLoadPlan(exe_path, &host_symbol);
  EXPECT_FALSE(plan.segments.empty());
  const auto image = ObjectReader{}.LoadEncodedObject(exe_path, "mfma_plan_probe");
  EXPECT_GE(image.kernel_descriptor.accum_offset, 4u);
  if (const auto it = image.metadata.values.find("agpr_count"); it != image.metadata.values.end()) {
    EXPECT_EQ(std::to_string(image.kernel_descriptor.agpr_count), it->second);
  }

  std::filesystem::remove_all(temp_dir);
}

}  // namespace
}  // namespace gpu_model
