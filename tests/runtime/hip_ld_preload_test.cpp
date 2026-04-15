#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

#include "program/program_object/object_reader.h"
#include "runtime/hip_runtime/hip_runtime.h"
#include "runtime/model_runtime/runtime_session.h"
#include "tests/test_utils/hipcc_cache_test_utils.h"

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

std::string ShellQuote(const std::string& text) {
  std::string quoted = "'";
  for (char ch : text) {
    if (ch == '\'') {
      quoted += "'\\''";
    } else {
      quoted.push_back(ch);
    }
  }
  quoted.push_back('\'');
  return quoted;
}

std::filesystem::path ResolveAsanRuntimePath(
    const std::filesystem::path& abi_library_path) {
  const std::string command =
      "ldd " + ShellQuote(abi_library_path.string()) + " 2>/dev/null";
  std::array<char, 4096> buffer{};
  std::string output;
  FILE* pipe = popen(command.c_str(), "r");
  if (pipe == nullptr) {
    return {};
  }
  while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
    output += buffer.data();
  }
  const int status = pclose(pipe);
  if (status != 0) {
    return {};
  }

  std::istringstream input(output);
  std::string line;
  while (std::getline(input, line)) {
    if (line.find("libasan") == std::string::npos) {
      continue;
    }
    const auto arrow = line.find("=>");
    if (arrow == std::string::npos) {
      continue;
    }
    const auto begin = line.find_first_not_of(' ', arrow + 2);
    if (begin == std::string::npos) {
      continue;
    }
    const auto end = line.find(' ', begin);
    const auto path = line.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
    if (!path.empty() && std::filesystem::exists(path)) {
      return path;
    }
  }
  return {};
}

std::string MakeLdPreloadValue(const std::filesystem::path& abi_library_path) {
  const auto asan_runtime = ResolveAsanRuntimePath(abi_library_path);
  if (asan_runtime.empty()) {
    return abi_library_path.string();
  }
  return asan_runtime.string() + ":" + abi_library_path.string();
}

TEST(HipLdPreloadTest, LaunchesHipVecAddExecutableThroughRegisteredHostFunction) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hip_ld_preload_vecadd");
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

  const std::string command = test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  HipRuntime state;
  state.ResetAbiState();
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

TEST(HipLdPreloadTest, LaunchesHipVecAddExecutableInCycleModeThroughRegisteredHostFunction) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hip_ld_preload_vecadd_cycle");
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

  const std::string command = test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  HipRuntime state;
  state.ResetAbiState();
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

TEST(HipLdPreloadTest, BuildsExecutableLoadPlanThroughRegisteredHostFunction) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hip_ld_preload_load_plan");
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

  const std::string command = test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  HipRuntime state;
  state.ResetAbiState();
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

TEST(HipLdPreloadTest, LaunchesHipFmaLoopExecutableThroughRegisteredHostFunction) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hip_ld_preload_fma_loop");
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

  const std::string command = test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  HipRuntime state;
  state.ResetAbiState();
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

TEST(HipLdPreloadTest, LaunchesHipBiasChainExecutableThroughRegisteredHostFunction) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hip_ld_preload_bias_chain");
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

  const std::string command = test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  HipRuntime state;
  state.ResetAbiState();
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

TEST(HipLdPreloadTest, LaunchesHipByValueAggregateExecutableThroughRegisteredHostFunction) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hip_ld_preload_by_value_aggregate");
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

  const std::string command = test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  HipRuntime state;
  state.ResetAbiState();
  static int host_symbol = 0;
  state.RegisterFunction(&host_symbol, "by_value_aggregate");

  const auto image = ObjectReader{}.LoadProgramObject(exe_path, "by_value_aggregate");
  ASSERT_TRUE(image.metadata().values.contains("arg_layout"));
  EXPECT_NE(image.metadata().values.at("arg_layout").find("by_value:"), std::string::npos);

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

TEST(HipLdPreloadTest, LaunchesHipThreeDimensionalHiddenArgsExecutableThroughRegisteredHostFunction) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hip_ld_preload_hidden_args_3d");
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

  const std::string command = test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  HipRuntime state;
  state.ResetAbiState();
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

TEST(HipLdPreloadTest, LaunchesHipThreeDimensionalBuiltinIdsExecutableThroughRegisteredHostFunction) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hip_ld_preload_builtin_ids_3d");
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

  const std::string command = test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  HipRuntime state;
  state.ResetAbiState();
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

TEST(HipLdPreloadTest, LaunchesHipVecAddExecutableThroughRegisteredHostFunctionAtLargeScale) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hip_ld_preload_vecadd_large");
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

  const std::string command = test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  HipRuntime state;
  state.ResetAbiState();
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

TEST(HipLdPreloadTest, LaunchesHipVecAddExecutableThroughManagedAllocations) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hip_ld_preload_managed_vecadd");
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

  const std::string command = test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  HipRuntime state;
  state.ResetAbiState();
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
  EXPECT_EQ(GetRuntimeSession().memory().pool_memory_size(MemoryPoolKind::Managed),
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

TEST(HipLdPreloadTest, RunsHipHostExecutableThroughLdPreloadHipLdPreload) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto build_dir = BuildDirPath();
  const auto abi_library_path = build_dir / "libgpu_model_hip_ld_preload.so";
  if (!std::filesystem::exists(abi_library_path)) {
    GTEST_SKIP() << "missing hip runtime abi library: " << abi_library_path;
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

  const std::string compile_command = "env -u LD_PRELOAD " + test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(compile_command.c_str()), 0);

  const std::string run_command =
      "env LD_PRELOAD=" + MakeLdPreloadValue(abi_library_path) +
      " GPU_MODEL_LOG_MODULES=hip_ld_preload GPU_MODEL_LOG_LEVEL=info " + exe_path.string() + " > " +
      stdout_path.string() + " 2>&1";
  ASSERT_EQ(std::system(run_command.c_str()), 0);

  std::ifstream in(stdout_path);
  ASSERT_TRUE(static_cast<bool>(in));
  const std::string output((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  EXPECT_NE(output.find("ld_preload host path ok"), std::string::npos);

  std::filesystem::remove_all(temp_dir);
}

TEST(HipLdPreloadTest, RunsHipHostExecutableWithEventsThroughLdPreloadHipLdPreload) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto build_dir = BuildDirPath();
  const auto abi_library_path = build_dir / "libgpu_model_hip_ld_preload.so";
  if (!std::filesystem::exists(abi_library_path)) {
    GTEST_SKIP() << "missing hip runtime abi library: " << abi_library_path;
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
      "env -u LD_PRELOAD " + test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(compile_command.c_str()), 0);

  const std::string run_command =
      "env LD_PRELOAD=" + MakeLdPreloadValue(abi_library_path) +
      " GPU_MODEL_LOG_MODULES=hip_ld_preload GPU_MODEL_LOG_LEVEL=info " + exe_path.string() + " > " + stdout_path.string() +
      " 2>&1";
  ASSERT_EQ(std::system(run_command.c_str()), 0);

  std::ifstream in(stdout_path);
  ASSERT_TRUE(static_cast<bool>(in));
  const std::string output((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  EXPECT_NE(output.find("ld_preload events path ok"), std::string::npos);

  std::filesystem::remove_all(temp_dir);
}

TEST(HipLdPreloadTest, RunsHipHostExecutableQueryingDevicePropertiesThroughLdPreloadHipLdPreload) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto build_dir = BuildDirPath();
  const auto abi_library_path = build_dir / "libgpu_model_hip_ld_preload.so";
  if (!std::filesystem::exists(abi_library_path)) {
    GTEST_SKIP() << "missing hip runtime abi library: " << abi_library_path;
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
  if (std::strcmp(props.name, "mac500") != 0) return 14;
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
      "env -u LD_PRELOAD " + test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(compile_command.c_str()), 0);

  const std::string run_command =
      "env LD_PRELOAD=" + MakeLdPreloadValue(abi_library_path) +
      " GPU_MODEL_LOG_MODULES=hip_ld_preload GPU_MODEL_LOG_LEVEL=info " + exe_path.string() + " > " + stdout_path.string() +
      " 2>&1";
  ASSERT_EQ(std::system(run_command.c_str()), 0);

  std::ifstream in(stdout_path);
  ASSERT_TRUE(static_cast<bool>(in));
  const std::string output((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  EXPECT_NE(output.find("ld_preload property path ok"), std::string::npos);

  std::filesystem::remove_all(temp_dir);
}

TEST(HipLdPreloadTest, EnforcesSingleStreamBoundaryThroughLdPreloadHipLdPreload) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto build_dir = BuildDirPath();
  const auto abi_library_path = build_dir / "libgpu_model_hip_ld_preload.so";
  if (!std::filesystem::exists(abi_library_path)) {
    GTEST_SKIP() << "missing hip runtime abi library: " << abi_library_path;
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
      "env -u LD_PRELOAD " + test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(compile_command.c_str()), 0);

  const std::string run_command =
      "env LD_PRELOAD=" + MakeLdPreloadValue(abi_library_path) +
      " GPU_MODEL_LOG_MODULES=hip_ld_preload GPU_MODEL_LOG_LEVEL=info " + exe_path.string() + " > " + stdout_path.string() +
      " 2>&1";
  ASSERT_EQ(std::system(run_command.c_str()), 0);

  std::ifstream in(stdout_path);
  ASSERT_TRUE(static_cast<bool>(in));
  const std::string output((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  EXPECT_NE(output.find("ld_preload single stream boundary ok"), std::string::npos);

  std::filesystem::remove_all(temp_dir);
}

TEST(HipLdPreloadTest, RunsHipHostExecutableWithMemsetAsyncThroughLdPreloadHipLdPreload) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto build_dir = BuildDirPath();
  const auto abi_library_path = build_dir / "libgpu_model_hip_ld_preload.so";
  if (!std::filesystem::exists(abi_library_path)) {
    GTEST_SKIP() << "missing hip runtime abi library: " << abi_library_path;
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
      "env -u LD_PRELOAD " + test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(compile_command.c_str()), 0);

  const std::string run_command =
      "env LD_PRELOAD=" + MakeLdPreloadValue(abi_library_path) +
      " GPU_MODEL_LOG_MODULES=hip_ld_preload GPU_MODEL_LOG_LEVEL=info " + exe_path.string() + " > " + stdout_path.string() +
      " 2>&1";
  ASSERT_EQ(std::system(run_command.c_str()), 0);

  std::ifstream in(stdout_path);
  ASSERT_TRUE(static_cast<bool>(in));
  const std::string output((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  EXPECT_NE(output.find("ld_preload memset async ok"), std::string::npos);

  std::filesystem::remove_all(temp_dir);
}

TEST(HipLdPreloadTest, RunsHipHostExecutableWithPureMemoryApisThroughLdPreloadHipLdPreload) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto build_dir = BuildDirPath();
  const auto abi_library_path = build_dir / "libgpu_model_hip_ld_preload.so";
  if (!std::filesystem::exists(abi_library_path)) {
    GTEST_SKIP() << "missing hip runtime abi library: " << abi_library_path;
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hip_ld_preload_memory_only");
  const auto src_path = temp_dir / "hip_ld_preload_memory_only.cpp";
  const auto exe_path = temp_dir / "hip_ld_preload_memory_only.out";
  const auto stdout_path = temp_dir / "stdout.txt";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << R"(
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>

int main() {
  constexpr int count = 16;
  std::vector<unsigned int> input(count);
  std::vector<unsigned int> output(count, 0);
  std::vector<unsigned int> copied(count, 0);
  std::vector<unsigned int> filled(count, 0);
  std::vector<unsigned short> half_filled(count, 0);
  std::vector<unsigned char> byte_filled(count, 0);
  for (int i = 0; i < count; ++i) {
    input[i] = 100u + static_cast<unsigned int>(i) * 9u;
  }

  void* src = nullptr;
  void* dst = nullptr;
  void* fill = nullptr;
  void* fill16 = nullptr;
  void* fill8 = nullptr;
  if (hipMalloc(&src, count * sizeof(unsigned int)) != hipSuccess) return 10;
  if (hipMalloc(&dst, count * sizeof(unsigned int)) != hipSuccess) return 11;
  if (hipMalloc(&fill, count * sizeof(unsigned int)) != hipSuccess) return 12;
  if (hipMalloc(&fill16, count * sizeof(unsigned short)) != hipSuccess) return 13;
  if (hipMalloc(&fill8, count * sizeof(unsigned char)) != hipSuccess) return 14;

  if (hipMemcpy(src, input.data(), count * sizeof(unsigned int), hipMemcpyHostToDevice) != hipSuccess) return 15;
  if (hipMemset(dst, 0, count * sizeof(unsigned int)) != hipSuccess) return 16;
  if (hipMemcpy(dst, src, count * sizeof(unsigned int), hipMemcpyDeviceToDevice) != hipSuccess) return 17;
  if (hipMemcpy(output.data(), dst, count * sizeof(unsigned int), hipMemcpyDeviceToHost) != hipSuccess) return 18;
  for (int i = 0; i < count; ++i) {
    if (output[i] != input[i]) return 20;
  }

  if (hipMemsetD8(reinterpret_cast<hipDeviceptr_t>(fill8), 0x5a, count) != hipSuccess) return 21;
  if (hipMemcpy(byte_filled.data(), fill8, count * sizeof(unsigned char), hipMemcpyDeviceToHost) != hipSuccess) return 22;
  for (int i = 0; i < count; ++i) {
    if (byte_filled[i] != 0x5au) return 23;
  }

  if (hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(fill), 0xdeadbeef, count) != hipSuccess) return 24;
  if (hipMemcpy(filled.data(), fill, count * sizeof(unsigned int), hipMemcpyDeviceToHost) != hipSuccess) return 25;
  for (int i = 0; i < count; ++i) {
    if (filled[i] != 0xdeadbeefu) return 26;
  }

  if (hipMemsetD16(reinterpret_cast<hipDeviceptr_t>(fill16), 0xbeef, count) != hipSuccess) return 27;
  if (hipMemcpy(half_filled.data(), fill16, count * sizeof(unsigned short), hipMemcpyDeviceToHost) != hipSuccess) return 28;
  for (int i = 0; i < count; ++i) {
    if (half_filled[i] != 0xbeefu) return 29;
  }

  if (hipMemcpy(copied.data(), src, count * sizeof(unsigned int), hipMemcpyDeviceToHost) != hipSuccess) return 30;
  for (int i = 0; i < count; ++i) {
    if (copied[i] != input[i]) return 31;
  }

  if (hipFree(src) != hipSuccess) return 32;
  if (hipFree(dst) != hipSuccess) return 33;
  if (hipFree(fill) != hipSuccess) return 34;
  if (hipFree(fill16) != hipSuccess) return 35;
  if (hipFree(fill8) != hipSuccess) return 36;
  std::puts("ld_preload pure memory api ok");
  return 0;
}
)";
  }

  const std::string compile_command =
      "env -u LD_PRELOAD " + test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " +
      exe_path.string();
  ASSERT_EQ(std::system(compile_command.c_str()), 0);

  const std::string run_command =
      "env LD_PRELOAD=" + MakeLdPreloadValue(abi_library_path) +
      " GPU_MODEL_LOG_MODULES=hip_ld_preload GPU_MODEL_LOG_LEVEL=info " + exe_path.string() + " > " + stdout_path.string() +
      " 2>&1";
  ASSERT_EQ(std::system(run_command.c_str()), 0);

  std::ifstream in(stdout_path);
  ASSERT_TRUE(static_cast<bool>(in));
  const std::string output((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  EXPECT_NE(output.find("ld_preload pure memory api ok"), std::string::npos);

  std::filesystem::remove_all(temp_dir);
}

TEST(HipLdPreloadTest, ReturnsInvalidValueForInvalidMemcpyKindsAndPointersThroughLdPreloadHipLdPreload) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto build_dir = BuildDirPath();
  const auto abi_library_path = build_dir / "libgpu_model_hip_ld_preload.so";
  if (!std::filesystem::exists(abi_library_path)) {
    GTEST_SKIP() << "missing hip runtime abi library: " << abi_library_path;
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hip_ld_preload_invalid_memcpy");
  const auto src_path = temp_dir / "hip_ld_preload_invalid_memcpy.cpp";
  const auto exe_path = temp_dir / "hip_ld_preload_invalid_memcpy.out";
  const auto stdout_path = temp_dir / "stdout.txt";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << R"(
#include <hip/hip_runtime.h>
#include <cstdio>

int main() {
  int input = 7;
  int output = 0;
  void* dev = nullptr;
  if (hipMalloc(&dev, sizeof(int)) != hipSuccess) return 10;

  if (hipMemcpy(&output, &input, sizeof(int), static_cast<hipMemcpyKind>(999)) != hipErrorInvalidValue) return 11;
  if (hipMemcpy(&output, &input, sizeof(int), hipMemcpyHostToDevice) != hipErrorInvalidValue) return 12;
  if (hipMemcpy(&output, &input, sizeof(int), hipMemcpyDeviceToHost) != hipErrorInvalidValue) return 13;
  if (hipMemcpy(&output, &input, sizeof(int), hipMemcpyDeviceToDevice) != hipErrorInvalidValue) return 14;

  if (hipMemcpy(dev, &input, sizeof(int), hipMemcpyHostToDevice) != hipSuccess) return 15;
  if (hipMemcpy(&output, dev, sizeof(int), hipMemcpyDeviceToHost) != hipSuccess) return 16;
  if (output != input) return 17;
  if (hipFree(dev) != hipSuccess) return 18;
  std::puts("ld_preload invalid memcpy api ok");
  return 0;
}
)";
  }

  const std::string compile_command =
      "env -u LD_PRELOAD " + test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " +
      exe_path.string();
  ASSERT_EQ(std::system(compile_command.c_str()), 0);

  const std::string run_command =
      "env LD_PRELOAD=" + MakeLdPreloadValue(abi_library_path) +
      " GPU_MODEL_LOG_MODULES=hip_ld_preload GPU_MODEL_LOG_LEVEL=info " + exe_path.string() + " > " + stdout_path.string() +
      " 2>&1";
  ASSERT_EQ(std::system(run_command.c_str()), 0);

  std::ifstream in(stdout_path);
  ASSERT_TRUE(static_cast<bool>(in));
  const std::string output((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  EXPECT_NE(output.find("ld_preload invalid memcpy api ok"), std::string::npos);

  std::filesystem::remove_all(temp_dir);
}

TEST(HipLdPreloadTest, RejectsInteriorPointerFreeWithoutInvalidatingBaseThroughLdPreloadHipLdPreload) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto build_dir = BuildDirPath();
  const auto abi_library_path = build_dir / "libgpu_model_hip_ld_preload.so";
  if (!std::filesystem::exists(abi_library_path)) {
    GTEST_SKIP() << "missing hip runtime abi library: " << abi_library_path;
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hip_ld_preload_interior_free");
  const auto src_path = temp_dir / "hip_ld_preload_interior_free.cpp";
  const auto exe_path = temp_dir / "hip_ld_preload_interior_free.out";
  const auto stdout_path = temp_dir / "stdout.txt";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << R"(
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>

int main() {
  std::uint32_t input = 1234;
  std::uint32_t output = 0;
  unsigned char* dev = nullptr;
  if (hipMalloc(reinterpret_cast<void**>(&dev), sizeof(std::uint32_t) * 2) != hipSuccess) return 10;
  if (hipMemcpy(dev, &input, sizeof(input), hipMemcpyHostToDevice) != hipSuccess) return 11;
  if (hipFree(dev + 1) != hipErrorInvalidValue) return 12;
  if (hipMemcpy(&output, dev, sizeof(output), hipMemcpyDeviceToHost) != hipSuccess) return 13;
  if (output != input) return 14;
  if (hipFree(dev) != hipSuccess) return 15;
  std::puts("ld_preload interior free ok");
  return 0;
}
)";
  }

  const std::string compile_command =
      "env -u LD_PRELOAD " + test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " +
      exe_path.string();
  ASSERT_EQ(std::system(compile_command.c_str()), 0);

  const std::string run_command =
      "env LD_PRELOAD=" + MakeLdPreloadValue(abi_library_path) +
      " GPU_MODEL_LOG_MODULES=hip_ld_preload GPU_MODEL_LOG_LEVEL=info " + exe_path.string() + " > " + stdout_path.string() +
      " 2>&1";
  ASSERT_EQ(std::system(run_command.c_str()), 0);

  std::ifstream in(stdout_path);
  ASSERT_TRUE(static_cast<bool>(in));
  const std::string output((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  EXPECT_NE(output.find("ld_preload interior free ok"), std::string::npos);

  std::filesystem::remove_all(temp_dir);
}

TEST(HipLdPreloadTest, RunsHipHostExecutableWithManagedMemorySyncThroughLdPreloadHipLdPreload) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto build_dir = BuildDirPath();
  const auto abi_library_path = build_dir / "libgpu_model_hip_ld_preload.so";
  if (!std::filesystem::exists(abi_library_path)) {
    GTEST_SKIP() << "missing hip runtime abi library: " << abi_library_path;
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hip_ld_preload_managed_memory_only");
  const auto src_path = temp_dir / "hip_ld_preload_managed_memory_only.cpp";
  const auto exe_path = temp_dir / "hip_ld_preload_managed_memory_only.out";
  const auto stdout_path = temp_dir / "stdout.txt";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << R"(
#include <hip/hip_runtime.h>
#include <cstdio>

int main() {
  constexpr int count = 8;
  int* ptr = nullptr;
  if (hipMallocManaged(&ptr, count * sizeof(int)) != hipSuccess) return 10;

  for (int i = 0; i < count; ++i) {
    ptr[i] = 10 + i;
  }
  if (hipDeviceSynchronize() != hipSuccess) return 11;

  for (int i = 0; i < count; ++i) {
    if (ptr[i] != 10 + i) return 20;
  }

  if (hipMemset(ptr, 0, count * sizeof(int)) != hipSuccess) return 21;
  if (hipDeviceSynchronize() != hipSuccess) return 22;
  for (int i = 0; i < count; ++i) {
    if (ptr[i] != 0) return 23;
  }

  if (hipFree(ptr) != hipSuccess) return 24;
  std::puts("ld_preload managed memory sync ok");
  return 0;
}
)";
  }

  const std::string compile_command =
      "env -u LD_PRELOAD " + test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " +
      exe_path.string();
  ASSERT_EQ(std::system(compile_command.c_str()), 0);

  const std::string run_command =
      "env LD_PRELOAD=" + MakeLdPreloadValue(abi_library_path) +
      " GPU_MODEL_LOG_MODULES=hip_ld_preload GPU_MODEL_LOG_LEVEL=info " + exe_path.string() + " > " + stdout_path.string() +
      " 2>&1";
  ASSERT_EQ(std::system(run_command.c_str()), 0);

  std::ifstream in(stdout_path);
  ASSERT_TRUE(static_cast<bool>(in));
  const std::string output((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  EXPECT_NE(output.find("ld_preload managed memory sync ok"), std::string::npos);

  std::filesystem::remove_all(temp_dir);
}

TEST(HipLdPreloadTest, RunsHipHostExecutableWithMemcpyAsyncThroughLdPreloadHipLdPreload) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto build_dir = BuildDirPath();
  const auto abi_library_path = build_dir / "libgpu_model_hip_ld_preload.so";
  if (!std::filesystem::exists(abi_library_path)) {
    GTEST_SKIP() << "missing hip runtime abi library: " << abi_library_path;
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hip_ld_preload_memcpy_async_only");
  const auto src_path = temp_dir / "hip_ld_preload_memcpy_async_only.cpp";
  const auto exe_path = temp_dir / "hip_ld_preload_memcpy_async_only.out";
  const auto stdout_path = temp_dir / "stdout.txt";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << R"(
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>

int main() {
  constexpr int count = 16;
  std::vector<int> input(count);
  std::vector<int> output(count, -1);
  for (int i = 0; i < count; ++i) {
    input[i] = 3 * i + 1;
  }

  hipStream_t stream = nullptr;
  if (hipStreamCreate(&stream) != hipSuccess) return 10;

  int* ptr = nullptr;
  if (hipMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(int)) != hipSuccess) return 11;
  if (hipMemcpyAsync(ptr, input.data(), count * sizeof(int), hipMemcpyHostToDevice, stream) != hipSuccess) return 12;
  if (hipMemcpyAsync(output.data(), ptr, count * sizeof(int), hipMemcpyDeviceToHost, stream) != hipSuccess) return 13;
  if (hipStreamSynchronize(stream) != hipSuccess) return 14;

  for (int i = 0; i < count; ++i) {
    if (output[i] != input[i]) return 20;
  }

  if (hipFree(ptr) != hipSuccess) return 21;
  if (hipStreamDestroy(stream) != hipSuccess) return 22;
  std::puts("ld_preload memcpy async abi ok");
  return 0;
}
)";
  }

  const std::string compile_command =
      "env -u LD_PRELOAD " + test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " +
      exe_path.string();
  ASSERT_EQ(std::system(compile_command.c_str()), 0);

  const std::string run_command =
      "env LD_PRELOAD=" + MakeLdPreloadValue(abi_library_path) +
      " GPU_MODEL_LOG_MODULES=hip_ld_preload GPU_MODEL_LOG_LEVEL=info " + exe_path.string() + " > " + stdout_path.string() +
      " 2>&1";
  ASSERT_EQ(std::system(run_command.c_str()), 0);

  std::ifstream in(stdout_path);
  ASSERT_TRUE(static_cast<bool>(in));
  const std::string output((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  EXPECT_NE(output.find("ld_preload memcpy async abi ok"), std::string::npos);

  std::filesystem::remove_all(temp_dir);
}

TEST(HipLdPreloadTest, RunsHipHostExecutableCheckingLastErrorApisThroughLdPreloadHipLdPreload) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto build_dir = BuildDirPath();
  const auto abi_library_path = build_dir / "libgpu_model_hip_ld_preload.so";
  if (!std::filesystem::exists(abi_library_path)) {
    GTEST_SKIP() << "missing hip runtime abi library: " << abi_library_path;
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hip_ld_preload_last_error_only");
  const auto src_path = temp_dir / "hip_ld_preload_last_error_only.cpp";
  const auto exe_path = temp_dir / "hip_ld_preload_last_error_only.out";
  const auto stdout_path = temp_dir / "stdout.txt";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << R"(
#include <hip/hip_runtime.h>
#include <cstdio>

int main() {
  if (hipGetLastError() != hipSuccess) return 10;
  if (hipPeekAtLastError() != hipSuccess) return 11;

  int* ptr = nullptr;
  if (hipMalloc(reinterpret_cast<void**>(&ptr), sizeof(int)) != hipSuccess) return 12;
  if (hipPeekAtLastError() != hipSuccess) return 13;
  if (hipGetLastError() != hipSuccess) return 14;
  if (hipGetLastError() != hipSuccess) return 15;
  if (hipFree(ptr) != hipSuccess) return 16;

  std::puts("ld_preload last error api ok");
  return 0;
}
)";
  }

  const std::string compile_command =
      "env -u LD_PRELOAD " + test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " +
      exe_path.string();
  ASSERT_EQ(std::system(compile_command.c_str()), 0);

  const std::string run_command =
      "env LD_PRELOAD=" + MakeLdPreloadValue(abi_library_path) +
      " GPU_MODEL_LOG_MODULES=hip_ld_preload GPU_MODEL_LOG_LEVEL=info " + exe_path.string() + " > " + stdout_path.string() +
      " 2>&1";
  ASSERT_EQ(std::system(run_command.c_str()), 0);

  std::ifstream in(stdout_path);
  ASSERT_TRUE(static_cast<bool>(in));
  const std::string output((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  EXPECT_NE(output.find("ld_preload last error api ok"), std::string::npos);

  std::filesystem::remove_all(temp_dir);
}

TEST(HipLdPreloadTest, LaunchesHipSharedReverseExecutableThroughRegisteredHostFunction) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hip_ld_preload_shared_reverse");
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

  const std::string command = test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  HipRuntime state;
  state.ResetAbiState();
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

TEST(HipLdPreloadTest, LaunchesHipDynamicSharedExecutableThroughRegisteredHostFunction) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hip_ld_preload_dynamic_shared");
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

  const std::string command = test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  HipRuntime state;
  state.ResetAbiState();
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

TEST(HipLdPreloadTest, LaunchesHipAtomicCountExecutableThroughRegisteredHostFunction) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hip_ld_preload_atomic_count");
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

  const std::string command = test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  HipRuntime state;
  state.ResetAbiState();
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

TEST(HipLdPreloadTest, LaunchesHipSoftmaxExecutableThroughRegisteredHostFunction) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hip_ld_preload_softmax");
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

  const std::string command = test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  HipRuntime state;
  state.ResetAbiState();
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

TEST(HipLdPreloadTest, LaunchesHipBlockReduceExecutableThroughRegisteredHostFunction) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hip_ld_preload_block_reduce");
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

  const std::string command = test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  HipRuntime state;
  state.ResetAbiState();
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

TEST(HipLdPreloadTest, LaunchesHipMfmaExecutableThroughRegisteredHostFunction) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hip_ld_preload_mfma");
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
      test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + exe_path.string();
  if (std::system(command.c_str()) != 0) {
    GTEST_SKIP() << "gfx90a mfma compilation not available";
  }

  HipRuntime state;
  state.ResetAbiState();
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

TEST(HipLdPreloadTest, BuildsExecutableLoadPlanForHipMfmaWithTypedTensorAbi) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hip_ld_preload_mfma_plan");
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
      test_utils::HipccCacheCommand() + " " + src_path.string() + " -o " + exe_path.string();
  if (std::system(command.c_str()) != 0) {
    GTEST_SKIP() << "gfx90a mfma compilation not available";
  }

  HipRuntime state;
  state.ResetAbiState();
  static int host_symbol = 0;
  state.RegisterFunction(&host_symbol, "mfma_plan_probe");

  const auto plan = state.BuildExecutableLoadPlan(exe_path, &host_symbol);
  EXPECT_FALSE(plan.segments.empty());
  const auto image = ObjectReader{}.LoadProgramObject(exe_path, "mfma_plan_probe");
  EXPECT_GE(image.kernel_descriptor().accum_offset, 4u);
  if (const auto it = image.metadata().values.find("agpr_count"); it != image.metadata().values.end()) {
    EXPECT_EQ(std::to_string(image.kernel_descriptor().agpr_count), it->second);
  }

  std::filesystem::remove_all(temp_dir);
}

}  // namespace
}  // namespace gpu_model
