#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "gpu_model/program/object_reader.h"
#include "gpu_model/runtime/hip_runtime.h"
#include "gpu_model/runtime/runtime_engine.h"

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

struct RunResult {
  std::vector<float> output;
  double elapsed_ms = 0.0;
};

TEST(HipccParallelExecutionTest, ThreeDimensionalVecaddAddsMatchesBetweenStAndMt) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hipcc_parallel_vecadd_3d");
  const auto src_path = temp_dir / "vecadd_3d_adds.cpp";
  const auto exe_path = temp_dir / "vecadd_3d_adds.out";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << "#include <hip/hip_runtime.h>\n"
           "extern \"C\" __global__ void vecadd_3d_adds(const float* a, const float* b, float* c,\n"
           "                                            int width, int height, int depth) {\n"
           "  int x = blockIdx.x * blockDim.x + threadIdx.x;\n"
           "  int y = blockIdx.y * blockDim.y + threadIdx.y;\n"
           "  int z = blockIdx.z * blockDim.z + threadIdx.z;\n"
           "  if (x < width && y < height && z < depth) {\n"
           "    int idx = (z * height + y) * width + x;\n"
           "    float acc = a[idx] + b[idx];\n"
           "    acc = acc + 0.125f;\n"
           "    acc = acc + 0.25f;\n"
           "    acc = acc + 0.5f;\n"
           "    c[idx] = acc;\n"
           "  }\n"
           "}\n"
           "int main() { return 0; }\n";
  }

  const std::string command = "hipcc " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  constexpr uint32_t width = 17;
  constexpr uint32_t height = 9;
  constexpr uint32_t depth = 9;
  constexpr uint32_t total = width * height * depth;

  std::vector<float> a(total), b(total), expect(total);
  for (uint32_t i = 0; i < total; ++i) {
    a[i] = 0.5f * static_cast<float>(i % 97);
    b[i] = 1.25f + 0.25f * static_cast<float>(i % 13);
    expect[i] = a[i] + b[i] + 0.125f + 0.25f + 0.5f;
  }

  const auto run_mode = [&](FunctionalExecutionMode mode, uint32_t worker_threads) {
    RuntimeEngine runtime;
    runtime.SetFunctionalExecutionConfig(
        FunctionalExecutionConfig{.mode = mode, .worker_threads = worker_threads});
    HipRuntime hooks(&runtime);

    const uint64_t a_addr = hooks.Malloc(total * sizeof(float));
    const uint64_t b_addr = hooks.Malloc(total * sizeof(float));
    const uint64_t c_addr = hooks.Malloc(total * sizeof(float));
    for (uint32_t i = 0; i < total; ++i) {
      runtime.memory().StoreGlobalValue<float>(a_addr + i * sizeof(float), a[i]);
      runtime.memory().StoreGlobalValue<float>(b_addr + i * sizeof(float), b[i]);
      runtime.memory().StoreGlobalValue<float>(c_addr + i * sizeof(float), -1.0f);
    }
    KernelArgPack args;
    args.PushU64(a_addr);
    args.PushU64(b_addr);
    args.PushU64(c_addr);
    args.PushU32(width);
    args.PushU32(height);
    args.PushU32(depth);

    const auto begin = std::chrono::steady_clock::now();
    const auto launch = hooks.LaunchEncodedProgramObject(
        ObjectReader{}.LoadEncodedObject(exe_path, "vecadd_3d_adds"),
        LaunchConfig{
            .grid_dim_x = (width + 3) / 4,
            .grid_dim_y = (height + 3) / 4,
            .grid_dim_z = (depth + 3) / 4,
            .block_dim_x = 4,
            .block_dim_y = 4,
            .block_dim_z = 4,
        },
        std::move(args),
        ExecutionMode::Functional,
        "c500",
        nullptr);
    const auto end = std::chrono::steady_clock::now();
    EXPECT_TRUE(launch.ok) << launch.error_message;

    RunResult result;
    result.elapsed_ms =
        std::chrono::duration<double, std::milli>(end - begin).count();
    result.output.resize(total);
    for (uint32_t i = 0; i < total; ++i) {
      result.output[i] = runtime.memory().LoadGlobalValue<float>(c_addr + i * sizeof(float));
    }
    return result;
  };

  const auto st = run_mode(FunctionalExecutionMode::SingleThreaded, 0);
  const auto mt = run_mode(FunctionalExecutionMode::MultiThreaded, 2);

  for (uint32_t i = 0; i < total; ++i) {
    EXPECT_FLOAT_EQ(st.output[i], expect[i]);
    EXPECT_FLOAT_EQ(mt.output[i], expect[i]);
  }

  std::fprintf(stderr,
               "[gpu_model] hipcc_3d_vecadd_adds functional_st_ms=%.3f functional_mt_ms=%.3f\n",
               st.elapsed_ms,
               mt.elapsed_ms);

  std::filesystem::remove_all(temp_dir);
}

}  // namespace
}  // namespace gpu_model
