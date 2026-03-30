#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "gpu_model/loader/amdgpu_code_object_decoder.h"
#include "gpu_model/isa/target_isa.h"
#include "gpu_model/runtime/runtime_engine.h"
#include "gpu_model/runtime/hip_runtime.h"

namespace gpu_model {
namespace {

bool HasLlvmMcAmdgpuToolchain() {
  return std::system("command -v llvm-mc >/dev/null 2>&1") == 0 &&
         std::system("command -v llvm-objcopy >/dev/null 2>&1") == 0 &&
         std::system("command -v llvm-objdump >/dev/null 2>&1") == 0 &&
         std::system("command -v llvm-readelf >/dev/null 2>&1") == 0 &&
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

std::string ShellQuote(const std::filesystem::path& path) {
  return "'" + path.string() + "'";
}

std::string ReadTextFile(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("failed to read text file: " + path.string());
  }
  std::ostringstream buffer;
  buffer << input.rdbuf();
  return buffer.str();
}

std::filesystem::path AssembleLlvmMcFixture(const std::string& stem,
                                            const std::filesystem::path& fixture_path) {
  const auto temp_dir = MakeUniqueTempDir(stem);
  const auto asm_path = temp_dir / fixture_path.filename();
  const auto obj_path = temp_dir / (fixture_path.stem().string() + ".o");
  {
    std::ofstream out(asm_path);
    if (!out) {
      throw std::runtime_error("failed to create asm file: " + asm_path.string());
    }
    out << ReadTextFile(fixture_path);
  }
  const std::string command =
      "llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj " +
      ShellQuote(asm_path) + " -o " + ShellQuote(obj_path);
  if (std::system(command.c_str()) != 0) {
    throw std::runtime_error("llvm-mc failed for fixture: " + fixture_path.string());
  }
  return obj_path;
}

TEST(RawCodeObjectLaunchTest, RuntimeEngineLaunchesExplicitEncodedRawInput) {
  if (!HasLlvmMcAmdgpuToolchain()) {
    GTEST_SKIP() << "required llvm-mc/LLVM/binutils tools not available";
  }

  const auto obj_path = AssembleLlvmMcFixture(
      "gpu_model_runtime_raw_code_object",
      std::filesystem::path("tests/asm_cases/loader/kernarg_aggregate_by_value.s"));
  const auto image = AmdgpuCodeObjectDecoder{}.Decode(obj_path, "asm_kernarg_aggregate_by_value");

  RuntimeEngine runtime;
  const uint64_t out_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(out_addr, 0);

  struct AggregateArg {
    int32_t x;
    int32_t y;
    int32_t z;
  } aggregate{3, 5, 7};

  LaunchRequest request;
  request.arch_name = "c500";
  request.raw_code_object = &image;
  request.config = LaunchConfig{.grid_dim_x = 1, .block_dim_x = 64};
  request.args.PushU64(out_addr);
  request.args.PushBytes(&aggregate, sizeof(aggregate));

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  EXPECT_EQ(runtime.memory().LoadGlobalValue<int32_t>(out_addr), 15);

  std::filesystem::remove_all(obj_path.parent_path());
}

}  // namespace
}  // namespace gpu_model
