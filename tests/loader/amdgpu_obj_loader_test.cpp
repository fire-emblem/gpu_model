#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include "program/program_object/object_reader.h"
#include "gpu_arch/chip_config/amdgpu_target_config.h"
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
  const auto suffix = std::to_string(
      std::chrono::steady_clock::now().time_since_epoch().count());
  const auto path =
      std::filesystem::temp_directory_path() / (stem + "_" + suffix);
  std::filesystem::remove_all(path);
  std::filesystem::create_directories(path);
  return path;
}

TEST(AmdgpuObjLoaderTest, LoadsExecutableKernelFromAmdgpuObjectFile) {
  if (std::system("command -v llc >/dev/null 2>&1") != 0 ||
      std::system("command -v llvm-objdump >/dev/null 2>&1") != 0 ||
      std::system("command -v readelf >/dev/null 2>&1") != 0) {
    GTEST_SKIP() << "required LLVM/binutils tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_amdgpu_obj_loader");
  const auto ir_path = temp_dir / "empty_kernel.ll";
  const auto obj_path = temp_dir / "empty_kernel.out";

  {
    std::ofstream out(ir_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << "target triple = \"amdgcn-amd-amdhsa\"\n\n"
           "define amdgpu_kernel void @empty_kernel() {\n"
           "entry:\n"
           "  ret void\n"
           "}\n";
  }

  const std::string command =
      "llc -march=amdgcn -mcpu=" + std::string(kProjectAmdgpuMcpu) + " -filetype=obj " +
      ir_path.string() + " -o " + obj_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  const auto image = ObjectReader{}.LoadProgramObject(obj_path);
  EXPECT_EQ(image.kernel_name(), "empty_kernel");
  EXPECT_FALSE(image.instructions().empty());
  EXPECT_EQ(image.metadata().values.at("entry"), "empty_kernel");

  std::filesystem::remove_all(temp_dir);
}

TEST(AmdgpuObjLoaderTest, LoadsExecutableKernelFromHipHostObjectWithFatbin) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_hip_host_obj_loader");
  const auto src_path = temp_dir / "hip_empty_kernel.cpp";
  const auto obj_path = temp_dir / "hip_empty_kernel.o";

  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << "#include <hip/hip_runtime.h>\n\n"
           "extern \"C\" __global__ void empty_kernel() {}\n\n"
           "int main() {\n"
           "  return 0;\n"
           "}\n";
  }

  const std::string command =
      test_utils::HipccCacheCommand() + " -c " + src_path.string() + " -o " + obj_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  const auto image = ObjectReader{}.LoadProgramObject(obj_path, "empty_kernel");
  EXPECT_EQ(image.kernel_name(), "empty_kernel");
  EXPECT_FALSE(image.instructions().empty());
  EXPECT_EQ(image.metadata().values.at("entry"), "empty_kernel");

  std::filesystem::remove_all(temp_dir);
}

}  // namespace
}  // namespace gpu_model
