#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>

#include "gpu_model/loader/amdgpu_obj_loader.h"

namespace gpu_model {
namespace {

TEST(AmdgpuObjLoaderTest, LoadsKernelProgramFromAmdgpuObjectFile) {
  if (std::system("command -v llc >/dev/null 2>&1") != 0 ||
      std::system("command -v llvm-objdump >/dev/null 2>&1") != 0 ||
      std::system("command -v readelf >/dev/null 2>&1") != 0) {
    GTEST_SKIP() << "required LLVM/binutils tools not available";
  }

  const auto temp_dir = std::filesystem::temp_directory_path() / "gpu_model_amdgpu_obj_loader";
  std::filesystem::create_directories(temp_dir);
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
      "llc -march=amdgcn -mcpu=gfx900 -filetype=obj " + ir_path.string() + " -o " + obj_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  const auto image = AmdgpuObjLoader{}.LoadFromObject(obj_path);
  EXPECT_EQ(image.kernel_name(), "empty_kernel");
  EXPECT_NE(image.assembly_text().find("s_endpgm"), std::string::npos);

  std::filesystem::remove_all(temp_dir);
}

}  // namespace
}  // namespace gpu_model
