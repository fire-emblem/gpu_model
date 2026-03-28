#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>

#include "gpu_model/loader/amdgpu_code_object_decoder.h"

namespace gpu_model {
namespace {

bool HasHipHostToolchain() {
  return std::system("command -v hipcc >/dev/null 2>&1") == 0 &&
         std::system("command -v clang-offload-bundler >/dev/null 2>&1") == 0 &&
         std::system("command -v llvm-objcopy >/dev/null 2>&1") == 0 &&
         std::system("command -v llvm-objdump >/dev/null 2>&1") == 0 &&
         std::system("command -v llvm-readelf >/dev/null 2>&1") == 0 &&
         std::system("command -v readelf >/dev/null 2>&1") == 0;
}

std::filesystem::path MakeUniqueTempDir(const std::string& stem) {
  const auto suffix = std::to_string(
      std::chrono::steady_clock::now().time_since_epoch().count());
  const auto path = std::filesystem::temp_directory_path() / (stem + "_" + suffix);
  std::filesystem::remove_all(path);
  std::filesystem::create_directories(path);
  return path;
}

TEST(AmdgpuCodeObjectDecoderTest, DecodesRawInstructionsFromAmdgpuObject) {
  if (std::system("command -v llc >/dev/null 2>&1") != 0 ||
      std::system("command -v llvm-objdump >/dev/null 2>&1") != 0 ||
      std::system("command -v readelf >/dev/null 2>&1") != 0) {
    GTEST_SKIP() << "required LLVM/binutils tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_code_object_decoder");
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

  const auto image = AmdgpuCodeObjectDecoder{}.Decode(obj_path, "empty_kernel");
  EXPECT_EQ(image.kernel_name, "empty_kernel");
  ASSERT_FALSE(image.instructions.empty());
  EXPECT_EQ(image.instructions.front().mnemonic, "s_endpgm");
  EXPECT_EQ(image.instructions.front().size_bytes, 4u);
  EXPECT_EQ(image.instructions.front().format_class, GcnInstFormatClass::Sopp);
  EXPECT_EQ(image.instructions.front().encoding_id, 1u);
  EXPECT_EQ(image.code_bytes.size(), 4u);
}

TEST(AmdgpuCodeObjectDecoderTest, DecodesRawInstructionsFromHipExecutable) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_code_object_hip");
  const auto src_path = temp_dir / "hip_vecadd.cpp";
  const auto exe_path = temp_dir / "hip_vecadd.out";
  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << "#include <hip/hip_runtime.h>\n"
           "extern \"C\" __global__ void vecadd(const float* a, const float* b, float* c, int n) {\n"
           "  int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
           "  if (i < n) c[i] = a[i] + b[i];\n"
           "}\n"
           "int main() { return 0; }\n";
  }
  const std::string command = "hipcc " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  const auto image = AmdgpuCodeObjectDecoder{}.Decode(exe_path, "vecadd");
  EXPECT_EQ(image.kernel_name, "vecadd");
  EXPECT_EQ(image.metadata.values.at("arg_count"), "4");
  ASSERT_FALSE(image.instructions.empty());
  EXPECT_EQ(image.instructions.front().mnemonic, "s_load_dword");
  EXPECT_EQ(image.instructions.front().format_class, GcnInstFormatClass::Smrd);
  EXPECT_EQ(image.instructions.front().encoding_id, 2u);
  ASSERT_EQ(image.instructions.front().decoded_operands.size(), 3u);
  EXPECT_FALSE(image.instructions.front().decoded_operands[0].text.empty());
  EXPECT_EQ(image.instructions.front().decoded_operands[0].info.reg_first, 0u);
  EXPECT_EQ(image.instructions.front().decoded_operands[1].info.reg_first, 4u);
  EXPECT_EQ(image.instructions.front().decoded_operands[1].info.reg_count, 2u);
  const auto v_lshl_it = std::find_if(
      image.instructions.begin(), image.instructions.end(),
      [](const RawGcnInstruction& inst) { return inst.mnemonic == "v_lshlrev_b64"; });
  ASSERT_NE(v_lshl_it, image.instructions.end());
  ASSERT_EQ(v_lshl_it->decoded_operands.size(), 3u);
  EXPECT_EQ(v_lshl_it->decoded_operands[0].text, "v[0:1]");
  EXPECT_EQ(v_lshl_it->decoded_operands[0].info.reg_count, 2u);
  EXPECT_EQ(v_lshl_it->decoded_operands[2].text, "v[0:1]");
  EXPECT_EQ(v_lshl_it->decoded_operands[2].info.reg_first, 0u);
  EXPECT_EQ(v_lshl_it->decoded_operands[2].info.reg_count, 2u);
  EXPECT_GT(image.instructions.size(), 20u);
  EXPECT_GT(image.code_bytes.size(), 100u);
}

TEST(AmdgpuCodeObjectDecoderTest, DecodesRawInstructionsFromHipFmaLoopExecutable) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_code_object_hip_fma");
  const auto src_path = temp_dir / "hip_fma_loop.cpp";
  const auto exe_path = temp_dir / "hip_fma_loop.out";
  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << "#include <hip/hip_runtime.h>\n"
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
           "}\n"
           "int main() { return 0; }\n";
  }
  const std::string command = "hipcc " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  const auto image = AmdgpuCodeObjectDecoder{}.Decode(exe_path, "fma_loop");
  EXPECT_EQ(image.kernel_name, "fma_loop");
  EXPECT_EQ(image.metadata.values.at("arg_count"), "5");
  EXPECT_EQ(image.metadata.values.at("arg_layout"),
            "global_buffer:8,global_buffer:8,global_buffer:8,by_value:4,by_value:4");

  const auto cmp_lt_it = std::find_if(
      image.instructions.begin(), image.instructions.end(),
      [](const RawGcnInstruction& inst) { return inst.mnemonic == "s_cmp_lt_i32"; });
  ASSERT_NE(cmp_lt_it, image.instructions.end());
  ASSERT_EQ(cmp_lt_it->decoded_operands.size(), 2u);
  EXPECT_EQ(cmp_lt_it->decoded_operands[0].text, "s9");
  EXPECT_EQ(cmp_lt_it->decoded_operands[1].text, "1");

  const auto fma_it = std::find_if(
      image.instructions.begin(), image.instructions.end(),
      [](const RawGcnInstruction& inst) { return inst.mnemonic == "v_fma_f32"; });
  ASSERT_NE(fma_it, image.instructions.end());
  ASSERT_EQ(fma_it->decoded_operands.size(), 4u);
  EXPECT_EQ(fma_it->decoded_operands[0].text, "v2");
  EXPECT_EQ(fma_it->decoded_operands[1].text, "v3");
  EXPECT_EQ(fma_it->decoded_operands[2].text, "v2");
  EXPECT_EQ(fma_it->decoded_operands[3].text, "v4");

  const auto branch_it = std::find_if(
      image.instructions.begin(), image.instructions.end(),
      [](const RawGcnInstruction& inst) { return inst.mnemonic == "s_cbranch_scc0"; });
  ASSERT_NE(branch_it, image.instructions.end());
  ASSERT_EQ(branch_it->decoded_operands.size(), 1u);
  EXPECT_EQ(branch_it->decoded_operands[0].text, "-6");
  EXPECT_TRUE(branch_it->decoded_operands[0].info.has_immediate);
  EXPECT_EQ(branch_it->decoded_operands[0].info.immediate, -6);
}

}  // namespace
}  // namespace gpu_model
