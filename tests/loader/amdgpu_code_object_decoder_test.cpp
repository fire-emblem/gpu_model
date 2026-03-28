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

TEST(AmdgpuCodeObjectDecoderTest, DecodesRawInstructionsFromHipBiasChainExecutable) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_code_object_hip_bias");
  const auto src_path = temp_dir / "hip_bias_chain.cpp";
  const auto exe_path = temp_dir / "hip_bias_chain.out";
  {
    std::ofstream out(src_path);
    ASSERT_TRUE(static_cast<bool>(out));
    out << "#include <hip/hip_runtime.h>\n"
           "extern \"C\" __global__ void bias_chain(const float* a, const float* b, float* c, int n, float b0, float b1, float b2) {\n"
           "  int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
           "  if (i < n) {\n"
           "    c[i] = a[i] + b[i] + b0 + b1 + b2;\n"
           "  }\n"
           "}\n"
           "int main() { return 0; }\n";
  }
  const std::string command = "hipcc " + src_path.string() + " -o " + exe_path.string();
  ASSERT_EQ(std::system(command.c_str()), 0);

  const auto image = AmdgpuCodeObjectDecoder{}.Decode(exe_path, "bias_chain");
  EXPECT_EQ(image.kernel_name, "bias_chain");
  EXPECT_EQ(image.metadata.values.at("arg_count"), "7");
  EXPECT_EQ(image.metadata.values.at("arg_layout"),
            "global_buffer:8,global_buffer:8,global_buffer:8,by_value:4,by_value:4,by_value:4,by_value:4");

  const auto add_float_count = std::count_if(
      image.instructions.begin(), image.instructions.end(),
      [](const RawGcnInstruction& inst) { return inst.mnemonic == "v_add_f32_e32"; });
  EXPECT_GE(add_float_count, 4);

  const auto load_scalar_count = std::count_if(
      image.instructions.begin(), image.instructions.end(),
      [](const RawGcnInstruction& inst) {
        return inst.mnemonic == "s_load_dword" || inst.mnemonic == "s_load_dwordx4" ||
               inst.mnemonic == "s_load_dwordx2";
      });
  EXPECT_GE(load_scalar_count, 4);
}

TEST(AmdgpuCodeObjectDecoderTest, DecodesHipSoftmaxExecutableWithoutUnknownInstructions) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_code_object_hip_softmax");
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

  const auto image = AmdgpuCodeObjectDecoder{}.Decode(exe_path, "softmax_row");
  EXPECT_EQ(image.kernel_name, "softmax_row");
  EXPECT_EQ(image.metadata.values.at("arg_count"), "3");
  const auto barrier_count = std::count_if(
      image.instructions.begin(), image.instructions.end(),
      [](const RawGcnInstruction& inst) { return inst.mnemonic == "s_barrier"; });
  const auto ds_write_count = std::count_if(
      image.instructions.begin(), image.instructions.end(),
      [](const RawGcnInstruction& inst) { return inst.mnemonic == "ds_write_b32"; });
  const auto ds_read_count = std::count_if(
      image.instructions.begin(), image.instructions.end(),
      [](const RawGcnInstruction& inst) { return inst.mnemonic == "ds_read_b32"; });
  const auto vmax_count = std::count_if(
      image.instructions.begin(), image.instructions.end(),
      [](const RawGcnInstruction& inst) { return inst.mnemonic == "v_max_f32_e32"; });
  const auto unknown_count = std::count_if(
      image.instructions.begin(), image.instructions.end(),
      [](const RawGcnInstruction& inst) { return inst.mnemonic == "unknown"; });
  EXPECT_GT(barrier_count, 0);
  EXPECT_GT(ds_write_count, 0);
  EXPECT_GT(ds_read_count, 0);
  EXPECT_GT(vmax_count, 0);
  EXPECT_EQ(unknown_count, 0);
}

TEST(AmdgpuCodeObjectDecoderTest, DecodesHipMfmaExecutableWithoutUnknownInstructions) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_code_object_hip_mfma");
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

  const auto image = AmdgpuCodeObjectDecoder{}.Decode(exe_path, "mfma_probe");
  EXPECT_EQ(image.kernel_name, "mfma_probe");
  const auto unknown_count = std::count_if(
      image.instructions.begin(), image.instructions.end(),
      [](const RawGcnInstruction& inst) { return inst.mnemonic == "unknown"; });
  EXPECT_EQ(unknown_count, 0);
  const auto mfma_it = std::find_if(
      image.instructions.begin(), image.instructions.end(),
      [](const RawGcnInstruction& inst) {
        return inst.mnemonic == "v_mfma_f32_16x16x4f32";
      });
  ASSERT_NE(mfma_it, image.instructions.end());
}

TEST(AmdgpuCodeObjectDecoderTest, DecodesHipSharedReverseExecutable) {
  if (!HasHipHostToolchain()) {
    GTEST_SKIP() << "required HIP/LLVM tools not available";
  }

  const auto temp_dir = MakeUniqueTempDir("gpu_model_code_object_hip_shared_reverse");
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

  const auto image = AmdgpuCodeObjectDecoder{}.Decode(exe_path, "shared_reverse");
  EXPECT_EQ(image.kernel_name, "shared_reverse");
  EXPECT_EQ(image.metadata.values.at("arg_count"), "3");
  EXPECT_EQ(image.metadata.values.at("group_segment_fixed_size"), "256");
  const auto ds_write_it = std::find_if(
      image.instructions.begin(), image.instructions.end(),
      [](const RawGcnInstruction& inst) { return inst.mnemonic == "ds_write_b32"; });
  ASSERT_NE(ds_write_it, image.instructions.end());
  const auto ds_read_it = std::find_if(
      image.instructions.begin(), image.instructions.end(),
      [](const RawGcnInstruction& inst) { return inst.mnemonic == "ds_read_b32"; });
  ASSERT_NE(ds_read_it, image.instructions.end());
  const auto barrier_it = std::find_if(
      image.instructions.begin(), image.instructions.end(),
      [](const RawGcnInstruction& inst) { return inst.mnemonic == "s_barrier"; });
  ASSERT_NE(barrier_it, image.instructions.end());
}

}  // namespace
}  // namespace gpu_model
