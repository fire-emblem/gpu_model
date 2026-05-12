#include "tests/test_utils/llvm_mc_test_support.h"

#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unistd.h>
#include <vector>

#include "instruction/decode/encoded/instruction_object.h"
#include "program/program_object/object_reader.h"
#include "program/loader/amdgpu_target_config.h"

namespace gpu_model::test_utils {

namespace {

std::filesystem::path MakeUniqueTempDir(const std::string& stem) {
  const auto suffix = std::to_string(
      std::chrono::steady_clock::now().time_since_epoch().count());
  const auto path = std::filesystem::temp_directory_path() / (stem + "_" + suffix);
  std::filesystem::remove_all(path);
  std::filesystem::create_directories(path);
  return path;
}

std::string ShellQuote(std::string_view text) {
  std::string quoted;
  quoted.push_back('\'');
  for (char c : text) {
    if (c == '\'') {
      quoted += "'\\''";
    } else {
      quoted.push_back(c);
    }
  }
  quoted.push_back('\'');
  return quoted;
}

std::vector<std::byte> ReadBinaryFile(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("failed to read binary file: " + path.string());
  }
  std::vector<char> chars((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
  std::vector<std::byte> bytes;
  bytes.reserve(chars.size());
  for (char ch : chars) {
    bytes.push_back(static_cast<std::byte>(static_cast<unsigned char>(ch)));
  }
  return bytes;
}

std::string NormalizeAssemblyTarget(std::string text) {
  const std::string needle = ".amdgcn_target";
  const std::string replacement =
      ".amdgcn_target \"" + std::string(kProjectAmdgpuTargetId) + "\"";
  size_t pos = 0;
  while ((pos = text.find(needle, pos)) != std::string::npos) {
    const size_t line_end = text.find('\n', pos);
    text.replace(pos, line_end == std::string::npos ? text.size() - pos : line_end - pos,
                 replacement);
    pos += replacement.size();
  }
  return text;
}

std::vector<std::filesystem::path> CandidateToolDirs() {
  std::vector<std::filesystem::path> dirs;

  const auto append_env_dir = [&](const char* env_name, std::string_view suffix = {}) {
    if (const char* value = std::getenv(env_name); value != nullptr && value[0] != '\0') {
      std::filesystem::path path(value);
      if (!suffix.empty()) {
        path /= suffix;
      }
      dirs.emplace_back(std::move(path));
    }
  };

  append_env_dir("GPU_MODEL_TOOLCHAIN_DIR");
  append_env_dir("ROCM_PATH", "llvm/bin");
  append_env_dir("ROCM_PATH", "lib/llvm/bin");
  append_env_dir("ROCM_PATH", "bin");
  append_env_dir("ROCM_HOME", "llvm/bin");
  append_env_dir("ROCM_HOME", "lib/llvm/bin");
  append_env_dir("ROCM_HOME", "bin");
  append_env_dir("HIP_PATH", "llvm/bin");
  append_env_dir("HIP_PATH", "lib/llvm/bin");
  append_env_dir("HIP_PATH", "bin");

  dirs.emplace_back("/opt/rocm/bin");
  dirs.emplace_back("/opt/rocm/llvm/bin");
  dirs.emplace_back("/opt/rocm/lib/llvm/bin");
  dirs.emplace_back("/opt/rocm-6.0.2/bin");
  dirs.emplace_back("/opt/rocm-6.0.2/llvm/bin");
  dirs.emplace_back("/opt/rocm-6.0.2/lib/llvm/bin");
  dirs.emplace_back("/opt/rocm-6.2.0/bin");
  dirs.emplace_back("/opt/rocm-6.2.0/llvm/bin");
  dirs.emplace_back("/opt/rocm-6.2.0/lib/llvm/bin");

  if (const char* home = std::getenv("HOME"); home != nullptr && home[0] != '\0') {
    const std::filesystem::path home_path(home);
    dirs.emplace_back(home_path / "tools/rocm/rocm/opt/rocm/bin");
    dirs.emplace_back(home_path / "tools/rocm/rocm/opt/rocm/llvm/bin");
    dirs.emplace_back(home_path / "tools/rocm/rocm/opt/rocm/lib/llvm/bin");
    dirs.emplace_back(home_path / "tools/rocm/rocm/opt/rocm-6.0.2/bin");
    dirs.emplace_back(home_path / "tools/rocm/rocm/opt/rocm-6.0.2/llvm/bin");
    dirs.emplace_back(home_path / "tools/rocm/rocm/opt/rocm-6.0.2/lib/llvm/bin");
    dirs.emplace_back(home_path / "tools/rocm/rocm/opt/rocm-6.2.0/bin");
    dirs.emplace_back(home_path / "tools/rocm/rocm/opt/rocm-6.2.0/llvm/bin");
    dirs.emplace_back(home_path / "tools/rocm/rocm/opt/rocm-6.2.0/lib/llvm/bin");
  }

  if (const char* repo = std::getenv("GPU_MODEL_TEST_REPO_ROOT"); repo != nullptr && repo[0] != '\0') {
    dirs.emplace_back(std::filesystem::path(repo) / "tools/hipcc/bin");
  }

  std::array<char, 4096> buffer{};
  const ssize_t length = ::readlink("/proc/self/exe", buffer.data(), buffer.size() - 1);
  if (length > 0) {
    buffer[static_cast<size_t>(length)] = '\0';
    const auto build_dir = std::filesystem::path(buffer.data()).parent_path().parent_path();
    dirs.emplace_back(build_dir.parent_path() / "tools/hipcc/bin");
  }

  return dirs;
}

bool CommandExists(std::string_view command) {
  return std::system(("command -v " + std::string(command) + " >/dev/null 2>&1").c_str()) == 0;
}

}  // namespace

std::string ResolveTestTool(std::string_view tool_name) {
  for (const auto& dir : CandidateToolDirs()) {
    const auto candidate = dir / tool_name;
    if (std::filesystem::exists(candidate)) {
      return candidate.string();
    }
  }
  if (CommandExists(tool_name)) {
    return std::string(tool_name);
  }
  throw std::runtime_error("missing required test tool: " + std::string(tool_name));
}

bool HasLlvmMcAmdgpuToolchain() {
  (void)ResolveTestTool("llvm-mc");
  (void)ResolveTestTool("llvm-objcopy");
  (void)ResolveTestTool("llvm-objdump");
  (void)ResolveTestTool("llvm-readelf");
  (void)ResolveTestTool("readelf");
  return true;
}

AssembledModule AssembleAndDecodeLlvmMcModule(const std::string& stem,
                                              const std::string& kernel_name,
                                              const std::string& assembly_text,
                                              std::string_view mcpu) {
  const auto temp_dir = MakeUniqueTempDir(stem);
  const auto asm_path = temp_dir / (kernel_name + ".s");
  const auto obj_path = temp_dir / (kernel_name + ".o");
  {
    std::ofstream out(asm_path);
    if (!out) {
      throw std::runtime_error("failed to create asm file: " + asm_path.string());
    }
    out << NormalizeAssemblyTarget(assembly_text);
  }

  const std::string assemble_command =
      ShellQuote(ResolveTestTool("llvm-mc")) + " -triple=" + std::string(kProjectAmdgpuTriple) +
      " -mcpu=" + std::string(mcpu) + " -filetype=obj " +
      ShellQuote(asm_path.string()) + " -o " + ShellQuote(obj_path.string());
  if (std::system(assemble_command.c_str()) != 0) {
    throw std::runtime_error("llvm-mc failed for asm module: " + kernel_name);
  }

  return AssembledModule{
      .temp_dir = temp_dir,
      .asm_path = asm_path,
      .obj_path = obj_path,
      .image = ObjectReader{}.LoadProgramObject(obj_path, kernel_name),
  };
}

AssembledInstructionStream AssembleInstructionStream(const std::string& stem,
                                                     const std::string& assembly_text,
                                                     uint64_t start_pc,
                                                     std::string_view mcpu) {
  const auto temp_dir = MakeUniqueTempDir(stem);
  const auto asm_path = temp_dir / (stem + ".s");
  const auto obj_path = temp_dir / (stem + ".o");
  const auto text_path = temp_dir / (stem + ".text.bin");
  {
    std::ofstream out(asm_path);
    if (!out) {
      throw std::runtime_error("failed to create asm file: " + asm_path.string());
    }
    out << NormalizeAssemblyTarget(assembly_text);
  }

  const std::string assemble_command =
      ShellQuote(ResolveTestTool("llvm-mc")) + " -triple=" + std::string(kProjectAmdgpuTriple) +
      " -mcpu=" + std::string(mcpu) + " -filetype=obj " +
      ShellQuote(asm_path.string()) + " -o " + ShellQuote(obj_path.string());
  if (std::system(assemble_command.c_str()) != 0) {
    throw std::runtime_error("llvm-mc failed for instruction stream: " + stem);
  }

  const std::string objcopy_command =
      ShellQuote(ResolveTestTool("llvm-objcopy")) + " --dump-section .text=" +
      ShellQuote(text_path.string()) + " " + ShellQuote(obj_path.string());
  if (std::system(objcopy_command.c_str()) != 0) {
    throw std::runtime_error("llvm-objcopy failed to extract .text for instruction stream: " + stem);
  }

  return AssembledInstructionStream{
      .temp_dir = temp_dir,
      .asm_path = asm_path,
      .obj_path = obj_path,
      .text_path = text_path,
      .parsed = InstructionArrayParser::Parse(ReadBinaryFile(text_path), start_pc),
  };
}

std::string WrapAmdgpuKernelAssembly(const std::string& kernel_name,
                                     const std::string& kernel_body_asm,
                                     uint32_t next_free_sgpr,
                                     uint32_t next_free_vgpr,
                                     uint32_t kernarg_segment_size,
                                     uint32_t kernarg_segment_align) {
  std::ostringstream out;
  out << ".amdgcn_target \"" << kProjectAmdgpuTargetId << "\"\n\n";
  out << ".text\n";
  out << ".globl " << kernel_name << "\n";
  out << ".p2align 8\n";
  out << ".type " << kernel_name << ",@function\n";
  out << kernel_name << ":\n";
  out << kernel_body_asm;
  if (!kernel_body_asm.empty() && kernel_body_asm.back() != '\n') {
    out << '\n';
  }
  out << ".Lfunc_end0:\n";
  out << "  .size " << kernel_name << ", .Lfunc_end0-" << kernel_name << "\n\n";
  out << ".rodata\n";
  out << ".p2align 6\n";
  out << ".amdhsa_kernel " << kernel_name << "\n";
  if (kernarg_segment_size != 0) {
    out << "  .amdhsa_user_sgpr_kernarg_segment_ptr 1\n";
  }
  out << "  .amdhsa_next_free_vgpr " << next_free_vgpr << "\n";
  out << "  .amdhsa_next_free_sgpr " << next_free_sgpr << "\n";
  out << "  .amdhsa_accum_offset 4\n";
  out << ".end_amdhsa_kernel\n\n";
  out << ".amdgpu_metadata\n";
  out << "---\n";
  out << "amdhsa.version:\n";
  out << "  - 1\n";
  out << "  - 0\n";
  out << "amdhsa.kernels:\n";
  out << "  - .name: " << kernel_name << "\n";
  out << "    .symbol: " << kernel_name << ".kd\n";
  out << "    .kernarg_segment_size: " << kernarg_segment_size << "\n";
  out << "    .group_segment_fixed_size: 0\n";
  out << "    .private_segment_fixed_size: 0\n";
  out << "    .kernarg_segment_align: " << kernarg_segment_align << "\n";
  out << "    .wavefront_size: 64\n";
  out << "    .sgpr_count: " << next_free_sgpr << "\n";
  out << "    .vgpr_count: " << next_free_vgpr << "\n";
  out << "    .max_flat_workgroup_size: 256\n";
  out << "    .args: []\n";
  out << "...\n";
  out << ".end_amdgpu_metadata\n";
  return out.str();
}

}  // namespace gpu_model::test_utils
