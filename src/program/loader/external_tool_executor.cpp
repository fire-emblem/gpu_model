#include "program/loader/external_tool_executor.h"

#include <array>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <unistd.h>

#include "program/loader/amdgpu_target_config.h"

namespace gpu_model {

namespace {

std::string ShellQuote(const std::string& text) {
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

std::vector<std::filesystem::path> CandidateToolDirs() {
  std::vector<std::filesystem::path> search_dirs;

  const auto append_env_dir = [&](const char* env_name, std::string_view suffix = {}) {
    if (const char* value = std::getenv(env_name); value != nullptr && value[0] != '\0') {
      std::filesystem::path path(value);
      if (!suffix.empty()) {
        path /= suffix;
      }
      search_dirs.emplace_back(std::move(path));
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

  search_dirs.emplace_back("/opt/rocm/bin");
  search_dirs.emplace_back("/opt/rocm/llvm/bin");
  search_dirs.emplace_back("/opt/rocm/lib/llvm/bin");
  search_dirs.emplace_back("/opt/rocm-6.0.2/bin");
  search_dirs.emplace_back("/opt/rocm-6.0.2/llvm/bin");
  search_dirs.emplace_back("/opt/rocm-6.0.2/lib/llvm/bin");
  search_dirs.emplace_back("/opt/rocm-6.2.0/bin");
  search_dirs.emplace_back("/opt/rocm-6.2.0/llvm/bin");
  search_dirs.emplace_back("/opt/rocm-6.2.0/lib/llvm/bin");

  if (const char* home = std::getenv("HOME"); home != nullptr && home[0] != '\0') {
    const std::filesystem::path home_path(home);
    search_dirs.emplace_back(home_path / "tools/rocm/rocm/opt/rocm/bin");
    search_dirs.emplace_back(home_path / "tools/rocm/rocm/opt/rocm/llvm/bin");
    search_dirs.emplace_back(home_path / "tools/rocm/rocm/opt/rocm/lib/llvm/bin");
    search_dirs.emplace_back(home_path / "tools/rocm/rocm/opt/rocm-6.0.2/bin");
    search_dirs.emplace_back(home_path / "tools/rocm/rocm/opt/rocm-6.0.2/llvm/bin");
    search_dirs.emplace_back(home_path / "tools/rocm/rocm/opt/rocm-6.0.2/lib/llvm/bin");
    search_dirs.emplace_back(home_path / "tools/rocm/rocm/opt/rocm-6.2.0/bin");
    search_dirs.emplace_back(home_path / "tools/rocm/rocm/opt/rocm-6.2.0/llvm/bin");
    search_dirs.emplace_back(home_path / "tools/rocm/rocm/opt/rocm-6.2.0/lib/llvm/bin");
  }

  if (const char* repo = std::getenv("GPU_MODEL_TEST_REPO_ROOT"); repo != nullptr && repo[0] != '\0') {
    search_dirs.emplace_back(std::filesystem::path(repo) / "tools/hipcc/bin");
  }

  try {
    std::array<char, 4096> buf{};
    const ssize_t len =
        readlink("/proc/self/exe", buf.data(), buf.size() - 1);
    if (len > 0) {
      buf[static_cast<size_t>(len)] = '\0';
      std::filesystem::path exe(buf.data());
      auto build_dir = exe.parent_path().parent_path();
      if (std::filesystem::exists(build_dir / "tools/hipcc/bin")) {
        search_dirs.push_back(build_dir / "tools/hipcc/bin");
      }
      if (std::filesystem::exists(build_dir.parent_path() / "tools/hipcc/bin")) {
        search_dirs.push_back(build_dir.parent_path() / "tools/hipcc/bin");
      }
    }
  } catch (...) {
  }

  return search_dirs;
}

std::optional<std::string> FindTool(std::string_view tool_name) {
  for (const auto& dir : CandidateToolDirs()) {
    auto candidate = dir / tool_name;
    if (std::filesystem::exists(candidate)) {
      return candidate.string();
    }
  }

  const std::string probe =
      "command -v " + std::string(tool_name) + " >/dev/null 2>&1";
  if (std::system(probe.c_str()) == 0) {
    return std::string(tool_name);
  }
  return std::nullopt;
}

}  // namespace

std::string ExternalToolExecutor::RunCommand(const std::string& command) {
  std::array<char, 4096> buffer{};
  std::string output;
  const std::string wrapped = "env -u LD_PRELOAD " + command;
  FILE* pipe = popen(wrapped.c_str(), "r");
  if (pipe == nullptr) {
    throw std::runtime_error("failed to execute command: " + command);
  }
  while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
    output += buffer.data();
  }
  const int status = pclose(pipe);
  if (status != 0) {
    throw std::runtime_error("command failed: " + command);
  }
  return output;
}

bool ExternalToolExecutor::HasCommand(std::string_view command_name) {
  return FindTool(command_name).has_value();
}

std::string ExternalToolExecutor::ResolveTool(std::string_view tool_name) {
  if (auto resolved = FindTool(tool_name)) {
    return *resolved;
  }
  throw std::runtime_error("missing required external tool: " + std::string(tool_name));
}

std::string ExternalToolExecutor::ReadElfHeader(const std::filesystem::path& path) {
  return RunCommand(ResolveTool("readelf") + " -h " + ShellQuote(path.string()));
}

std::string ExternalToolExecutor::ReadElfSectionTable(const std::filesystem::path& path) {
  return RunCommand(ResolveTool("readelf") + " -S " + ShellQuote(path.string()));
}

std::string ExternalToolExecutor::ReadElfSymbolTable(const std::filesystem::path& path) {
  return RunCommand(ResolveTool("readelf") + " -Ws " + ShellQuote(path.string()));
}

std::string ExternalToolExecutor::ReadElfNotes(const std::filesystem::path& path) {
  return RunCommand(ResolveTool("llvm-readelf") + " --notes " + ShellQuote(path.string()));
}

std::string ExternalToolExecutor::ReadAmdgpuFileHeaders(const std::filesystem::path& path) {
  return RunCommand(ResolveTool("llvm-readobj") + " --file-headers " + ShellQuote(path.string()));
}

void ExternalToolExecutor::DumpElfSection(const std::filesystem::path& path,
                                          const std::string& section_name,
                                          const std::filesystem::path& output_path) {
  RunCommand(ResolveTool("llvm-objcopy") + " --dump-section " + section_name + "=" +
             ShellQuote(output_path.string()) + " " + ShellQuote(path.string()));
}

std::string ExternalToolExecutor::ListOffloadBundles(const std::filesystem::path& fatbin_path) {
  return RunCommand(ResolveTool("clang-offload-bundler") + " --list --type=o --input=" +
                    ShellQuote(fatbin_path.string()));
}

void ExternalToolExecutor::UnbundleOffloadTarget(const std::filesystem::path& fatbin_path,
                                                 const std::string& target,
                                                 const std::filesystem::path& output_path) {
  RunCommand(ResolveTool("clang-offload-bundler") + " --unbundle --type=o --input=" +
             ShellQuote(fatbin_path.string()) + " --targets=" +
             ShellQuote(target) + " --output=" +
             ShellQuote(output_path.string()));
}

std::string ExternalToolExecutor::DisassembleHexByteStreamWithLlvmMc(
    const std::filesystem::path& input_path) {
  return RunCommand(ResolveTool("llvm-mc") + " --disassemble --triple=" + std::string(kProjectAmdgpuTriple) +
                    " --mcpu=" + std::string(kProjectAmdgpuMcpu) + " " +
                    ShellQuote(input_path.string()));
}

bool ExternalToolExecutor::IsAmdgpuElf(const std::filesystem::path& path) {
  std::string header = ReadElfHeader(path);
  // Match the exact string format from readelf -h output
  return header.find("Machine:                           AMD GPU") != std::string::npos;
}

bool ExternalToolExecutor::HasHipFatbin(const std::filesystem::path& path) {
  // Check for .hip_fatbin section in ELF
  std::string sections = ReadElfSectionTable(path);
  return sections.find(".hip_fatbin") != std::string::npos;
}

bool ExternalToolExecutor::HasLlvmMc() {
  return HasCommand("llvm-mc");
}

}  // namespace gpu_model
