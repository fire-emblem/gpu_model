#include "program/loader/external_tool_executor.h"

#include <array>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

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
  const std::string probe = "command -v " + std::string(command_name) + " >/dev/null 2>&1";
  return std::system(probe.c_str()) == 0;
}

std::string ExternalToolExecutor::ReadElfHeader(const std::filesystem::path& path) {
  return RunCommand("readelf -h " + ShellQuote(path.string()));
}

std::string ExternalToolExecutor::ReadElfSectionTable(const std::filesystem::path& path) {
  return RunCommand("readelf -S " + ShellQuote(path.string()));
}

std::string ExternalToolExecutor::ReadElfSymbolTable(const std::filesystem::path& path) {
  return RunCommand("readelf -Ws " + ShellQuote(path.string()));
}

std::string ExternalToolExecutor::ReadElfNotes(const std::filesystem::path& path) {
  return RunCommand("llvm-readelf --notes " + ShellQuote(path.string()));
}

std::string ExternalToolExecutor::ReadAmdgpuFileHeaders(const std::filesystem::path& path) {
  return RunCommand("llvm-readobj --file-headers " + ShellQuote(path.string()));
}

void ExternalToolExecutor::DumpElfSection(const std::filesystem::path& path,
                                          const std::string& section_name,
                                          const std::filesystem::path& output_path) {
  RunCommand("llvm-objcopy --dump-section " + section_name + "=" +
             ShellQuote(output_path.string()) + " " + ShellQuote(path.string()));
}

std::string ExternalToolExecutor::ListOffloadBundles(const std::filesystem::path& fatbin_path) {
  return RunCommand("clang-offload-bundler --list --type=o --input=" +
                    ShellQuote(fatbin_path.string()));
}

void ExternalToolExecutor::UnbundleOffloadTarget(const std::filesystem::path& fatbin_path,
                                                 const std::string& target,
                                                 const std::filesystem::path& output_path) {
  RunCommand("clang-offload-bundler --unbundle --type=o --input=" +
             ShellQuote(fatbin_path.string()) + " --targets=" +
             ShellQuote(target) + " --output=" +
             ShellQuote(output_path.string()));
}

std::string ExternalToolExecutor::DisassembleHexByteStreamWithLlvmMc(
    const std::filesystem::path& input_path) {
  return RunCommand("llvm-mc --disassemble --triple=" + std::string(kProjectAmdgpuTriple) +
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
