#include "gpu_model/loader/amdgpu_code_object_decoder.h"

#include <array>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <vector>

#include "gpu_model/loader/amdgpu_obj_loader.h"

namespace gpu_model {

namespace {

std::string Trim(std::string_view text) {
  size_t begin = 0;
  size_t end = text.size();
  while (begin < end && std::isspace(static_cast<unsigned char>(text[begin])) != 0) {
    ++begin;
  }
  while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1])) != 0) {
    --end;
  }
  return std::string(text.substr(begin, end - begin));
}

std::string ShellQuote(const std::string& text) {
  std::string quoted = "'";
  for (const char ch : text) {
    if (ch == '\'') {
      quoted += "'\\''";
    } else {
      quoted.push_back(ch);
    }
  }
  quoted.push_back('\'');
  return quoted;
}

std::string RunCommand(std::string command) {
  std::array<char, 4096> buffer{};
  std::string output;
  FILE* pipe = popen(command.c_str(), "r");
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

class ScopedTempDir {
 public:
  ScopedTempDir() {
    std::string pattern =
        (std::filesystem::temp_directory_path() / "gpu_model_code_object_XXXXXX").string();
    buffer_.assign(pattern.begin(), pattern.end());
    buffer_.push_back('\0');
    char* created = ::mkdtemp(buffer_.data());
    if (created == nullptr) {
      throw std::runtime_error("failed to create temp directory for code-object decode");
    }
    path_ = created;
  }

  ~ScopedTempDir() {
    std::error_code ec;
    std::filesystem::remove_all(path_, ec);
  }

  const std::filesystem::path& path() const { return path_; }

 private:
  std::vector<char> buffer_;
  std::filesystem::path path_;
};

bool IsAmdgpuElf(const std::filesystem::path& path) {
  const std::string header = RunCommand("readelf -h " + ShellQuote(path.string()));
  return header.find("Machine:                           AMD GPU") != std::string::npos;
}

bool HasHipFatbin(const std::filesystem::path& path) {
  const std::string sections = RunCommand("readelf -S " + ShellQuote(path.string()));
  return sections.find(".hip_fatbin") != std::string::npos;
}

std::filesystem::path MaterializeDeviceCodeObject(const std::filesystem::path& path,
                                                  const ScopedTempDir& temp_dir) {
  if (IsAmdgpuElf(path)) {
    return path;
  }
  if (!HasHipFatbin(path)) {
    throw std::runtime_error("ELF is neither AMDGPU code object nor HIP fatbin host artifact: " +
                             path.string());
  }

  const auto fatbin_path = temp_dir.path() / "kernel.hip_fatbin";
  const auto device_path = temp_dir.path() / "kernel_device.co";
  RunCommand("llvm-objcopy --dump-section .hip_fatbin=" + ShellQuote(fatbin_path.string()) + " " +
             ShellQuote(path.string()));
  const std::string bundles = RunCommand("clang-offload-bundler --list --type=o --input=" +
                                         ShellQuote(fatbin_path.string()));
  std::istringstream bundle_stream(bundles);
  std::string bundle;
  std::string target;
  while (std::getline(bundle_stream, bundle)) {
    if (bundle.find("amdgcn-amd-amdhsa") != std::string::npos) {
      target = Trim(bundle);
      break;
    }
  }
  if (target.empty()) {
    throw std::runtime_error("HIP fatbin does not contain an AMDGPU device bundle");
  }
  RunCommand("clang-offload-bundler --unbundle --type=o --input=" + ShellQuote(fatbin_path.string()) +
             " --targets=" + ShellQuote(target) + " --output=" + ShellQuote(device_path.string()));
  return device_path;
}

bool IsFunctionHeader(const std::string& line) {
  const auto trimmed = Trim(line);
  return !trimmed.empty() && trimmed.back() == ':' && trimmed.find('<') != std::string::npos &&
         trimmed.find('>') != std::string::npos;
}

std::string ExtractFunctionName(const std::string& line) {
  const auto trimmed = Trim(line);
  const size_t begin = trimmed.find('<');
  const size_t end = trimmed.find('>');
  if (begin == std::string::npos || end == std::string::npos || end <= begin + 1) {
    throw std::runtime_error("failed to parse function header: " + line);
  }
  return trimmed.substr(begin + 1, end - begin - 1);
}

std::vector<uint32_t> ParseRawWords(const std::string& line) {
  const size_t comment = line.find("//");
  if (comment == std::string::npos) {
    return {};
  }
  std::string comment_text = Trim(std::string_view(line).substr(comment + 2));
  const size_t colon = comment_text.find(':');
  if (colon == std::string::npos) {
    return {};
  }
  comment_text = Trim(std::string_view(comment_text).substr(colon + 1));
  std::istringstream input(comment_text);
  std::vector<uint32_t> words;
  std::string token;
  while (input >> token) {
    if (token.size() != 8) {
      continue;
    }
    bool all_hex = true;
    for (const char ch : token) {
      if (std::isxdigit(static_cast<unsigned char>(ch)) == 0) {
        all_hex = false;
        break;
      }
    }
    if (all_hex) {
      words.push_back(static_cast<uint32_t>(std::stoul(token, nullptr, 16)));
    }
  }
  return words;
}

RawGcnInstruction ParseInstructionLine(const std::string& line) {
  RawGcnInstruction instruction;
  const auto trimmed = Trim(line);
  const size_t comment = trimmed.find("//");
  std::string asm_text = comment == std::string::npos ? trimmed : Trim(std::string_view(trimmed).substr(0, comment));
  size_t address_colon = asm_text.find(':');
  if (address_colon != std::string::npos) {
    bool all_hex = true;
    for (size_t i = 0; i < address_colon; ++i) {
      if (std::isxdigit(static_cast<unsigned char>(asm_text[i])) == 0) {
        all_hex = false;
        break;
      }
    }
    if (all_hex) {
      instruction.pc = std::stoull(asm_text.substr(0, address_colon), nullptr, 16);
      asm_text = Trim(std::string_view(asm_text).substr(address_colon + 1));
    }
  }
  const size_t space = asm_text.find_first_of(" \t");
  instruction.mnemonic = space == std::string::npos ? asm_text : asm_text.substr(0, space);
  instruction.operands = space == std::string::npos ? "" : Trim(std::string_view(asm_text).substr(space + 1));
  instruction.words = ParseRawWords(line);
  instruction.size_bytes = static_cast<uint32_t>(instruction.words.size() * sizeof(uint32_t));
  instruction.format_class = ClassifyGcnInstFormat(instruction.words);
  return instruction;
}

}  // namespace

AmdgpuCodeObjectImage AmdgpuCodeObjectDecoder::Decode(const std::filesystem::path& path,
                                                      std::optional<std::string> kernel_name) const {
  ScopedTempDir temp_dir;
  const auto device_path = MaterializeDeviceCodeObject(path, temp_dir);

  const ProgramImage image = AmdgpuObjLoader{}.LoadFromObject(path, kernel_name);
  AmdgpuCodeObjectImage code_object;
  code_object.kernel_name = image.kernel_name();
  code_object.metadata = image.metadata();

  const std::string disassembly =
      RunCommand("llvm-objdump -d " + ShellQuote(device_path.string()));
  std::string current_function;
  std::istringstream input(disassembly);
  std::string line;
  while (std::getline(input, line)) {
    if (line.find("Disassembly of section") != std::string::npos) {
      continue;
    }
    if (IsFunctionHeader(line)) {
      current_function = ExtractFunctionName(line);
      continue;
    }
    if (current_function != code_object.kernel_name) {
      continue;
    }
    const auto instruction = ParseInstructionLine(line);
    if (instruction.mnemonic.empty() || instruction.words.empty()) {
      continue;
    }
    code_object.instructions.push_back(instruction);
    for (uint32_t word : instruction.words) {
      code_object.code_bytes.push_back(static_cast<std::byte>(word & 0xffu));
      code_object.code_bytes.push_back(static_cast<std::byte>((word >> 8u) & 0xffu));
      code_object.code_bytes.push_back(static_cast<std::byte>((word >> 16u) & 0xffu));
      code_object.code_bytes.push_back(static_cast<std::byte>((word >> 24u) & 0xffu));
    }
  }

  if (code_object.instructions.empty()) {
    throw std::runtime_error("failed to decode AMDGPU kernel instructions: " + code_object.kernel_name);
  }
  return code_object;
}

}  // namespace gpu_model
