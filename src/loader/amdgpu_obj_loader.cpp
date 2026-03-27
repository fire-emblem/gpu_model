#include "gpu_model/loader/amdgpu_obj_loader.h"

#include <array>
#include <cctype>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

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

std::string ExtractInstruction(const std::string& line) {
  std::string text = line;
  const auto comment = text.find("//");
  if (comment != std::string::npos) {
    text.resize(comment);
  }
  text = Trim(text);
  if (text.empty()) {
    return {};
  }
  if (text.find('\t') != std::string::npos) {
    text = Trim(text.substr(text.find_last_of('\t') + 1));
  }
  const auto colon = text.find(':');
  if (colon != std::string::npos) {
    bool hex_prefix = true;
    for (size_t i = 0; i < colon; ++i) {
      if (std::isxdigit(static_cast<unsigned char>(text[i])) == 0) {
        hex_prefix = false;
        break;
      }
    }
    if (hex_prefix && colon + 1 < text.size()) {
      text = Trim(text.substr(colon + 1));
    }
  }
  return text;
}

}  // namespace

ProgramImage AmdgpuObjLoader::LoadFromObject(const std::filesystem::path& path,
                                             std::optional<std::string> kernel_name) const {
  if (!std::filesystem::exists(path)) {
    throw std::runtime_error("missing AMDGPU object file: " + path.string());
  }

  const std::string quoted = ShellQuote(path.string());
  const std::string header = RunCommand("readelf -h " + quoted);
  if (header.find("Machine:                           AMD GPU") == std::string::npos) {
    throw std::runtime_error("object is not an AMDGPU ELF file: " + path.string());
  }

  const std::string disassembly = RunCommand("llvm-objdump -d " + quoted);
  std::unordered_map<std::string, std::vector<std::string>> functions;
  std::string current_function;

  std::istringstream input(disassembly);
  std::string line;
  while (std::getline(input, line)) {
    if (line.find("Disassembly of section") != std::string::npos) {
      continue;
    }
    if (IsFunctionHeader(line)) {
      current_function = ExtractFunctionName(line);
      functions[current_function];
      continue;
    }
    if (current_function.empty()) {
      continue;
    }
    const std::string instruction = ExtractInstruction(line);
    if (!instruction.empty()) {
      functions[current_function].push_back(instruction);
    }
  }

  if (functions.empty()) {
    throw std::runtime_error("no AMDGPU functions found in object: " + path.string());
  }

  const std::string selected = kernel_name.has_value() ? *kernel_name : functions.begin()->first;
  const auto it = functions.find(selected);
  if (it == functions.end()) {
    throw std::runtime_error("requested kernel not found in object: " + selected);
  }

  std::ostringstream asm_text;
  asm_text << selected << ":\n";
  for (const auto& instruction : it->second) {
    asm_text << "  " << instruction << '\n';
  }

  MetadataBlob metadata;
  metadata.values["entry"] = selected;
  return ProgramImage(selected, asm_text.str(), std::move(metadata));
}

}  // namespace gpu_model
