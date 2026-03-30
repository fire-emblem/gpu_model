#include "gpu_model/program/object_reader.h"

#include "gpu_model/loader/amdgpu_obj_loader.h"

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

#include "gpu_model/isa/kernel_metadata.h"
#include "gpu_model/isa/target_isa.h"
#include "gpu_model/loader/amdgpu_binary_decoder.h"

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

class ScopedTempDir {
 public:
  ScopedTempDir() {
    std::string pattern =
        (std::filesystem::temp_directory_path() / "gpu_model_hip_bundle_XXXXXX").string();
    buffer_.assign(pattern.begin(), pattern.end());
    buffer_.push_back('\0');
    char* created = ::mkdtemp(buffer_.data());
    if (created == nullptr) {
      throw std::runtime_error("failed to create temporary directory for HIP bundle extraction");
    }
    path_ = created;
  }

  ~ScopedTempDir() {
    if (path_.empty()) {
      return;
    }
    std::error_code ec;
    std::filesystem::remove_all(path_, ec);
  }

  const std::filesystem::path& path() const { return path_; }

 private:
  std::vector<char> buffer_;
  std::filesystem::path path_;
};

bool IsAmdgpuElfHeader(const std::string& header) {
  return header.find("Machine:                           AMD GPU") != std::string::npos;
}

bool HasHipFatbinSection(const std::string& sections) {
  return sections.find(".hip_fatbin") != std::string::npos;
}

std::string SelectAmdgpuBundleTarget(const std::string& bundle_list) {
  std::istringstream input(bundle_list);
  std::string line;
  while (std::getline(input, line)) {
    const std::string trimmed = Trim(line);
    if (trimmed.find("amdgcn-amd-amdhsa") != std::string::npos) {
      return trimmed;
    }
  }
  throw std::runtime_error("HIP fatbin does not contain an AMDGPU device bundle");
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

struct NoteKernelArgLayoutEntry {
  KernelArgValueKind arg_kind = KernelArgValueKind::Unknown;
  KernelHiddenArgKind hidden_kind = KernelHiddenArgKind::Unknown;
  std::string kind_name;
  uint32_t offset = 0;
  uint32_t size = 0;
};

struct NoteKernelMetadata {
  std::string name;
  std::vector<NoteKernelArgLayoutEntry> args;
};

std::vector<NoteKernelMetadata> ParseKernelMetadataNotes(const std::string& notes) {
  std::vector<NoteKernelMetadata> kernels;
  std::optional<NoteKernelMetadata> current;
  std::optional<NoteKernelArgLayoutEntry> current_arg;

  const auto finalize_arg = [&]() {
    if (current.has_value() && current_arg.has_value() && !current_arg->kind_name.empty() &&
        current_arg->size != 0 && current_arg->hidden_kind == KernelHiddenArgKind::Unknown) {
      current->args.push_back(*current_arg);
    }
    current_arg.reset();
  };
  const auto finalize_kernel = [&]() {
    finalize_arg();
    if (current.has_value()) {
      kernels.push_back(*current);
      current.reset();
    }
  };

  std::istringstream input(notes);
  std::string line;
  while (std::getline(input, line)) {
    const size_t indent = line.find_first_not_of(' ');
    const std::string trimmed = Trim(line);
    if (trimmed == "amdhsa.kernels:" || trimmed.empty()) {
      continue;
    }
    if (trimmed.rfind("- .", 0) == 0 && indent != std::string::npos && indent <= 2) {
      finalize_kernel();
      current = NoteKernelMetadata{};
      if (trimmed == "- .args:") {
        continue;
      }
    }
    if (trimmed == "- .args:") {
      continue;
    }
    if (!current.has_value()) {
      continue;
    }
    if (trimmed.rfind("- .", 0) == 0) {
      finalize_arg();
      current_arg = NoteKernelArgLayoutEntry{};
      const auto size_pos = trimmed.find(".size:");
      if (size_pos != std::string::npos) {
        current_arg->size = static_cast<uint32_t>(
            std::stoul(Trim(std::string_view(trimmed).substr(size_pos + 6))));
      }
      const auto offset_pos = trimmed.find(".offset:");
      if (offset_pos != std::string::npos) {
        current_arg->offset = static_cast<uint32_t>(
            std::stoul(Trim(std::string_view(trimmed).substr(offset_pos + 8))));
      }
      const auto kind_pos = trimmed.find(".value_kind:");
      if (kind_pos != std::string::npos) {
        current_arg->kind_name = Trim(std::string_view(trimmed).substr(kind_pos + 12));
        current_arg->arg_kind = ParseKernelArgValueKind(current_arg->kind_name);
        current_arg->hidden_kind = ParseKernelHiddenArgKind(current_arg->kind_name);
      }
      continue;
    }
    if (trimmed.rfind(".size:", 0) == 0 && current_arg.has_value()) {
      current_arg->size =
          static_cast<uint32_t>(std::stoul(Trim(std::string_view(trimmed).substr(6))));
      continue;
    }
    if (trimmed.rfind(".offset:", 0) == 0 && current_arg.has_value()) {
      current_arg->offset =
          static_cast<uint32_t>(std::stoul(Trim(std::string_view(trimmed).substr(8))));
      continue;
    }
    if (trimmed.rfind(".value_kind:", 0) == 0 && current_arg.has_value()) {
      current_arg->kind_name = Trim(std::string_view(trimmed).substr(12));
      current_arg->arg_kind = ParseKernelArgValueKind(current_arg->kind_name);
      current_arg->hidden_kind = ParseKernelHiddenArgKind(current_arg->kind_name);
      continue;
    }
    if (trimmed.rfind(".name:", 0) == 0) {
      current->name = Trim(std::string_view(trimmed).substr(6));
      continue;
    }
  }
  finalize_kernel();
  return kernels;
}

void PopulateMetadataFromNotes(const std::filesystem::path& path,
                               const std::string& selected_kernel,
                               MetadataBlob& metadata) {
  const std::string notes = RunCommand("llvm-readelf --notes " + ShellQuote(path.string()));
  const auto kernels = ParseKernelMetadataNotes(notes);
  for (const auto& kernel : kernels) {
    if (kernel.name != selected_kernel) {
      continue;
    }
    std::ostringstream layout;
    uint32_t expected_offset = 0;
    for (size_t i = 0; i < kernel.args.size(); ++i) {
      if (i != 0) {
        layout << ',';
      }
      layout << kernel.args[i].kind_name << ':';
      if (kernel.args[i].offset != expected_offset) {
        layout << kernel.args[i].offset << ':';
      }
      layout << kernel.args[i].size;
      expected_offset = kernel.args[i].offset + kernel.args[i].size;
    }
    metadata.values["arg_layout"] = layout.str();
    metadata.values["arg_count"] = std::to_string(kernel.args.size());
    break;
  }
}

ProgramObject DecodeAmdgpuElfToProgramImage(const std::filesystem::path& path,
                                            std::optional<std::string> kernel_name,
                                            MetadataBlob metadata = {}) {
  if (!std::filesystem::exists(path)) {
    throw std::runtime_error("missing AMDGPU object file: " + path.string());
  }

  const std::string quoted = ShellQuote(path.string());
  const std::string header = RunCommand("readelf -h " + quoted);
  if (!IsAmdgpuElfHeader(header)) {
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

  metadata.values["entry"] = selected;
  PopulateMetadataFromNotes(path, selected, metadata);
  if (metadata.values.find("artifact_path") == metadata.values.end()) {
    metadata.values["artifact_path"] = path.string();
  }
  if (metadata.values.find("target_isa") == metadata.values.end()) {
    SetTargetIsa(metadata, TargetIsa::GcnRawAsm);
  }
  return ProgramObject(selected, asm_text.str(), std::move(metadata));
}

class ObjdumpAmdgpuBinaryDecoder final : public IAmdgpuBinaryDecoder {
 public:
  ProgramObject Decode(const std::filesystem::path& path,
                      std::optional<std::string> kernel_name) const override {
    return DecodeAmdgpuElfToProgramImage(path, std::move(kernel_name));
  }
};

struct BinaryDecoderBinding {
  const IAmdgpuBinaryDecoder* decoder = nullptr;
};

const IAmdgpuBinaryDecoder& DefaultAmdgpuBinaryDecoder() {
  static const ObjdumpAmdgpuBinaryDecoder kObjdumpDecoder;
  static const std::vector<BinaryDecoderBinding> kBindings = {
      {.decoder = &kObjdumpDecoder},
  };
  return *kBindings.front().decoder;
}

ProgramObject LoadFromHipFatbinHostElf(const std::filesystem::path& path,
                                       std::optional<std::string> kernel_name) {
  ScopedTempDir temp_dir;
  const auto fatbin_path = temp_dir.path() / "kernel.hip_fatbin";
  const auto device_path = temp_dir.path() / "kernel_device.co";

  const std::string quoted = ShellQuote(path.string());
  RunCommand("llvm-objcopy --dump-section .hip_fatbin=" + ShellQuote(fatbin_path.string()) + " " +
             quoted);

  const std::string bundle_list = RunCommand("clang-offload-bundler --list --type=o --input=" +
                                             ShellQuote(fatbin_path.string()));
  const std::string bundle_target = SelectAmdgpuBundleTarget(bundle_list);
  RunCommand("clang-offload-bundler --unbundle --type=o --input=" +
             ShellQuote(fatbin_path.string()) + " --targets=" + ShellQuote(bundle_target) +
             " --output=" + ShellQuote(device_path.string()));

  MetadataBlob metadata;
  metadata.values["bundle_target"] = bundle_target;
  metadata.values["loader_source"] = "hip_fatbin";
  metadata.values["artifact_path"] = path.string();
  SetTargetIsa(metadata, TargetIsa::GcnRawAsm);
  return DecodeAmdgpuElfToProgramImage(device_path, std::move(kernel_name), std::move(metadata));
}

}  // namespace

ProgramObject ObjectReader::LoadFromObject(const std::filesystem::path& path,
                                           std::optional<std::string> kernel_name) const {
  if (!std::filesystem::exists(path)) {
    throw std::runtime_error("missing AMDGPU object file: " + path.string());
  }

  const std::string quoted = ShellQuote(path.string());
  const std::string header = RunCommand("readelf -h " + quoted);
  if (IsAmdgpuElfHeader(header)) {
    return DefaultAmdgpuBinaryDecoder().Decode(path, std::move(kernel_name));
  }

  const std::string sections = RunCommand("readelf -S " + quoted);
  if (HasHipFatbinSection(sections)) {
    return LoadFromHipFatbinHostElf(path, std::move(kernel_name));
  }

  throw std::runtime_error("ELF is neither AMDGPU code object nor HIP fatbin host artifact: " +
                           path.string());
}

ProgramObject AmdgpuObjLoader::LoadFromObject(const std::filesystem::path& path,
                                              std::optional<std::string> kernel_name) const {
  return ObjectReader{}.LoadFromObject(path, std::move(kernel_name));
}

}  // namespace gpu_model
