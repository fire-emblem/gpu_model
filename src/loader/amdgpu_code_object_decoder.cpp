#include "gpu_model/loader/amdgpu_code_object_decoder.h"

#include <array>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <vector>

#include "gpu_model/decode/gcn_inst_encoding_def.h"
#include "gpu_model/decode/gcn_inst_decoder.h"
#include "gpu_model/isa/target_isa.h"

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

std::vector<std::string> SplitWhitespace(std::string_view text) {
  std::istringstream input{std::string(text)};
  std::vector<std::string> tokens;
  std::string token;
  while (input >> token) {
    tokens.push_back(token);
  }
  return tokens;
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

struct TextSectionInfo {
  uint64_t addr = 0;
  uint64_t offset = 0;
  uint64_t size = 0;
};

struct SymbolInfo {
  std::string name;
  uint64_t value = 0;
  uint64_t size = 0;
};

struct KernelArgLayoutEntry {
  std::string value_kind;
  uint32_t size = 0;
};

struct NoteKernelMetadata {
  std::string name;
  std::vector<KernelArgLayoutEntry> args;
  uint32_t group_segment_fixed_size = 0;
};

TextSectionInfo ParseTextSectionInfo(const std::string& sections) {
  std::istringstream input(sections);
  std::string line;
  while (std::getline(input, line)) {
    if (line.find(".text") == std::string::npos) {
      continue;
    }
    std::string next_line;
    if (!std::getline(input, next_line)) {
      break;
    }
    const auto head_tokens = SplitWhitespace(line);
    const auto tail_tokens = SplitWhitespace(next_line);
    if (head_tokens.size() < 2 || tail_tokens.empty()) {
      continue;
    }
    const std::string addr_hex = head_tokens[head_tokens.size() - 2];
    const std::string off_hex = head_tokens[head_tokens.size() - 1];
    const std::string size_hex = tail_tokens[0];
    return TextSectionInfo{
        .addr = std::stoull(addr_hex, nullptr, 16),
        .offset = std::stoull(off_hex, nullptr, 16),
        .size = std::stoull(size_hex, nullptr, 16),
    };
  }
  throw std::runtime_error("failed to locate .text section");
}

std::vector<SymbolInfo> ParseFunctionSymbols(const std::string& symbols) {
  std::vector<SymbolInfo> results;
  std::istringstream input(symbols);
  std::string line;
  while (std::getline(input, line)) {
    const std::string trimmed = Trim(line);
    if (trimmed.empty()) {
      continue;
    }
    std::istringstream row(trimmed);
    std::string num;
    std::string value_hex;
    std::string size_dec;
    std::string type;
    std::string bind;
    std::string vis;
    std::string ndx;
    std::string name;
    row >> num >> value_hex >> size_dec >> type >> bind >> vis >> ndx >> name;
    if (type == "FUNC") {
      results.push_back(SymbolInfo{
          .name = name,
          .value = std::stoull(value_hex, nullptr, 16),
          .size = std::stoull(size_dec),
      });
    }
  }
  return results;
}

SymbolInfo SelectKernelSymbol(const std::vector<SymbolInfo>& symbols,
                              std::optional<std::string> kernel_name) {
  if (kernel_name.has_value()) {
    for (const auto& symbol : symbols) {
      if (symbol.name == *kernel_name) {
        return symbol;
      }
    }
    throw std::runtime_error("failed to locate kernel symbol: " + *kernel_name);
  }
  if (!symbols.empty()) {
    return symbols.front();
  }
  throw std::runtime_error("failed to locate any kernel symbol");
}

std::vector<NoteKernelMetadata> ParseKernelMetadataNotes(const std::string& notes) {
  std::vector<NoteKernelMetadata> kernels;
  std::optional<NoteKernelMetadata> current;
  std::optional<KernelArgLayoutEntry> current_arg;

  const auto finalize_arg = [&]() {
    if (current.has_value() && current_arg.has_value() && !current_arg->value_kind.empty() &&
        current_arg->size != 0 && current_arg->value_kind.rfind("hidden_", 0) != 0) {
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
      current_arg = KernelArgLayoutEntry{};
      continue;
    }
    if (trimmed.rfind(".size:", 0) == 0 && current_arg.has_value()) {
      current_arg->size =
          static_cast<uint32_t>(std::stoul(Trim(std::string_view(trimmed).substr(6))));
      continue;
    }
    if (trimmed.rfind(".value_kind:", 0) == 0 && current_arg.has_value()) {
      current_arg->value_kind = Trim(std::string_view(trimmed).substr(12));
      continue;
    }
    if (trimmed.rfind(".name:", 0) == 0) {
      current->name = Trim(std::string_view(trimmed).substr(6));
      continue;
    }
    if (trimmed.rfind(".group_segment_fixed_size:", 0) == 0) {
      current->group_segment_fixed_size =
          static_cast<uint32_t>(std::stoul(Trim(std::string_view(trimmed).substr(26))));
      continue;
    }
  }
  finalize_kernel();
  return kernels;
}

MetadataBlob BuildMetadataFromNotes(const std::filesystem::path& note_source_path,
                                    const std::filesystem::path& artifact_path,
                                    const std::string& kernel_name) {
  MetadataBlob metadata;
  metadata.values["entry"] = kernel_name;
  metadata.values["artifact_path"] = artifact_path.string();
  SetTargetIsa(metadata, TargetIsa::GcnRawAsm);

  const std::string notes =
      RunCommand("llvm-readelf --notes " + ShellQuote(note_source_path.string()));
  const auto kernels = ParseKernelMetadataNotes(notes);
  for (const auto& kernel : kernels) {
    if (kernel.name != kernel_name) {
      continue;
    }
    std::ostringstream layout;
    for (size_t i = 0; i < kernel.args.size(); ++i) {
      if (i != 0) {
        layout << ',';
      }
      layout << kernel.args[i].value_kind << ':' << kernel.args[i].size;
    }
    metadata.values["arg_layout"] = layout.str();
    metadata.values["arg_count"] = std::to_string(kernel.args.size());
    metadata.values["group_segment_fixed_size"] =
        std::to_string(kernel.group_segment_fixed_size);
    break;
  }
  return metadata;
}

std::vector<std::byte> ReadBinaryFile(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("failed to open binary file: " + path.string());
  }
  input.seekg(0, std::ios::end);
  const std::streamsize size = input.tellg();
  input.seekg(0, std::ios::beg);
  std::vector<std::byte> bytes(static_cast<size_t>(size));
  if (size > 0) {
    input.read(reinterpret_cast<char*>(bytes.data()), size);
  }
  return bytes;
}

uint32_t InstructionSizeForFormat(const std::vector<uint32_t>& words,
                                 GcnInstFormatClass format_class) {
  const uint32_t low = words.empty() ? 0u : words[0];
  switch (format_class) {
    case GcnInstFormatClass::Sopp:
    case GcnInstFormatClass::Sopk:
      return 4;
    case GcnInstFormatClass::Sop2:
    case GcnInstFormatClass::Sopc:
      return ((low & 0xffu) == 255u || ((low >> 8u) & 0xffu) == 255u) ? 8u : 4u;
    case GcnInstFormatClass::Sop1:
      return (low & 0xffu) == 255u ? 8u : 4u;
    case GcnInstFormatClass::Vop2:
    case GcnInstFormatClass::Vopc:
      return (low & 0x1ffu) == 255u ? 8u : 4u;
    case GcnInstFormatClass::Vop1:
      return (low & 0x1ffu) == 255u ? 8u : 4u;
    case GcnInstFormatClass::Smrd:
    case GcnInstFormatClass::Smem:
    case GcnInstFormatClass::Vop3a:
    case GcnInstFormatClass::Vop3b:
    case GcnInstFormatClass::Vop3p:
    case GcnInstFormatClass::Ds:
    case GcnInstFormatClass::Flat:
    case GcnInstFormatClass::Mubuf:
    case GcnInstFormatClass::Mtbuf:
    case GcnInstFormatClass::Mimg:
    case GcnInstFormatClass::Exp:
      return 8;
    case GcnInstFormatClass::Vintrp:
      return (((low >> 26u) & 0x3fu) == 0x32u) ? 4u : 8u;
    case GcnInstFormatClass::Unknown:
      break;
  }
  throw std::runtime_error("failed to determine raw instruction size");
}

std::vector<uint32_t> ReadWords(const std::vector<std::byte>& bytes, size_t offset, uint32_t size_bytes) {
  std::vector<uint32_t> words;
  words.reserve(size_bytes / 4);
  for (uint32_t i = 0; i < size_bytes; i += 4) {
    uint32_t word = 0;
    std::memcpy(&word, bytes.data() + offset + i, sizeof(word));
    words.push_back(word);
  }
  return words;
}

}  // namespace

AmdgpuCodeObjectImage AmdgpuCodeObjectDecoder::Decode(const std::filesystem::path& path,
                                                      std::optional<std::string> kernel_name) const {
  ScopedTempDir temp_dir;
  const auto device_path = MaterializeDeviceCodeObject(path, temp_dir);
  AmdgpuCodeObjectImage code_object;
  const auto symbols =
      ParseFunctionSymbols(RunCommand("readelf -Ws " + ShellQuote(device_path.string())));
  const auto selected_symbol = SelectKernelSymbol(symbols, kernel_name);
  code_object.kernel_name = selected_symbol.name;
  code_object.metadata = BuildMetadataFromNotes(device_path, path, code_object.kernel_name);

  const auto text_dump_path = temp_dir.path() / "text.bin";
  RunCommand("llvm-objcopy --dump-section .text=" + ShellQuote(text_dump_path.string()) + " " +
             ShellQuote(device_path.string()));
  const auto text_bytes = ReadBinaryFile(text_dump_path);
  const auto section_info = ParseTextSectionInfo(
      RunCommand("readelf -S " + ShellQuote(device_path.string())));
  const auto symbol_info = selected_symbol;

  const uint64_t kernel_offset = symbol_info.value - section_info.addr;
  if (kernel_offset + symbol_info.size > text_bytes.size()) {
    throw std::runtime_error("kernel symbol range exceeds dumped .text bytes");
  }

  size_t offset = static_cast<size_t>(kernel_offset);
  const size_t end = static_cast<size_t>(kernel_offset + symbol_info.size);
  while (offset < end) {
    uint32_t low = 0;
    std::memcpy(&low, text_bytes.data() + offset, sizeof(low));
    const auto format_class = ClassifyGcnInstFormat({low});
    const uint32_t size_guess = InstructionSizeForFormat({low}, format_class);
    const uint32_t size_bytes = size_guess;
    if (offset + size_bytes > end) {
      throw std::runtime_error("raw instruction exceeds kernel symbol bounds");
    }
    RawGcnInstruction instruction;
    instruction.pc = symbol_info.value + (offset - static_cast<size_t>(kernel_offset));
    instruction.words = ReadWords(text_bytes, offset, size_bytes);
    instruction.size_bytes = size_bytes;
    instruction.format_class = format_class;
    if (const auto* def = FindGcnInstEncodingDef(instruction.words)) {
      instruction.encoding_id = def->id;
      instruction.mnemonic = std::string(def->mnemonic);
    } else {
      instruction.mnemonic = "unknown";
    }
    DecodeGcnOperands(instruction);
    if (instruction.operands.empty() && !instruction.decoded_operands.empty()) {
      std::ostringstream operand_text;
      for (size_t i = 0; i < instruction.decoded_operands.size(); ++i) {
        if (i != 0) {
          operand_text << ", ";
        }
        operand_text << instruction.decoded_operands[i].text;
      }
      instruction.operands = operand_text.str();
    }
    code_object.instructions.push_back(instruction);
    code_object.decoded_instructions.push_back(GcnInstDecoder{}.Decode(instruction));
    for (uint32_t word : instruction.words) {
      code_object.code_bytes.push_back(static_cast<std::byte>(word & 0xffu));
      code_object.code_bytes.push_back(static_cast<std::byte>((word >> 8u) & 0xffu));
      code_object.code_bytes.push_back(static_cast<std::byte>((word >> 16u) & 0xffu));
      code_object.code_bytes.push_back(static_cast<std::byte>((word >> 24u) & 0xffu));
    }
    offset += size_bytes;
  }

  if (code_object.instructions.empty()) {
    throw std::runtime_error("failed to decode AMDGPU kernel instructions: " + code_object.kernel_name);
  }
  return code_object;
}

}  // namespace gpu_model
