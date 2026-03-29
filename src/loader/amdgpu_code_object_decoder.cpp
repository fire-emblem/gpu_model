#include "gpu_model/loader/amdgpu_code_object_decoder.h"

#include <array>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <vector>

#include "gpu_model/decode/gcn_inst_encoding_def.h"
#include "gpu_model/decode/gcn_inst_decoder.h"
#include "gpu_model/isa/kernel_metadata.h"
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

struct SectionInfo {
  std::string name;
  uint64_t addr = 0;
  uint64_t offset = 0;
  uint64_t size = 0;
};

struct SymbolInfo {
  std::string name;
  std::string type;
  uint64_t value = 0;
  uint64_t size = 0;
};

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
  std::vector<NoteKernelArgLayoutEntry> hidden_args;
  uint32_t group_segment_fixed_size = 0;
  uint32_t kernarg_segment_size = 0;
  uint32_t private_segment_fixed_size = 0;
  uint32_t sgpr_count = 0;
  uint32_t vgpr_count = 0;
  uint32_t wavefront_size = 0;
  bool uniform_work_group_size = false;
  std::string symbol;
};

SectionInfo ParseSectionInfo(const std::string& sections, std::string_view section_name) {
  std::istringstream input(sections);
  std::string line;
  while (std::getline(input, line)) {
    if (line.find(std::string(section_name)) == std::string::npos) {
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
    return SectionInfo{
        .name = std::string(section_name),
        .addr = std::stoull(addr_hex, nullptr, 16),
        .offset = std::stoull(off_hex, nullptr, 16),
        .size = std::stoull(size_hex, nullptr, 16),
    };
  }
  throw std::runtime_error("failed to locate section: " + std::string(section_name));
}

std::vector<SymbolInfo> ParseSymbols(const std::string& symbols) {
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
    if (!(row >> num >> value_hex >> size_dec >> type >> bind >> vis >> ndx)) {
      continue;
    }
    std::getline(row, name);
    name = Trim(name);
    try {
      results.push_back(SymbolInfo{
          .name = name,
          .type = type,
          .value = std::stoull(value_hex, nullptr, 16),
          .size = std::stoull(size_dec),
      });
    } catch (const std::exception&) {
      continue;
    }
  }
  return results;
}

SymbolInfo SelectKernelSymbol(const std::vector<SymbolInfo>& symbols,
                              std::optional<std::string> kernel_name) {
  if (kernel_name.has_value()) {
    for (const auto& symbol : symbols) {
      if (symbol.type == "FUNC" && symbol.name == *kernel_name) {
        return symbol;
      }
    }
    throw std::runtime_error("failed to locate kernel symbol: " + *kernel_name);
  }
  for (const auto& symbol : symbols) {
    if (symbol.type == "FUNC") {
      return symbol;
    }
  }
  throw std::runtime_error("failed to locate any kernel symbol");
}

SymbolInfo SelectDescriptorSymbol(const std::vector<SymbolInfo>& symbols,
                                  const std::string& descriptor_name) {
  for (const auto& symbol : symbols) {
    if (symbol.type == "OBJECT" && symbol.name == descriptor_name) {
      return symbol;
    }
  }
  throw std::runtime_error("failed to locate kernel descriptor symbol: " + descriptor_name);
}

std::vector<NoteKernelMetadata> ParseKernelMetadataNotes(const std::string& notes) {
  std::vector<NoteKernelMetadata> kernels;
  std::optional<NoteKernelMetadata> current;
  std::optional<NoteKernelArgLayoutEntry> current_arg;

  const auto finalize_arg = [&]() {
    if (current.has_value() && current_arg.has_value() && !current_arg->kind_name.empty() &&
        current_arg->size != 0) {
      if (current_arg->hidden_kind != KernelHiddenArgKind::Unknown) {
        current->hidden_args.push_back(*current_arg);
      } else {
        current->args.push_back(*current_arg);
      }
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
      const auto inline_field = Trim(std::string_view(trimmed).substr(2));
      if (inline_field.rfind(".offset:", 0) == 0) {
        current_arg->offset =
            static_cast<uint32_t>(std::stoul(Trim(std::string_view(inline_field).substr(8))));
      } else if (inline_field.rfind(".size:", 0) == 0) {
        current_arg->size =
            static_cast<uint32_t>(std::stoul(Trim(std::string_view(inline_field).substr(6))));
      } else if (inline_field.rfind(".value_kind:", 0) == 0) {
        current_arg->kind_name = Trim(std::string_view(inline_field).substr(12));
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
    if (trimmed.rfind(".symbol:", 0) == 0) {
      current->symbol = Trim(std::string_view(trimmed).substr(8));
      continue;
    }
    if (trimmed.rfind(".group_segment_fixed_size:", 0) == 0) {
      current->group_segment_fixed_size =
          static_cast<uint32_t>(std::stoul(Trim(std::string_view(trimmed).substr(26))));
      continue;
    }
    if (trimmed.rfind(".kernarg_segment_size:", 0) == 0) {
      current->kernarg_segment_size =
          static_cast<uint32_t>(std::stoul(Trim(std::string_view(trimmed).substr(22))));
      continue;
    }
    if (trimmed.rfind(".private_segment_fixed_size:", 0) == 0) {
      current->private_segment_fixed_size =
          static_cast<uint32_t>(std::stoul(Trim(std::string_view(trimmed).substr(28))));
      continue;
    }
    if (trimmed.rfind(".sgpr_count:", 0) == 0) {
      current->sgpr_count =
          static_cast<uint32_t>(std::stoul(Trim(std::string_view(trimmed).substr(12))));
      continue;
    }
    if (trimmed.rfind(".vgpr_count:", 0) == 0) {
      current->vgpr_count =
          static_cast<uint32_t>(std::stoul(Trim(std::string_view(trimmed).substr(12))));
      continue;
    }
    if (trimmed.rfind(".wavefront_size:", 0) == 0) {
      current->wavefront_size =
          static_cast<uint32_t>(std::stoul(Trim(std::string_view(trimmed).substr(16))));
      continue;
    }
    if (trimmed.rfind(".uniform_work_group_size:", 0) == 0) {
      current->uniform_work_group_size =
          std::stoul(Trim(std::string_view(trimmed).substr(25))) != 0;
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
      layout << kernel.args[i].kind_name << ':' << kernel.args[i].size;
    }
    metadata.values["arg_layout"] = layout.str();
    metadata.values["arg_count"] = std::to_string(kernel.args.size());
    metadata.values["group_segment_fixed_size"] =
        std::to_string(kernel.group_segment_fixed_size);
    metadata.values["kernarg_segment_size"] =
        std::to_string(kernel.kernarg_segment_size);
    metadata.values["private_segment_fixed_size"] =
        std::to_string(kernel.private_segment_fixed_size);
    metadata.values["sgpr_count"] = std::to_string(kernel.sgpr_count);
    metadata.values["vgpr_count"] = std::to_string(kernel.vgpr_count);
    metadata.values["wavefront_size"] = std::to_string(kernel.wavefront_size);
    metadata.values["uniform_work_group_size"] =
        kernel.uniform_work_group_size ? "1" : "0";
    metadata.values["descriptor_symbol"] = kernel.symbol;
    std::ostringstream hidden_layout;
    for (size_t i = 0; i < kernel.hidden_args.size(); ++i) {
      if (i != 0) {
        hidden_layout << ',';
      }
      hidden_layout << kernel.hidden_args[i].kind_name << ':'
                    << kernel.hidden_args[i].offset << ':'
                    << kernel.hidden_args[i].size;
    }
    metadata.values["hidden_arg_layout"] = hidden_layout.str();
    break;
  }
  return metadata;
}

uint32_t LoadU32(std::span<const std::byte> bytes, size_t offset) {
  uint32_t value = 0;
  std::memcpy(&value, bytes.data() + offset, sizeof(value));
  return value;
}

uint64_t LoadU64(std::span<const std::byte> bytes, size_t offset) {
  uint64_t value = 0;
  std::memcpy(&value, bytes.data() + offset, sizeof(value));
  return value;
}

AmdgpuKernelDescriptor ParseKernelDescriptor(std::span<const std::byte> bytes) {
  if (bytes.size() < 64) {
    throw std::runtime_error("kernel descriptor is smaller than 64 bytes");
  }
  AmdgpuKernelDescriptor descriptor;
  descriptor.group_segment_fixed_size = LoadU32(bytes, 0);
  descriptor.private_segment_fixed_size = LoadU32(bytes, 4);
  descriptor.kernarg_size = LoadU32(bytes, 8);
  descriptor.kernel_code_entry_byte_offset = static_cast<int64_t>(LoadU64(bytes, 16));
  descriptor.compute_pgm_rsrc3 = LoadU32(bytes, 44);
  descriptor.compute_pgm_rsrc1 = LoadU32(bytes, 48);
  descriptor.compute_pgm_rsrc2 = LoadU32(bytes, 52);
  descriptor.setup_word = LoadU32(bytes, 56);

  descriptor.enable_private_segment = (descriptor.compute_pgm_rsrc2 & 0x1u) != 0;
  descriptor.user_sgpr_count = static_cast<uint8_t>((descriptor.compute_pgm_rsrc2 >> 1u) & 0x1fu);
  descriptor.enable_sgpr_workgroup_id_x = ((descriptor.compute_pgm_rsrc2 >> 7u) & 0x1u) != 0;
  descriptor.enable_sgpr_workgroup_id_y = ((descriptor.compute_pgm_rsrc2 >> 8u) & 0x1u) != 0;
  descriptor.enable_sgpr_workgroup_id_z = ((descriptor.compute_pgm_rsrc2 >> 9u) & 0x1u) != 0;
  descriptor.enable_sgpr_workgroup_info = ((descriptor.compute_pgm_rsrc2 >> 10u) & 0x1u) != 0;
  descriptor.enable_vgpr_workitem_id =
      static_cast<uint8_t>((descriptor.compute_pgm_rsrc2 >> 11u) & 0x3u);

  descriptor.enable_sgpr_private_segment_buffer = (descriptor.setup_word & 0x1u) != 0;
  descriptor.enable_sgpr_dispatch_ptr = ((descriptor.setup_word >> 1u) & 0x1u) != 0;
  descriptor.enable_sgpr_queue_ptr = ((descriptor.setup_word >> 2u) & 0x1u) != 0;
  descriptor.enable_sgpr_kernarg_segment_ptr = ((descriptor.setup_word >> 3u) & 0x1u) != 0;
  descriptor.enable_sgpr_dispatch_id = ((descriptor.setup_word >> 4u) & 0x1u) != 0;
  descriptor.enable_sgpr_flat_scratch_init = ((descriptor.setup_word >> 5u) & 0x1u) != 0;
  descriptor.enable_sgpr_private_segment_size = ((descriptor.setup_word >> 6u) & 0x1u) != 0;
  descriptor.enable_wavefront_size32 = ((descriptor.setup_word >> 10u) & 0x1u) != 0;
  descriptor.uses_dynamic_stack = ((descriptor.setup_word >> 11u) & 0x1u) != 0;
  descriptor.kernarg_preload_spec_length =
      static_cast<uint8_t>((descriptor.setup_word >> 16u) & 0x7fu);
  descriptor.kernarg_preload_spec_offset =
      static_cast<uint16_t>((descriptor.setup_word >> 23u) & 0x1ffu);
  return descriptor;
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

}  // namespace

AmdgpuCodeObjectImage AmdgpuCodeObjectDecoder::Decode(const std::filesystem::path& path,
                                                      std::optional<std::string> kernel_name) const {
  ScopedTempDir temp_dir;
  const auto device_path = MaterializeDeviceCodeObject(path, temp_dir);
  AmdgpuCodeObjectImage code_object;
  const auto symbols =
      ParseSymbols(RunCommand("readelf -Ws " + ShellQuote(device_path.string())));
  const auto selected_symbol = SelectKernelSymbol(symbols, kernel_name);
  code_object.kernel_name = selected_symbol.name;
  code_object.metadata = BuildMetadataFromNotes(device_path, path, code_object.kernel_name);

  const auto descriptor_symbol_name_it = code_object.metadata.values.find("descriptor_symbol");
  if (descriptor_symbol_name_it != code_object.metadata.values.end() &&
      !descriptor_symbol_name_it->second.empty()) {
    const auto descriptor_symbol = SelectDescriptorSymbol(symbols, descriptor_symbol_name_it->second);
    const auto rodata_dump_path = temp_dir.path() / "rodata.bin";
    RunCommand("llvm-objcopy --dump-section .rodata=" + ShellQuote(rodata_dump_path.string()) + " " +
               ShellQuote(device_path.string()));
    const auto rodata_bytes = ReadBinaryFile(rodata_dump_path);
    const auto rodata_info = ParseSectionInfo(
        RunCommand("readelf -S " + ShellQuote(device_path.string())), ".rodata");
    const uint64_t descriptor_offset = descriptor_symbol.value - rodata_info.addr;
    if (descriptor_offset + descriptor_symbol.size > rodata_bytes.size()) {
      throw std::runtime_error("kernel descriptor range exceeds dumped .rodata bytes");
    }
    std::span<const std::byte> descriptor_bytes{
        rodata_bytes.data() + static_cast<size_t>(descriptor_offset),
        static_cast<size_t>(descriptor_symbol.size),
    };
    code_object.kernel_descriptor = ParseKernelDescriptor(descriptor_bytes);
  }

  const auto text_dump_path = temp_dir.path() / "text.bin";
  RunCommand("llvm-objcopy --dump-section .text=" + ShellQuote(text_dump_path.string()) + " " +
             ShellQuote(device_path.string()));
  const auto text_bytes = ReadBinaryFile(text_dump_path);
  const auto section_info = ParseSectionInfo(
      RunCommand("readelf -S " + ShellQuote(device_path.string())), ".text");
  const auto symbol_info = selected_symbol;

  const uint64_t kernel_offset = symbol_info.value - section_info.addr;
  if (kernel_offset + symbol_info.size > text_bytes.size()) {
    throw std::runtime_error("kernel symbol range exceeds dumped .text bytes");
  }

  const auto code_begin = static_cast<size_t>(kernel_offset);
  const auto code_size = static_cast<size_t>(symbol_info.size);
  std::span<const std::byte> kernel_text{text_bytes.data() + code_begin, code_size};
  auto parsed = RawGcnInstructionArrayParser::Parse(kernel_text, symbol_info.value);
  code_object.instructions = std::move(parsed.raw_instructions);
  code_object.decoded_instructions = std::move(parsed.decoded_instructions);
  code_object.instruction_objects = std::move(parsed.instruction_objects);
  code_object.code_bytes.assign(kernel_text.begin(), kernel_text.end());

  for (auto& instruction : code_object.instructions) {
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
  }

  if (code_object.instructions.empty()) {
    throw std::runtime_error("failed to decode AMDGPU kernel instructions: " + code_object.kernel_name);
  }
  return code_object;
}

}  // namespace gpu_model
