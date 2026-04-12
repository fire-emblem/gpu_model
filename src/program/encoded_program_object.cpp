#include "gpu_model/program/object_reader.h"

#include <array>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <vector>

#include "gpu_model/instruction/encoded/internal/encoded_gcn_encoding_def.h"
#include "gpu_model/instruction/encoded/instruction_object.h"
#include "gpu_model/isa/kernel_metadata.h"
#include "gpu_model/loader/artifact_parser.h"
#include "gpu_model/loader/external_tool_executor.h"
#include "gpu_model/loader/temp_dir_manager.h"
#include "gpu_model/target/amdgpu_target_config.h"

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

bool IsAmdgpuElf(const std::filesystem::path& path) {
  const std::string header = ExternalToolExecutor::ReadElfHeader(path);
  return header.find("AMDGPU") != std::string::npos;
}

bool HasHipFatbin(const std::filesystem::path& path) {
  const std::string sections = ExternalToolExecutor::ReadElfSectionTable(path);
  return sections.find(".hip_fatbin") != std::string::npos;
}

struct MaterializedCodeObject {
  std::filesystem::path path;
  MetadataBlob metadata;
};

MaterializedCodeObject MaterializeDeviceCodeObject(const std::filesystem::path& path,
                                                   const ScopedTempDir& temp_dir) {
  if (IsAmdgpuElf(path)) {
    return MaterializedCodeObject{
        .path = path,
        .metadata = MetadataBlob{},
    };
  }
  if (!HasHipFatbin(path)) {
    throw std::runtime_error("ELF is neither AMDGPU code object nor HIP fatbin host artifact: " +
                             path.string());
  }

  const auto fatbin_path = temp_dir.path() / "kernel.hip_fatbin";
  const auto device_path = temp_dir.path() / "kernel_device.co";
  ExternalToolExecutor::DumpElfSection(path, ".hip_fatbin", fatbin_path);
  const std::string bundles = ExternalToolExecutor::ListOffloadBundles(fatbin_path);
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
  ExternalToolExecutor::UnbundleOffloadTarget(fatbin_path, target, device_path);
  return MaterializedCodeObject{
      .path = device_path,
      .metadata = MetadataBlob{},
  };
}

std::string EncodeArgLayoutToken(const NoteKernelArgLayoutEntry& arg, uint32_t expected_offset) {
  std::ostringstream out;
  out << arg.kind_name << ':';
  if (arg.offset != expected_offset) {
    out << arg.offset << ':';
  }
  out << arg.size;
  return out.str();
}

std::string ExtractMetadataScalar(const std::string& notes, std::string_view key) {
  const std::string prefix = std::string(key) + ':';
  std::istringstream input(notes);
  std::string line;
  while (std::getline(input, line)) {
    const std::string trimmed = Trim(line);
    if (trimmed.rfind(prefix, 0) != 0) {
      continue;
    }
    return Trim(std::string_view(trimmed).substr(prefix.size()));
  }
  return {};
}

MetadataBlob BuildMetadataFromNotes(const std::filesystem::path& note_source_path,
                                    const std::string& kernel_name,
                                    MetadataBlob metadata = {}) {
  metadata.values["entry"] = kernel_name;

  const std::string notes = ExternalToolExecutor::ReadElfNotes(note_source_path);
  if (const std::string amdhsa_target = ExtractMetadataScalar(notes, "amdhsa.target");
      !amdhsa_target.empty()) {
    metadata.values["amdhsa_target"] = amdhsa_target;
  }
  const auto kernels = ArtifactParser::ParseKernelMetadataNotes(notes);
  for (const auto& kernel : kernels) {
    if (kernel.name != kernel_name) {
      continue;
    }
    std::ostringstream layout;
    uint32_t expected_offset = 0;
    for (size_t i = 0; i < kernel.args.size(); ++i) {
      if (i != 0) {
        layout << ',';
      }
      layout << EncodeArgLayoutToken(kernel.args[i], expected_offset);
      expected_offset = kernel.args[i].offset + kernel.args[i].size;
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
    metadata.values["agpr_count"] = std::to_string(kernel.agpr_count);
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

bool HasLlvmMc() {
  static const bool available = ExternalToolExecutor::HasLlvmMc();
  return available;
}

std::string McpuFromMetadata(const MetadataBlob& metadata) {
  const auto it = metadata.values.find("amdhsa_target");
  if (it == metadata.values.end() || it->second.empty()) {
    return {};
  }
  return NormalizeAmdgpuMcpu(it->second);
}

std::string McpuFromFileHeaders(const std::filesystem::path& path) {
  static constexpr std::string_view kPrefix = "EF_AMDGPU_MACH_AMDGCN_";
  const std::string headers = ExternalToolExecutor::ReadAmdgpuFileHeaders(path);
  std::istringstream input(headers);
  std::string line;
  while (std::getline(input, line)) {
    const std::string trimmed = Trim(line);
    const size_t pos = trimmed.find(kPrefix);
    if (pos == std::string::npos) {
      continue;
    }
    return NormalizeAmdgpuMcpu(trimmed.substr(pos + kPrefix.size()));
  }
  return {};
}

std::string ResolveArtifactAmdgpuMcpu(const MetadataBlob& metadata,
                                      const std::filesystem::path& path) {
  if (std::string mcpu = McpuFromMetadata(metadata); !mcpu.empty()) {
    return mcpu;
  }
  if (std::string mcpu = McpuFromFileHeaders(path); !mcpu.empty()) {
    return mcpu;
  }
  return {};
}

void ValidateProjectAmdgpuTarget(const std::filesystem::path& path,
                                 const MetadataBlob& metadata) {
  const std::string actual = ResolveArtifactAmdgpuMcpu(metadata, path);
  if (!actual.empty() && IsProjectAmdgpuMcpu(actual)) {
    return;
  }
  throw std::runtime_error(ProjectAmdgpuTargetErrorMessage(actual.empty() ? "<unknown>" : actual) +
                           " [artifact=" + path.string() + "]");
}

std::string FormatHexByteStream(std::span<const std::byte> bytes) {
  std::ostringstream out;
  out << std::hex << std::setfill('0');
  for (size_t i = 0; i < bytes.size(); ++i) {
    if (i != 0) {
      out << ' ';
    }
    out << "0x" << std::setw(2) << std::to_integer<unsigned int>(bytes[i]);
  }
  return out.str();
}

struct LlvmMcInstructionLine {
  std::string op;
  std::string text;
};

std::vector<LlvmMcInstructionLine> DisassembleCodeSegmentWithLlvmMc(
    std::span<const std::byte> code_bytes,
    const ScopedTempDir& temp_dir) {
  std::vector<LlvmMcInstructionLine> lines;
  if (code_bytes.empty() || !HasLlvmMc()) {
    return lines;
  }

  const auto input_path = temp_dir.path() / "kernel_disasm_input.txt";
  {
    std::ofstream out(input_path);
    if (!out) {
      throw std::runtime_error("failed to create llvm-mc disassembly input");
    }
    out << FormatHexByteStream(code_bytes) << '\n';
  }

  const std::string output = ExternalToolExecutor::DisassembleHexByteStreamWithLlvmMc(input_path);
  std::istringstream input(output);
  std::string line;
  while (std::getline(input, line)) {
    const std::string trimmed = Trim(line);
    if (trimmed.empty() || trimmed == ".text") {
      continue;
    }
    const size_t split = trimmed.find_first_of(" \t");
    lines.push_back(LlvmMcInstructionLine{
        .op = split == std::string::npos ? trimmed : trimmed.substr(0, split),
        .text = trimmed,
    });
  }
  return lines;
}

std::string ExtractAsmOperands(const LlvmMcInstructionLine& line) {
  const size_t split = line.text.find_first_of(" \t");
  if (split == std::string::npos) {
    return {};
  }
  return Trim(std::string_view(line.text).substr(split + 1));
}

void BindLlvmMcDisassembly(ParsedInstructionArray& parsed,
                           const std::vector<LlvmMcInstructionLine>& disassembly) {
  if (disassembly.empty()) {
    return;
  }
  if (disassembly.size() != parsed.raw_instructions.size() ||
      disassembly.size() != parsed.decoded_instructions.size()) {
    throw std::runtime_error("llvm-mc disassembly count does not match decoded instruction count");
  }

  for (size_t i = 0; i < disassembly.size(); ++i) {
    parsed.raw_instructions[i].asm_op = disassembly[i].op;
    parsed.raw_instructions[i].asm_text = disassembly[i].text;
    if (parsed.raw_instructions[i].operands.empty()) {
      parsed.raw_instructions[i].operands = ExtractAsmOperands(disassembly[i]);
    }
    parsed.decoded_instructions[i].asm_op = disassembly[i].op;
    parsed.decoded_instructions[i].asm_text = disassembly[i].text;
  }
}

std::string JoinLlvmMcAssemblyText(const std::vector<LlvmMcInstructionLine>& disassembly) {
  std::ostringstream out;
  for (size_t i = 0; i < disassembly.size(); ++i) {
    if (i != 0) {
      out << '\n';
    }
    out << disassembly[i].text;
  }
  return out.str();
}

}  // namespace

ProgramObject ObjectReader::LoadProgramObject(const std::filesystem::path& path,
                                              std::optional<std::string> kernel_name) const {
  ScopedTempDir temp_dir;
  const auto materialized = MaterializeDeviceCodeObject(path, temp_dir);
  const auto& device_path = materialized.path;
  ProgramObject code_object;
  const auto symbols = ArtifactParser::ParseSymbols(
      ExternalToolExecutor::ReadElfSymbolTable(device_path));
  const auto selected_symbol = ArtifactParser::SelectKernelSymbol(symbols, kernel_name);
  code_object.set_kernel_name(selected_symbol.name);
  code_object.set_metadata(
      BuildMetadataFromNotes(device_path, code_object.kernel_name(), materialized.metadata));
  ValidateProjectAmdgpuTarget(device_path, code_object.metadata());

  const auto descriptor_symbol_name_it = code_object.metadata().values.find("descriptor_symbol");
  if (descriptor_symbol_name_it != code_object.metadata().values.end() &&
      !descriptor_symbol_name_it->second.empty()) {
    const auto descriptor_symbol = ArtifactParser::SelectDescriptorSymbol(
        symbols, descriptor_symbol_name_it->second);
    const auto rodata_dump_path = temp_dir.path() / "rodata.bin";
    ExternalToolExecutor::DumpElfSection(device_path, ".rodata", rodata_dump_path);
    const auto rodata_bytes = ReadBinaryFile(rodata_dump_path);
    const auto rodata_info = ArtifactParser::ParseSectionInfo(
        ExternalToolExecutor::ReadElfSectionTable(device_path), ".rodata");
    const uint64_t descriptor_offset = descriptor_symbol.value - rodata_info.addr;
    if (descriptor_offset + descriptor_symbol.size > rodata_bytes.size()) {
      throw std::runtime_error("kernel descriptor range exceeds dumped .rodata bytes");
    }
    std::span<const std::byte> descriptor_bytes{
        rodata_bytes.data() + static_cast<size_t>(descriptor_offset),
        static_cast<size_t>(descriptor_symbol.size),
    };
    auto kernel_descriptor = ArtifactParser::ParseKernelDescriptor(descriptor_bytes);
    const auto agpr_count_it = code_object.metadata().values.find("agpr_count");
    if (agpr_count_it != code_object.metadata().values.end()) {
      kernel_descriptor.agpr_count = static_cast<uint16_t>(std::stoul(agpr_count_it->second));
    }
    code_object.set_kernel_descriptor(std::move(kernel_descriptor));
  }

  const auto text_dump_path = temp_dir.path() / "text.bin";
  ExternalToolExecutor::DumpElfSection(device_path, ".text", text_dump_path);
  const auto text_bytes = ReadBinaryFile(text_dump_path);
  const auto section_info = ArtifactParser::ParseSectionInfo(
      ExternalToolExecutor::ReadElfSectionTable(device_path), ".text");
  const auto symbol_info = selected_symbol;

  const uint64_t kernel_offset = symbol_info.value - section_info.addr;
  if (kernel_offset + symbol_info.size > text_bytes.size()) {
    throw std::runtime_error("kernel symbol range exceeds dumped .text bytes");
  }

  const auto code_begin = static_cast<size_t>(kernel_offset);
  const auto code_size = static_cast<size_t>(symbol_info.size);
  std::span<const std::byte> kernel_text{text_bytes.data() + code_begin, code_size};
  ParsedInstructionArray parsed;
  parsed.raw_instructions = InstructionArrayParser::ParseRaw(kernel_text, symbol_info.value);
  parsed.decoded_instructions = InstructionArrayParser::Decode(parsed.raw_instructions);
  const auto llvm_mc_disassembly = DisassembleCodeSegmentWithLlvmMc(kernel_text, temp_dir);
  BindLlvmMcDisassembly(parsed, llvm_mc_disassembly);
  for (auto& instruction : parsed.raw_instructions) {
    if (!instruction.asm_op.empty()) {
      instruction.mnemonic = instruction.asm_op;
      DecodeEncodedGcnOperands(instruction);
    }
  }
  parsed.decoded_instructions = InstructionArrayParser::Decode(parsed.raw_instructions);
  parsed.instruction_objects = InstructionArrayParser::Parse(parsed.decoded_instructions);
  if (!llvm_mc_disassembly.empty()) {
    code_object.set_assembly_text(JoinLlvmMcAssemblyText(llvm_mc_disassembly));
  }
  code_object.set_instructions(std::move(parsed.raw_instructions));
  code_object.set_decoded_instructions(std::move(parsed.decoded_instructions));
  code_object.set_instruction_objects(std::move(parsed.instruction_objects));
  code_object.set_code_bytes(std::vector<std::byte>(kernel_text.begin(), kernel_text.end()));

  auto instructions = code_object.instructions();
  for (auto& instruction : instructions) {
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
  code_object.set_instructions(std::move(instructions));

  if (code_object.instructions().empty()) {
    throw std::runtime_error("failed to decode AMDGPU kernel instructions: " +
                             code_object.kernel_name());
  }
  return code_object;
}

}  // namespace gpu_model
