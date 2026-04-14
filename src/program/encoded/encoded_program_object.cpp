#include "program/program_object/object_reader.h"

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

#include "instruction/decode/encoded/internal/encoded_gcn_encoding_def.h"
#include "instruction/decode/encoded/instruction_object.h"
#include "instruction/isa/kernel_metadata.h"
#include "program/loader/artifact_parser.h"
#include "program/loader/code_object_materializer.h"
#include "program/loader/elf_section_loader.h"
#include "program/loader/external_tool_executor.h"
#include "program/loader/temp_dir_manager.h"
#include "program/loader/amdgpu_target_config.h"

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
    const auto rodata = LoadElfSection(device_path, ".rodata", temp_dir);
    const uint64_t descriptor_offset = descriptor_symbol.value - rodata.info.addr;
    if (descriptor_offset + descriptor_symbol.size > rodata.bytes.size()) {
      throw std::runtime_error("kernel descriptor range exceeds dumped .rodata bytes");
    }
    std::span<const std::byte> descriptor_bytes{
        rodata.bytes.data() + static_cast<size_t>(descriptor_offset),
        static_cast<size_t>(descriptor_symbol.size),
    };
    auto kernel_descriptor = ArtifactParser::ParseKernelDescriptor(descriptor_bytes);
    const auto agpr_count_it = code_object.metadata().values.find("agpr_count");
    if (agpr_count_it != code_object.metadata().values.end()) {
      kernel_descriptor.agpr_count = static_cast<uint16_t>(std::stoul(agpr_count_it->second));
    }
    code_object.set_kernel_descriptor(std::move(kernel_descriptor));
  }

  const auto text = LoadElfSection(device_path, ".text", temp_dir);
  const auto symbol_info = selected_symbol;

  const uint64_t kernel_offset = symbol_info.value - text.info.addr;
  if (kernel_offset + symbol_info.size > text.bytes.size()) {
    throw std::runtime_error("kernel symbol range exceeds dumped .text bytes");
  }

  const auto code_begin = static_cast<size_t>(kernel_offset);
  const auto code_size = static_cast<size_t>(symbol_info.size);
  std::span<const std::byte> kernel_text{text.bytes.data() + code_begin, code_size};
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
