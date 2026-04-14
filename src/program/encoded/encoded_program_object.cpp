#include "program/program_object/object_reader.h"

#include <cctype>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

#include "instruction/decode/encoded/internal/encoded_gcn_encoding_def.h"
#include "program/loader/artifact_parser.h"
#include "program/loader/code_object_binding.h"
#include "program/loader/code_object_disassembly.h"
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

  if (const auto descriptor_symbol_name_it = code_object.metadata().values.find("descriptor_symbol");
      descriptor_symbol_name_it != code_object.metadata().values.end() &&
      !descriptor_symbol_name_it->second.empty()) {
    const auto rodata = LoadElfSection(device_path, ".rodata", temp_dir);
    code_object.set_kernel_descriptor(
        BuildKernelDescriptor(code_object.metadata(), symbols, rodata));
  }

  const auto text = LoadElfSection(device_path, ".text", temp_dir);
  const auto bound_code = BindKernelCodeSlice(selected_symbol, text);
  const auto kernel_text = bound_code.code_bytes;
  ParsedInstructionArray parsed;
  parsed.raw_instructions = InstructionArrayParser::ParseRaw(kernel_text, bound_code.entry_pc);
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

  FillMissingOperandTextFromDecodedOperands(code_object);

  if (code_object.instructions().empty()) {
    throw std::runtime_error("failed to decode AMDGPU kernel instructions: " +
                             code_object.kernel_name());
  }
  return code_object;
}

}  // namespace gpu_model
