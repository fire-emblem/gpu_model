#include "program/loader/code_object_builder.h"

#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

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

ProgramObject LoadEncodedProgramObject(const std::filesystem::path& path,
                                       std::optional<std::string> kernel_name) {
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
