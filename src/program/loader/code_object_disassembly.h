#pragma once

#include <cstddef>
#include <span>
#include <string>
#include <vector>

#include "instruction/decode/encoded/instruction_object.h"
#include "program/program_object/program_object.h"
#include "program/loader/temp_dir_manager.h"

namespace gpu_model {

struct LlvmMcInstructionLine {
  std::string op;
  std::string text;
};

std::vector<LlvmMcInstructionLine> DisassembleCodeSegmentWithLlvmMc(
    std::span<const std::byte> code_bytes,
    const ScopedTempDir& temp_dir);

void BindLlvmMcDisassembly(ParsedInstructionArray& parsed,
                           const std::vector<LlvmMcInstructionLine>& disassembly);

std::string JoinLlvmMcAssemblyText(const std::vector<LlvmMcInstructionLine>& disassembly);

void FillMissingOperandTextFromDecodedOperands(ProgramObject& code_object);

}  // namespace gpu_model
