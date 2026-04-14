#pragma once

#include "gpu_arch/issue_config/issue_config.h"
#include "instruction/decode/encoded/internal/encoded_instruction_descriptor.h"

namespace gpu_model {

inline ArchitecturalIssueType ArchitecturalIssueTypeForEncodedInstruction(
    const DecodedInstruction& instruction,
    const EncodedInstructionDescriptor& descriptor) {
  if (instruction.mnemonic == "s_barrier") {
    return ArchitecturalIssueType::Special;
  }
  if (instruction.mnemonic.starts_with("s_cbranch")) {
    return ArchitecturalIssueType::Branch;
  }

  switch (descriptor.category) {
    case EncodedInstructionCategory::ScalarMemory:
    case EncodedInstructionCategory::Scalar:
      return ArchitecturalIssueType::ScalarAluOrMemory;
    case EncodedInstructionCategory::Memory:
      return instruction.mnemonic.starts_with("ds_")
                 ? ArchitecturalIssueType::LocalDataShare
                 : ArchitecturalIssueType::VectorMemory;
    case EncodedInstructionCategory::Vector:
      return instruction.mnemonic.starts_with("ds_")
                 ? ArchitecturalIssueType::LocalDataShare
                 : ArchitecturalIssueType::VectorAlu;
    case EncodedInstructionCategory::Unknown:
    default:
      return ArchitecturalIssueType::Special;
  }
}

}  // namespace gpu_model
