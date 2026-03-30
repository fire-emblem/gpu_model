#pragma once

#include <string_view>

#include "gpu_model/decode/generated_gcn_full_opcode_table.h"
#include "gpu_model/instruction/encoded/decoded_instruction.h"

namespace gpu_model {

enum class EncodedInstructionCategory {
  Unknown,
  ScalarMemory,
  Scalar,
  Vector,
  Memory,
};

struct EncodedInstructionDescriptor {
  const GcnIsaOpcodeDescriptor* opcode_descriptor = nullptr;
  EncodedInstructionCategory category = EncodedInstructionCategory::Unknown;
  std::string_view placeholder_op_type_name = "unknown";
  std::string_view placeholder_class_name = "unknown_placeholder";

  bool known() const { return opcode_descriptor != nullptr; }
};

EncodedInstructionDescriptor DescribeEncodedInstruction(const DecodedInstruction& instruction);
std::string_view ToString(EncodedInstructionCategory category);

}  // namespace gpu_model
