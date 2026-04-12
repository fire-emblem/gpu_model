#pragma once

#include <string_view>

#include "instruction/decode/encoded/internal/encoded_gcn_encoding_def.h"
#include "instruction/decode/encoded/decoded_instruction.h"

namespace gpu_model {

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
