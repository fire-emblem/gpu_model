#pragma once

#include <string_view>

#include "gpu_model/decode/decoded_gcn_instruction.h"
#include "gpu_model/decode/generated_gcn_full_opcode_table.h"
#include "gpu_model/instruction/encoded/decoded_instruction.h"

namespace gpu_model {

enum class RawGcnInstructionCategory {
  Unknown,
  ScalarMemory,
  Scalar,
  Vector,
  Memory,
};

struct RawGcnInstructionDescriptor {
  const GcnIsaOpcodeDescriptor* opcode_descriptor = nullptr;
  RawGcnInstructionCategory category = RawGcnInstructionCategory::Unknown;
  std::string_view placeholder_op_type_name = "unknown";
  std::string_view placeholder_class_name = "unknown_placeholder";

  bool known() const { return opcode_descriptor != nullptr; }
};

RawGcnInstructionDescriptor DescribeRawGcnInstruction(const DecodedInstruction& instruction);
std::string_view ToString(RawGcnInstructionCategory category);

}  // namespace gpu_model
