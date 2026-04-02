#include "gpu_model/instruction/encoded/internal/encoded_instruction_descriptor.h"

#include "gpu_model/instruction/encoded/internal/encoded_gcn_encoding_def.h"

namespace gpu_model {

namespace {

EncodedInstructionDescriptor MakeUnknownDescriptor() {
  return EncodedInstructionDescriptor{};
}

EncodedInstructionDescriptor MakeDescriptor(const GcnIsaOpcodeDescriptor* opcode_descriptor,
                                          EncodedInstructionCategory category,
                                          std::string_view placeholder_op_type_name,
                                          std::string_view placeholder_class_name) {
  return EncodedInstructionDescriptor{
      .opcode_descriptor = opcode_descriptor,
      .category = category,
      .placeholder_op_type_name = placeholder_op_type_name,
      .placeholder_class_name = placeholder_class_name,
  };
}

}  // namespace

std::string_view ToString(EncodedInstructionCategory category) {
  switch (category) {
    case EncodedInstructionCategory::Unknown:
      return "unknown";
    case EncodedInstructionCategory::ScalarMemory:
      return "scalar_memory";
    case EncodedInstructionCategory::Scalar:
      return "scalar";
    case EncodedInstructionCategory::Vector:
      return "vector";
    case EncodedInstructionCategory::Memory:
      return "memory";
  }
  return "unknown";
}

EncodedInstructionDescriptor DescribeEncodedInstruction(const DecodedInstruction& instruction) {
  const auto* match = FindEncodedGcnMatchRecord(instruction.words);
  if (match == nullptr) {
    return MakeUnknownDescriptor();
  }
  switch (match->opcode_descriptor->op_type) {
    case GcnIsaOpType::Smrd:
    case GcnIsaOpType::Smem:
      return MakeDescriptor(match->opcode_descriptor, match->category, "scalar_memory",
                            "scalar_memory_placeholder");
    case GcnIsaOpType::Sop1:
      return MakeDescriptor(match->opcode_descriptor, match->category, "sop1",
                            "sop1_placeholder");
    case GcnIsaOpType::Sop2:
      return MakeDescriptor(match->opcode_descriptor, match->category, "sop2",
                            "sop2_placeholder");
    case GcnIsaOpType::Sopk:
      return MakeDescriptor(match->opcode_descriptor, match->category, "sopk",
                            "sopk_placeholder");
    case GcnIsaOpType::Sopc:
      return MakeDescriptor(match->opcode_descriptor, match->category, "sopc",
                            "sopc_placeholder");
    case GcnIsaOpType::Sopp:
      return MakeDescriptor(match->opcode_descriptor, match->category, "sopp",
                            "sopp_placeholder");
    case GcnIsaOpType::Vop1:
      return MakeDescriptor(match->opcode_descriptor, match->category, "vop1",
                            "vop1_placeholder");
    case GcnIsaOpType::Vop2:
      return MakeDescriptor(match->opcode_descriptor, match->category, "vop2",
                            "vop2_placeholder");
    case GcnIsaOpType::Vop3a:
      return MakeDescriptor(match->opcode_descriptor, match->category, "vop3a",
                            "vop3a_placeholder");
    case GcnIsaOpType::Vop3b:
      return MakeDescriptor(match->opcode_descriptor, match->category, "vop3b",
                            "vop3b_placeholder");
    case GcnIsaOpType::Vop3p:
      return MakeDescriptor(match->opcode_descriptor, match->category, "vop3p",
                            "vop3p_placeholder");
    case GcnIsaOpType::Vopc:
      return MakeDescriptor(match->opcode_descriptor, match->category, "vopc",
                            "vopc_placeholder");
    case GcnIsaOpType::Flat:
      return MakeDescriptor(match->opcode_descriptor, match->category, "flat",
                            "flat_placeholder");
    case GcnIsaOpType::Ds:
      return MakeDescriptor(match->opcode_descriptor, match->category, "ds",
                            "ds_placeholder");
    case GcnIsaOpType::Mubuf:
      return MakeDescriptor(match->opcode_descriptor, match->category, "mubuf",
                            "mubuf_placeholder");
    case GcnIsaOpType::Mtbuf:
      return MakeDescriptor(match->opcode_descriptor, match->category, "mtbuf",
                            "mtbuf_placeholder");
    case GcnIsaOpType::Mimg:
      return MakeDescriptor(match->opcode_descriptor, match->category, "mimg",
                            "mimg_placeholder");
    case GcnIsaOpType::Vintrp:
      return MakeDescriptor(match->opcode_descriptor, match->category, "vintrp",
                            "vintrp_placeholder");
    case GcnIsaOpType::Exp:
      return MakeDescriptor(match->opcode_descriptor, match->category, "exp",
                            "exp_placeholder");
  }
  return MakeUnknownDescriptor();
}

}  // namespace gpu_model
