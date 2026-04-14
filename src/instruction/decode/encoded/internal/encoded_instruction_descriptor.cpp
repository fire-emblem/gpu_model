#include "instruction/decode/encoded/internal/encoded_instruction_descriptor.h"

#include "instruction/decode/encoded/internal/encoded_gcn_encoding_def.h"

namespace gpu_model {

namespace {

EncodedInstructionDescriptor MakeUnknownDescriptor() {
  return EncodedInstructionDescriptor{};
}

EncodedInstructionDescriptor MakeDescriptor(const GcnIsaOpcodeDescriptor* opcode_descriptor,
                                            EncodedInstructionCategory category,
                                            std::string_view fallback_op_type_name,
                                            std::string_view fallback_class_name) {
  return EncodedInstructionDescriptor{
      .opcode_descriptor = opcode_descriptor,
      .category = category,
      .fallback_op_type_name = fallback_op_type_name,
      .fallback_class_name = fallback_class_name,
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
    case GcnIsaOpType::Unknown:
      return MakeUnknownDescriptor();
    case GcnIsaOpType::Smrd:
    case GcnIsaOpType::Smem:
      return MakeDescriptor(match->opcode_descriptor, match->category, "scalar_memory",
                            "scalar_memory_unsupported");
    case GcnIsaOpType::Sop1:
      return MakeDescriptor(match->opcode_descriptor, match->category, "sop1", "sop1_unsupported");
    case GcnIsaOpType::Sop2:
      return MakeDescriptor(match->opcode_descriptor, match->category, "sop2", "sop2_unsupported");
    case GcnIsaOpType::Sopk:
      return MakeDescriptor(match->opcode_descriptor, match->category, "sopk", "sopk_unsupported");
    case GcnIsaOpType::Sopc:
      return MakeDescriptor(match->opcode_descriptor, match->category, "sopc", "sopc_unsupported");
    case GcnIsaOpType::Sopp:
      return MakeDescriptor(match->opcode_descriptor, match->category, "sopp", "sopp_unsupported");
    case GcnIsaOpType::Vop1:
      return MakeDescriptor(match->opcode_descriptor, match->category, "vop1", "vop1_unsupported");
    case GcnIsaOpType::Vop2:
      return MakeDescriptor(match->opcode_descriptor, match->category, "vop2", "vop2_unsupported");
    case GcnIsaOpType::Vop3a:
      return MakeDescriptor(match->opcode_descriptor, match->category, "vop3a",
                            "vop3a_unsupported");
    case GcnIsaOpType::Vop3b:
      return MakeDescriptor(match->opcode_descriptor, match->category, "vop3b",
                            "vop3b_unsupported");
    case GcnIsaOpType::Vop3p:
      return MakeDescriptor(match->opcode_descriptor, match->category, "vop3p",
                            "vop3p_unsupported");
    case GcnIsaOpType::Vopc:
      return MakeDescriptor(match->opcode_descriptor, match->category, "vopc", "vopc_unsupported");
    case GcnIsaOpType::Flat:
      return MakeDescriptor(match->opcode_descriptor, match->category, "flat", "flat_unsupported");
    case GcnIsaOpType::Ds:
      return MakeDescriptor(match->opcode_descriptor, match->category, "ds", "ds_unsupported");
    case GcnIsaOpType::Mubuf:
      return MakeDescriptor(match->opcode_descriptor, match->category, "mubuf",
                            "mubuf_unsupported");
    case GcnIsaOpType::Mtbuf:
      return MakeDescriptor(match->opcode_descriptor, match->category, "mtbuf",
                            "mtbuf_unsupported");
    case GcnIsaOpType::Mimg:
      return MakeDescriptor(match->opcode_descriptor, match->category, "mimg", "mimg_unsupported");
    case GcnIsaOpType::Vintrp:
      return MakeDescriptor(match->opcode_descriptor, match->category, "vintrp",
                            "vintrp_unsupported");
    case GcnIsaOpType::Exp:
      return MakeDescriptor(match->opcode_descriptor, match->category, "exp", "exp_unsupported");
  }
  return MakeUnknownDescriptor();
}

}  // namespace gpu_model
