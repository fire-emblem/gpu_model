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
  const auto* descriptor = FindEncodedGcnFallbackOpcodeDescriptor(instruction.words);
  if (descriptor == nullptr) {
    return MakeUnknownDescriptor();
  }

  switch (descriptor->op_type) {
    case GcnIsaOpType::Smrd:
    case GcnIsaOpType::Smem:
      return MakeDescriptor(descriptor, EncodedInstructionCategory::ScalarMemory,
                            "scalar_memory", "scalar_memory_placeholder");
    case GcnIsaOpType::Sop1:
      return MakeDescriptor(descriptor, EncodedInstructionCategory::Scalar, "sop1",
                            "sop1_placeholder");
    case GcnIsaOpType::Sop2:
      return MakeDescriptor(descriptor, EncodedInstructionCategory::Scalar, "sop2",
                            "sop2_placeholder");
    case GcnIsaOpType::Sopk:
      return MakeDescriptor(descriptor, EncodedInstructionCategory::Scalar, "sopk",
                            "sopk_placeholder");
    case GcnIsaOpType::Sopc:
      return MakeDescriptor(descriptor, EncodedInstructionCategory::Scalar, "sopc",
                            "sopc_placeholder");
    case GcnIsaOpType::Sopp:
      return MakeDescriptor(descriptor, EncodedInstructionCategory::Scalar, "sopp",
                            "sopp_placeholder");
    case GcnIsaOpType::Vop1:
      return MakeDescriptor(descriptor, EncodedInstructionCategory::Vector, "vop1",
                            "vop1_placeholder");
    case GcnIsaOpType::Vop2:
      return MakeDescriptor(descriptor, EncodedInstructionCategory::Vector, "vop2",
                            "vop2_placeholder");
    case GcnIsaOpType::Vop3a:
      return MakeDescriptor(descriptor, EncodedInstructionCategory::Vector, "vop3a",
                            "vop3a_placeholder");
    case GcnIsaOpType::Vop3b:
      return MakeDescriptor(descriptor, EncodedInstructionCategory::Vector, "vop3b",
                            "vop3b_placeholder");
    case GcnIsaOpType::Vop3p:
      return MakeDescriptor(descriptor, EncodedInstructionCategory::Vector, "vop3p",
                            "vop3p_placeholder");
    case GcnIsaOpType::Vopc:
      return MakeDescriptor(descriptor, EncodedInstructionCategory::Vector, "vopc",
                            "vopc_placeholder");
    case GcnIsaOpType::Flat:
      return MakeDescriptor(descriptor, EncodedInstructionCategory::Memory, "flat",
                            "flat_placeholder");
    case GcnIsaOpType::Ds:
      return MakeDescriptor(descriptor, EncodedInstructionCategory::Memory, "ds",
                            "ds_placeholder");
    case GcnIsaOpType::Mubuf:
      return MakeDescriptor(descriptor, EncodedInstructionCategory::Memory, "mubuf",
                            "mubuf_placeholder");
    case GcnIsaOpType::Mtbuf:
      return MakeDescriptor(descriptor, EncodedInstructionCategory::Memory, "mtbuf",
                            "mtbuf_placeholder");
    case GcnIsaOpType::Mimg:
      return MakeDescriptor(descriptor, EncodedInstructionCategory::Memory, "mimg",
                            "mimg_placeholder");
    case GcnIsaOpType::Vintrp:
      return MakeDescriptor(descriptor, EncodedInstructionCategory::Memory, "vintrp",
                            "vintrp_placeholder");
    case GcnIsaOpType::Exp:
      return MakeDescriptor(descriptor, EncodedInstructionCategory::Memory, "exp",
                            "exp_placeholder");
  }
  return MakeUnknownDescriptor();
}

}  // namespace gpu_model
