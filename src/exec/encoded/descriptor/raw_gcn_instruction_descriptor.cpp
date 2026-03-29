#include "gpu_model/exec/encoded/descriptor/raw_gcn_instruction_descriptor.h"

#include "gpu_model/decode/gcn_inst_encoding_def.h"

namespace gpu_model {

namespace {

RawGcnInstructionDescriptor MakeUnknownDescriptor() {
  return RawGcnInstructionDescriptor{};
}

RawGcnInstructionDescriptor MakeDescriptor(const GcnIsaOpcodeDescriptor* opcode_descriptor,
                                          RawGcnInstructionCategory category,
                                          std::string_view placeholder_op_type_name,
                                          std::string_view placeholder_class_name) {
  return RawGcnInstructionDescriptor{
      .opcode_descriptor = opcode_descriptor,
      .category = category,
      .placeholder_op_type_name = placeholder_op_type_name,
      .placeholder_class_name = placeholder_class_name,
  };
}

}  // namespace

std::string_view ToString(RawGcnInstructionCategory category) {
  switch (category) {
    case RawGcnInstructionCategory::Unknown:
      return "unknown";
    case RawGcnInstructionCategory::ScalarMemory:
      return "scalar_memory";
    case RawGcnInstructionCategory::Scalar:
      return "scalar";
    case RawGcnInstructionCategory::Vector:
      return "vector";
    case RawGcnInstructionCategory::Memory:
      return "memory";
  }
  return "unknown";
}

RawGcnInstructionDescriptor DescribeRawGcnInstruction(const DecodedGcnInstruction& instruction) {
  const auto* descriptor = FindGcnFallbackOpcodeDescriptor(instruction.words);
  if (descriptor == nullptr) {
    return MakeUnknownDescriptor();
  }

  switch (descriptor->op_type) {
    case GcnIsaOpType::Smrd:
    case GcnIsaOpType::Smem:
      return MakeDescriptor(descriptor, RawGcnInstructionCategory::ScalarMemory,
                            "scalar_memory", "scalar_memory_placeholder");
    case GcnIsaOpType::Sop1:
      return MakeDescriptor(descriptor, RawGcnInstructionCategory::Scalar, "sop1",
                            "sop1_placeholder");
    case GcnIsaOpType::Sop2:
      return MakeDescriptor(descriptor, RawGcnInstructionCategory::Scalar, "sop2",
                            "sop2_placeholder");
    case GcnIsaOpType::Sopk:
      return MakeDescriptor(descriptor, RawGcnInstructionCategory::Scalar, "sopk",
                            "sopk_placeholder");
    case GcnIsaOpType::Sopc:
      return MakeDescriptor(descriptor, RawGcnInstructionCategory::Scalar, "sopc",
                            "sopc_placeholder");
    case GcnIsaOpType::Sopp:
      return MakeDescriptor(descriptor, RawGcnInstructionCategory::Scalar, "sopp",
                            "sopp_placeholder");
    case GcnIsaOpType::Vop1:
      return MakeDescriptor(descriptor, RawGcnInstructionCategory::Vector, "vop1",
                            "vop1_placeholder");
    case GcnIsaOpType::Vop2:
      return MakeDescriptor(descriptor, RawGcnInstructionCategory::Vector, "vop2",
                            "vop2_placeholder");
    case GcnIsaOpType::Vop3a:
      return MakeDescriptor(descriptor, RawGcnInstructionCategory::Vector, "vop3a",
                            "vop3a_placeholder");
    case GcnIsaOpType::Vop3b:
      return MakeDescriptor(descriptor, RawGcnInstructionCategory::Vector, "vop3b",
                            "vop3b_placeholder");
    case GcnIsaOpType::Vop3p:
      return MakeDescriptor(descriptor, RawGcnInstructionCategory::Vector, "vop3p",
                            "vop3p_placeholder");
    case GcnIsaOpType::Vopc:
      return MakeDescriptor(descriptor, RawGcnInstructionCategory::Vector, "vopc",
                            "vopc_placeholder");
    case GcnIsaOpType::Flat:
      return MakeDescriptor(descriptor, RawGcnInstructionCategory::Memory, "flat",
                            "flat_placeholder");
    case GcnIsaOpType::Ds:
      return MakeDescriptor(descriptor, RawGcnInstructionCategory::Memory, "ds",
                            "ds_placeholder");
    case GcnIsaOpType::Mubuf:
      return MakeDescriptor(descriptor, RawGcnInstructionCategory::Memory, "mubuf",
                            "mubuf_placeholder");
    case GcnIsaOpType::Mtbuf:
      return MakeDescriptor(descriptor, RawGcnInstructionCategory::Memory, "mtbuf",
                            "mtbuf_placeholder");
    case GcnIsaOpType::Mimg:
      return MakeDescriptor(descriptor, RawGcnInstructionCategory::Memory, "mimg",
                            "mimg_placeholder");
    case GcnIsaOpType::Vintrp:
      return MakeDescriptor(descriptor, RawGcnInstructionCategory::Memory, "vintrp",
                            "vintrp_placeholder");
    case GcnIsaOpType::Exp:
      return MakeDescriptor(descriptor, RawGcnInstructionCategory::Memory, "exp",
                            "exp_placeholder");
  }
  return MakeUnknownDescriptor();
}

}  // namespace gpu_model
