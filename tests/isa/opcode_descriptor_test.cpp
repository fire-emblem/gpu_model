#include <gtest/gtest.h>

#include "instruction/isa/opcode_info.h"
#include "instruction/isa/opcode_descriptor.h"

namespace gpu_model {
namespace {

TEST(OpcodeDescriptorTest, ClassifiesVectorFloatAdd) {
  const auto& descriptor = GetOpcodeDescriptor(Opcode::VAddF32);
  EXPECT_EQ(descriptor.mnemonic, "v_add_f32");
  EXPECT_EQ(descriptor.category, OpcodeCategory::VectorAlu);
  EXPECT_TRUE(descriptor.is_vector);
  EXPECT_FALSE(descriptor.is_memory);

  const auto& exec_info = GetOpcodeExecutionInfo(Opcode::VAddF32);
  EXPECT_EQ(exec_info.family, SemanticFamily::VectorAluFloat);
  ASSERT_TRUE(exec_info.issue_type.has_value());
  EXPECT_EQ(*exec_info.issue_type, ArchitecturalIssueType::VectorAlu);
}

TEST(OpcodeDescriptorTest, ClassifiesPracticalVectorIsaOps) {
  EXPECT_EQ(GetOpcodeDescriptor(Opcode::VNotB32).mnemonic, "v_not_b32_e32");
  EXPECT_EQ(GetOpcodeExecutionInfo(Opcode::VNotB32).family, SemanticFamily::VectorAluInt);

  EXPECT_EQ(GetOpcodeDescriptor(Opcode::VCvtF32I32).mnemonic, "v_cvt_f32_i32_e32");
  EXPECT_EQ(GetOpcodeExecutionInfo(Opcode::VCvtF32I32).family, SemanticFamily::VectorAluFloat);

  EXPECT_EQ(GetOpcodeDescriptor(Opcode::VCvtI32F32).mnemonic, "v_cvt_i32_f32_e32");
  EXPECT_EQ(GetOpcodeExecutionInfo(Opcode::VCvtI32F32).family, SemanticFamily::VectorAluFloat);

  EXPECT_EQ(GetOpcodeDescriptor(Opcode::VMadU64U32).mnemonic, "v_mad_u64_u32");
  EXPECT_EQ(GetOpcodeExecutionInfo(Opcode::VMadU64U32).family, SemanticFamily::VectorAluInt);

  EXPECT_EQ(GetOpcodeDescriptor(Opcode::VMadU32U24).mnemonic, "v_mad_u32_u24");
  EXPECT_EQ(GetOpcodeExecutionInfo(Opcode::VMadU32U24).family, SemanticFamily::VectorAluInt);
}

TEST(OpcodeDescriptorTest, ClassifiesMaskAndBranchOps) {
  EXPECT_EQ(GetOpcodeExecutionInfo(Opcode::MaskAndExecCmask).family, SemanticFamily::Mask);
  EXPECT_TRUE(GetOpcodeExecutionInfo(Opcode::BIfNoexec).may_branch);
}

}  // namespace
}  // namespace gpu_model
