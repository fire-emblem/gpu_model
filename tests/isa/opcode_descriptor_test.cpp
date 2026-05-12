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

  EXPECT_EQ(GetOpcodeDescriptor(Opcode::VLshlrevB32).mnemonic, "v_lshlrev_b32_e32");
  EXPECT_EQ(GetOpcodeExecutionInfo(Opcode::VLshlrevB32).family, SemanticFamily::VectorAluInt);

  EXPECT_EQ(GetOpcodeDescriptor(Opcode::VSubrevU32).mnemonic, "v_subrev_u32_e32");
  EXPECT_EQ(GetOpcodeExecutionInfo(Opcode::VSubrevU32).family, SemanticFamily::VectorAluInt);

  EXPECT_EQ(GetOpcodeDescriptor(Opcode::VOr3B32).mnemonic, "v_or3_b32");
  EXPECT_EQ(GetOpcodeExecutionInfo(Opcode::VOr3B32).family, SemanticFamily::VectorAluInt);

  EXPECT_EQ(GetOpcodeDescriptor(Opcode::VAdd3U32).mnemonic, "v_add3_u32");
  EXPECT_EQ(GetOpcodeExecutionInfo(Opcode::VAdd3U32).family, SemanticFamily::VectorAluInt);

  EXPECT_EQ(GetOpcodeDescriptor(Opcode::VMulU32U24).mnemonic, "v_mul_u32_u24_e32");
  EXPECT_EQ(GetOpcodeExecutionInfo(Opcode::VMulU32U24).family, SemanticFamily::VectorAluInt);

  EXPECT_EQ(GetOpcodeDescriptor(Opcode::VFmacF32).mnemonic, "v_fmac_f32_e32");
  EXPECT_EQ(GetOpcodeExecutionInfo(Opcode::VFmacF32).family, SemanticFamily::VectorAluFloat);
}

TEST(OpcodeDescriptorTest, ClassifiesPracticalScalarIsaOps) {
  EXPECT_EQ(GetOpcodeDescriptor(Opcode::SMinU32).mnemonic, "s_min_u32");
  EXPECT_EQ(GetOpcodeExecutionInfo(Opcode::SMinU32).family, SemanticFamily::ScalarAlu);

  EXPECT_EQ(GetOpcodeDescriptor(Opcode::SMaxU32).mnemonic, "s_max_u32");
  EXPECT_EQ(GetOpcodeExecutionInfo(Opcode::SMaxU32).family, SemanticFamily::ScalarAlu);

  EXPECT_EQ(GetOpcodeDescriptor(Opcode::SFF1I32B32).mnemonic, "s_ff1_i32_b32");
  EXPECT_EQ(GetOpcodeExecutionInfo(Opcode::SFF1I32B32).family, SemanticFamily::ScalarAlu);
  ASSERT_TRUE(GetOpcodeExecutionInfo(Opcode::SFF1I32B32).issue_type.has_value());
  EXPECT_EQ(*GetOpcodeExecutionInfo(Opcode::SFF1I32B32).issue_type,
            ArchitecturalIssueType::ScalarAluOrMemory);
}

TEST(OpcodeDescriptorTest, ClassifiesMaskAndBranchOps) {
  EXPECT_EQ(GetOpcodeExecutionInfo(Opcode::MaskAndExecCmask).family, SemanticFamily::Mask);
  EXPECT_TRUE(GetOpcodeExecutionInfo(Opcode::BIfNoexec).may_branch);
}

}  // namespace
}  // namespace gpu_model
