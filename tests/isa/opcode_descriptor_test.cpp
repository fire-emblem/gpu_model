#include <gtest/gtest.h>

#include "gpu_model/exec/opcode_execution_info.h"
#include "gpu_model/isa/opcode_descriptor.h"

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

TEST(OpcodeDescriptorTest, ClassifiesMaskAndBranchOps) {
  EXPECT_EQ(GetOpcodeExecutionInfo(Opcode::MaskAndExecCmask).family, SemanticFamily::Mask);
  EXPECT_TRUE(GetOpcodeExecutionInfo(Opcode::BIfNoexec).may_branch);
}

}  // namespace
}  // namespace gpu_model
