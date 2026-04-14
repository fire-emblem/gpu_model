#include <gtest/gtest.h>

#include "execution/internal/issue_logic/encoded_issue_type.h"

namespace gpu_model {
namespace {

DecodedInstruction MakeDecoded(std::string mnemonic, EncodedGcnInstFormatClass format_class) {
  DecodedInstruction decoded;
  decoded.mnemonic = std::move(mnemonic);
  decoded.format_class = format_class;
  return decoded;
}

TEST(EncodedIssueTypeTest, ClassifiesRepresentativeScalarAndVectorInstructions) {
  EXPECT_EQ(
      ArchitecturalIssueTypeForEncodedInstruction(
          MakeDecoded("s_add_i32", EncodedGcnInstFormatClass::Sop2),
          EncodedInstructionDescriptor{.category = EncodedInstructionCategory::Scalar}),
      ArchitecturalIssueType::ScalarAluOrMemory);
  EXPECT_EQ(
      ArchitecturalIssueTypeForEncodedInstruction(
          MakeDecoded("v_add_f32_e32", EncodedGcnInstFormatClass::Vop2),
          EncodedInstructionDescriptor{.category = EncodedInstructionCategory::Vector}),
      ArchitecturalIssueType::VectorAlu);
}

TEST(EncodedIssueTypeTest, ClassifiesSharedAndGlobalMemoryInstructions) {
  EXPECT_EQ(
      ArchitecturalIssueTypeForEncodedInstruction(
          MakeDecoded("ds_read_b32", EncodedGcnInstFormatClass::Ds),
          EncodedInstructionDescriptor{.category = EncodedInstructionCategory::Memory}),
      ArchitecturalIssueType::LocalDataShare);
  EXPECT_EQ(
      ArchitecturalIssueTypeForEncodedInstruction(
          MakeDecoded("buffer_load_dword", EncodedGcnInstFormatClass::Mubuf),
          EncodedInstructionDescriptor{.category = EncodedInstructionCategory::Memory}),
      ArchitecturalIssueType::VectorMemory);
}

TEST(EncodedIssueTypeTest, ClassifiesBarrierAndBranchAsSpecializedPipes) {
  EXPECT_EQ(
      ArchitecturalIssueTypeForEncodedInstruction(
          MakeDecoded("s_barrier", EncodedGcnInstFormatClass::Sopp),
          EncodedInstructionDescriptor{.category = EncodedInstructionCategory::Vector}),
      ArchitecturalIssueType::Special);
  EXPECT_EQ(
      ArchitecturalIssueTypeForEncodedInstruction(
          MakeDecoded("s_cbranch_scc0", EncodedGcnInstFormatClass::Sopp),
          EncodedInstructionDescriptor{.category = EncodedInstructionCategory::Scalar}),
      ArchitecturalIssueType::Branch);
}

}  // namespace
}  // namespace gpu_model
