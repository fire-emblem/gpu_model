#include <gtest/gtest.h>

#include "gpu_model/isa/target_isa.h"
#include "gpu_model/loader/program_lowering.h"

namespace gpu_model {
namespace {

TEST(ProgramLoweringTest, LowersCanonicalAsmProgramImage) {
  MetadataBlob metadata;
  SetTargetIsa(metadata, TargetIsa::CanonicalAsm);
  ProgramImage image("canonical_exit", "s_endpgm\n", metadata);

  const auto kernel = ProgramLoweringRegistry::Lower(image);
  ASSERT_EQ(kernel.instructions().size(), 1u);
  EXPECT_EQ(kernel.instructions()[0].opcode, Opcode::BExit);
}

TEST(ProgramLoweringTest, LowersGcnAsmSubsetProgramImage) {
  MetadataBlob metadata;
  SetTargetIsa(metadata, TargetIsa::GcnAsm);
  ProgramImage image("gcn_subset_exit", "s_endpgm\n", metadata);

  const auto kernel = ProgramLoweringRegistry::Lower(image);
  ASSERT_EQ(kernel.instructions().size(), 1u);
  EXPECT_EQ(kernel.instructions()[0].opcode, Opcode::BExit);
}

}  // namespace
}  // namespace gpu_model
