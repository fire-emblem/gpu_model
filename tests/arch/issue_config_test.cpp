#include <gtest/gtest.h>

#include "gpu_arch/issue_config/issue_config.h"

namespace gpu_model {
namespace {

TEST(IssueConfigTest, MapsRepresentativeOpcodesToWhitepaperIssueTypes) {
  EXPECT_EQ(ArchitecturalIssueTypeForOpcode(Opcode::BBranch), ArchitecturalIssueType::Branch);
  EXPECT_EQ(ArchitecturalIssueTypeForOpcode(Opcode::SAdd), ArchitecturalIssueType::ScalarAluOrMemory);
  EXPECT_EQ(ArchitecturalIssueTypeForOpcode(Opcode::SBufferLoadDword),
            ArchitecturalIssueType::ScalarAluOrMemory);
  EXPECT_EQ(ArchitecturalIssueTypeForOpcode(Opcode::VAdd), ArchitecturalIssueType::VectorAlu);
  EXPECT_EQ(ArchitecturalIssueTypeForOpcode(Opcode::MLoadGlobal),
            ArchitecturalIssueType::VectorMemory);
  EXPECT_EQ(ArchitecturalIssueTypeForOpcode(Opcode::MLoadShared),
            ArchitecturalIssueType::LocalDataShare);
  EXPECT_EQ(ArchitecturalIssueTypeForOpcode(Opcode::SyncBarrier),
            ArchitecturalIssueType::Special);
}

TEST(IssueConfigTest, DefaultArchitecturalIssueLimitsAllowOnePerType) {
  const auto limits = DefaultArchitecturalIssueLimits();
  EXPECT_EQ(limits.branch, 1u);
  EXPECT_EQ(limits.scalar_alu_or_memory, 1u);
  EXPECT_EQ(limits.vector_alu, 1u);
  EXPECT_EQ(limits.vector_memory, 1u);
  EXPECT_EQ(limits.local_data_share, 1u);
  EXPECT_EQ(limits.global_data_share_or_export, 1u);
  EXPECT_EQ(limits.special, 1u);
}

}  // namespace
}  // namespace gpu_model
