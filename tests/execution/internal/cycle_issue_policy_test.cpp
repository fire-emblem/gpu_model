#include <gtest/gtest.h>

#include <vector>

#include "gpu_model/arch/arch_registry.h"
#include "gpu_model/execution/internal/cycle_issue_policy.h"
#include "gpu_model/execution/internal/issue_scheduler.h"

namespace gpu_model {
namespace {

TEST(CycleIssuePolicyTest, ReturnsConfiguredIssueLimitsFromSpec) {
  const auto spec = ArchRegistry::Get("mac500");
  ASSERT_NE(spec, nullptr);

  const auto limits = CycleIssueLimitsForSpec(*spec);
  EXPECT_EQ(limits.branch, 1u);
  EXPECT_EQ(limits.scalar_alu_or_memory, 1u);
  EXPECT_EQ(limits.vector_alu, 1u);
  EXPECT_EQ(limits.vector_memory, 1u);
  EXPECT_EQ(limits.local_data_share, 1u);
  EXPECT_EQ(limits.global_data_share_or_export, 1u);
  EXPECT_EQ(limits.special, 1u);
}

TEST(CycleIssuePolicyTest, ReturnsConfiguredIssuePolicyFromSpec) {
  const auto spec = ArchRegistry::Get("mac500");
  ASSERT_NE(spec, nullptr);

  const auto policy = CycleIssuePolicyForSpec(*spec);
  EXPECT_EQ(policy.type_limits.branch, 1u);
  EXPECT_EQ(policy.group_limits[0], 1u);
  EXPECT_EQ(policy.group_limits[6], 1u);
  EXPECT_EQ(policy.type_to_group[0], 0u);
  EXPECT_EQ(policy.type_to_group[6], 0u);
  EXPECT_EQ(CycleEligibleWaveSelectionPolicyForSpec(*spec),
            EligibleWaveSelectionPolicy::RoundRobin);
}

TEST(CycleIssuePolicyTest, Mac500DefaultPolicyMakesBranchAndSpecialConflictInScheduler) {
  const auto spec = ArchRegistry::Get("mac500");
  ASSERT_NE(spec, nullptr);

  const std::vector<IssueSchedulerCandidate> candidates{
      {.candidate_index = 0,
       .wave_id = 0,
       .issue_type = ArchitecturalIssueType::Branch,
       .ready = true},
      {.candidate_index = 1,
       .wave_id = 1,
       .issue_type = ArchitecturalIssueType::Special,
       .ready = true},
  };

  const auto result =
      IssueScheduler::SelectIssueBundle(candidates, 0, CycleIssuePolicyForSpec(*spec));
  ASSERT_EQ(result.selected_candidate_indices.size(), 1u);
  EXPECT_EQ(result.selected_candidate_indices[0], 0u);
}

TEST(CycleIssuePolicyTest, ApplyingLimitsToGroupedPolicyPreservesTypeGrouping) {
  const auto spec = ArchRegistry::Get("mac500");
  ASSERT_NE(spec, nullptr);

  ArchitecturalIssueLimits widened_limits = CycleIssueLimitsForSpec(*spec);
  widened_limits.branch = 2;
  widened_limits.vector_alu = 3;

  const auto policy = CycleIssuePolicyWithLimits(CycleIssuePolicyForSpec(*spec), widened_limits);
  EXPECT_EQ(policy.type_limits.branch, 2u);
  EXPECT_EQ(policy.type_limits.vector_alu, 3u);
  EXPECT_EQ(policy.type_to_group[0], 0u);
  EXPECT_EQ(policy.type_to_group[6], 0u);
  EXPECT_EQ(policy.group_limits[0], 2u);
  EXPECT_EQ(policy.group_limits[2], 3u);
}

}  // namespace
}  // namespace gpu_model
