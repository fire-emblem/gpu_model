#include <gtest/gtest.h>

#include "gpu_model/execution/internal/issue_scheduler.h"

namespace gpu_model {
namespace {

TEST(IssueSchedulerTest, AllowsAtMostOnePerIssueTypeByDefault) {
  const std::vector<IssueSchedulerCandidate> candidates{
      {.candidate_index = 0,
       .wave_id = 0,
       .issue_type = ArchitecturalIssueType::VectorAlu,
       .ready = true},
      {.candidate_index = 1,
       .wave_id = 1,
       .issue_type = ArchitecturalIssueType::VectorAlu,
       .ready = true},
      {.candidate_index = 2,
       .wave_id = 2,
       .issue_type = ArchitecturalIssueType::ScalarAluOrMemory,
       .ready = true},
  };

  const auto result = IssueScheduler::SelectIssueBundle(candidates, 0);
  ASSERT_EQ(result.selected_candidate_indices.size(), 2u);
  EXPECT_EQ(result.selected_candidate_indices[0], 0u);
  EXPECT_EQ(result.selected_candidate_indices[1], 2u);
}

TEST(IssueSchedulerTest, AllowsAtMostOneInstructionPerWavefront) {
  const std::vector<IssueSchedulerCandidate> candidates{
      {.candidate_index = 0,
       .wave_id = 7,
       .issue_type = ArchitecturalIssueType::ScalarAluOrMemory,
       .ready = true},
      {.candidate_index = 1,
       .wave_id = 7,
       .issue_type = ArchitecturalIssueType::VectorAlu,
       .ready = true},
      {.candidate_index = 2,
       .wave_id = 8,
       .issue_type = ArchitecturalIssueType::VectorMemory,
       .ready = true},
  };

  const auto result = IssueScheduler::SelectIssueBundle(candidates, 0);
  ASSERT_EQ(result.selected_candidate_indices.size(), 2u);
  EXPECT_EQ(result.selected_candidate_indices[0], 0u);
  EXPECT_EQ(result.selected_candidate_indices[1], 2u);
}

TEST(IssueSchedulerTest, RespectsRoundRobinStartIndex) {
  const std::vector<IssueSchedulerCandidate> candidates{
      {.candidate_index = 0,
       .wave_id = 0,
       .issue_type = ArchitecturalIssueType::ScalarAluOrMemory,
       .ready = true},
      {.candidate_index = 1,
       .wave_id = 1,
       .issue_type = ArchitecturalIssueType::VectorAlu,
       .ready = true},
      {.candidate_index = 2,
       .wave_id = 2,
       .issue_type = ArchitecturalIssueType::VectorMemory,
       .ready = true},
  };

  const auto result = IssueScheduler::SelectIssueBundle(candidates, 1);
  ASSERT_EQ(result.selected_candidate_indices.size(), 3u);
  EXPECT_EQ(result.selected_candidate_indices[0], 1u);
  EXPECT_EQ(result.selected_candidate_indices[1], 2u);
  EXPECT_EQ(result.selected_candidate_indices[2], 0u);
  EXPECT_EQ(result.next_round_robin_index, 2u);
}

TEST(IssueSchedulerTest, AllowsMixedPipeBundleWhenLimitsPermit) {
  const std::vector<IssueSchedulerCandidate> candidates{
      {.candidate_index = 0,
       .wave_id = 0,
       .issue_type = ArchitecturalIssueType::ScalarAluOrMemory,
       .ready = true},
      {.candidate_index = 1,
       .wave_id = 1,
       .issue_type = ArchitecturalIssueType::VectorAlu,
       .ready = true},
      {.candidate_index = 2,
       .wave_id = 2,
       .issue_type = ArchitecturalIssueType::VectorMemory,
       .ready = true},
      {.candidate_index = 3,
       .wave_id = 3,
       .issue_type = ArchitecturalIssueType::LocalDataShare,
       .ready = true},
      {.candidate_index = 4,
       .wave_id = 4,
       .issue_type = ArchitecturalIssueType::Special,
       .ready = true},
  };

  const auto result = IssueScheduler::SelectIssueBundle(candidates, 0);
  ASSERT_EQ(result.selected_candidate_indices.size(), 5u);
}

TEST(IssueSchedulerTest, RespectsPerPipeOverrideLimits) {
  const std::vector<IssueSchedulerCandidate> candidates{
      {.candidate_index = 0,
       .wave_id = 0,
       .issue_type = ArchitecturalIssueType::VectorAlu,
       .ready = true},
      {.candidate_index = 1,
       .wave_id = 1,
       .issue_type = ArchitecturalIssueType::VectorAlu,
       .ready = true},
      {.candidate_index = 2,
       .wave_id = 2,
       .issue_type = ArchitecturalIssueType::VectorAlu,
       .ready = true},
  };

  ArchitecturalIssueLimits limits = DefaultArchitecturalIssueLimits();
  limits.vector_alu = 2;
  const auto result = IssueScheduler::SelectIssueBundle(candidates, 0, limits);
  ASSERT_EQ(result.selected_candidate_indices.size(), 2u);
  EXPECT_EQ(result.selected_candidate_indices[0], 0u);
  EXPECT_EQ(result.selected_candidate_indices[1], 1u);
}

TEST(IssueSchedulerTest, KeepsSpecialPipeIndependentFromBranchPipe) {
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

  const auto result = IssueScheduler::SelectIssueBundle(candidates, 0);
  ASSERT_EQ(result.selected_candidate_indices.size(), 2u);
}

TEST(IssueSchedulerTest, SharedIssueGroupCanMakeTwoTypesConflict) {
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

  auto policy = DefaultArchitecturalIssuePolicy();
  policy.type_to_group[0] = 0;  // Branch
  policy.type_to_group[6] = 0;  // Special
  policy.group_limits[0] = 1;

  const auto result = IssueScheduler::SelectIssueBundle(candidates, 0, policy);
  ASSERT_EQ(result.selected_candidate_indices.size(), 1u);
  EXPECT_EQ(result.selected_candidate_indices[0], 0u);
}

TEST(IssueSchedulerTest, ExplicitRoundRobinPolicyRespectsSelectionCursorOnUnrotatedCandidates) {
  const std::vector<IssueSchedulerCandidate> candidates{
      {.candidate_index = 0,
       .wave_id = 0,
       .age_order_key = 30,
       .issue_type = ArchitecturalIssueType::ScalarAluOrMemory,
       .ready = true},
      {.candidate_index = 1,
       .wave_id = 1,
       .age_order_key = 10,
       .issue_type = ArchitecturalIssueType::VectorAlu,
       .ready = true},
      {.candidate_index = 2,
       .wave_id = 2,
       .age_order_key = 20,
       .issue_type = ArchitecturalIssueType::VectorMemory,
       .ready = true},
  };

  const auto result = IssueScheduler::SelectIssueBundle(
      candidates, 1, EligibleWaveSelectionPolicy::RoundRobin, DefaultArchitecturalIssuePolicy());
  ASSERT_EQ(result.selected_candidate_indices.size(), 3u);
  EXPECT_EQ(result.selected_candidate_indices[0], 1u);
  EXPECT_EQ(result.selected_candidate_indices[1], 2u);
  EXPECT_EQ(result.selected_candidate_indices[2], 0u);
  EXPECT_EQ(result.next_round_robin_index, 2u);
}

TEST(IssueSchedulerTest, ExplicitOldestFirstPolicyUsesAgeOrderKeyInsteadOfSelectionCursor) {
  const std::vector<IssueSchedulerCandidate> candidates{
      {.candidate_index = 0,
       .wave_id = 0,
       .age_order_key = 30,
       .issue_type = ArchitecturalIssueType::ScalarAluOrMemory,
       .ready = true},
      {.candidate_index = 1,
       .wave_id = 1,
       .age_order_key = 10,
       .issue_type = ArchitecturalIssueType::VectorAlu,
       .ready = true},
      {.candidate_index = 2,
       .wave_id = 2,
       .age_order_key = 20,
       .issue_type = ArchitecturalIssueType::VectorMemory,
       .ready = true},
  };

  const auto result = IssueScheduler::SelectIssueBundle(
      candidates, 2, EligibleWaveSelectionPolicy::OldestFirst, DefaultArchitecturalIssuePolicy());
  ASSERT_EQ(result.selected_candidate_indices.size(), 3u);
  EXPECT_EQ(result.selected_candidate_indices[0], 1u);
  EXPECT_EQ(result.selected_candidate_indices[1], 2u);
  EXPECT_EQ(result.selected_candidate_indices[2], 0u);
  EXPECT_EQ(result.next_round_robin_index, 0u);
}

}  // namespace
}  // namespace gpu_model
