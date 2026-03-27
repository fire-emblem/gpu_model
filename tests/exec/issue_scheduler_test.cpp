#include <gtest/gtest.h>

#include "gpu_model/exec/issue_scheduler.h"

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

}  // namespace
}  // namespace gpu_model
