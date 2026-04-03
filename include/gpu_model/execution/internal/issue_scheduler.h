#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "gpu_model/execution/internal/issue_model.h"

namespace gpu_model {

struct IssueSchedulerCandidate {
  size_t candidate_index = 0;
  uint32_t wave_id = 0;
  ArchitecturalIssueType issue_type = ArchitecturalIssueType::ScalarAluOrMemory;
  bool ready = false;
};

struct IssueSchedulerResult {
  std::vector<size_t> selected_candidate_indices;
  size_t next_round_robin_index = 0;
};

class IssueScheduler {
 public:
  static IssueSchedulerResult SelectIssueBundle(
      const std::vector<IssueSchedulerCandidate>& candidates,
      size_t round_robin_start_index,
      const ArchitecturalIssueLimits& limits = DefaultArchitecturalIssueLimits());
  static IssueSchedulerResult SelectIssueBundle(
      const std::vector<IssueSchedulerCandidate>& candidates,
      size_t round_robin_start_index,
      const ArchitecturalIssuePolicy& policy);
};

}  // namespace gpu_model
