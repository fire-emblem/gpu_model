#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "gpu_model/gpu_arch/issue_config/issue_config.h"

namespace gpu_model {

// Blocked reason for eligibility projection diagnostics
enum class IssueBlockedReason {
  None,                // Not blocked
  NotYetEligible,      // next_issue_cycle > current_cycle
  WaveWaiting,         // Wave in wait state (waitcnt, barrier, etc.)
  NoValidInstruction,  // No valid instruction at PC
  BundleConflict,      // Type/group limit reached in bundle
  SwitchPenalty,       // Wave switch penalty not yet satisfied
};

struct IssueSchedulerCandidate {
  // Core fields consumed by scheduler
  size_t candidate_index = 0;
  uint32_t wave_id = 0;
  uint64_t age_order_key = 0;
  ArchitecturalIssueType issue_type = ArchitecturalIssueType::ScalarAluOrMemory;
  bool ready = false;

  // Extended eligibility projection fields (for engine diagnostics)
  // These are NOT consumed by scheduler, but available for engine-side analysis
  uint64_t eligible_since_cycle = 0;
  uint64_t next_issue_earliest_global_cycle = 0;
  IssueBlockedReason blocked_reason = IssueBlockedReason::None;
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
  static IssueSchedulerResult SelectIssueBundle(
      const std::vector<IssueSchedulerCandidate>& candidates,
      size_t selection_cursor,
      EligibleWaveSelectionPolicy selection_policy,
      const ArchitecturalIssueLimits& limits);
  static IssueSchedulerResult SelectIssueBundle(
      const std::vector<IssueSchedulerCandidate>& candidates,
      size_t selection_cursor,
      EligibleWaveSelectionPolicy selection_policy,
      const ArchitecturalIssuePolicy& policy);
};

}  // namespace gpu_model
