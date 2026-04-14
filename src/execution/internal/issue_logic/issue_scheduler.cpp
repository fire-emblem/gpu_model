#include "execution/internal/issue_logic/issue_scheduler.h"

#include <algorithm>
#include <array>
#include <numeric>
#include <set>
#include <tuple>

namespace gpu_model {

namespace {

size_t IssueTypeIndex(ArchitecturalIssueType type) {
  switch (type) {
    case ArchitecturalIssueType::Branch:
      return 0;
    case ArchitecturalIssueType::ScalarAluOrMemory:
      return 1;
    case ArchitecturalIssueType::VectorAlu:
      return 2;
    case ArchitecturalIssueType::VectorMemory:
      return 3;
    case ArchitecturalIssueType::LocalDataShare:
      return 4;
    case ArchitecturalIssueType::GlobalDataShareOrExport:
      return 5;
    case ArchitecturalIssueType::Special:
      return 6;
  }
  return 0;
}

std::array<uint32_t, 7> LimitsToArray(const ArchitecturalIssueLimits& limits) {
  return {limits.branch,
          limits.scalar_alu_or_memory,
          limits.vector_alu,
          limits.vector_memory,
          limits.local_data_share,
          limits.global_data_share_or_export,
          limits.special};
}

std::vector<size_t> BuildSelectionTraversalOrder(
    const std::vector<IssueSchedulerCandidate>& candidates,
    size_t selection_cursor,
    EligibleWaveSelectionPolicy selection_policy) {
  std::vector<size_t> order(candidates.size());
  std::iota(order.begin(), order.end(), 0);
  if (order.empty()) {
    return order;
  }
  switch (selection_policy) {
    case EligibleWaveSelectionPolicy::RoundRobin: {
      const size_t start = selection_cursor % order.size();
      std::rotate(order.begin(), order.begin() + start, order.end());
      return order;
    }
    case EligibleWaveSelectionPolicy::OldestFirst:
      std::stable_sort(order.begin(),
                       order.end(),
                       [&candidates](size_t lhs, size_t rhs) {
                         return std::tie(candidates[lhs].age_order_key,
                                         candidates[lhs].candidate_index) <
                                std::tie(candidates[rhs].age_order_key,
                                         candidates[rhs].candidate_index);
                       });
      return order;
  }
  return order;
}

size_t AdvanceSelectionCursor(size_t selection_cursor,
                              size_t count,
                              EligibleWaveSelectionPolicy selection_policy,
                              bool selected_any) {
  if (count == 0) {
    return 0;
  }
  switch (selection_policy) {
    case EligibleWaveSelectionPolicy::RoundRobin: {
      const size_t cursor = selection_cursor % count;
      return selected_any ? ((cursor + 1) % count) : cursor;
    }
    case EligibleWaveSelectionPolicy::OldestFirst:
      return 0;
  }
  return 0;
}

}  // namespace

IssueSchedulerResult IssueScheduler::SelectIssueBundle(
    const std::vector<IssueSchedulerCandidate>& candidates,
    size_t round_robin_start_index,
    const ArchitecturalIssueLimits& limits) {
  return SelectIssueBundle(candidates,
                           round_robin_start_index,
                           EligibleWaveSelectionPolicy::RoundRobin,
                           ArchitecturalIssuePolicyFromLimits(limits));
}

IssueSchedulerResult IssueScheduler::SelectIssueBundle(
    const std::vector<IssueSchedulerCandidate>& candidates,
    size_t round_robin_start_index,
    const ArchitecturalIssuePolicy& policy) {
  return SelectIssueBundle(candidates,
                           round_robin_start_index,
                           EligibleWaveSelectionPolicy::RoundRobin,
                           policy);
}

IssueSchedulerResult IssueScheduler::SelectIssueBundle(
    const std::vector<IssueSchedulerCandidate>& candidates,
    size_t selection_cursor,
    EligibleWaveSelectionPolicy selection_policy,
    const ArchitecturalIssueLimits& limits) {
  return SelectIssueBundle(candidates,
                           selection_cursor,
                           selection_policy,
                           ArchitecturalIssuePolicyFromLimits(limits));
}

IssueSchedulerResult IssueScheduler::SelectIssueBundle(
    const std::vector<IssueSchedulerCandidate>& candidates,
    size_t selection_cursor,
    EligibleWaveSelectionPolicy selection_policy,
    const ArchitecturalIssuePolicy& policy) {
  IssueSchedulerResult result;
  if (candidates.empty()) {
    return result;
  }

  const auto remaining_limits = LimitsToArray(policy.type_limits);
  std::array<uint32_t, 7> used_per_type{};
  std::array<uint32_t, 7> used_per_group{};
  std::set<uint32_t> used_waves;

  const std::vector<size_t> traversal_order =
      BuildSelectionTraversalOrder(candidates, selection_cursor, selection_policy);
  for (size_t index : traversal_order) {
    const auto& candidate = candidates[index];
    if (!candidate.ready) {
      continue;
    }
    if (used_waves.find(candidate.wave_id) != used_waves.end()) {
      continue;
    }

    const size_t type_index = IssueTypeIndex(candidate.issue_type);
    const size_t group_index = policy.type_to_group[type_index];
    if (used_per_type[type_index] >= remaining_limits[type_index]) {
      continue;
    }
    if (used_per_group[group_index] >= policy.group_limits[group_index]) {
      continue;
    }

    result.selected_candidate_indices.push_back(candidate.candidate_index);
    used_waves.insert(candidate.wave_id);
    ++used_per_type[type_index];
    ++used_per_group[group_index];
  }

  result.next_round_robin_index = AdvanceSelectionCursor(selection_cursor,
                                                         candidates.size(),
                                                         selection_policy,
                                                         !result.selected_candidate_indices.empty());
  return result;
}

}  // namespace gpu_model
