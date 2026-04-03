#include "gpu_model/execution/internal/issue_scheduler.h"

#include <array>
#include <set>

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

}  // namespace

IssueSchedulerResult IssueScheduler::SelectIssueBundle(
    const std::vector<IssueSchedulerCandidate>& candidates,
    size_t round_robin_start_index,
    const ArchitecturalIssueLimits& limits) {
  auto policy = DefaultArchitecturalIssuePolicy();
  policy.type_limits = limits;
  policy.group_limits = {limits.branch,
                         limits.scalar_alu_or_memory,
                         limits.vector_alu,
                         limits.vector_memory,
                         limits.local_data_share,
                         limits.global_data_share_or_export,
                         limits.special};
  return SelectIssueBundle(candidates, round_robin_start_index, policy);
}

IssueSchedulerResult IssueScheduler::SelectIssueBundle(
    const std::vector<IssueSchedulerCandidate>& candidates,
    size_t round_robin_start_index,
    const ArchitecturalIssuePolicy& policy) {
  IssueSchedulerResult result;
  if (candidates.empty()) {
    return result;
  }

  const auto remaining_limits = LimitsToArray(policy.type_limits);
  std::array<uint32_t, 7> used_per_type{};
  std::array<uint32_t, 7> used_per_group{};
  std::set<uint32_t> used_waves;

  const size_t count = candidates.size();
  const size_t start = round_robin_start_index % count;
  for (size_t offset = 0; offset < count; ++offset) {
    const size_t index = (start + offset) % count;
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

  if (!result.selected_candidate_indices.empty()) {
    result.next_round_robin_index = (start + 1) % count;
  } else {
    result.next_round_robin_index = start;
  }
  return result;
}

}  // namespace gpu_model
