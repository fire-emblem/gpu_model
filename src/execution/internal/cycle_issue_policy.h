#pragma once

#include "gpu_arch/chip_config/gpu_arch_spec.h"

namespace gpu_model {

inline ArchitecturalIssueLimits CycleIssueLimitsForSpec(const GpuArchSpec& spec) {
  return spec.cycle_resources.issue_limits;
}

inline ArchitecturalIssuePolicy CycleIssuePolicyForSpec(const GpuArchSpec& spec) {
  return spec.cycle_resources.issue_policy;
}

inline EligibleWaveSelectionPolicy CycleEligibleWaveSelectionPolicyForSpec(const GpuArchSpec& spec) {
  return spec.cycle_resources.eligible_wave_selection_policy;
}

inline ArchitecturalIssuePolicy CycleIssuePolicyWithLimits(
    const ArchitecturalIssuePolicy& base_policy,
    const ArchitecturalIssueLimits& limits) {
  ArchitecturalIssuePolicy policy = base_policy;
  policy.type_limits = limits;
  policy.group_limits.fill(0);

  const std::array<uint32_t, 7> type_limits = {limits.branch,
                                               limits.scalar_alu_or_memory,
                                               limits.vector_alu,
                                               limits.vector_memory,
                                               limits.local_data_share,
                                               limits.global_data_share_or_export,
                                               limits.special};

  for (size_t type_index = 0; type_index < type_limits.size(); ++type_index) {
    const size_t group_index = policy.type_to_group[type_index];
    if (policy.group_limits[group_index] < type_limits[type_index]) {
      policy.group_limits[group_index] = type_limits[type_index];
    }
  }

  return policy;
}

}  // namespace gpu_model
