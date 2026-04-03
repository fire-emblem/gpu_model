#pragma once

#include "gpu_model/arch/gpu_arch_spec.h"

namespace gpu_model {

inline ArchitecturalIssueLimits CycleIssueLimitsForSpec(const GpuArchSpec& spec) {
  return spec.cycle_resources.issue_limits;
}

inline ArchitecturalIssuePolicy CycleIssuePolicyForSpec(const GpuArchSpec& spec) {
  return spec.cycle_resources.issue_policy;
}

}  // namespace gpu_model
