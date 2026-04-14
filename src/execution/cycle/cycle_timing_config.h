#pragma once

#include <optional>

#include "gpu_arch/device/gpu_arch_spec.h"

namespace gpu_model {

struct CycleTimingConfig {
  CacheModelSpec cache_model;
  SharedBankModelSpec shared_bank_model;
  LaunchTimingSpec launch_timing;
  IssueCycleClassOverridesSpec issue_cycle_class_overrides;
  IssueCycleOpOverridesSpec issue_cycle_op_overrides;
  ArchitecturalIssueLimits issue_limits;
  std::optional<ArchitecturalIssuePolicy> issue_policy;
  EligibleWaveSelectionPolicy eligible_wave_selection_policy =
      EligibleWaveSelectionPolicy::RoundRobin;
};

}  // namespace gpu_model
