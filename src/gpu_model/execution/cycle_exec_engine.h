#pragma once

#include <optional>

#include "gpu_model/arch/gpu_arch_spec.h"
#include "gpu_model/execution/internal/execution_engine.h"

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

class CycleExecEngine final : public IExecutionEngine {
 public:
  explicit CycleExecEngine(CycleTimingConfig timing_config = {}) : timing_config_(timing_config) {}

  uint64_t Run(ExecutionContext& context) override;

 private:
  CycleTimingConfig timing_config_;
  Semantics semantics_;
};

}  // namespace gpu_model
