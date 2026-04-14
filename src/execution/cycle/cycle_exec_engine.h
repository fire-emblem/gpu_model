#pragma once

#include <optional>

#include "gpu_arch/device/gpu_arch_spec.h"
#include "execution/internal/plan/execution_engine.h"
#include "runtime/model_runtime/program_cycle_stats.h"

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
  std::optional<ProgramCycleStats> TakeProgramCycleStats() const { return program_cycle_stats_; }

 private:
  CycleTimingConfig timing_config_;
  Semantics semantics_;
  std::optional<ProgramCycleStats> program_cycle_stats_;
};

}  // namespace gpu_model
