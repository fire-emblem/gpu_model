#pragma once

#include <memory>
#include <optional>

#include "execution/cycle/cycle_timing_config.h"
#include "execution/execution_engine.h"
#include "runtime/model_runtime/stats/program_cycle_stats.h"

namespace gpu_model {

class Semantics;

class CycleExecEngine final : public IExecutionEngine {
 public:
  explicit CycleExecEngine(CycleTimingConfig timing_config = {});
  ~CycleExecEngine();

  uint64_t Run(ExecutionContext& context) override;
  std::optional<ProgramCycleStats> TakeProgramCycleStats() const { return program_cycle_stats_; }

 private:
  CycleTimingConfig timing_config_;
  std::unique_ptr<Semantics> semantics_;
  std::optional<ProgramCycleStats> program_cycle_stats_;
};

}  // namespace gpu_model
