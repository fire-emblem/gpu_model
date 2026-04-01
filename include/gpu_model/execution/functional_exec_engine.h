#pragma once

#include <cstdint>
#include <optional>

#include "gpu_model/execution/internal/semantics.h"
#include "gpu_model/runtime/program_cycle_stats.h"

namespace gpu_model {

class FunctionalExecEngine {
 public:
  explicit FunctionalExecEngine(ExecutionContext& context) : context_(context) {}

  uint64_t RunSequential();
  uint64_t RunParallelBlocks(uint32_t worker_threads);
  std::optional<ProgramCycleStats> TakeProgramCycleStats() const {
    return program_cycle_stats_;
  }

 private:
  ExecutionContext& context_;
  std::optional<ProgramCycleStats> program_cycle_stats_;
};

}  // namespace gpu_model
