#pragma once

#include "gpu_model/exec/execution_engine.h"
#include "gpu_model/exec/functional_execution_mode.h"
#include "gpu_model/exec/functional_executor.h"

namespace gpu_model {

class ParallelWaveExecutor final : public IExecutionEngine {
 public:
  explicit ParallelWaveExecutor(FunctionalExecutionConfig config = {}) : config_(config) {}

  uint64_t Run(ExecutionContext& context) override;

 private:
  FunctionalExecutionConfig config_;
};

}  // namespace gpu_model
