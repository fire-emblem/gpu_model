#pragma once

#include "gpu_model/exec/execution_engine.h"

namespace gpu_model {

class CycleExecutor final : public IExecutionEngine {
 public:
  explicit CycleExecutor(uint64_t fixed_global_latency = 20)
      : fixed_global_latency_(fixed_global_latency) {}

  uint64_t Run(ExecutionContext& context) override;

 private:
  uint64_t fixed_global_latency_ = 20;
  Semantics semantics_;
};

}  // namespace gpu_model
