#pragma once

#include "gpu_model/exec/execution_engine.h"

namespace gpu_model {

class FunctionalExecutor final : public IExecutionEngine {
 public:
  uint64_t Run(ExecutionContext& context) override;

 private:
  Semantics semantics_;
};

}  // namespace gpu_model
