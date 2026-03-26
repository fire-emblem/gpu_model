#pragma once

#include "gpu_model/arch/gpu_arch_spec.h"
#include "gpu_model/exec/execution_engine.h"

namespace gpu_model {

struct CycleTimingConfig {
  CacheModelSpec cache_model;
  SharedBankModelSpec shared_bank_model;
};

class CycleExecutor final : public IExecutionEngine {
 public:
  explicit CycleExecutor(CycleTimingConfig timing_config = {})
      : timing_config_(timing_config) {}

  uint64_t Run(ExecutionContext& context) override;

 private:
  CycleTimingConfig timing_config_;
  Semantics semantics_;
};

}  // namespace gpu_model
