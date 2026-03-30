#pragma once

#include <cstdint>

#include "gpu_model/exec/semantics.h"

namespace gpu_model {

class FunctionalExecEngine {
 public:
  explicit FunctionalExecEngine(ExecutionContext& context) : context_(context) {}

  uint64_t RunSequential();
  uint64_t RunParallelBlocks(uint32_t worker_threads);

 private:
  ExecutionContext& context_;
};

}  // namespace gpu_model
