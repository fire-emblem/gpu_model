#pragma once

#include <cstdint>

#include "execution/execution_context.h"

namespace gpu_model {

class IExecutionEngine {
 public:
  virtual ~IExecutionEngine() = default;
  virtual uint64_t Run(ExecutionContext& context) = 0;
};

}  // namespace gpu_model
