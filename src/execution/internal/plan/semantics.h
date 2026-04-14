#pragma once

#include "execution/execution_context.h"
#include "execution/internal/plan/op_plan.h"

namespace gpu_model {

class Semantics {
 public:
  OpPlan BuildPlan(const Instruction& instruction,
                   const WaveContext& wave,
                   const ExecutionContext& context) const;
};

}  // namespace gpu_model
