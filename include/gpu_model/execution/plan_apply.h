#pragma once

#include "gpu_model/exec/op_plan_apply.h"
#include "gpu_model/execution/wave_context.h"

namespace gpu_model {

inline void ApplyExecutionPlanRegisterWrites(const OpPlan& plan, WaveContext& wave) {
  ApplyPlanRegisterWrites(plan, wave);
}

inline std::optional<std::string> MaybeFormatExecutionMaskUpdate(const OpPlan& plan,
                                                                 const WaveContext& wave) {
  return MaybeFormatExecMaskUpdate(plan, wave);
}

inline void ApplyExecutionPlanControlFlow(const OpPlan& plan,
                                          WaveContext& wave,
                                          bool set_valid_entry,
                                          bool clear_branch_pending) {
  ApplyPlanControlFlow(plan, wave, set_valid_entry, clear_branch_pending);
}

}  // namespace gpu_model
