#pragma once

#include "gpu_model/execution/plan_apply.h"
#include "gpu_model/state/wave_state.h"

namespace gpu_model {

inline void ApplyPlanRegisterWrites(const OpPlan& plan, WaveState& wave) {
  ApplyExecutionPlanRegisterWrites(plan, wave);
}

inline std::optional<std::string> MaybeFormatExecMaskUpdate(const OpPlan& plan, const WaveState& wave) {
  return MaybeFormatExecutionMaskUpdate(plan, wave);
}

inline void ApplyPlanControlFlow(const OpPlan& plan,
                                 WaveState& wave,
                                 bool set_valid_entry,
                                 bool clear_branch_pending) {
  ApplyExecutionPlanControlFlow(plan, wave, set_valid_entry, clear_branch_pending);
}

}  // namespace gpu_model
