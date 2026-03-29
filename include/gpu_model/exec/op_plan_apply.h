#pragma once

#include <optional>
#include <string>

#include "gpu_model/exec/op_plan.h"
#include "gpu_model/state/wave_state.h"

namespace gpu_model {

void ApplyPlanRegisterWrites(const OpPlan& plan, WaveState& wave);

std::optional<std::string> MaybeFormatExecMaskUpdate(const OpPlan& plan, const WaveState& wave);

void ApplyPlanControlFlow(const OpPlan& plan,
                          WaveState& wave,
                          bool set_valid_entry,
                          bool clear_branch_pending);

}  // namespace gpu_model
