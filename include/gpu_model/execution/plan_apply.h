#pragma once

#include <optional>
#include <string>

#include "gpu_model/exec/op_plan.h"
#include "gpu_model/execution/wave_context.h"
#include "gpu_model/state/wave_state.h"

namespace gpu_model {

void ApplyExecutionPlanRegisterWrites(const OpPlan& plan, WaveContext& wave);

std::optional<std::string> MaybeFormatExecutionMaskUpdate(const OpPlan& plan,
                                                          const WaveContext& wave);

void ApplyExecutionPlanControlFlow(const OpPlan& plan,
                                   WaveContext& wave,
                                   bool set_valid_entry,
                                   bool clear_branch_pending);

}  // namespace gpu_model
