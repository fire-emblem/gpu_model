#pragma once

#include <optional>
#include <string>

#include "execution/internal/op_plan.h"
#include "program/executable/executable_kernel.h"
#include "state/wave/wave_runtime_state.h"

namespace gpu_model {

void ApplyExecutionPlanRegisterWrites(const OpPlan& plan, WaveContext& wave);

std::optional<std::string> MaybeFormatExecutionMaskUpdate(const OpPlan& plan,
                                                          const WaveContext& wave);

void ApplyExecutionPlanControlFlow(const ExecutableKernel& kernel,
                                   const OpPlan& plan,
                                   WaveContext& wave,
                                   bool set_valid_entry,
                                   bool clear_branch_pending);

}  // namespace gpu_model
