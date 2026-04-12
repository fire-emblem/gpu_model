#pragma once

#include <string>

#include "debug/trace/step_detail.h"
#include "state/wave/wave_runtime_state.h"
#include "instruction/isa/instruction.h"

namespace gpu_model {

// Format a simple message for WaveStep (backward compatible)
std::string FormatWaveStepMessage(const Instruction& instruction, const WaveContext& wave);

// Format complete assembly text: "v_add_f32 v0, v1, v2"
std::string FormatAssemblyText(const Instruction& instruction);

// Build structured step detail for WaveStep
TraceWaveStepDetail BuildWaveStepDetail(const Instruction& instruction, const WaveContext& wave);

}  // namespace gpu_model
