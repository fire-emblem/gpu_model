#pragma once

#include <string>

#include "gpu_model/isa/instruction.h"
#include "gpu_model/state/wave_state.h"

namespace gpu_model {

std::string FormatWaveStepMessage(const Instruction& instruction, const WaveState& wave);

}  // namespace gpu_model
