#pragma once

#include <string>

#include "gpu_model/execution/wave_context.h"
#include "gpu_model/isa/instruction.h"

namespace gpu_model {

std::string FormatWaveStepMessage(const Instruction& instruction, const WaveContext& wave);

}  // namespace gpu_model
