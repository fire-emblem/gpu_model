#pragma once

#include <optional>
#include <string_view>

#include "gpu_model/isa/instruction.h"
#include "gpu_model/execution/wave_context.h"

namespace gpu_model {

std::optional<WaveWaitReason> MapWaitcntStringToWaveWaitReason(std::string_view reason);
bool EnterWaitStateFromInstruction(const Instruction& instruction, WaveContext& wave);
bool WaitReasonSatisfied(const WaveContext& wave);
bool ResumeWaveIfWaitReasonSatisfied(WaveContext& wave);

}  // namespace gpu_model
