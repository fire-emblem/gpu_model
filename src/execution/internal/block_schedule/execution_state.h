#pragma once

#include "runtime/model_runtime/core/mapper.h"
#include "state/ap/ap_runtime_state.h"

namespace gpu_model {

WaveContext BuildInitialWaveContext(const BlockPlacement& block_placement,
                                    const WavePlacement& wave_placement);

}  // namespace gpu_model
