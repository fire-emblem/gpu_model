#pragma once

#include "gpu_model/runtime/mapper.h"
#include "gpu_model/state/ap/ap_runtime_state.h"

namespace gpu_model {

WaveContext BuildInitialWaveContext(const BlockPlacement& block_placement,
                                    const WavePlacement& wave_placement);

}  // namespace gpu_model
