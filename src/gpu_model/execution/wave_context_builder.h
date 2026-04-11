#pragma once

#include <vector>

#include "gpu_model/runtime/mapper.h"
#include "gpu_model/runtime/launch_config.h"
#include "gpu_model/state/ap/ap_runtime_state.h"

namespace gpu_model {

std::vector<ExecutionBlockState> BuildWaveContextBlocks(const PlacementMap& placement,
                                                        const LaunchConfig& launch_config);

using WaveContextBuilder = decltype(&BuildWaveContextBlocks);

}  // namespace gpu_model
