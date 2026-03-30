#pragma once

#include <vector>

#include "gpu_model/exec/execution_state.h"
#include "gpu_model/runtime/launch_config.h"

namespace gpu_model {

std::vector<ExecutionBlockState> BuildWaveContextBlocks(const PlacementMap& placement,
                                                        const LaunchConfig& launch_config);

using WaveContextBuilder = decltype(&BuildWaveContextBlocks);

}  // namespace gpu_model
