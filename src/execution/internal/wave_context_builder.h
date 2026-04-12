#pragma once

#include <vector>

#include "runtime/mapper.h"
#include "runtime/launch_config.h"
#include "state/ap/ap_runtime_state.h"

namespace gpu_model {

std::vector<ExecutionBlockState> BuildWaveContextBlocks(const PlacementMap& placement,
                                                        const LaunchConfig& launch_config);

using WaveContextBuilder = decltype(&BuildWaveContextBlocks);

}  // namespace gpu_model
