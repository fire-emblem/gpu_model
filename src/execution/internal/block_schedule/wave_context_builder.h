#pragma once

#include <vector>

#include "runtime/model_runtime/core/mapper.h"
#include "runtime/config/launch_config.h"
#include "state/ap/ap_runtime_state.h"

namespace gpu_model {

std::vector<ExecutionBlockState> BuildWaveContextBlocks(const PlacementMap& placement,
                                                        const LaunchConfig& launch_config);

using WaveContextBuilder = decltype(&BuildWaveContextBlocks);

}  // namespace gpu_model
