#pragma once

#include "gpu_model/exec/execution_state_builder.h"

namespace gpu_model {

inline std::vector<ExecutionBlockState> BuildWaveContextBlocks(const PlacementMap& placement,
                                                               const LaunchConfig& launch_config) {
  return BuildExecutionBlockStates(placement, launch_config);
}

using WaveContextBuilder = decltype(&BuildWaveContextBlocks);

}  // namespace gpu_model
