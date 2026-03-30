#pragma once

#include "gpu_model/execution/wave_context_builder.h"

namespace gpu_model {

inline std::vector<ExecutionBlockState> BuildExecutionBlockStates(const PlacementMap& placement,
                                                                  const LaunchConfig& launch_config) {
  return BuildWaveContextBlocks(placement, launch_config);
}

}  // namespace gpu_model
