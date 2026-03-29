#pragma once

#include <vector>

#include "gpu_model/exec/execution_state.h"
#include "gpu_model/runtime/launch_config.h"

namespace gpu_model {

std::vector<ExecutionBlockState> BuildExecutionBlockStates(const PlacementMap& placement,
                                                           const LaunchConfig& launch_config);

}  // namespace gpu_model
