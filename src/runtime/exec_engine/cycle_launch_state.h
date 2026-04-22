#pragma once

#include <cstdint>

#include "utils/config/launch_request.h"

namespace gpu_model {

struct CycleLaunchState {
  uint64_t device_cycle = 0;
  bool has_cycle_launch_history = false;
};

void ResetCycleLaunchState(CycleLaunchState& state);
void PrepareCycleLaunchResult(const LaunchRequest& request,
                              const GpuArchSpec& spec,
                              const CycleLaunchState& state,
                              LaunchResult& result);
void CommitCycleLaunchResult(const LaunchRequest& request,
                             const LaunchResult& result,
                             CycleLaunchState& state);

}  // namespace gpu_model
