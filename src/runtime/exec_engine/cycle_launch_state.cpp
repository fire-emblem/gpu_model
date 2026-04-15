#include "runtime/exec_engine/cycle_launch_state.h"

namespace gpu_model {

void ResetCycleLaunchState(CycleLaunchState& state) {
  state.device_cycle = 0;
  state.has_cycle_launch_history = false;
}

void PrepareCycleLaunchResult(const LaunchRequest& request,
                              const GpuArchSpec& spec,
                              const CycleLaunchState& state,
                              LaunchResult& result) {
  if (request.mode != ExecutionMode::Cycle) {
    return;
  }
  result.submit_cycle =
      state.has_cycle_launch_history ? state.device_cycle + spec.launch_timing.kernel_launch_gap_cycles
                                     : 0;
  result.begin_cycle = result.submit_cycle + spec.launch_timing.kernel_launch_cycles;
}

void CommitCycleLaunchResult(const LaunchRequest& request,
                             const LaunchResult& result,
                             CycleLaunchState& state) {
  if (request.mode != ExecutionMode::Cycle) {
    return;
  }
  state.device_cycle = result.end_cycle;
  state.has_cycle_launch_history = true;
}

}  // namespace gpu_model
