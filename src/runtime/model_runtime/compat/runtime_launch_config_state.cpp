#include "runtime/model_runtime/compat/runtime_launch_config_state.h"

namespace gpu_model {

void RuntimeLaunchConfigState::Reset() {
  pending_launch_config_.reset();
}

void RuntimeLaunchConfigState::Push(LaunchConfig config) {
  pending_launch_config_ = config;
}

std::optional<LaunchConfig> RuntimeLaunchConfigState::Pop() {
  auto config = pending_launch_config_;
  pending_launch_config_.reset();
  return config;
}

}  // namespace gpu_model
