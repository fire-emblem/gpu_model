#pragma once

#include <optional>

#include "runtime/config/launch_config.h"

namespace gpu_model {

class RuntimeLaunchConfigState {
 public:
  void Reset();
  void Push(LaunchConfig config);
  std::optional<LaunchConfig> Pop();

 private:
  std::optional<LaunchConfig> pending_launch_config_;
};

}  // namespace gpu_model
