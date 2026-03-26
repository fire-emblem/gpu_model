#pragma once

#include <cstdint>

namespace gpu_model {

struct LaunchConfig {
  uint32_t grid_dim_x = 1;
  uint32_t block_dim_x = 1;
};

}  // namespace gpu_model
