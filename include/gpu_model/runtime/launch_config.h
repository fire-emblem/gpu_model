#pragma once

#include <cstdint>

namespace gpu_model {

struct LaunchConfig {
  uint32_t grid_dim_x = 1;
  uint32_t grid_dim_y = 1;
  uint32_t block_dim_x = 1;
  uint32_t block_dim_y = 1;
  uint32_t shared_memory_bytes = 0;
};

}  // namespace gpu_model
