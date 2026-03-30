#pragma once

#include <cstdint>

namespace gpu_model {

struct LaunchConfig {
  uint32_t grid_dim_x = 1;
  uint32_t grid_dim_y = 1;
  uint32_t grid_dim_z = 1;
  uint32_t block_dim_x = 1;
  uint32_t block_dim_y = 1;
  uint32_t block_dim_z = 1;
  uint32_t shared_memory_bytes = 0;
  uint64_t global_offset_x = 0;
  uint64_t global_offset_y = 0;
  uint64_t global_offset_z = 0;
};

}  // namespace gpu_model
