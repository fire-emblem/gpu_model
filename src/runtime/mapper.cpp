#include "gpu_model/runtime/mapper.h"

#include <algorithm>
#include <stdexcept>

#include "gpu_model/state/wave/wave_runtime_state.h"

namespace gpu_model {

PlacementMap Mapper::Place(const GpuArchSpec& spec, const LaunchConfig& config) {
  if (config.grid_dim_x == 0 || config.grid_dim_y == 0 || config.grid_dim_z == 0 ||
      config.block_dim_x == 0 || config.block_dim_y == 0 || config.block_dim_z == 0) {
    throw std::invalid_argument("grid and block dimensions must be non-zero");
  }
  if (spec.wave_size != kWaveSize) {
    throw std::invalid_argument("wave size mismatch with runtime wave state");
  }

  PlacementMap placement;
  const uint32_t total_blocks = config.grid_dim_x * config.grid_dim_y * config.grid_dim_z;
  placement.blocks.reserve(total_blocks);

  const uint32_t total_aps = spec.total_ap_count();
  const uint32_t threads_per_block = config.block_dim_x * config.block_dim_y * config.block_dim_z;
  const uint32_t waves_per_block = (threads_per_block + spec.wave_size - 1) / spec.wave_size;

  for (uint32_t block_id = 0; block_id < total_blocks; ++block_id) {
    const uint32_t xy = config.grid_dim_x * config.grid_dim_y;
    const uint32_t block_idx_x = block_id % config.grid_dim_x;
    const uint32_t block_idx_y = (block_id / config.grid_dim_x) % config.grid_dim_y;
    const uint32_t block_idx_z = block_id / xy;
    const uint32_t global_ap_id = block_id % total_aps;
    const uint32_t dpc_id = global_ap_id / spec.ap_per_dpc;
    const uint32_t ap_id = global_ap_id % spec.ap_per_dpc;

    BlockPlacement block{
        .block_id = block_id,
        .block_idx_x = block_idx_x,
        .block_idx_y = block_idx_y,
        .block_idx_z = block_idx_z,
        .dpc_id = dpc_id,
        .ap_id = ap_id,
        .global_ap_id = global_ap_id,
        .waves = {},
    };

    block.waves.reserve(waves_per_block);
    for (uint32_t wave_id = 0; wave_id < waves_per_block; ++wave_id) {
      const uint32_t wave_base_lane = wave_id * spec.wave_size;
      const uint32_t remaining = threads_per_block > wave_base_lane
                                     ? threads_per_block - wave_base_lane
                                     : 0;
      const uint32_t lane_count = std::min(spec.wave_size, remaining);
      block.waves.push_back(WavePlacement{
          .wave_id = wave_id,
          .peu_id = wave_id % spec.peu_per_ap,
          .lane_count = lane_count,
      });
    }

    placement.blocks.push_back(std::move(block));
  }

  return placement;
}

}  // namespace gpu_model
