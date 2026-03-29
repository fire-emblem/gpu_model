#include "gpu_model/exec/execution_state_builder.h"

namespace gpu_model {

WaveState BuildInitialWaveState(const BlockPlacement& block_placement,
                                const WavePlacement& wave_placement) {
  WaveState wave;
  wave.block_id = block_placement.block_id;
  wave.block_idx_x = block_placement.block_idx_x;
  wave.block_idx_y = block_placement.block_idx_y;
  wave.block_idx_z = block_placement.block_idx_z;
  wave.dpc_id = block_placement.dpc_id;
  wave.wave_id = wave_placement.wave_id;
  wave.peu_id = wave_placement.peu_id;
  wave.ap_id = block_placement.ap_id;
  wave.thread_count = wave_placement.lane_count;
  wave.ResetInitialExec();
  return wave;
}

std::vector<ExecutionBlockState> BuildExecutionBlockStates(const PlacementMap& placement,
                                                           const LaunchConfig& launch_config) {
  std::vector<ExecutionBlockState> blocks;
  blocks.reserve(placement.blocks.size());

  for (const auto& block_placement : placement.blocks) {
    ExecutionBlockState block{
        .block_id = block_placement.block_id,
        .dpc_id = block_placement.dpc_id,
        .ap_id = block_placement.ap_id,
        .global_ap_id = block_placement.global_ap_id,
        .barrier_generation = 0,
        .barrier_arrivals = 0,
        .shared_memory = std::vector<std::byte>(launch_config.shared_memory_bytes),
        .waves = {},
    };
    block.waves.reserve(block_placement.waves.size());
    for (const auto& wave_placement : block_placement.waves) {
      block.waves.push_back(BuildInitialWaveState(block_placement, wave_placement));
    }
    blocks.push_back(std::move(block));
  }

  return blocks;
}

}  // namespace gpu_model
