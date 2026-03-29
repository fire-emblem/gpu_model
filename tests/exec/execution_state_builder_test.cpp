#include <gtest/gtest.h>

#include "gpu_model/exec/execution_state_builder.h"

namespace gpu_model {
namespace {

TEST(ExecutionStateBuilderTest, MaterializesBlockAndWaveStateFromPlacement) {
  PlacementMap placement;
  placement.blocks.push_back(BlockPlacement{
      .block_id = 7,
      .block_idx_x = 1,
      .block_idx_y = 2,
      .block_idx_z = 3,
      .dpc_id = 4,
      .ap_id = 5,
      .global_ap_id = 6,
      .waves =
          {
              WavePlacement{.wave_id = 0, .peu_id = 1, .lane_count = 64},
              WavePlacement{.wave_id = 1, .peu_id = 3, .lane_count = 32},
          },
  });

  const auto blocks = BuildExecutionBlockStates(
      placement, LaunchConfig{.grid_dim_x = 1, .block_dim_x = 64, .shared_memory_bytes = 96});
  ASSERT_EQ(blocks.size(), 1u);

  const auto& block = blocks.front();
  EXPECT_EQ(block.block_id, 7u);
  EXPECT_EQ(block.dpc_id, 4u);
  EXPECT_EQ(block.ap_id, 5u);
  EXPECT_EQ(block.global_ap_id, 6u);
  EXPECT_EQ(block.barrier_generation, 0u);
  EXPECT_EQ(block.barrier_arrivals, 0u);
  EXPECT_EQ(block.shared_memory.size(), 96u);

  ASSERT_EQ(block.waves.size(), 2u);
  EXPECT_EQ(block.waves[0].block_id, 7u);
  EXPECT_EQ(block.waves[0].block_idx_x, 1u);
  EXPECT_EQ(block.waves[0].block_idx_y, 2u);
  EXPECT_EQ(block.waves[0].block_idx_z, 3u);
  EXPECT_EQ(block.waves[0].dpc_id, 4u);
  EXPECT_EQ(block.waves[0].ap_id, 5u);
  EXPECT_EQ(block.waves[0].peu_id, 1u);
  EXPECT_EQ(block.waves[0].wave_id, 0u);
  EXPECT_EQ(block.waves[0].thread_count, 64u);
  EXPECT_TRUE(block.waves[0].valid_entry);
  EXPECT_TRUE(block.waves[0].exec.test(63));

  EXPECT_EQ(block.waves[1].peu_id, 3u);
  EXPECT_EQ(block.waves[1].wave_id, 1u);
  EXPECT_EQ(block.waves[1].thread_count, 32u);
  EXPECT_TRUE(block.waves[1].exec.test(31));
  EXPECT_FALSE(block.waves[1].exec.test(32));
}

}  // namespace
}  // namespace gpu_model
