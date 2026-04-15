#include <gtest/gtest.h>

#include "gpu_arch/chip_config/arch_registry.h"
#include "runtime/model_runtime/core/mapper.h"

namespace gpu_model {
namespace {

TEST(MapperTest, MapsBlocksToApsAndWavesToPeusForMac500) {
  const auto spec = ArchRegistry::Get("mac500");
  ASSERT_NE(spec, nullptr);

  LaunchConfig config{.grid_dim_x = 2, .block_dim_x = 128};
  const auto placement = Mapper::Place(*spec, config);

  ASSERT_EQ(placement.blocks.size(), 2u);
  EXPECT_EQ(placement.blocks[0].dpc_id, 0u);
  EXPECT_EQ(placement.blocks[0].ap_id, 0u);
  EXPECT_EQ(placement.blocks[0].waves.size(), 2u);
  EXPECT_EQ(placement.blocks[0].waves[0].peu_id, 0u);
  EXPECT_EQ(placement.blocks[0].waves[1].peu_id, 1u);
}

TEST(MapperTest, TailWaveUsesPartialLaneCount) {
  const auto spec = ArchRegistry::Get("mac500");
  ASSERT_NE(spec, nullptr);

  LaunchConfig config{.grid_dim_x = 1, .block_dim_x = 96};
  const auto placement = Mapper::Place(*spec, config);

  ASSERT_EQ(placement.blocks.size(), 1u);
  ASSERT_EQ(placement.blocks[0].waves.size(), 2u);
  EXPECT_EQ(placement.blocks[0].waves[1].lane_count, 32u);
}

TEST(MapperTest, Supports2DGridAndBlockCoordinates) {
  const auto spec = ArchRegistry::Get("mac500");
  ASSERT_NE(spec, nullptr);

  LaunchConfig config{
      .grid_dim_x = 3,
      .grid_dim_y = 2,
      .block_dim_x = 8,
      .block_dim_y = 4,
  };
  const auto placement = Mapper::Place(*spec, config);

  ASSERT_EQ(placement.blocks.size(), 6u);
  EXPECT_EQ(placement.blocks[0].block_idx_x, 0u);
  EXPECT_EQ(placement.blocks[0].block_idx_y, 0u);
  EXPECT_EQ(placement.blocks[0].block_idx_z, 0u);
  EXPECT_EQ(placement.blocks[1].block_idx_x, 1u);
  EXPECT_EQ(placement.blocks[1].block_idx_y, 0u);
  EXPECT_EQ(placement.blocks[3].block_idx_x, 0u);
  EXPECT_EQ(placement.blocks[3].block_idx_y, 1u);
  ASSERT_EQ(placement.blocks[0].waves.size(), 1u);
  EXPECT_EQ(placement.blocks[0].waves[0].lane_count, 32u);
}

TEST(MapperTest, Supports3DGridAndBlockCoordinates) {
  const auto spec = ArchRegistry::Get("mac500");
  ASSERT_NE(spec, nullptr);

  LaunchConfig config{
      .grid_dim_x = 2,
      .grid_dim_y = 2,
      .grid_dim_z = 2,
      .block_dim_x = 4,
      .block_dim_y = 2,
      .block_dim_z = 2,
  };
  const auto placement = Mapper::Place(*spec, config);

  ASSERT_EQ(placement.blocks.size(), 8u);
  EXPECT_EQ(placement.blocks[0].block_idx_x, 0u);
  EXPECT_EQ(placement.blocks[0].block_idx_y, 0u);
  EXPECT_EQ(placement.blocks[0].block_idx_z, 0u);
  EXPECT_EQ(placement.blocks[3].block_idx_x, 1u);
  EXPECT_EQ(placement.blocks[3].block_idx_y, 1u);
  EXPECT_EQ(placement.blocks[3].block_idx_z, 0u);
  EXPECT_EQ(placement.blocks[4].block_idx_x, 0u);
  EXPECT_EQ(placement.blocks[4].block_idx_y, 0u);
  EXPECT_EQ(placement.blocks[4].block_idx_z, 1u);
  ASSERT_EQ(placement.blocks[0].waves.size(), 1u);
  EXPECT_EQ(placement.blocks[0].waves[0].lane_count, 16u);
}

}  // namespace
}  // namespace gpu_model
