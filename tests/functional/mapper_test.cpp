#include <gtest/gtest.h>

#include "gpu_model/arch/arch_registry.h"
#include "gpu_model/runtime/mapper.h"

namespace gpu_model {
namespace {

TEST(MapperTest, MapsBlocksToApsAndWavesToPeusForC500) {
  const auto spec = ArchRegistry::Get("c500");
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
  const auto spec = ArchRegistry::Get("c500");
  ASSERT_NE(spec, nullptr);

  LaunchConfig config{.grid_dim_x = 1, .block_dim_x = 96};
  const auto placement = Mapper::Place(*spec, config);

  ASSERT_EQ(placement.blocks.size(), 1u);
  ASSERT_EQ(placement.blocks[0].waves.size(), 2u);
  EXPECT_EQ(placement.blocks[0].waves[1].lane_count, 32u);
}

}  // namespace
}  // namespace gpu_model
