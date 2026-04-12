#include <gtest/gtest.h>

#include "state/ap/ap_runtime_state.h"

namespace gpu_model {
namespace {

TEST(ApRuntimeStateTest, DeclaresApAndExecutionBlockRuntimeContainers) {
  ApState ap;
  ExecutionBlockState block;

  EXPECT_FALSE(ap.barrier.armed);
  EXPECT_TRUE(ap.peus.empty());
  EXPECT_TRUE(ap.shared_memory.empty());

  EXPECT_EQ(block.barrier_generation, 0u);
  EXPECT_EQ(block.barrier_arrivals, 0u);
  EXPECT_TRUE(block.shared_memory.empty());
  EXPECT_TRUE(block.waves.empty());
}

TEST(ApRuntimeStateTest, ExecutionBlockStateTracksWaveStorageSeparatelyFromApState) {
  ExecutionBlockState block;
  block.shared_memory.resize(16);
  block.waves.resize(2);

  EXPECT_EQ(block.shared_memory.size(), 16u);
  EXPECT_EQ(block.waves.size(), 2u);
}

}  // namespace
}  // namespace gpu_model
