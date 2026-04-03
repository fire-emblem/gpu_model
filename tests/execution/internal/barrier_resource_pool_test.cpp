#include <gtest/gtest.h>

#include "gpu_model/execution/internal/barrier_resource_pool.h"

namespace gpu_model {
namespace {

TEST(BarrierResourcePoolTest, FirstWaveOfBarrierGenerationAcquiresSingleSlot) {
  uint32_t slots_in_use = 0;
  bool acquired = false;

  EXPECT_TRUE(TryAcquireBarrierSlot(16, slots_in_use, acquired));
  EXPECT_TRUE(acquired);
  EXPECT_EQ(slots_in_use, 1u);

  EXPECT_TRUE(TryAcquireBarrierSlot(16, slots_in_use, acquired));
  EXPECT_EQ(slots_in_use, 1u);
}

TEST(BarrierResourcePoolTest, FullPoolRejectsNewBarrierGeneration) {
  uint32_t slots_in_use = 16;
  bool acquired = false;

  EXPECT_FALSE(TryAcquireBarrierSlot(16, slots_in_use, acquired));
  EXPECT_FALSE(acquired);
  EXPECT_EQ(slots_in_use, 16u);
}

TEST(BarrierResourcePoolTest, ReleaseReturnsSlotOnlyOncePerGeneration) {
  uint32_t slots_in_use = 1;
  bool acquired = true;

  ReleaseBarrierSlot(slots_in_use, acquired);
  EXPECT_FALSE(acquired);
  EXPECT_EQ(slots_in_use, 0u);

  ReleaseBarrierSlot(slots_in_use, acquired);
  EXPECT_EQ(slots_in_use, 0u);
}

}  // namespace
}  // namespace gpu_model
