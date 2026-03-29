#include <gtest/gtest.h>

#include "gpu_model/exec/execution_sync_ops.h"

namespace gpu_model {
namespace {

TEST(ExecutionSyncOpsTest, MarksWaveAtBarrier) {
  WaveState wave;
  wave.thread_count = 64;
  wave.ResetInitialExec();
  uint32_t arrivals = 0;

  execution_sync_ops::MarkWaveAtBarrier(wave, 9, arrivals, true);

  EXPECT_EQ(wave.status, WaveStatus::Stalled);
  EXPECT_TRUE(wave.waiting_at_barrier);
  EXPECT_EQ(wave.barrier_generation, 9u);
  EXPECT_FALSE(wave.valid_entry);
  EXPECT_EQ(arrivals, 1u);
}

TEST(ExecutionSyncOpsTest, ReleasesBarrierOnlyWhenAllActiveWavesWait) {
  std::vector<WaveState> waves(2);
  for (uint32_t i = 0; i < waves.size(); ++i) {
    waves[i].thread_count = 64;
    waves[i].ResetInitialExec();
    waves[i].pc = i;
  }

  uint64_t generation = 3;
  uint32_t arrivals = 0;
  execution_sync_ops::MarkWaveAtBarrier(waves[0], generation, arrivals, false);
  EXPECT_FALSE(execution_sync_ops::ReleaseBarrierIfReady(
      waves, generation, arrivals, 1, true));

  execution_sync_ops::MarkWaveAtBarrier(waves[1], generation, arrivals, false);
  EXPECT_TRUE(execution_sync_ops::ReleaseBarrierIfReady(
      waves, generation, arrivals, 1, true));
  EXPECT_EQ(arrivals, 0u);
  EXPECT_EQ(generation, 4u);
  EXPECT_FALSE(waves[0].waiting_at_barrier);
  EXPECT_FALSE(waves[1].waiting_at_barrier);
  EXPECT_EQ(waves[0].status, WaveStatus::Active);
  EXPECT_EQ(waves[1].status, WaveStatus::Active);
  EXPECT_TRUE(waves[0].valid_entry);
  EXPECT_TRUE(waves[1].valid_entry);
  EXPECT_EQ(waves[0].pc, 1u);
  EXPECT_EQ(waves[1].pc, 2u);
}

}  // namespace
}  // namespace gpu_model
