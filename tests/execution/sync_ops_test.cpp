#include <gtest/gtest.h>

#include "gpu_model/execution/sync_ops.h"

namespace gpu_model {
namespace {

TEST(ExecutionSyncOpsTest, MarksWaveAtBarrier) {
  WaveContext wave;
  wave.thread_count = 64;
  wave.ResetInitialExec();
  uint32_t arrivals = 0;

  sync_ops::MarkWaveAtBarrier(wave, 9, arrivals, true);

  EXPECT_EQ(wave.status, WaveStatus::Stalled);
  EXPECT_TRUE(wave.waiting_at_barrier);
  EXPECT_EQ(wave.barrier_generation, 9u);
  EXPECT_FALSE(wave.valid_entry);
  EXPECT_EQ(arrivals, 1u);
}

TEST(ExecutionSyncOpsTest, ReleasesBarrierOnlyWhenAllActiveWavesWait) {
  std::vector<WaveContext> waves(2);
  for (uint32_t i = 0; i < waves.size(); ++i) {
    waves[i].thread_count = 64;
    waves[i].ResetInitialExec();
    waves[i].pc = i;
  }

  uint64_t generation = 3;
  uint32_t arrivals = 0;
  sync_ops::MarkWaveAtBarrier(waves[0], generation, arrivals, false);
  EXPECT_FALSE(sync_ops::ReleaseBarrierIfReady(
      waves, generation, arrivals, 1, true));

  sync_ops::MarkWaveAtBarrier(waves[1], generation, arrivals, false);
  EXPECT_TRUE(sync_ops::ReleaseBarrierIfReady(
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

TEST(ExecutionSyncOpsTest, ReleasesBarrierThroughWavePointers) {
  std::vector<WaveContext> owned_waves(2);
  for (uint32_t i = 0; i < owned_waves.size(); ++i) {
    owned_waves[i].thread_count = 64;
    owned_waves[i].ResetInitialExec();
    owned_waves[i].pc = 10 + i;
  }

  std::vector<WaveContext*> wave_ptrs;
  for (auto& wave : owned_waves) {
    wave_ptrs.push_back(&wave);
  }

  uint64_t generation = 7;
  uint32_t arrivals = 0;
  sync_ops::MarkWaveAtBarrier(owned_waves[0], generation, arrivals, true);
  sync_ops::MarkWaveAtBarrier(owned_waves[1], generation, arrivals, true);

  EXPECT_TRUE(sync_ops::ReleaseBarrierIfReady(
      wave_ptrs, generation, arrivals, 4, false));
  EXPECT_EQ(arrivals, 0u);
  EXPECT_EQ(generation, 8u);
  EXPECT_FALSE(owned_waves[0].waiting_at_barrier);
  EXPECT_FALSE(owned_waves[1].waiting_at_barrier);
  EXPECT_EQ(owned_waves[0].status, WaveStatus::Active);
  EXPECT_EQ(owned_waves[1].status, WaveStatus::Active);
  EXPECT_FALSE(owned_waves[0].valid_entry);
  EXPECT_FALSE(owned_waves[1].valid_entry);
  EXPECT_EQ(owned_waves[0].pc, 14u);
  EXPECT_EQ(owned_waves[1].pc, 15u);
}

}  // namespace
}  // namespace gpu_model
