#include <gtest/gtest.h>

#include "execution/internal/sync_ops/sync_ops.h"
#include "state/wave/barrier_wait_ops.h"

namespace gpu_model {
namespace {

TEST(SyncOpsTest, MarksWaveAtBarrier) {
  WaveContext wave;
  wave.thread_count = 64;
  wave.ResetInitialExec();
  wave.run_state = WaveRunState::Runnable;
  wave.wait_reason = WaveWaitReason::None;
  uint32_t arrivals = 0;

  MarkWaveAtBarrier(wave, 9, arrivals, true);

  EXPECT_EQ(wave.status, WaveStatus::Stalled);
  EXPECT_TRUE(wave.waiting_at_barrier);
  EXPECT_EQ(wave.barrier_generation, 9u);
  EXPECT_EQ(wave.run_state, WaveRunState::Waiting);
  EXPECT_EQ(wave.wait_reason, WaveWaitReason::BlockBarrier);
  EXPECT_FALSE(wave.valid_entry);
  EXPECT_EQ(arrivals, 1u);
}

TEST(SyncOpsTest, ReleasesBarrierOnlyWhenAllActiveWavesWait) {
  std::vector<WaveContext> waves(2);
  for (uint32_t i = 0; i < waves.size(); ++i) {
    waves[i].thread_count = 64;
    waves[i].ResetInitialExec();
    waves[i].pc = i;
  }

  uint64_t generation = 3;
  uint32_t arrivals = 0;
  MarkWaveAtBarrier(waves[0], generation, arrivals, false);
  waves[0].run_state = WaveRunState::Waiting;
  waves[0].wait_reason = WaveWaitReason::BlockBarrier;
  EXPECT_FALSE(sync_ops::ReleaseBarrierIfReady(
      waves, generation, arrivals, 1, true));

  MarkWaveAtBarrier(waves[1], generation, arrivals, false);
  waves[1].run_state = WaveRunState::Waiting;
  waves[1].wait_reason = WaveWaitReason::BlockBarrier;
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

TEST(SyncOpsTest, ReleasesBarrierThroughWavePointers) {
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
  MarkWaveAtBarrier(owned_waves[0], generation, arrivals, true);
  MarkWaveAtBarrier(owned_waves[1], generation, arrivals, true);
  owned_waves[0].run_state = WaveRunState::Waiting;
  owned_waves[0].wait_reason = WaveWaitReason::BlockBarrier;
  owned_waves[1].run_state = WaveRunState::Waiting;
  owned_waves[1].wait_reason = WaveWaitReason::BlockBarrier;

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

TEST(SyncOpsTest, ReleasesBarrierOnlyWhenAllBlockWavesWait) {
  WaveContext a;
  WaveContext b;
  a.status = WaveStatus::Stalled;
  a.run_state = WaveRunState::Waiting;
  a.wait_reason = WaveWaitReason::BlockBarrier;
  a.waiting_at_barrier = true;
  a.barrier_generation = 0;
  b.status = WaveStatus::Stalled;
  b.run_state = WaveRunState::Runnable;
  b.wait_reason = WaveWaitReason::None;
  b.waiting_at_barrier = true;
  b.barrier_generation = 0;

  std::vector<WaveContext*> waves{&a, &b};
  uint64_t target_generation = 0;
  uint32_t barrier_arrivals = 2;
  const bool released =
      sync_ops::ReleaseBarrierIfReady(waves, target_generation, barrier_arrivals, 1, false);

  EXPECT_FALSE(released);
  EXPECT_EQ(target_generation, 0u);
  EXPECT_EQ(barrier_arrivals, 2u);
  EXPECT_EQ(a.run_state, WaveRunState::Waiting);
  EXPECT_EQ(a.wait_reason, WaveWaitReason::BlockBarrier);
  EXPECT_EQ(b.run_state, WaveRunState::Runnable);
  EXPECT_EQ(b.wait_reason, WaveWaitReason::None);
}

TEST(SyncOpsTest, BarrierReleaseResumesRunStateAndClearsBarrierWaitReason) {
  std::vector<WaveContext> waves(2);
  for (auto& wave : waves) {
    wave.thread_count = 64;
    wave.ResetInitialExec();
  }

  uint64_t generation = 2;
  uint32_t arrivals = 0;
  MarkWaveAtBarrier(waves[0], generation, arrivals, false);
  MarkWaveAtBarrier(waves[1], generation, arrivals, false);
  waves[0].run_state = WaveRunState::Waiting;
  waves[1].run_state = WaveRunState::Waiting;
  waves[0].wait_reason = WaveWaitReason::BlockBarrier;
  waves[1].wait_reason = WaveWaitReason::BlockBarrier;

  ASSERT_TRUE(sync_ops::ReleaseBarrierIfReady(waves, generation, arrivals, 1, false));
  EXPECT_EQ(waves[0].run_state, WaveRunState::Runnable);
  EXPECT_EQ(waves[1].run_state, WaveRunState::Runnable);
  EXPECT_EQ(waves[0].wait_reason, WaveWaitReason::None);
  EXPECT_EQ(waves[1].wait_reason, WaveWaitReason::None);
}

}  // namespace
}  // namespace gpu_model
