#include <gtest/gtest.h>

#include "gpu_model/execution/wave_context.h"

namespace gpu_model {
namespace {

TEST(WaveContextTest, InitializesExecAndPredicateMasks) {
  WaveContext wave;
  wave.thread_count = 10;
  wave.ResetInitialExec();

  EXPECT_EQ(wave.exec.count(), 10u);
  EXPECT_EQ(wave.cmask.count(), 0u);
  EXPECT_EQ(wave.smask, 0u);
}

TEST(WaveContextTest, InitializesRunStateAsRunnable) {
  WaveContext wave;
  EXPECT_EQ(wave.run_state, WaveRunState::Runnable);
  EXPECT_EQ(wave.wait_reason, WaveWaitReason::None);
  wave.thread_count = 8;
  wave.ResetInitialExec();

  EXPECT_EQ(wave.run_state, WaveRunState::Runnable);
  EXPECT_EQ(wave.wait_reason, WaveWaitReason::None);
}

TEST(WaveContextTest, ClearsBarrierWaitStateOnReset) {
  WaveContext wave;
  wave.waiting_at_barrier = true;
  wave.run_state = WaveRunState::Waiting;
  wave.wait_reason = WaveWaitReason::BlockBarrier;
  wave.thread_count = 4;

  wave.ResetInitialExec();

  EXPECT_FALSE(wave.waiting_at_barrier);
  EXPECT_EQ(wave.run_state, WaveRunState::Runnable);
  EXPECT_EQ(wave.wait_reason, WaveWaitReason::None);
}

TEST(WaveContextTest, SupportsMemoryDomainWaitReasons) {
  EXPECT_NE(WaveWaitReason::PendingGlobalMemory, WaveWaitReason::PendingSharedMemory);
  EXPECT_NE(WaveWaitReason::PendingPrivateMemory, WaveWaitReason::PendingScalarBufferMemory);
}

TEST(WaveContextTest, ResetClearsMemoryWaitReasonBackToNone) {
  WaveContext wave;
  wave.run_state = WaveRunState::Waiting;
  wave.wait_reason = WaveWaitReason::PendingGlobalMemory;
  wave.thread_count = 8;

  wave.ResetInitialExec();

  EXPECT_EQ(wave.run_state, WaveRunState::Runnable);
  EXPECT_EQ(wave.wait_reason, WaveWaitReason::None);
}

}  // namespace
}  // namespace gpu_model
