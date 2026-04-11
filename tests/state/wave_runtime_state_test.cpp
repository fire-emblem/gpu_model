#include <array>
#include <gtest/gtest.h>

#include "gpu_model/state/wave/wave_runtime_state.h"

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

namespace {
constexpr std::array kMemoryDomainWaitReasons{
    WaveWaitReason::PendingGlobalMemory,
    WaveWaitReason::PendingSharedMemory,
    WaveWaitReason::PendingPrivateMemory,
    WaveWaitReason::PendingScalarBufferMemory,
};
}  // namespace

TEST(WaveContextTest, SupportsMemoryDomainWaitReasons) {
  for (size_t i = 0; i < kMemoryDomainWaitReasons.size(); ++i) {
    for (size_t j = i + 1; j < kMemoryDomainWaitReasons.size(); ++j) {
      EXPECT_NE(kMemoryDomainWaitReasons[i], kMemoryDomainWaitReasons[j]);
    }
  }

  WaveContext wave;
  for (auto wait_reason : kMemoryDomainWaitReasons) {
    wave.wait_reason = wait_reason;
    EXPECT_EQ(wave.wait_reason, wait_reason);
  }
}

TEST(WaveContextTest, ResetClearsMemoryWaitReasonBackToNone) {
  WaveContext wave;
  wave.thread_count = 8;

  for (auto wait_reason : kMemoryDomainWaitReasons) {
    wave.run_state = WaveRunState::Waiting;
    wave.wait_reason = wait_reason;

    wave.ResetInitialExec();

    EXPECT_EQ(wave.run_state, WaveRunState::Runnable);
    EXPECT_EQ(wave.wait_reason, WaveWaitReason::None);
  }
}

}  // namespace
}  // namespace gpu_model
