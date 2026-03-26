#include <gtest/gtest.h>

#include "gpu_model/state/wave_state.h"

namespace gpu_model {
namespace {

TEST(WaveStateTest, InitializesExecAndPredicateMasks) {
  WaveState wave;
  wave.thread_count = 10;
  wave.ResetInitialExec();

  EXPECT_EQ(wave.exec.count(), 10u);
  EXPECT_EQ(wave.cmask.count(), 0u);
  EXPECT_EQ(wave.smask, 0u);
}

}  // namespace
}  // namespace gpu_model
