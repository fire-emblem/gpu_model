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

}  // namespace
}  // namespace gpu_model
