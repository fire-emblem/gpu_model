#include <gtest/gtest.h>

#include "gpu_model/runtime/program_cycle_aggregator.h"

namespace gpu_model {
namespace {

TEST(ExecutedFlowProgramCycleEstimatorTest,
     ProgramCycleAggregatorAccumulatesWaveWorkByTick) {
  ProgramCycleAggregator agg(ProgramCycleEstimatorConfig{});
  agg.BeginWaveWork(/*wave_id=*/0, ExecutedStepClass::VectorAlu, /*cost_cycles=*/4);
  agg.BeginWaveWork(/*wave_id=*/1, ExecutedStepClass::VectorAlu, /*cost_cycles=*/4);

  for (int i = 0; i < 4; ++i) {
    agg.AdvanceOneTick();
  }

  const auto estimate = agg.Finish();
  EXPECT_EQ(estimate.total_cycles, 4u);
  EXPECT_EQ(estimate.total_issued_work_cycles, 8u);
  EXPECT_EQ(estimate.vector_alu_cycles, 8u);
}

}  // namespace
}  // namespace gpu_model
