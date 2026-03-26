#include <gtest/gtest.h>

#include "gpu_model/isa/instruction_builder.h"
#include "gpu_model/runtime/host_runtime.h"

namespace gpu_model {
namespace {

TEST(CycleSmokeTest, ScalarAndVectorOpsConsumeFourCyclesEach) {
  InstructionBuilder builder;
  builder.SMov("s0", 7);
  builder.VMov("v0", "s0");
  builder.BExit();
  const auto kernel = builder.Build("tiny_cycle_kernel");

  HostRuntime runtime;
  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  EXPECT_EQ(result.total_cycles, 12u);
}

}  // namespace
}  // namespace gpu_model
