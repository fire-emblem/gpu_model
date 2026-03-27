#include <gtest/gtest.h>

#include <algorithm>

#include "gpu_model/debug/trace_sink.h"
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

TEST(CycleSmokeTest, ConsecutiveKernelLaunchesIncludeDeviceGap) {
  InstructionBuilder builder;
  builder.BExit();
  const auto kernel = builder.Build("launch_gap_kernel");

  HostRuntime runtime;
  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;

  const auto first = runtime.Launch(request);
  ASSERT_TRUE(first.ok) << first.error_message;
  EXPECT_EQ(first.submit_cycle, 0u);
  EXPECT_EQ(first.begin_cycle, 0u);
  EXPECT_EQ(first.end_cycle, 4u);
  EXPECT_EQ(first.total_cycles, 4u);

  const auto second = runtime.Launch(request);
  ASSERT_TRUE(second.ok) << second.error_message;
  EXPECT_EQ(second.submit_cycle, first.end_cycle + 8u);
  EXPECT_EQ(second.begin_cycle, second.submit_cycle);
  EXPECT_EQ(second.end_cycle, second.begin_cycle + 4u);
  EXPECT_EQ(second.total_cycles, 4u);
  EXPECT_EQ(runtime.device_cycle(), second.end_cycle);
}

TEST(CycleSmokeTest, QueuesBlocksWhenGridExceedsPhysicalApCount) {
  CollectingTraceSink trace;
  HostRuntime runtime(&trace);

  InstructionBuilder builder;
  builder.BExit();
  const auto kernel = builder.Build("queued_blocks_kernel");

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 57;
  request.config.block_dim_x = 64;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  EXPECT_EQ(result.total_cycles, 9u);

  uint32_t block_launches = 0;
  uint32_t wave_launches = 0;
  uint64_t wrapped_block_launch_cycle = 0;
  bool saw_warp_switch = false;
  for (const auto& event : trace.events()) {
    if (event.kind == TraceEventKind::BlockLaunch) {
      ++block_launches;
      if (event.block_id == 56u) {
        wrapped_block_launch_cycle = event.cycle;
      }
    } else if (event.kind == TraceEventKind::WaveLaunch) {
      ++wave_launches;
    } else if (event.kind == TraceEventKind::Stall && event.message == "warp_switch") {
      saw_warp_switch = true;
    }
  }

  EXPECT_EQ(block_launches, 57u);
  EXPECT_EQ(wave_launches, 57u);
  EXPECT_EQ(wrapped_block_launch_cycle, 4u);
  EXPECT_TRUE(saw_warp_switch);
}

}  // namespace
}  // namespace gpu_model
