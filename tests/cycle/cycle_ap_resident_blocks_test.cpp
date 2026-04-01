#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include "gpu_model/arch/arch_registry.h"
#include "gpu_model/debug/trace_sink.h"
#include "gpu_model/isa/instruction_builder.h"
#include "gpu_model/runtime/runtime_engine.h"

namespace gpu_model {
namespace {

uint32_t WrappedBlockId(const GpuArchSpec& spec, uint32_t ordinal) {
  return ordinal * spec.total_ap_count();
}

ExecutableKernel BuildCycleResidentExitKernel() {
  InstructionBuilder builder;
  builder.VMov("v0", 1);
  builder.BExit();
  return builder.Build("cycle_resident_exit_kernel");
}

size_t FirstBlockLaunchIndex(const std::vector<TraceEvent>& events, uint32_t block_id) {
  for (size_t i = 0; i < events.size(); ++i) {
    if (events[i].kind == TraceEventKind::BlockLaunch && events[i].block_id == block_id) {
      return i;
    }
  }
  return std::numeric_limits<size_t>::max();
}

size_t FirstWaveExitIndex(const std::vector<TraceEvent>& events, uint32_t block_id) {
  for (size_t i = 0; i < events.size(); ++i) {
    if (events[i].kind == TraceEventKind::WaveExit && events[i].block_id == block_id) {
      return i;
    }
  }
  return std::numeric_limits<size_t>::max();
}

LaunchRequest BuildCycleResidentLaunchRequest(const ExecutableKernel& kernel, const GpuArchSpec& spec) {
  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 2 * spec.total_ap_count() + 1;
  request.config.block_dim_x = 64;
  return request;
}

void ConfigureLaunchTiming(RuntimeEngine& runtime) {
  runtime.SetLaunchTimingProfile(/*kernel_launch_gap_cycles=*/8,
                                 /*kernel_launch_cycles=*/0,
                                 /*block_launch_cycles=*/0,
                                 /*wave_launch_cycles=*/0,
                                 /*warp_switch_cycles=*/1,
                                 /*arg_load_cycles=*/4);
}

TEST(CycleApResidentBlocksTest, SingleApAdmitsTwoResidentBlocksBeforeBackfillingThird) {
  CollectingTraceSink trace;
  RuntimeEngine runtime(&trace);
  ConfigureLaunchTiming(runtime);

  const auto spec = ArchRegistry::Get("c500");
  ASSERT_NE(spec, nullptr);
  const auto kernel = BuildCycleResidentExitKernel();

  const auto request = BuildCycleResidentLaunchRequest(kernel, *spec);
  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  const uint32_t block0 = WrappedBlockId(*spec, 0);
  const uint32_t block1 = WrappedBlockId(*spec, 1);
  const uint32_t block2 = WrappedBlockId(*spec, 2);

  const size_t block0_launch = FirstBlockLaunchIndex(trace.events(), block0);
  const size_t block1_launch = FirstBlockLaunchIndex(trace.events(), block1);
  const size_t block2_launch = FirstBlockLaunchIndex(trace.events(), block2);
  ASSERT_NE(block0_launch, std::numeric_limits<size_t>::max());
  ASSERT_NE(block1_launch, std::numeric_limits<size_t>::max());
  ASSERT_NE(block2_launch, std::numeric_limits<size_t>::max());

  EXPECT_EQ(trace.events()[block0_launch].cycle, 0u);
  EXPECT_EQ(trace.events()[block1_launch].cycle, 0u);
  EXPECT_GT(trace.events()[block2_launch].cycle, 0u);
}

TEST(CycleApResidentBlocksTest, RetiredBlockBackfillsPendingBlockOnSameAp) {
  CollectingTraceSink trace;
  RuntimeEngine runtime(&trace);
  ConfigureLaunchTiming(runtime);

  const auto spec = ArchRegistry::Get("c500");
  ASSERT_NE(spec, nullptr);
  const auto kernel = BuildCycleResidentExitKernel();

  const auto request = BuildCycleResidentLaunchRequest(kernel, *spec);
  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  const uint32_t block0 = WrappedBlockId(*spec, 0);
  const uint32_t block2 = WrappedBlockId(*spec, 2);

  const size_t block0_exit = FirstWaveExitIndex(trace.events(), block0);
  const size_t block2_launch = FirstBlockLaunchIndex(trace.events(), block2);
  ASSERT_NE(block0_exit, std::numeric_limits<size_t>::max());
  ASSERT_NE(block2_launch, std::numeric_limits<size_t>::max());

  EXPECT_LT(block0_exit, block2_launch);
}

}  // namespace
}  // namespace gpu_model
