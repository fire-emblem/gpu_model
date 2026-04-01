#include <gtest/gtest.h>

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

ExecutableKernel BuildCycleResidentAsyncLoadKernel(uint64_t base_addr) {
  InstructionBuilder builder;
  builder.SMov("s0", base_addr);
  builder.SMov("s1", 0);
  builder.MLoadGlobal("v0", "s0", "s1", 4);
  builder.BExit();
  return builder.Build("cycle_resident_async_load_kernel");
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

uint32_t CountWaveLaunchesForBlockAtCycle(const std::vector<TraceEvent>& events,
                                          uint32_t block_id,
                                          uint64_t cycle) {
  uint32_t count = 0;
  for (const auto& event : events) {
    if (event.kind == TraceEventKind::WaveLaunch && event.block_id == block_id &&
        event.cycle == cycle) {
      ++count;
    }
  }
  return count;
}

size_t FirstWaveLaunchIndexForBlock(const std::vector<TraceEvent>& events, uint32_t block_id) {
  for (size_t i = 0; i < events.size(); ++i) {
    if (events[i].kind == TraceEventKind::WaveLaunch && events[i].block_id == block_id) {
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

TEST(CycleApResidentBlocksTest, SingleApAdmitsTwoResidentBlocksBeforeBackfillingThird) {
  CollectingTraceSink trace;
  RuntimeEngine runtime(&trace);
  runtime.SetLaunchTimingProfile(/*kernel_launch_gap_cycles=*/8,
                                 /*kernel_launch_cycles=*/0,
                                 /*block_launch_cycles=*/0,
                                 /*wave_launch_cycles=*/0,
                                 /*warp_switch_cycles=*/1,
                                 /*arg_load_cycles=*/4);

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
  runtime.SetLaunchTimingProfile(/*kernel_launch_gap_cycles=*/8,
                                 /*kernel_launch_cycles=*/0,
                                 /*block_launch_cycles=*/0,
                                 /*wave_launch_cycles=*/0,
                                 /*warp_switch_cycles=*/1,
                                 /*arg_load_cycles=*/4);

  const auto spec = ArchRegistry::Get("c500");
  ASSERT_NE(spec, nullptr);
  const auto kernel = BuildCycleResidentExitKernel();

  const auto request = BuildCycleResidentLaunchRequest(kernel, *spec);
  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  const uint32_t block0 = WrappedBlockId(*spec, 0);
  const uint32_t block1 = WrappedBlockId(*spec, 1);
  const uint32_t block2 = WrappedBlockId(*spec, 2);

  const size_t block0_exit = FirstWaveExitIndex(trace.events(), block0);
  const size_t block1_exit = FirstWaveExitIndex(trace.events(), block1);
  const size_t block2_launch = FirstBlockLaunchIndex(trace.events(), block2);
  ASSERT_NE(block0_exit, std::numeric_limits<size_t>::max());
  ASSERT_NE(block1_exit, std::numeric_limits<size_t>::max());
  ASSERT_NE(block2_launch, std::numeric_limits<size_t>::max());

  EXPECT_LT(block0_exit, block2_launch);
  EXPECT_LT(block2_launch, block1_exit);
}

TEST(CycleApResidentBlocksTest, ResidentStandbyBlockDoesNotLaunchWavesUntilActiveSlotOpens) {
  CollectingTraceSink trace;
  RuntimeEngine runtime(&trace);
  runtime.SetFixedGlobalMemoryLatency(40);
  runtime.SetLaunchTimingProfile(/*kernel_launch_gap_cycles=*/8,
                                 /*kernel_launch_cycles=*/0,
                                 /*block_launch_cycles=*/0,
                                 /*wave_launch_cycles=*/0,
                                 /*warp_switch_cycles=*/1,
                                 /*arg_load_cycles=*/4);

  const auto spec = ArchRegistry::Get("c500");
  ASSERT_NE(spec, nullptr);
  const uint64_t base_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(base_addr, 7);
  const auto kernel = BuildCycleResidentAsyncLoadKernel(base_addr);

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = spec->total_ap_count() + 1;
  request.config.block_dim_x = 1024;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  const uint32_t block0 = WrappedBlockId(*spec, 0);
  const uint32_t block1 = WrappedBlockId(*spec, 1);
  const uint32_t active_window_waves_per_ap = spec->peu_per_ap * spec->max_issuable_waves;
  const size_t block0_launch = FirstBlockLaunchIndex(trace.events(), block0);
  const size_t block1_launch = FirstBlockLaunchIndex(trace.events(), block1);
  ASSERT_NE(block0_launch, std::numeric_limits<size_t>::max());
  ASSERT_NE(block1_launch, std::numeric_limits<size_t>::max());

  EXPECT_EQ(trace.events()[block0_launch].cycle, 0u);
  EXPECT_EQ(trace.events()[block1_launch].cycle, 0u);
  EXPECT_EQ(CountWaveLaunchesForBlockAtCycle(trace.events(), block0, 0u), active_window_waves_per_ap);
  EXPECT_EQ(CountWaveLaunchesForBlockAtCycle(trace.events(), block1, 0u), 0u);
}

TEST(CycleApResidentBlocksTest, StandbyWavePromotesAfterActiveWaveExits) {
  CollectingTraceSink trace;
  RuntimeEngine runtime(&trace);
  runtime.SetFixedGlobalMemoryLatency(40);
  runtime.SetLaunchTimingProfile(/*kernel_launch_gap_cycles=*/8,
                                 /*kernel_launch_cycles=*/0,
                                 /*block_launch_cycles=*/0,
                                 /*wave_launch_cycles=*/0,
                                 /*warp_switch_cycles=*/1,
                                 /*arg_load_cycles=*/4);

  const auto spec = ArchRegistry::Get("c500");
  ASSERT_NE(spec, nullptr);
  const uint64_t base_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(base_addr, 7);
  const auto kernel = BuildCycleResidentAsyncLoadKernel(base_addr);

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = spec->total_ap_count() + 1;
  request.config.block_dim_x = 1024;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  const uint32_t block0 = WrappedBlockId(*spec, 0);
  const uint32_t block1 = WrappedBlockId(*spec, 1);
  const size_t block0_block_launch = FirstBlockLaunchIndex(trace.events(), block0);
  const size_t block1_block_launch = FirstBlockLaunchIndex(trace.events(), block1);
  const size_t block0_exit = FirstWaveExitIndex(trace.events(), block0);
  const size_t block1_launch = FirstWaveLaunchIndexForBlock(trace.events(), block1);
  ASSERT_NE(block0_block_launch, std::numeric_limits<size_t>::max());
  ASSERT_NE(block1_block_launch, std::numeric_limits<size_t>::max());
  ASSERT_NE(block0_exit, std::numeric_limits<size_t>::max());
  ASSERT_NE(block1_launch, std::numeric_limits<size_t>::max());

  EXPECT_EQ(trace.events()[block0_block_launch].cycle, 0u);
  EXPECT_EQ(trace.events()[block1_block_launch].cycle, 0u);
  EXPECT_EQ(CountWaveLaunchesForBlockAtCycle(trace.events(), block1, 0u), 0u);
  // Event-index ordering is the oracle here because promotion can happen in the same cycle as an
  // exit, but the promoting launch must still be emitted after the exit event in the trace.
  EXPECT_LT(block0_exit, block1_launch);
}

}  // namespace
}  // namespace gpu_model
