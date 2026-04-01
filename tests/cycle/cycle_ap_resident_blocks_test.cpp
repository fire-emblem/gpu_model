#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <vector>

#include "gpu_model/arch/arch_registry.h"
#include "gpu_model/debug/trace_sink.h"
#include "gpu_model/isa/instruction_builder.h"
#include "gpu_model/runtime/runtime_engine.h"

namespace gpu_model {
namespace {

std::optional<uint64_t> BlockLaunchCycle(const std::vector<TraceEvent>& events,
                                          uint32_t block_id) {
  for (const auto& event : events) {
    if (event.kind == TraceEventKind::BlockLaunch && event.block_id == block_id) {
      return event.cycle;
    }
  }
  return std::nullopt;
}

std::optional<uint64_t> FirstWaveExitCycle(const std::vector<TraceEvent>& events,
                                            uint32_t block_id) {
  for (const auto& event : events) {
    if (event.kind == TraceEventKind::WaveExit && event.block_id == block_id) {
      return event.cycle;
    }
  }
  return std::nullopt;
}

LaunchRequest BuildSingleCycleLaunch(const ExecutableKernel& kernel, uint32_t grid_blocks) {
  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = grid_blocks;
  request.config.block_dim_x = 64;
  return request;
}

InstructionBuilder BuildResidentBlocksKernel() {
  InstructionBuilder builder;
  builder.BExit();
  return builder;
}

TEST(CycleApResidentBlocksTest, SingleApAdmitsTwoResidentBlocksBeforeBackfillingThird) {
  const auto spec = ArchRegistry::Get("c500");
  ASSERT_NE(spec, nullptr);

  CollectingTraceSink trace;
  RuntimeEngine runtime(&trace);

  InstructionBuilder builder = BuildResidentBlocksKernel();
  const auto kernel = builder.Build("resident_blocks_kernel");

  const uint32_t total_ap = spec->total_ap_count();
  const uint32_t grid_blocks = total_ap * 2 + 1;
  const auto request = BuildSingleCycleLaunch(kernel, grid_blocks);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  const auto second_block_cycle = BlockLaunchCycle(trace.events(), total_ap);
  ASSERT_TRUE(second_block_cycle.has_value()) << "missing wrapped block launch";
  EXPECT_EQ(*second_block_cycle, 0u);
}

TEST(CycleApResidentBlocksTest, RetiredBlockBackfillsPendingBlockOnSameAp) {
  const auto spec = ArchRegistry::Get("c500");
  ASSERT_NE(spec, nullptr);

  CollectingTraceSink trace;
  RuntimeEngine runtime(&trace);

  InstructionBuilder builder = BuildResidentBlocksKernel();
  const auto kernel = builder.Build("resident_blocks_backfill_kernel");

  const uint32_t total_ap = spec->total_ap_count();
  const uint32_t grid_blocks = total_ap * 2 + 1;
  const auto request = BuildSingleCycleLaunch(kernel, grid_blocks);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  const auto third_block_cycle = BlockLaunchCycle(trace.events(), total_ap * 2);
  ASSERT_TRUE(third_block_cycle.has_value()) << "missing third block launch";

  const auto second_block_exit_cycle =
      FirstWaveExitCycle(trace.events(), total_ap);
  ASSERT_TRUE(second_block_exit_cycle.has_value()) << "missing second block exit";
  EXPECT_LT(*third_block_cycle, *second_block_exit_cycle);
}

}  // namespace
}  // namespace gpu_model
