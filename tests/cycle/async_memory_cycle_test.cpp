#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <string_view>
#include <vector>

#include "gpu_model/debug/trace_sink.h"
#include "gpu_model/isa/instruction_builder.h"
#include "gpu_model/runtime/host_runtime.h"

namespace gpu_model {
namespace {

uint64_t NthWaveStepCycle(const std::vector<TraceEvent>& events,
                          std::string_view opcode,
                          size_t ordinal) {
  size_t seen = 0;
  for (const auto& event : events) {
    if (event.kind == TraceEventKind::WaveStep &&
        event.message.find(std::string(opcode)) != std::string::npos) {
      if (seen == ordinal) {
        return event.cycle;
      }
      ++seen;
    }
  }
  return std::numeric_limits<uint64_t>::max();
}

uint64_t FirstWaveStepCycle(const std::vector<TraceEvent>& events, std::string_view opcode) {
  return NthWaveStepCycle(events, opcode, 0);
}

TEST(AsyncMemoryCycleTest, LoadUsesIssuePlusArriveLatency) {
  CollectingTraceSink trace;
  HostRuntime runtime(&trace);
  runtime.SetFixedGlobalMemoryLatency(20);

  const uint64_t base_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(base_addr, 9);

  InstructionBuilder builder;
  builder.SMov("s0", base_addr);
  builder.SMov("s1", 0);
  builder.MLoadGlobal("v1", "s0", "s1", 4);
  builder.VAdd("v2", "v1", "v1");
  builder.BExit();
  const auto kernel = builder.Build("one_load");

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  EXPECT_EQ(result.total_cycles, 40u);
  EXPECT_EQ(FirstWaveStepCycle(trace.events(), "v_add"), 32u);
}

TEST(AsyncMemoryCycleTest, LoadBlocksWaveIssueUntilArriveEvenForIndependentOp) {
  CollectingTraceSink trace;
  HostRuntime runtime(&trace);
  runtime.SetFixedGlobalMemoryLatency(20);

  const uint64_t base_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(base_addr, 9);

  InstructionBuilder builder;
  builder.SMov("s0", base_addr);
  builder.SMov("s1", 0);
  builder.MLoadGlobal("v1", "s0", "s1", 4);
  builder.SMov("s2", 7);
  builder.BExit();
  const auto kernel = builder.Build("load_blocks_wave");

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  EXPECT_EQ(NthWaveStepCycle(trace.events(), "s_mov", 0), 0u);
  EXPECT_EQ(NthWaveStepCycle(trace.events(), "s_mov", 1), 4u);
  EXPECT_EQ(FirstWaveStepCycle(trace.events(), "m_load_global"), 8u);
  EXPECT_EQ(NthWaveStepCycle(trace.events(), "s_mov", 2), 32u);
  EXPECT_EQ(result.total_cycles, 40u);
}

}  // namespace
}  // namespace gpu_model
