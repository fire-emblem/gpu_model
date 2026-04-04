#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "gpu_model/debug/trace_event_builder.h"
#include "gpu_model/debug/trace_sink.h"
#include "gpu_model/isa/instruction_builder.h"
#include "gpu_model/runtime/runtime_engine.h"

namespace gpu_model {
namespace {

ExecutableKernel BuildCacheProbeKernel() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SMov("s1", 0);
  builder.MLoadGlobal("v1", "s0", "s1", 4);
  builder.VAdd("v2", "v1", "v1");
  builder.MLoadGlobal("v3", "s0", "s1", 4);
  builder.BExit();
  return builder.Build("cache_probe");
}

std::vector<uint64_t> ArriveCycles(const std::vector<TraceEvent>& events) {
  std::vector<uint64_t> cycles;
  for (const auto& event : events) {
    if (event.kind == TraceEventKind::Arrive && event.message == kTraceArriveLoadMessage) {
      cycles.push_back(event.cycle);
    }
  }
  return cycles;
}

TEST(CacheCycleTest, SecondLoadHitsCacheAfterFirstLoadArrives) {
  CollectingTraceSink trace;
  RuntimeEngine runtime(&trace);
  runtime.SetGlobalMemoryLatencyProfile(/*dram=*/40, /*l2=*/20, /*l1=*/8);

  const uint64_t base_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(base_addr, 21);

  const auto kernel = BuildCacheProbeKernel();
  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;
  request.args.PushU64(base_addr);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  const auto cycles = ArriveCycles(trace.events());
  ASSERT_EQ(cycles.size(), 2u);
  EXPECT_EQ(cycles[0], 52u);
  EXPECT_EQ(cycles[1], 68u);
  EXPECT_EQ(result.total_cycles, 68u);
}

}  // namespace
}  // namespace gpu_model
