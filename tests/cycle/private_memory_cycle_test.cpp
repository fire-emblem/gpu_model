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

KernelProgram BuildPrivateCycleKernel() {
  InstructionBuilder builder;
  builder.VMov("v0", 0);
  builder.VMov("v1", 7);
  builder.MStorePrivate("v0", "v1", 4);
  builder.MLoadPrivate("v2", "v0", 4);
  builder.BExit();
  return builder.Build("private_cycle");
}

uint64_t FirstCycle(const std::vector<TraceEvent>& events,
                    TraceEventKind kind,
                    std::string_view message = {}) {
  for (const auto& event : events) {
    if (event.kind == kind &&
        (message.empty() || event.message.find(std::string(message)) != std::string::npos)) {
      return event.cycle;
    }
  }
  return std::numeric_limits<uint64_t>::max();
}

TEST(PrivateMemoryCycleTest, PrivateLoadCompletesAtIssueCommitWithoutAsyncArrive) {
  CollectingTraceSink trace;
  HostRuntime runtime(&trace);

  const auto kernel = BuildPrivateCycleKernel();
  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  EXPECT_EQ(result.total_cycles, 20u);
  EXPECT_EQ(FirstCycle(trace.events(), TraceEventKind::WaveStep, "m_load_private"), 12u);
}

}  // namespace
}  // namespace gpu_model
