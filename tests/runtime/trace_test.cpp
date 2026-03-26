#include <gtest/gtest.h>

#include "gpu_model/debug/trace_sink.h"
#include "gpu_model/isa/instruction_builder.h"
#include "gpu_model/runtime/host_runtime.h"

namespace gpu_model {
namespace {

TEST(TraceTest, EmitsLaunchAndBlockPlacementEvents) {
  CollectingTraceSink trace;
  HostRuntime runtime(&trace);

  InstructionBuilder builder;
  builder.BExit();
  const auto kernel = builder.Build("trace_kernel");

  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 2;
  request.config.block_dim_x = 64;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok);
  ASSERT_GE(trace.events().size(), 3u);
  EXPECT_EQ(trace.events()[0].kind, TraceEventKind::Launch);
  EXPECT_EQ(trace.events()[1].kind, TraceEventKind::BlockPlaced);
}

}  // namespace
}  // namespace gpu_model
