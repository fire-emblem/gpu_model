#include <gtest/gtest.h>

#include <cstdint>

#include "gpu_model/debug/cycle_timeline.h"
#include "gpu_model/debug/trace_sink.h"
#include "gpu_model/isa/instruction_builder.h"
#include "gpu_model/runtime/host_runtime.h"

namespace gpu_model {
namespace {

KernelProgram BuildTimelineKernel() {
  InstructionBuilder builder;
  builder.SysGlobalIdX("v0");
  builder.VMov("v1", "v0");
  builder.VFma("v1", "v1", "v1", "v1");
  builder.BExit();
  return builder.Build("timeline_kernel");
}

TEST(CycleTimelineTest, RendersAsciiTimelineForMultipleWaves) {
  CollectingTraceSink trace;
  HostRuntime runtime(&trace);

  const auto kernel = BuildTimelineKernel();
  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 128;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  const std::string timeline = CycleTimelineRenderer::RenderAscii(trace.events());
  EXPECT_NE(timeline.find("cycle_timeline"), std::string::npos);
  EXPECT_NE(timeline.find("B0W0"), std::string::npos);
  EXPECT_NE(timeline.find("B0W1"), std::string::npos);
  EXPECT_NE(timeline.find("v_fma"), std::string::npos);
}

}  // namespace
}  // namespace gpu_model
