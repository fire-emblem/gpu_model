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
  EXPECT_NE(timeline.find("v_mad_i32"), std::string::npos);
}

TEST(CycleTimelineTest, CanGroupTimelineByBlock) {
  CollectingTraceSink trace;
  HostRuntime runtime(&trace);

  const auto kernel = BuildTimelineKernel();
  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 2;
  request.config.block_dim_x = 128;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  const std::string timeline = CycleTimelineRenderer::RenderAscii(
      trace.events(), CycleTimelineOptions{.max_columns = 120,
                                           .cycle_begin = std::nullopt,
                                           .cycle_end = std::nullopt,
                                           .group_by = CycleTimelineGroupBy::Block});
  EXPECT_NE(timeline.find("B0"), std::string::npos);
  EXPECT_NE(timeline.find("B1"), std::string::npos);
  EXPECT_EQ(timeline.find("B0W0"), std::string::npos);
}

TEST(CycleTimelineTest, RendersGoogleTraceForWaveTimeline) {
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

  const std::string timeline = CycleTimelineRenderer::RenderGoogleTrace(trace.events());
  EXPECT_NE(timeline.find("\"traceEvents\""), std::string::npos);
  EXPECT_NE(timeline.find("\"ph\":\"X\""), std::string::npos);
  EXPECT_NE(timeline.find("\"thread_name\""), std::string::npos);
  EXPECT_NE(timeline.find("B0W0"), std::string::npos);
  EXPECT_NE(timeline.find("\"name\":\"v_mad_i32\""), std::string::npos);
  EXPECT_NE(timeline.find("\"thread_sort_index\""), std::string::npos);
}

TEST(CycleTimelineTest, GoogleTraceCanGroupByBlock) {
  CollectingTraceSink trace;
  HostRuntime runtime(&trace);

  const auto kernel = BuildTimelineKernel();
  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 2;
  request.config.block_dim_x = 128;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  const std::string timeline = CycleTimelineRenderer::RenderGoogleTrace(
      trace.events(), CycleTimelineOptions{.max_columns = 120,
                                           .cycle_begin = std::nullopt,
                                           .cycle_end = std::nullopt,
                                           .group_by = CycleTimelineGroupBy::Block});
  EXPECT_NE(timeline.find("B0"), std::string::npos);
  EXPECT_NE(timeline.find("B1"), std::string::npos);
  EXPECT_EQ(timeline.find("\"name\":\"B0W0\""), std::string::npos);
}

TEST(CycleTimelineTest, GoogleTraceCanGroupByPeu) {
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

  const std::string timeline = CycleTimelineRenderer::RenderGoogleTrace(
      trace.events(), CycleTimelineOptions{.max_columns = 120,
                                           .cycle_begin = std::nullopt,
                                           .cycle_end = std::nullopt,
                                           .group_by = CycleTimelineGroupBy::Peu});
  EXPECT_NE(timeline.find("Device/D0/D0A0"), std::string::npos);
  EXPECT_NE(timeline.find("\"name\":\"D0A0P0\""), std::string::npos);
  EXPECT_NE(timeline.find("\"name\":\"D0A0P1\""), std::string::npos);
  EXPECT_NE(timeline.find("\"process_sort_index\""), std::string::npos);
}

TEST(CycleTimelineTest, HighlightsTensorOpsInAsciiAndGoogleTrace) {
  std::vector<TraceEvent> events{
      TraceEvent{
          .kind = TraceEventKind::WaveStep,
          .cycle = 10,
          .dpc_id = 0,
          .ap_id = 0,
          .peu_id = 0,
          .block_id = 0,
          .wave_id = 0,
          .pc = 0x100,
          .message = "pc=0x100 op=v_mfma_f32_16x16x4f32 exec_lanes=0x40",
      },
      TraceEvent{
          .kind = TraceEventKind::Commit,
          .cycle = 14,
          .dpc_id = 0,
          .ap_id = 0,
          .peu_id = 0,
          .block_id = 0,
          .wave_id = 0,
          .pc = 0x100,
          .message = "commit",
      },
  };

  const std::string ascii = CycleTimelineRenderer::RenderAscii(events);
  EXPECT_NE(ascii.find("T=tensor-op"), std::string::npos);
  EXPECT_NE(ascii.find("T=v_mfma_f32_16x16x4f32"), std::string::npos);

  const std::string trace = CycleTimelineRenderer::RenderGoogleTrace(events);
  EXPECT_NE(trace.find("\"cat\":\"tensor\""), std::string::npos);
  EXPECT_NE(trace.find("\"name\":\"v_mfma_f32_16x16x4f32\""), std::string::npos);
}

}  // namespace
}  // namespace gpu_model
