#include <gtest/gtest.h>

#include <cstdint>

#include "gpu_model/debug/cycle_timeline.h"
#include "gpu_model/debug/trace_sink.h"
#include "gpu_model/isa/instruction_builder.h"
#include "gpu_model/runtime/runtime_engine.h"

namespace gpu_model {
namespace {

ExecutableKernel BuildTimelineKernel() {
  InstructionBuilder builder;
  builder.SysGlobalIdX("v0");
  builder.VMov("v1", "v0");
  builder.VFma("v1", "v1", "v1", "v1");
  builder.BExit();
  return builder.Build("timeline_kernel");
}

ExecutableKernel BuildCycleOrderingKernel() {
  InstructionBuilder builder;
  builder.SMov("s0", 0);
  builder.SMov("s1", 0);
  builder.MLoadGlobal("v1", "s0", "s1", 4);
  builder.SWaitCnt(/*global_count=*/0, /*shared_count=*/UINT32_MAX,
                   /*private_count=*/UINT32_MAX, /*scalar_buffer_count=*/UINT32_MAX);
  builder.SMov("s2", 7);
  builder.BExit();
  return builder.Build("cycle_ordering_kernel");
}

ExecutableKernel BuildSharedBarrierCycleKernel() {
  InstructionBuilder builder;
  builder.SysGlobalIdX("v0");
  builder.SysBlockIdxX("s0");
  builder.SysBlockDimX("s1");
  builder.SMul("s2", "s0", "s1");
  builder.SMov("s3", static_cast<uint64_t>(-1));
  builder.SMul("s4", "s2", "s3");
  builder.VAdd("v1", "v0", "s4");
  builder.VMov("v2", 1);
  builder.MStoreShared("v1", "v2", 4);
  builder.SMov("s5", 64);
  builder.VCmpLtCmask("v1", "s5");
  builder.MaskSaveExec("s10");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("after_extra");
  builder.VMov("v3", 7);
  builder.Label("after_extra");
  builder.MaskRestoreExec("s10");
  builder.SyncBarrier();
  builder.MLoadShared("v4", "v1", 4);
  builder.SLoadArg("s6", 0);
  builder.SMov("s7", 0);
  builder.MLoadGlobal("v5", "s6", "s7", 4);
  builder.SWaitCnt(/*global_count=*/0, /*shared_count=*/UINT32_MAX,
                   /*private_count=*/UINT32_MAX, /*scalar_buffer_count=*/UINT32_MAX);
  builder.BExit();
  return builder.Build("shared_barrier_cycle_timeline");
}

uint64_t FirstEventCycle(const std::vector<TraceEvent>& events,
                         TraceEventKind kind,
                         std::string_view message_substr) {
  for (const auto& event : events) {
    if (event.kind != kind) {
      continue;
    }
    if (event.message.find(std::string(message_substr)) == std::string::npos) {
      continue;
    }
    return event.cycle;
  }
  return std::numeric_limits<uint64_t>::max();
}

TEST(CycleTimelineTest, RendersAsciiTimelineForMultipleWaves) {
  CollectingTraceSink trace;
  RuntimeEngine runtime(&trace);

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
  RuntimeEngine runtime(&trace);

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
  RuntimeEngine runtime(&trace);

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
  RuntimeEngine runtime(&trace);

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
  RuntimeEngine runtime(&trace);

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

TEST(CycleTimelineTest, PerfettoDumpPreservesCycleIssueAndCommitOrdering) {
  CollectingTraceSink trace;
  RuntimeEngine runtime(&trace);
  runtime.SetFixedGlobalMemoryLatency(20);

  const uint64_t base_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(base_addr, 9);

  const auto kernel = BuildCycleOrderingKernel();
  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  const auto& events = trace.events();
  const uint64_t issue_cycle = FirstEventCycle(events, TraceEventKind::WaveStep, "buffer_load_dword");
  const uint64_t stall_cycle =
      FirstEventCycle(events, TraceEventKind::Stall, "reason=waitcnt_global");
  const uint64_t arrive_cycle = FirstEventCycle(events, TraceEventKind::Arrive, "load_arrive");
  const uint64_t commit_cycle = FirstEventCycle(events, TraceEventKind::Commit, "s_waitcnt");

  ASSERT_NE(issue_cycle, std::numeric_limits<uint64_t>::max());
  ASSERT_NE(stall_cycle, std::numeric_limits<uint64_t>::max());
  ASSERT_NE(arrive_cycle, std::numeric_limits<uint64_t>::max());
  ASSERT_NE(commit_cycle, std::numeric_limits<uint64_t>::max());
  EXPECT_LT(issue_cycle, arrive_cycle);
  EXPECT_LT(stall_cycle, commit_cycle);

  const std::string timeline = CycleTimelineRenderer::RenderGoogleTrace(events);
  EXPECT_NE(timeline.find("\"name\":\"buffer_load_dword\""), std::string::npos);
  EXPECT_NE(timeline.find("\"name\":\"load_arrive\""), std::string::npos);
  EXPECT_NE(timeline.find("\"name\":\"stall_waitcnt_global\""), std::string::npos);
  EXPECT_NE(timeline.find("\"issue_cycle\":\""), std::string::npos);
  EXPECT_NE(timeline.find("\"commit_cycle\":\""), std::string::npos);
}

TEST(CycleTimelineTest, PerfettoDumpPreservesBarrierKernelStallTaxonomy) {
  CollectingTraceSink trace;
  RuntimeEngine runtime(&trace);
  runtime.SetFixedGlobalMemoryLatency(20);

  const auto kernel = BuildSharedBarrierCycleKernel();
  const uint64_t base_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(base_addr, 9);
  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 128;
  request.config.shared_memory_bytes = 128 * sizeof(int32_t);
  request.args.PushU64(base_addr);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  const auto& events = trace.events();
  const uint64_t stall_cycle =
      FirstEventCycle(events, TraceEventKind::Stall, "reason=waitcnt_global");
  EXPECT_NE(stall_cycle, std::numeric_limits<uint64_t>::max());

  const std::string timeline = CycleTimelineRenderer::RenderGoogleTrace(events);
  EXPECT_NE(timeline.find("\"name\":\"stall_waitcnt_global\""), std::string::npos);
}

}  // namespace
}  // namespace gpu_model
