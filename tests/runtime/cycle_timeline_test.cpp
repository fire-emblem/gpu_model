#include <gtest/gtest.h>

#include <cstdint>

#include "gpu_model/debug/cycle_timeline.h"
#include "gpu_model/debug/trace_event_builder.h"
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

TraceWaveView MakeWaveView(uint32_t slot_id, uint64_t pc = 0x100, uint32_t wave_id = 0) {
  return TraceWaveView{
      .dpc_id = 0,
      .ap_id = 0,
      .peu_id = 0,
      .slot_id = slot_id,
      .block_id = 0,
      .wave_id = wave_id,
      .pc = pc,
  };
}

TraceEvent MakeResidentWaveEvent(const TraceWaveView& wave,
                                 TraceEventKind kind,
                                 uint64_t cycle,
                                 std::string message,
                                 uint64_t pc = std::numeric_limits<uint64_t>::max()) {
  return MakeTraceWaveEvent(
      wave, kind, cycle, TraceSlotModelKind::ResidentFixed, std::move(message), pc);
}

TraceEvent MakeTypedStallEvent(const TraceWaveView& wave,
                               uint64_t cycle,
                               TraceSlotModelKind slot_model,
                               TraceStallReason stall_reason) {
  TraceEvent event = MakeTraceWaveEvent(wave, TraceEventKind::Stall, cycle, slot_model, {});
  event.stall_reason = stall_reason;
  event.message.clear();
  return event;
}

uint64_t FirstEventCycle(const std::vector<TraceEvent>& events,
                         TraceEventKind kind,
                         std::string_view message_substr) {
  for (const auto& event : events) {
    if (event.kind != kind) {
      continue;
    }
    if (kind == TraceEventKind::Stall &&
        TraceHasStallReason(event, TraceStallReasonFromMessage(message_substr))) {
      return event.cycle;
    }
    if (event.message.find(std::string(message_substr)) == std::string::npos) {
      continue;
    }
    return event.cycle;
  }
  return std::numeric_limits<uint64_t>::max();
}

uint64_t FirstCommitCycleAtOrAfter(const std::vector<TraceEvent>& events, uint64_t cycle) {
  for (const auto& event : events) {
    if (event.kind != TraceEventKind::Commit || event.cycle < cycle) {
      continue;
    }
    return event.cycle;
  }
  return std::numeric_limits<uint64_t>::max();
}

size_t CountOccurrences(std::string_view text, std::string_view needle) {
  size_t count = 0;
  size_t pos = 0;
  while ((pos = text.find(needle, pos)) != std::string_view::npos) {
    ++count;
    pos += needle.size();
  }
  return count;
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
  EXPECT_NE(timeline.find("S0"), std::string::npos);
  EXPECT_EQ(timeline.find("B0W0"), std::string::npos);
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
  EXPECT_NE(timeline.find("\"args\":{\"name\":\"S0\"}"), std::string::npos);
  EXPECT_EQ(timeline.find("B0W0"), std::string::npos);
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
  EXPECT_NE(timeline.find("\"name\":\"D0/A0\""), std::string::npos);
  EXPECT_NE(timeline.find("\"name\":\"P0\""), std::string::npos);
  EXPECT_NE(timeline.find("\"name\":\"P1\""), std::string::npos);
  EXPECT_NE(timeline.find("\"process_sort_index\""), std::string::npos);
}

TEST(CycleTimelineTest, HighlightsTensorOpsInAsciiAndGoogleTrace) {
  const TraceWaveView wave = MakeWaveView(/*slot_id=*/0);
  std::vector<TraceEvent> events{
      MakeResidentWaveEvent(
          wave, TraceEventKind::WaveStep, 10, "pc=0x100 op=v_mfma_f32_16x16x4f32 exec_lanes=0x40"),
      MakeTraceCommitEvent(wave, 14, TraceSlotModelKind::ResidentFixed),
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
  uint64_t arrive_cycle = std::numeric_limits<uint64_t>::max();
  for (const auto& event : events) {
    if (event.kind == TraceEventKind::Arrive && event.arrive_kind == TraceArriveKind::Load) {
      arrive_cycle = event.cycle;
      break;
    }
  }
  const uint64_t waitcnt_issue_cycle =
      FirstEventCycle(events, TraceEventKind::WaveStep, "s_waitcnt");
  const uint64_t commit_cycle = FirstCommitCycleAtOrAfter(events, waitcnt_issue_cycle);

  ASSERT_NE(issue_cycle, std::numeric_limits<uint64_t>::max());
  ASSERT_NE(stall_cycle, std::numeric_limits<uint64_t>::max());
  ASSERT_NE(arrive_cycle, std::numeric_limits<uint64_t>::max());
  ASSERT_NE(waitcnt_issue_cycle, std::numeric_limits<uint64_t>::max());
  ASSERT_NE(commit_cycle, std::numeric_limits<uint64_t>::max());
  EXPECT_LT(issue_cycle, arrive_cycle);
  EXPECT_LT(stall_cycle, commit_cycle);

  const std::string timeline = CycleTimelineRenderer::RenderGoogleTrace(events);
  EXPECT_NE(timeline.find("\"name\":\"buffer_load_dword\""), std::string::npos);
  EXPECT_NE(timeline.find("\"name\":\"load_arrive\""), std::string::npos);
  EXPECT_NE(timeline.find("\"name\":\"stall_waitcnt_global\""), std::string::npos);
  EXPECT_NE(timeline.find("\"issue_cycle\":"), std::string::npos);
  EXPECT_NE(timeline.find("\"commit_cycle\":"), std::string::npos);
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

TEST(CycleTimelineTest, GoogleTraceUsesSlotTracksAndPreservesWaveAsArgs) {
  const TraceWaveView wave = MakeWaveView(/*slot_id=*/3);
  std::vector<TraceEvent> events{
      MakeTraceWaveLaunchEvent(wave, 0, {}, TraceSlotModelKind::ResidentFixed),
      MakeResidentWaveEvent(wave, TraceEventKind::WaveStep, 2, "pc=0x100 op=v_add_i32"),
      MakeTraceCommitEvent(wave, 5, TraceSlotModelKind::ResidentFixed),
  };

  const std::string trace = CycleTimelineRenderer::RenderGoogleTrace(events);
  EXPECT_NE(trace.find("\"args\":{\"name\":\"S3\"}"), std::string::npos);
  EXPECT_NE(trace.find("\"name\":\"v_add_i32\""), std::string::npos);
  EXPECT_NE(trace.find("\"slot\":3"), std::string::npos);
  EXPECT_NE(trace.find("\"slot_model\":\"resident_fixed\""), std::string::npos);
  EXPECT_NE(trace.find("\"wave\":0"), std::string::npos);
  EXPECT_NE(trace.find("\"issue_cycle\":2"), std::string::npos);
  EXPECT_NE(trace.find("\"commit_cycle\":5"), std::string::npos);
  EXPECT_EQ(trace.find("\"args\":{\"name\":\"B0W0\"}"), std::string::npos);
  EXPECT_NE(trace.find("\"slot_models\":[\"resident_fixed\"]"),
            std::string::npos);
}

TEST(CycleTimelineTest, GoogleTraceDoesNotRenderBubbleAsDurationSlice) {
  const TraceWaveView wave = MakeWaveView(/*slot_id=*/1);
  std::vector<TraceEvent> events{
      MakeResidentWaveEvent(wave, TraceEventKind::WaveStep, 1, "op=v_add_i32"),
      MakeTraceCommitEvent(wave, 2, TraceSlotModelKind::ResidentFixed),
      MakeTypedStallEvent(wave, 10, TraceSlotModelKind::ResidentFixed,
                          TraceStallReason::WaitCntGlobal),
  };

  const std::string trace = CycleTimelineRenderer::RenderGoogleTrace(events);
  EXPECT_EQ(CountOccurrences(trace, "\"ph\":\"X\""), 1u);
  EXPECT_NE(trace.find("\"name\":\"stall_waitcnt_global\""), std::string::npos);
}

TEST(CycleTimelineTest, GoogleTracePrefersTypedSchemaFieldsWhenLegacyStringsAreEmpty) {
  const TraceWaveView wave = MakeWaveView(/*slot_id=*/2, /*pc=*/0x80, /*wave_id=*/3);
  std::vector<TraceEvent> events{
      MakeTraceWaveStepEvent(
          wave, 1, TraceSlotModelKind::LogicalUnbounded, "op=v_add_i32"),
      MakeTraceCommitEvent(wave, 4, TraceSlotModelKind::LogicalUnbounded),
      MakeTypedStallEvent(wave, 7, TraceSlotModelKind::LogicalUnbounded,
                          TraceStallReason::WaitCntGlobal),
  };

  const std::string trace = CycleTimelineRenderer::RenderGoogleTrace(events);
  EXPECT_NE(trace.find("\"name\":\"stall_waitcnt_global\""), std::string::npos);
  EXPECT_NE(trace.find("\"slot_model\":\"logical_unbounded\""), std::string::npos);
  EXPECT_NE(trace.find("\"stall_reason\":\"waitcnt_global\""), std::string::npos);
  EXPECT_NE(
      trace.find("\"metadata\":{\"time_unit\":\"cycle\",\"slot_models\":[\"logical_unbounded\"]"),
      std::string::npos);
}

TEST(CycleTimelineTest, GoogleTraceUsesCanonicalBarrierAndArriveNamesFromTypedFields) {
  const TraceWaveView wave = MakeWaveView(/*slot_id=*/1);
  std::vector<TraceEvent> events{
      TraceEvent{.kind = TraceEventKind::Barrier,
                 .cycle = 3,
                 .dpc_id = wave.dpc_id,
                 .ap_id = wave.ap_id,
                 .peu_id = wave.peu_id,
                 .slot_id = wave.slot_id,
                 .slot_model_kind = TraceSlotModelKind::ResidentFixed,
                 .slot_model = {},
                 .block_id = wave.block_id,
                 .wave_id = wave.wave_id,
                 .pc = wave.pc,
                 .barrier_kind = TraceBarrierKind::Arrive,
                 .display_name = "arrive",
                 .message = {}},
      TraceEvent{.kind = TraceEventKind::Arrive,
                 .cycle = 4,
                 .dpc_id = wave.dpc_id,
                 .ap_id = wave.ap_id,
                 .peu_id = wave.peu_id,
                 .slot_id = wave.slot_id,
                 .slot_model_kind = TraceSlotModelKind::ResidentFixed,
                 .slot_model = {},
                 .block_id = wave.block_id,
                 .wave_id = wave.wave_id,
                 .pc = wave.pc,
                 .arrive_kind = TraceArriveKind::Load,
                 .display_name = "load",
                 .message = {}},
  };

  const std::string trace = CycleTimelineRenderer::RenderGoogleTrace(events);
  EXPECT_NE(trace.find("\"name\":\"barrier_arrive\""), std::string::npos);
  EXPECT_NE(trace.find("\"name\":\"load_arrive\""), std::string::npos);
}

TEST(CycleTimelineTest, TimelineCanRenderCanonicalNamesWithoutLegacyMessages) {
  const TraceWaveView wave = MakeWaveView(/*slot_id=*/0);
  TraceEvent barrier_arrive =
      MakeTraceBarrierArriveEvent(wave, 1, TraceSlotModelKind::ResidentFixed);
  barrier_arrive.message.clear();

  const std::string trace = CycleTimelineRenderer::RenderGoogleTrace({barrier_arrive});
  EXPECT_NE(trace.find("\"name\":\"barrier_arrive\""), std::string::npos);
}

TEST(CycleTimelineTest, RuntimePerfettoDumpCarriesWaveLifecycleOnSlotTracks) {
  CollectingTraceSink trace;
  RuntimeEngine runtime(&trace);

  const auto kernel = BuildTimelineKernel();
  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 320;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  const std::string timeline = CycleTimelineRenderer::RenderGoogleTrace(trace.events());
  EXPECT_NE(timeline.find("\"args\":{\"name\":\"S1\"}"), std::string::npos);
  EXPECT_NE(timeline.find("\"name\":\"wave_launch\""), std::string::npos);
  EXPECT_NE(timeline.find(std::string("\"message\":\"") + std::string(kTraceWaveStartMessage)),
            std::string::npos);
  EXPECT_NE(timeline.find("\"name\":\"wave_exit\""), std::string::npos);
  EXPECT_NE(timeline.find(std::string("\"message\":\"") + std::string(kTraceWaveEndMessage) + "\""),
            std::string::npos);
  EXPECT_NE(timeline.find("\"slot\":1"), std::string::npos);
}

}  // namespace
}  // namespace gpu_model
