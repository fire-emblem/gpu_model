#include <gtest/gtest.h>

#include <cstdint>

#include "gpu_model/debug/recorder/recorder.h"
#include "gpu_model/debug/timeline/cycle_timeline.h"
#include "gpu_model/debug/trace/event_factory.h"
#include "gpu_model/debug/trace/sink.h"
#include "gpu_model/isa/instruction_builder.h"
#include "gpu_model/runtime/exec_engine.h"

namespace gpu_model {
namespace {

CycleTimelineOptions FullMarkerOptions() {
  CycleTimelineOptions options;
  options.marker_detail = CycleTimelineMarkerDetail::Full;
  return options;
}

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

Recorder MakeRecorder(std::initializer_list<TraceEvent> events) {
  Recorder recorder;
  for (const auto& event : events) {
    recorder.Record(event);
  }
  return recorder;
}

Recorder MakeRecorder(const std::vector<TraceEvent>& events) {
  Recorder recorder;
  for (const auto& event : events) {
    recorder.Record(event);
  }
  return recorder;
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

size_t CountOccurrences(std::string_view text, std::string_view needle) {
  size_t count = 0;
  size_t pos = 0;
  while ((pos = text.find(needle, pos)) != std::string_view::npos) {
    ++count;
    pos += needle.size();
  }
  return count;
}

TEST(CycleTimelineTest, RendersGoogleTraceForWaveTimeline) {
  CollectingTraceSink trace;
  ExecEngine runtime(&trace);

  const auto kernel = BuildTimelineKernel();
  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 128;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  const std::string timeline =
      CycleTimelineRenderer::RenderGoogleTrace(MakeRecorder(trace.events()));
  EXPECT_NE(timeline.find("\"traceEvents\""), std::string::npos);
  EXPECT_NE(timeline.find("\"ph\":\"X\""), std::string::npos);
  EXPECT_NE(timeline.find("\"thread_name\""), std::string::npos);
  EXPECT_NE(timeline.find("\"args\":{\"name\":\"WAVE_SLOT_00\"}"), std::string::npos);
  EXPECT_EQ(timeline.find("B0W0"), std::string::npos);
  EXPECT_NE(timeline.find("\"name\":\"v_mad_i32\""), std::string::npos);
  EXPECT_NE(timeline.find("\"thread_sort_index\""), std::string::npos);
}

TEST(CycleTimelineTest, GoogleTraceUsesZeroBasedSlotIdsForWaveTracks) {
  const TraceWaveView wave0 = MakeWaveView(/*slot_id=*/0, /*pc=*/0x100, /*wave_id=*/0);
  const TraceWaveView wave1 = MakeWaveView(/*slot_id=*/1, /*pc=*/0x104, /*wave_id=*/1);
  const std::string timeline = CycleTimelineRenderer::RenderGoogleTrace(MakeRecorder({
      MakeTraceWaveLaunchEvent(
          wave0, /*cycle=*/0, {}, TraceSlotModelKind::ResidentFixed),
      MakeTraceWaveStepEvent(wave0, /*cycle=*/0, TraceSlotModelKind::ResidentFixed, "op=s_mov_b32"),
      MakeTraceCommitEvent(wave0, /*cycle=*/4, TraceSlotModelKind::ResidentFixed, wave0.pc),
      MakeTraceWaveLaunchEvent(
          wave1, /*cycle=*/0, {}, TraceSlotModelKind::ResidentFixed),
      MakeTraceWaveStepEvent(wave1, /*cycle=*/4, TraceSlotModelKind::ResidentFixed, "op=s_mov_b32"),
      MakeTraceCommitEvent(wave1, /*cycle=*/8, TraceSlotModelKind::ResidentFixed, wave1.pc),
  }));

  EXPECT_NE(timeline.find("\"tid\":0,\"args\":{\"name\":\"WAVE_SLOT_00\"}"), std::string::npos);
  EXPECT_NE(timeline.find("\"tid\":1,\"args\":{\"name\":\"WAVE_SLOT_01\"}"), std::string::npos);
}

TEST(CycleTimelineTest, GoogleTraceCanGroupByBlock) {
  CollectingTraceSink trace;
  ExecEngine runtime(&trace);

  const auto kernel = BuildTimelineKernel();
  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 2;
  request.config.block_dim_x = 128;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  const std::string timeline = CycleTimelineRenderer::RenderGoogleTrace(
      MakeRecorder(trace.events()), CycleTimelineOptions{.max_columns = 120,
                                                         .cycle_begin = std::nullopt,
                                                         .cycle_end = std::nullopt,
                                                         .group_by = CycleTimelineGroupBy::Block});
  EXPECT_NE(timeline.find("B0"), std::string::npos);
  EXPECT_NE(timeline.find("B1"), std::string::npos);
  EXPECT_EQ(timeline.find("\"name\":\"B0W0\""), std::string::npos);
}

TEST(CycleTimelineTest, GoogleTraceCanGroupByPeu) {
  CollectingTraceSink trace;
  ExecEngine runtime(&trace);

  const auto kernel = BuildTimelineKernel();
  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 128;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  const std::string timeline = CycleTimelineRenderer::RenderGoogleTrace(
      MakeRecorder(trace.events()), CycleTimelineOptions{.max_columns = 120,
                                                         .cycle_begin = std::nullopt,
                                                         .cycle_end = std::nullopt,
                                                         .group_by = CycleTimelineGroupBy::Peu});
  EXPECT_NE(timeline.find("\"name\":\"DPC_00/AP_00\""), std::string::npos);
  EXPECT_NE(timeline.find("\"name\":\"PEU_00\""), std::string::npos);
  EXPECT_NE(timeline.find("\"name\":\"PEU_01\""), std::string::npos);
  EXPECT_NE(timeline.find("\"process_sort_index\""), std::string::npos);
}

TEST(CycleTimelineTest, HighlightsTensorOpsInGoogleTrace) {
  const TraceWaveView wave = MakeWaveView(/*slot_id=*/0);
  const Recorder recorder = MakeRecorder({
      MakeResidentWaveEvent(
          wave, TraceEventKind::WaveStep, 10, "pc=0x100 op=v_mfma_f32_16x16x4f32 exec_lanes=0x40"),
      MakeTraceCommitEvent(wave, 14, TraceSlotModelKind::ResidentFixed),
  });

  const std::string trace = CycleTimelineRenderer::RenderGoogleTrace(recorder);
  EXPECT_NE(trace.find("\"cat\":\"tensor\""), std::string::npos);
  EXPECT_NE(trace.find("\"name\":\"v_mfma_f32_16x16x4f32\""), std::string::npos);
}

TEST(CycleTimelineTest, PerfettoDumpPreservesCycleIssueAndCommitOrdering) {
  CollectingTraceSink trace;
  ExecEngine runtime(&trace);
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

  ASSERT_NE(issue_cycle, std::numeric_limits<uint64_t>::max());
  ASSERT_NE(stall_cycle, std::numeric_limits<uint64_t>::max());
  ASSERT_NE(arrive_cycle, std::numeric_limits<uint64_t>::max());
  ASSERT_NE(waitcnt_issue_cycle, std::numeric_limits<uint64_t>::max());
  EXPECT_LT(issue_cycle, arrive_cycle);
  EXPECT_LT(waitcnt_issue_cycle, stall_cycle);
  EXPECT_LT(stall_cycle, arrive_cycle);

  const Recorder recorder = [&] {
    Recorder r;
    for (const auto& event : events) {
      r.Record(event);
    }
    return r;
  }();
  const std::string timeline = CycleTimelineRenderer::RenderGoogleTrace(recorder);
  EXPECT_NE(timeline.find("\"name\":\"buffer_load_dword\""), std::string::npos);
  EXPECT_NE(timeline.find("load_arrive"), std::string::npos);
  EXPECT_NE(timeline.find("\"name\":\"stall_waitcnt_global\""), std::string::npos);
  EXPECT_NE(timeline.find("\"issue_cycle\":"), std::string::npos);
  EXPECT_NE(timeline.find("\"commit_cycle\":"), std::string::npos);
}

TEST(CycleTimelineTest, PerfettoDumpPreservesBarrierKernelStallTaxonomy) {
  CollectingTraceSink trace;
  ExecEngine runtime(&trace);
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

  const Recorder recorder = [&] {
    Recorder r;
    for (const auto& event : events) {
      r.Record(event);
    }
    return r;
  }();
  const std::string timeline = CycleTimelineRenderer::RenderGoogleTrace(recorder);
  EXPECT_NE(timeline.find("\"name\":\"stall_waitcnt_global\""), std::string::npos);
}

TEST(CycleTimelineTest, GoogleTraceUsesSlotTracksAndPreservesWaveAsArgs) {
  const TraceWaveView wave = MakeWaveView(/*slot_id=*/3);
  Recorder recorder;
  for (const TraceEvent& event : std::vector<TraceEvent>{
      MakeTraceWaveLaunchEvent(wave, 0, {}, TraceSlotModelKind::ResidentFixed),
      MakeTraceWaveStepEvent(wave,
                             2,
                             TraceSlotModelKind::ResidentFixed,
                             "pc=0x100 op=v_add_i32",
                             std::numeric_limits<uint64_t>::max(),
                             /*issue_duration_cycles=*/4),
      MakeTraceCommitEvent(wave, 5, TraceSlotModelKind::ResidentFixed),
  }) {
    recorder.Record(event);
  }

  const std::string trace = CycleTimelineRenderer::RenderGoogleTrace(recorder);
  EXPECT_NE(trace.find("\"args\":{\"name\":\"WAVE_SLOT_03\"}"), std::string::npos);
  EXPECT_NE(trace.find("\"name\":\"v_add_i32\""), std::string::npos);
  EXPECT_NE(trace.find("\"slot\":3"), std::string::npos);
  EXPECT_NE(trace.find("\"slot_model\":\"resident_fixed\""), std::string::npos);
  EXPECT_NE(trace.find("\"wave\":0"), std::string::npos);
  EXPECT_NE(trace.find("\"issue_cycle\":2"), std::string::npos);
  EXPECT_NE(trace.find("\"commit_cycle\":5"), std::string::npos);
  EXPECT_NE(trace.find("\"dur\":4"), std::string::npos);
  EXPECT_EQ(trace.find("\"args\":{\"name\":\"B0W0\"}"), std::string::npos);
  EXPECT_NE(trace.find("\"slot_models\":[\"resident_fixed\"]"),
            std::string::npos);
}

TEST(CycleTimelineTest, GoogleTraceRendersOrdinaryInstructionAsFixedFourCycleSlice) {
  const TraceWaveView wave = MakeWaveView(/*slot_id=*/0);
  std::vector<TraceEvent> events{
      MakeTraceWaveStepEvent(wave,
                             10,
                             TraceSlotModelKind::ResidentFixed,
                             "pc=0x100 op=v_add_i32",
                             std::numeric_limits<uint64_t>::max(),
                             /*issue_duration_cycles=*/4),
      MakeTraceCommitEvent(wave, 11, TraceSlotModelKind::ResidentFixed),
  };

  const std::string trace = CycleTimelineRenderer::RenderGoogleTrace(MakeRecorder(events));
  EXPECT_NE(trace.find("\"name\":\"v_add_i32\""), std::string::npos);
  EXPECT_NE(trace.find("\"ts\":10"), std::string::npos);
  EXPECT_NE(trace.find("\"dur\":4"), std::string::npos);
  EXPECT_NE(trace.find("\"issue_cycle\":10"), std::string::npos);
  EXPECT_NE(trace.find("\"commit_cycle\":11"), std::string::npos);
}

TEST(CycleTimelineTest, GoogleTraceDoesNotRenderInstructionSliceWithoutRecordedCycleRange) {
  const TraceWaveView wave = MakeWaveView(/*slot_id=*/0);
  Recorder recorder;
  recorder.Record(
      MakeResidentWaveEvent(wave, TraceEventKind::WaveStep, 10, "pc=0x100 op=v_add_i32"));

  const std::string trace = CycleTimelineRenderer::RenderGoogleTrace(recorder);
  EXPECT_EQ(trace.find("\"name\":\"v_add_i32\""), std::string::npos);
  EXPECT_EQ(trace.find("\"render_duration_cycles\":"), std::string::npos);
  EXPECT_EQ(trace.find("\"ph\":\"X\""), std::string::npos);
}

TEST(CycleTimelineTest, GoogleTraceUsesSourceRecordedInstructionRangeWithoutCommitInference) {
  const TraceWaveView wave = MakeWaveView(/*slot_id=*/0);
  Recorder recorder;
  recorder.Record(MakeTraceWaveStepEvent(wave,
                                         10,
                                         TraceSlotModelKind::ResidentFixed,
                                         "pc=0x100 op=v_add_i32",
                                         std::numeric_limits<uint64_t>::max(),
                                         /*issue_duration_cycles=*/12));
  recorder.Record(MakeTraceCommitEvent(wave, 11, TraceSlotModelKind::ResidentFixed));

  const std::string trace = CycleTimelineRenderer::RenderGoogleTrace(recorder);
  EXPECT_NE(trace.find("\"name\":\"v_add_i32\""), std::string::npos);
  EXPECT_NE(trace.find("\"issue_cycle\":10"), std::string::npos);
  EXPECT_NE(trace.find("\"commit_cycle\":11"), std::string::npos);
  EXPECT_NE(trace.find("\"render_duration_cycles\":12"), std::string::npos);
  EXPECT_NE(trace.find("\"dur\":12"), std::string::npos);
}

TEST(CycleTimelineTest, GoogleTraceFillsFourCyclesIndependentlyAcrossPeusAndSlots) {
  const TraceWaveView wave0{
      .dpc_id = 0,
      .ap_id = 0,
      .peu_id = 0,
      .slot_id = 1,
      .block_id = 0,
      .wave_id = 0,
      .pc = 0x100,
  };
  const TraceWaveView wave1{
      .dpc_id = 0,
      .ap_id = 0,
      .peu_id = 1,
      .slot_id = 3,
      .block_id = 1,
      .wave_id = 0,
      .pc = 0x200,
  };

  std::vector<TraceEvent> events{
      MakeTraceWaveStepEvent(wave0,
                             20,
                             TraceSlotModelKind::ResidentFixed,
                             "pc=0x100 op=v_add_i32",
                             std::numeric_limits<uint64_t>::max(),
                             /*issue_duration_cycles=*/4),
      MakeTraceCommitEvent(wave0, 21, TraceSlotModelKind::ResidentFixed),
      MakeTraceWaveStepEvent(wave1,
                             20,
                             TraceSlotModelKind::ResidentFixed,
                             "pc=0x200 op=s_mov_b32",
                             std::numeric_limits<uint64_t>::max(),
                             /*issue_duration_cycles=*/4),
      MakeTraceCommitEvent(wave1, 20, TraceSlotModelKind::ResidentFixed),
  };

  const std::string trace = CycleTimelineRenderer::RenderGoogleTrace(MakeRecorder(events));
  EXPECT_EQ(CountOccurrences(trace, "\"dur\":4"), 2u);
  EXPECT_NE(trace.find("\"peu\":0,\"slot\":1"), std::string::npos);
  EXPECT_NE(trace.find("\"peu\":1,\"slot\":3"), std::string::npos);
  EXPECT_NE(trace.find("\"name\":\"v_add_i32\""), std::string::npos);
  EXPECT_NE(trace.find("\"name\":\"s_mov_b32\""), std::string::npos);
}

TEST(CycleTimelineTest, GoogleTraceDoesNotRenderBubbleAsDurationSlice) {
  const TraceWaveView wave = MakeWaveView(/*slot_id=*/1);
  std::vector<TraceEvent> events{
      MakeResidentWaveEvent(wave, TraceEventKind::WaveStep, 1, "op=v_add_i32"),
      MakeTraceCommitEvent(wave, 2, TraceSlotModelKind::ResidentFixed),
      MakeTypedStallEvent(wave, 10, TraceSlotModelKind::ResidentFixed,
                          TraceStallReason::WaitCntGlobal),
  };

  const std::string trace = CycleTimelineRenderer::RenderGoogleTrace(MakeRecorder(events));
  EXPECT_EQ(CountOccurrences(trace, "\"ph\":\"X\""), 1u);
  EXPECT_NE(trace.find("\"name\":\"stall_waitcnt_global\""), std::string::npos);
}

TEST(CycleTimelineTest, GoogleTraceDoesNotRenderInstructionSliceWithoutCommit) {
  const TraceWaveView wave = MakeWaveView(/*slot_id=*/0);
  std::vector<TraceEvent> events{
      MakeResidentWaveEvent(wave, TraceEventKind::WaveStep, 10, "pc=0x100 op=v_add_i32"),
  };

  const std::string trace = CycleTimelineRenderer::RenderGoogleTrace(MakeRecorder(events));
  EXPECT_EQ(CountOccurrences(trace, "\"ph\":\"X\""), 0u);
  EXPECT_EQ(trace.find("\"name\":\"v_add_i32\""), std::string::npos);
}

TEST(CycleTimelineTest, GoogleTraceRendersIssueSelectAsMarkerNotInstructionSlice) {
  const TraceWaveView wave = MakeWaveView(/*slot_id=*/2);
  std::vector<TraceEvent> events{
      MakeTraceWaveEvent(wave,
                         TraceEventKind::IssueSelect,
                         /*cycle=*/7,
                         TraceSlotModelKind::ResidentFixed,
                         "selected",
                         wave.pc),
  };

  const std::string trace =
      CycleTimelineRenderer::RenderGoogleTrace(MakeRecorder(events), FullMarkerOptions());
  EXPECT_EQ(CountOccurrences(trace, "\"ph\":\"X\""), 0u);
  EXPECT_NE(trace.find("\"name\":\"issue_select\""), std::string::npos);
}

TEST(CycleTimelineTest, GoogleTraceDefaultHidesIssueSelectAndWaveSwitchAwayMarkers) {
  const TraceWaveView wave = MakeWaveView(/*slot_id=*/2);
  std::vector<TraceEvent> events{
      MakeTraceWaveEvent(wave,
                         TraceEventKind::IssueSelect,
                         /*cycle=*/7,
                         TraceSlotModelKind::ResidentFixed,
                         "selected",
                         wave.pc),
      MakeTraceWaveSwitchAwayEvent(wave, /*cycle=*/8, TraceSlotModelKind::ResidentFixed),
  };

  const std::string trace = CycleTimelineRenderer::RenderGoogleTrace(MakeRecorder(events));
  EXPECT_EQ(trace.find("\"name\":\"issue_select\""), std::string::npos);
  EXPECT_EQ(trace.find("\"name\":\"wave_switch_away\""), std::string::npos);
}

TEST(CycleTimelineTest, GoogleTraceDoesNotRenderArriveBarrierOrStallAsInstructionSlice) {
  const TraceWaveView wave = MakeWaveView(/*slot_id=*/1);
  std::vector<TraceEvent> events{
      MakeTypedStallEvent(wave, 5, TraceSlotModelKind::ResidentFixed,
                          TraceStallReason::WaitCntGlobal),
      MakeTraceMemoryArriveEvent(
          wave, 6, TraceMemoryArriveKind::Load, TraceSlotModelKind::ResidentFixed),
      MakeTraceBarrierArriveEvent(wave, 7, TraceSlotModelKind::ResidentFixed),
  };

  const std::string trace = CycleTimelineRenderer::RenderGoogleTrace(MakeRecorder(events));
  EXPECT_EQ(CountOccurrences(trace, "\"ph\":\"X\""), 0u);
  EXPECT_NE(trace.find("\"name\":\"stall_waitcnt_global\""), std::string::npos);
  EXPECT_NE(trace.find("\"name\":\"load_arrive\""), std::string::npos);
  EXPECT_NE(trace.find("\"name\":\"barrier_arrive\""), std::string::npos);
}

TEST(CycleTimelineTest, GoogleTraceRendersWaveFrontEndMarkersWithStableTypedNames) {
  const TraceWaveView wave = MakeWaveView(/*slot_id=*/2);
  std::vector<TraceEvent> events{
      MakeTraceWaveEvent(
          wave, TraceEventKind::WaveGenerate, /*cycle=*/3, TraceSlotModelKind::ResidentFixed, "generate"),
      MakeTraceWaveEvent(
          wave, TraceEventKind::WaveDispatch, /*cycle=*/4, TraceSlotModelKind::ResidentFixed, "dispatch"),
      MakeTraceWaveEvent(
          wave, TraceEventKind::SlotBind, /*cycle=*/5, TraceSlotModelKind::ResidentFixed, "bound"),
  };

  const std::string trace = CycleTimelineRenderer::RenderGoogleTrace(MakeRecorder(events));
  EXPECT_EQ(CountOccurrences(trace, "\"ph\":\"X\""), 0u);
  EXPECT_NE(trace.find("\"name\":\"wave_generate\""), std::string::npos);
  EXPECT_NE(trace.find("\"name\":\"wave_dispatch\""), std::string::npos);
  EXPECT_NE(trace.find("\"name\":\"slot_bind\""), std::string::npos);
}

TEST(CycleTimelineTest, GoogleTraceRendersWaveStateEdgeMarkersWithStableTypedNames) {
  const TraceWaveView wave = MakeWaveView(/*slot_id=*/2);
  std::vector<TraceEvent> events{
      MakeTraceActivePromoteEvent(wave, /*cycle=*/3, TraceSlotModelKind::ResidentFixed),
      MakeTraceWaveWaitEvent(
          wave, /*cycle=*/4, TraceSlotModelKind::ResidentFixed, TraceStallReason::WaitCntGlobal),
      MakeTraceWaveArriveEvent(wave,
                               /*cycle=*/5,
                               TraceMemoryArriveKind::Load,
                               TraceSlotModelKind::ResidentFixed,
                               TraceArriveProgressKind::Resume),
      MakeTraceWaveResumeEvent(wave, /*cycle=*/6, TraceSlotModelKind::ResidentFixed),
      MakeTraceWaveSwitchAwayEvent(wave, /*cycle=*/7, TraceSlotModelKind::ResidentFixed),
  };

  const std::string trace =
      CycleTimelineRenderer::RenderGoogleTrace(MakeRecorder(events), FullMarkerOptions());
  EXPECT_EQ(CountOccurrences(trace, "\"ph\":\"X\""), 0u);
  EXPECT_NE(trace.find("\"name\":\"active_promote\""), std::string::npos);
  EXPECT_NE(trace.find("\"name\":\"wave_wait\""), std::string::npos);
  EXPECT_NE(trace.find("\"name\":\"wave_arrive\""), std::string::npos);
  EXPECT_NE(trace.find("\"name\":\"wave_resume\""), std::string::npos);
  EXPECT_NE(trace.find("\"name\":\"wave_switch_away\""), std::string::npos);
}

TEST(CycleTimelineTest, GoogleTraceKeepsRuntimeBlockEventsOffSlotTracks) {
  std::vector<TraceEvent> events{
      MakeTraceBlockAdmitEvent(/*dpc_id=*/0, /*ap_id=*/0, /*block_id=*/7, /*cycle=*/9, "admit"),
      MakeTraceBlockEvent(/*dpc_id=*/0,
                          /*ap_id=*/0,
                          /*block_id=*/7,
                          TraceEventKind::BlockLaunch,
                          /*cycle=*/10,
                          "launch"),
  };

  const std::string trace = CycleTimelineRenderer::RenderGoogleTrace(MakeRecorder(events));
  EXPECT_NE(trace.find("\"pid\":0,\"tid\":0"), std::string::npos);
  EXPECT_NE(trace.find("\"name\":\"block_admit\""), std::string::npos);
  EXPECT_NE(trace.find("\"name\":\"block_launch\""), std::string::npos);
  EXPECT_EQ(trace.find("\"args\":{\"name\":\"WAVE_SLOT_"), std::string::npos);
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

  const std::string trace = CycleTimelineRenderer::RenderGoogleTrace(MakeRecorder(events));
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
                 .waitcnt_state = {},
                 .has_cycle_range = false,
                 .range_end_cycle = 0,
                 .semantic_canonical_name = {},
                 .semantic_presentation_name = {},
                 .semantic_category = {},
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
                 .waitcnt_state = {},
                 .has_cycle_range = false,
                 .range_end_cycle = 0,
                 .semantic_canonical_name = {},
                 .semantic_presentation_name = {},
                 .semantic_category = {},
                 .display_name = "load",
                 .message = {}},
  };

  const std::string trace = CycleTimelineRenderer::RenderGoogleTrace(MakeRecorder(events));
  EXPECT_NE(trace.find("\"name\":\"barrier_arrive\""), std::string::npos);
  EXPECT_NE(trace.find("\"cat\":\"sync/barrier\""), std::string::npos);
  EXPECT_NE(trace.find("\"barrier_kind\":\"arrive\""), std::string::npos);
  EXPECT_NE(trace.find("\"presentation_name\":\"barrier_arrive\""), std::string::npos);
  EXPECT_NE(trace.find("\"name\":\"load_arrive\""), std::string::npos);
  EXPECT_NE(trace.find("\"cat\":\"memory/load_arrive\""), std::string::npos);
  EXPECT_NE(trace.find("\"arrive_kind\":\"load\""), std::string::npos);
  EXPECT_NE(trace.find("\"presentation_name\":\"load_arrive\""), std::string::npos);
}

TEST(CycleTimelineTest, GoogleTraceUsesArriveProgressTypedNamesAsMarkers) {
  const TraceWaveView wave = MakeWaveView(/*slot_id=*/1);
  std::vector<TraceEvent> events{
      TraceEvent{.kind = TraceEventKind::Arrive,
                 .cycle = 6,
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
                 .arrive_progress = TraceArriveProgressKind::StillBlocked,
                 .waitcnt_state = {},
                 .has_cycle_range = false,
                 .range_end_cycle = 0,
                 .semantic_canonical_name = {},
                 .semantic_presentation_name = {},
                 .semantic_category = {},
                 .display_name = "load",
                 .message = {}},
      TraceEvent{.kind = TraceEventKind::Arrive,
                 .cycle = 9,
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
                 .arrive_progress = TraceArriveProgressKind::Resume,
                 .waitcnt_state = {},
                 .has_cycle_range = false,
                 .range_end_cycle = 0,
                 .semantic_canonical_name = {},
                 .semantic_presentation_name = {},
                 .semantic_category = {},
                 .display_name = "load",
                 .message = {}},
  };

  const std::string trace = CycleTimelineRenderer::RenderGoogleTrace(MakeRecorder(events));
  EXPECT_EQ(CountOccurrences(trace, "\"ph\":\"X\""), 0u);
  EXPECT_NE(trace.find("\"name\":\"load_arrive_still_blocked\""), std::string::npos);
  EXPECT_NE(trace.find("\"name\":\"load_arrive_resume\""), std::string::npos);
  EXPECT_NE(trace.find("\"category\":\"memory/load_arrive/still_blocked\""), std::string::npos);
  EXPECT_NE(trace.find("\"category\":\"memory/load_arrive/resume\""), std::string::npos);
  EXPECT_NE(trace.find("\"arrive_progress\":\"still_blocked\""), std::string::npos);
  EXPECT_NE(trace.find("\"arrive_progress\":\"resume\""), std::string::npos);
}

TEST(CycleTimelineTest, GoogleTraceMarkerArgsShareTypedPresentationFields) {
  const TraceWaveView wave = MakeWaveView(/*slot_id=*/2, /*pc=*/0x80, /*wave_id=*/3);
  std::vector<TraceEvent> events{
      MakeTypedStallEvent(wave, 7, TraceSlotModelKind::LogicalUnbounded,
                          TraceStallReason::WaitCntGlobal),
  };

  const std::string trace = CycleTimelineRenderer::RenderGoogleTrace(MakeRecorder(events));
  EXPECT_NE(trace.find("\"name\":\"stall_waitcnt_global\""), std::string::npos);
  EXPECT_NE(trace.find("\"category\":\"stall/waitcnt_global\""), std::string::npos);
  EXPECT_NE(trace.find("\"presentation_name\":\"stall_waitcnt_global\""), std::string::npos);
  EXPECT_NE(trace.find("\"canonical_name\":\"stall_waitcnt_global\""), std::string::npos);
  EXPECT_NE(trace.find("\"stall_reason\":\"waitcnt_global\""), std::string::npos);
  EXPECT_NE(trace.find("\"slot_model\":\"logical_unbounded\""), std::string::npos);
}

TEST(CycleTimelineTest, GoogleTraceRuntimeArgsSharePresentationFields) {
  std::vector<TraceEvent> events{
      MakeTraceRuntimeLaunchEvent(/*cycle=*/5, "kernel=timeline_runtime arch=c500"),
      MakeTraceBlockEvent(/*dpc_id=*/0,
                          /*ap_id=*/0,
                          /*block_id=*/3,
                          TraceEventKind::BlockActivate,
                          /*cycle=*/8,
                          "activate"),
  };

  const std::string trace = CycleTimelineRenderer::RenderGoogleTrace(MakeRecorder(events));
  EXPECT_NE(trace.find("\"name\":\"launch\""), std::string::npos);
  EXPECT_NE(trace.find("\"cat\":\"runtime\""), std::string::npos);
  EXPECT_NE(trace.find("\"canonical_name\":\"launch\""), std::string::npos);
  EXPECT_NE(trace.find("\"presentation_name\":\"launch\""), std::string::npos);
  EXPECT_NE(trace.find("\"category\":\"runtime\""), std::string::npos);
  EXPECT_NE(trace.find("\"message\":\"kernel=timeline_runtime arch=c500\""), std::string::npos);
  EXPECT_NE(trace.find("\"name\":\"block_activate\""), std::string::npos);
  EXPECT_NE(trace.find("\"cat\":\"launch/block\""), std::string::npos);
  EXPECT_NE(trace.find("\"canonical_name\":\"block_activate\""), std::string::npos);
  EXPECT_NE(trace.find("\"presentation_name\":\"block_activate\""), std::string::npos);
  EXPECT_NE(trace.find("\"category\":\"launch/block\""), std::string::npos);
  EXPECT_EQ(trace.find("\"args\":{\"name\":\"WAVE_SLOT_"), std::string::npos);
}

TEST(CycleTimelineTest, TimelineCanRenderCanonicalNamesWithoutLegacyMessages) {
  const TraceWaveView wave = MakeWaveView(/*slot_id=*/0);
  TraceEvent barrier_arrive =
      MakeTraceBarrierArriveEvent(wave, 1, TraceSlotModelKind::ResidentFixed);
  barrier_arrive.message.clear();

  const std::string trace =
      CycleTimelineRenderer::RenderGoogleTrace(MakeRecorder({barrier_arrive}));
  EXPECT_NE(trace.find("\"name\":\"barrier_arrive\""), std::string::npos);
}

TEST(CycleTimelineTest, RuntimePerfettoDumpCarriesWaveLifecycleOnSlotTracks) {
  CollectingTraceSink trace;
  ExecEngine runtime(&trace);

  const auto kernel = BuildTimelineKernel();
  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 320;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  const std::string timeline = CycleTimelineRenderer::RenderGoogleTrace(MakeRecorder(trace.events()));
  EXPECT_NE(timeline.find("\"args\":{\"name\":\"WAVE_SLOT_01\"}"), std::string::npos);
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
