#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "debug/recorder/export.h"
#include "debug/recorder/recorder.h"
#include "debug/timeline/cycle_timeline.h"
#include "debug/trace/artifact_recorder.h"
#include "debug/trace/event_factory.h"
#include "debug/trace/event_view.h"
#include "instruction/isa/instruction_builder.h"
#include "runtime/exec_engine/exec_engine.h"
#include "tests/test_utils/llvm_mc_test_support.h"
#include "tests/test_utils/trace_test_support.h"

namespace gpu_model {
namespace {

using test::MakeUniqueTempDir;
using test::ReadTextFile;
using test::ExtractTraceEventsPayload;
using test::CountOccurrences;
using test::FindFirst;
using test::HasJsonField;
using test::HasEventArg;
using test::ExpectContainsTypedSlotFields;
using test::ExpectContainsTypedSlotFieldsJson;
using test::ExpectContainsTypedStallReasonFields;
using test::ExpectContainsTypedStallReasonFieldsJson;
using test::ParsedPerfettoTrackDescriptor;
using test::ParsedPerfettoTrackEvent;
using test::ParseTrackDescriptors;
using test::ParseTrackEvents;
using test::FullMarkerOptions;
using test::MakeRecorder;
using test::BuildWaitcntTraceKernel;
using test::BuildSamePeuWaitcntSiblingKernel;
using test::BuildCycleMultiWaveWaitcntKernelForTraceTest;
using test::BuildWaitcntThresholdProgressKernel;
using test::HasLlvmMcAmdgpuToolchain;

// =============================================================================
// Perfetto JSON Trace Structure Tests
// =============================================================================

TEST(TracePerfettoTest, PerfettoDumpContainsTraceEventsAndRequiredFields) {
  const auto out_dir = MakeUniqueTempDir("gpu_model_perfetto_structure");
  const struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() { std::filesystem::remove_all(path); }
  } cleanup{out_dir};

  TraceArtifactRecorder trace(out_dir, FullMarkerOptions());
  ExecEngine runtime(&trace);

  InstructionBuilder builder;
  builder.BExit();
  const auto kernel = builder.Build("perfetto_structure_kernel");

  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  trace.FlushTimeline();

  const auto timeline_path = out_dir / "timeline.perfetto.json";
  ASSERT_TRUE(std::filesystem::exists(timeline_path));

  const std::string text = ReadTextFile(timeline_path);
  const auto trace_events = ExtractTraceEventsPayload(text);
  ASSERT_FALSE(trace_events.empty());
  EXPECT_NE(text.find("\"traceEvents\""), std::string::npos);
  EXPECT_NE(trace_events.find('{'), std::string::npos);
  EXPECT_NE(trace_events.find("\"name\""), std::string::npos);
  EXPECT_NE(trace_events.find("\"ph\":"), std::string::npos);
  EXPECT_NE(trace_events.find("\"ts\":"), std::string::npos);
  EXPECT_TRUE(HasJsonField(trace_events, "\"args\""));
  EXPECT_TRUE(HasEventArg(trace_events, "name"));
}

TEST(TracePerfettoTest, PerfettoExportUsesCanonicalTypedNamesWithoutMessageParsing) {
  const TraceWaveView wave{
      .dpc_id = 0,
      .ap_id = 0,
      .peu_id = 0,
      .slot_id = 0,
      .block_id = 0,
      .wave_id = 0,
      .pc = 0,
  };

  const std::vector<TraceEvent> events{
      TraceEvent{.kind = TraceEventKind::WaveLaunch,
                 .cycle = 0,
                 .dpc_id = wave.dpc_id,
                 .ap_id = wave.ap_id,
                 .peu_id = wave.peu_id,
                 .slot_id = wave.slot_id,
                 .slot_model_kind = TraceSlotModelKind::ResidentFixed,
                 .slot_model = {},
                 .block_id = wave.block_id,
                 .wave_id = wave.wave_id,
                 .pc = wave.pc,
                 .lifecycle_stage = TraceLifecycleStage::Launch,
                 .waitcnt_state = {},
                 .has_cycle_range = false,
                 .range_end_cycle = 0,
                 .semantic_canonical_name = {},
                 .semantic_presentation_name = {},
                 .semantic_category = {},
                 .display_name = "launch",
                 .message = {}},
      TraceEvent{.kind = TraceEventKind::WaveExit,
                 .cycle = 5,
                 .dpc_id = wave.dpc_id,
                 .ap_id = wave.ap_id,
                 .peu_id = wave.peu_id,
                 .slot_id = wave.slot_id,
                 .slot_model_kind = TraceSlotModelKind::ResidentFixed,
                 .slot_model = {},
                 .block_id = wave.block_id,
                 .wave_id = wave.wave_id,
                 .pc = wave.pc,
                 .lifecycle_stage = TraceLifecycleStage::Exit,
                 .waitcnt_state = {},
                 .has_cycle_range = false,
                 .range_end_cycle = 0,
                 .semantic_canonical_name = {},
                 .semantic_presentation_name = {},
                 .semantic_category = {},
                 .display_name = "exit",
                 .message = {}},
  };

  const std::string trace = CycleTimelineRenderer::RenderPerfettoTraceProto(MakeRecorder(events));
  const auto parsed_events = ParseTrackEvents(trace);
  bool saw_wave_launch = false;
  bool saw_wave_exit = false;
  for (const auto& event : parsed_events) {
    if (event.type != 3u) {
      continue;
    }
    saw_wave_launch = saw_wave_launch || event.name == "wave_launch";
    saw_wave_exit = saw_wave_exit || event.name == "wave_exit";
  }

  EXPECT_TRUE(saw_wave_launch);
  EXPECT_TRUE(saw_wave_exit);
}

TEST(TracePerfettoTest, PerfettoProtoUsesCanonicalNamesForRuntimeAndFrontEndMarkers) {
  const TraceWaveView wave{
      .dpc_id = 0,
      .ap_id = 0,
      .peu_id = 1,
      .slot_id = 2,
      .block_id = 3,
      .wave_id = 4,
      .pc = 0x40,
  };

  const std::vector<TraceEvent> events{
      MakeTraceRuntimeLaunchEvent(/*cycle=*/1, "kernel=perfetto_runtime arch=mac500"),
      MakeTraceBlockEvent(/*dpc_id=*/0,
                          /*ap_id=*/0,
                          /*block_id=*/3,
                          TraceEventKind::BlockLaunch,
                          /*cycle=*/2,
                          {}),
      MakeTraceWaveEvent(
          wave, TraceEventKind::WaveGenerate, /*cycle=*/3, TraceSlotModelKind::ResidentFixed, {}),
      MakeTraceWaveEvent(
          wave, TraceEventKind::WaveDispatch, /*cycle=*/4, TraceSlotModelKind::ResidentFixed, {}),
      MakeTraceWaveEvent(
          wave, TraceEventKind::SlotBind, /*cycle=*/5, TraceSlotModelKind::ResidentFixed, {}),
      MakeTraceWaveEvent(
          wave, TraceEventKind::IssueSelect, /*cycle=*/6, TraceSlotModelKind::ResidentFixed, {}),
  };

  const std::string bytes =
      CycleTimelineRenderer::RenderPerfettoTraceProto(MakeRecorder(events), FullMarkerOptions());
  const auto parsed_events = ParseTrackEvents(bytes);

  bool saw_launch = false;
  bool saw_block_launch = false;
  bool saw_wave_generate = false;
  bool saw_wave_dispatch = false;
  bool saw_slot_bind = false;
  bool saw_issue_select = false;
  for (const auto& event : parsed_events) {
    if (event.type != 3u) {
      continue;
    }
    saw_launch = saw_launch || event.name == "launch";
    saw_block_launch = saw_block_launch || event.name == "block_launch";
    saw_wave_generate = saw_wave_generate || event.name == "wave_generate";
    saw_wave_dispatch = saw_wave_dispatch || event.name == "wave_dispatch";
    saw_slot_bind = saw_slot_bind || event.name == "slot_bind";
    saw_issue_select = saw_issue_select || event.name == "issue_select";
  }

  EXPECT_TRUE(saw_launch);
  EXPECT_TRUE(saw_block_launch);
  EXPECT_TRUE(saw_wave_generate);
  EXPECT_TRUE(saw_wave_dispatch);
  EXPECT_TRUE(saw_slot_bind);
  EXPECT_TRUE(saw_issue_select);
}

TEST(TracePerfettoTest, PerfettoDumpForMultiThreadedWaitKernelShowsWaitingAndResumeOrdering) {
  const auto out_dir = MakeUniqueTempDir("gpu_model_perfetto_mt_wait");
  const struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() { std::filesystem::remove_all(path); }
  } cleanup{out_dir};

  TraceArtifactRecorder trace(out_dir, FullMarkerOptions());
  ExecEngine runtime(&trace);
  runtime.SetFunctionalExecutionMode(FunctionalExecutionMode::MultiThreaded);

  const auto kernel = BuildWaitcntTraceKernel();
  const uint64_t base_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(base_addr, 11);

  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 2;
  request.config.block_dim_x = 64;
  request.args.PushU64(base_addr);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  trace.FlushTimeline();

  const std::string timeline = ReadTextFile(out_dir / "timeline.perfetto.json");
  const auto trace_events = ExtractTraceEventsPayload(timeline);
  ASSERT_FALSE(trace_events.empty());

  EXPECT_GE(CountOccurrences(trace_events, "\"name\":\"thread_name\""), 2u) << timeline;
  EXPECT_NE(FindFirst(trace_events, "\"args\":{\"name\":\"WAVE_SLOT_00\"}"), std::string::npos)
      << timeline;
  EXPECT_EQ(FindFirst(trace_events, "\"args\":{\"name\":\"B0W0\"}"), std::string::npos)
      << timeline;
  EXPECT_NE(FindFirst(trace_events, "\"name\":\"stall_waitcnt_global\""), std::string::npos)
      << timeline;
  EXPECT_NE(FindFirst(trace_events, "\"name\":\"wave_exit\""), std::string::npos) << timeline;
  EXPECT_NE(FindFirst(trace_events, "\"cycle\":"), std::string::npos) << timeline;
  EXPECT_LT(FindFirst(trace_events, "\"name\":\"stall_waitcnt_global\""),
            FindFirst(trace_events, "\"name\":\"wave_exit\""))
      << timeline;
}

TEST(TracePerfettoTest, PerfettoDumpForSingleThreadedWaitKernelUsesSharedSlotSchema) {
  const auto out_dir = MakeUniqueTempDir("gpu_model_perfetto_st_wait");
  const struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() { std::filesystem::remove_all(path); }
  } cleanup{out_dir};

  TraceArtifactRecorder trace(out_dir, FullMarkerOptions());
  ExecEngine runtime(&trace);
  runtime.SetFunctionalExecutionMode(FunctionalExecutionMode::SingleThreaded);
  runtime.SetFixedGlobalMemoryLatency(20);

  const auto kernel = BuildWaitcntTraceKernel();
  const uint64_t base_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(base_addr, 11);

  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;
  request.args.PushU64(base_addr);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  trace.FlushTimeline();

  const std::string timeline = ReadTextFile(out_dir / "timeline.perfetto.json");
  const auto trace_events = ExtractTraceEventsPayload(timeline);
  ASSERT_FALSE(trace_events.empty());
  EXPECT_NE(trace_events.find("\"args\":{\"name\":\"WAVE_SLOT_"), std::string::npos) << timeline;
  EXPECT_NE(trace_events.find("\"slot\":"), std::string::npos) << timeline;
  EXPECT_NE(trace_events.find("\"slot_model\":\"logical_unbounded\""), std::string::npos)
      << timeline;
  EXPECT_NE(trace_events.find("\"cycle\":"), std::string::npos) << timeline;
  EXPECT_EQ(trace_events.find("\"args\":{\"name\":\"B0W0\"}"), std::string::npos) << timeline;
  EXPECT_NE(timeline.find("\"metadata\":{\"time_unit\":\"cycle\",\"slot_models\":"),
            std::string::npos)
      << timeline;
  EXPECT_NE(timeline.find("\"hierarchy_levels\":[\"Device\",\"DPC\",\"AP\",\"PEU\",\"WAVE_SLOT\"]"),
            std::string::npos)
      << timeline;
  EXPECT_NE(timeline.find("\"perfetto_format\":\"chrome_json\""), std::string::npos)
      << timeline;
}

// =============================================================================
// TraceArtifactRecorder Tests
// =============================================================================

TEST(TracePerfettoTest, TraceArtifactRecorderWritesTraceAndPerfettoFiles) {
  const auto out_dir =
      std::filesystem::temp_directory_path() / "gpu_model_trace_artifact_recorder";
  std::filesystem::remove_all(out_dir);
  std::filesystem::create_directories(out_dir);

  {
    TraceArtifactRecorder trace(out_dir);
    trace.OnEvent(MakeTraceRuntimeLaunchEvent(0, "kernel=artifact_trace arch=mac500"));
    const TraceWaveView wave{
        .dpc_id = 0,
        .ap_id = 0,
        .peu_id = 0,
        .slot_id = 2,
        .block_id = 0,
        .wave_id = 0,
        .pc = 0,
    };
    trace.OnEvent(MakeTraceWaveLaunchEvent(
        wave, 0, "lanes=0x40 exec=0xffffffffffffffff", TraceSlotModelKind::ResidentFixed));
    trace.OnEvent(
        MakeTraceWaveStepEvent(wave, 1, TraceSlotModelKind::ResidentFixed, "op=v_add_i32"));
    trace.OnEvent(MakeTraceCommitEvent(wave, 4, TraceSlotModelKind::ResidentFixed));
    trace.OnEvent(MakeTraceWaveExitEvent(wave, 5, TraceSlotModelKind::ResidentFixed));
    trace.OnEvent(MakeTraceWaitStallEvent(wave,
                                          6,
                                          TraceStallReason::WaitCntGlobal,
                                          TraceSlotModelKind::ResidentFixed));
    trace.FlushTimeline();
  }

  const auto text_path = out_dir / "trace.txt";
  const auto json_path = out_dir / "trace.jsonl";
  const auto timeline_path = out_dir / "timeline.perfetto.json";

  std::ifstream text_in(text_path);
  std::ifstream json_in(json_path);
  std::ifstream timeline_in(timeline_path);
  ASSERT_TRUE(static_cast<bool>(text_in));
  ASSERT_TRUE(static_cast<bool>(json_in));
  ASSERT_TRUE(static_cast<bool>(timeline_in));

  std::ostringstream text_buffer;
  std::ostringstream json_buffer;
  std::ostringstream timeline_buffer;
  text_buffer << text_in.rdbuf();
  json_buffer << json_in.rdbuf();
  timeline_buffer << timeline_in.rdbuf();

  // Text trace now only includes wave_step and wave_exit for cleaner output.
  // JSON trace still contains all events.
  EXPECT_NE(json_buffer.str().find("\"kind\":\"Launch\""), std::string::npos);
  // JSON has slot_id as hex string
  EXPECT_NE(json_buffer.str().find("\"slot_id\":\"0x2\""), std::string::npos);
  // These checks validate the typed schema fields that should remain the primary contract.
  ExpectContainsTypedSlotFieldsJson(json_buffer.str(), "resident_fixed");
  ExpectContainsTypedStallReasonFieldsJson(json_buffer.str(), "waitcnt_global");
  EXPECT_NE(timeline_buffer.str().find("\"traceEvents\""), std::string::npos);
  EXPECT_NE(timeline_buffer.str().find("\"slot\":2"), std::string::npos);
  EXPECT_NE(timeline_buffer.str().find("\"slot_model\":\"resident_fixed\""), std::string::npos);
  EXPECT_NE(timeline_buffer.str().find("\"metadata\":{\"time_unit\":\"cycle\",\"slot_models\":"),
            std::string::npos);
  EXPECT_NE(timeline_buffer.str().find("\"hierarchy_levels\":[\"Device\",\"DPC\",\"AP\",\"PEU\",\"WAVE_SLOT\"]"),
            std::string::npos);
  EXPECT_NE(timeline_buffer.str().find("\"perfetto_format\":\"chrome_json\""),
            std::string::npos);
  EXPECT_NE(timeline_buffer.str().find("\"name\":\"stall_waitcnt_global\""), std::string::npos);

  std::filesystem::remove_all(out_dir);
}

TEST(TracePerfettoTest, TraceArtifactRecorderKeepsSchedulerMarkersInJsonWhenTimelineHidesThem) {
  const auto out_dir =
      std::filesystem::temp_directory_path() / "gpu_model_trace_scheduler_marker_visibility";
  std::filesystem::remove_all(out_dir);
  std::filesystem::create_directories(out_dir);

  {
    TraceArtifactRecorder trace(out_dir);
    const TraceWaveView wave{
        .dpc_id = 0,
        .ap_id = 0,
        .peu_id = 0,
        .slot_id = 2,
        .block_id = 1,
        .wave_id = 3,
        .pc = 0x120,
    };
    trace.OnEvent(MakeTraceWaveEvent(wave,
                                     TraceEventKind::IssueSelect,
                                     /*cycle=*/7,
                                     TraceSlotModelKind::ResidentFixed,
                                     "selected",
                                     wave.pc));
    trace.OnEvent(
        MakeTraceWaveSwitchAwayEvent(wave, /*cycle=*/8, TraceSlotModelKind::ResidentFixed));
    trace.FlushTimeline();
  }

  const std::string json = ReadTextFile(out_dir / "trace.jsonl");
  const std::string timeline = ReadTextFile(out_dir / "timeline.perfetto.json");

  EXPECT_NE(json.find("\"kind\":\"IssueSelect\""), std::string::npos);
  EXPECT_NE(json.find("\"kind\":\"WaveSwitchAway\""), std::string::npos);
  EXPECT_EQ(timeline.find("\"name\":\"issue_select\""), std::string::npos);
  EXPECT_EQ(timeline.find("\"name\":\"wave_switch_away\""), std::string::npos);

  std::filesystem::remove_all(out_dir);
}

TEST(TracePerfettoTest, TraceArtifactRecorderOwnsUnifiedRecorderState) {
  const auto out_dir =
      std::filesystem::temp_directory_path() / "gpu_model_trace_artifact_recorder_state";
  std::filesystem::remove_all(out_dir);
  std::filesystem::create_directories(out_dir);

  TraceArtifactRecorder trace(out_dir);
  const TraceWaveView wave{
      .dpc_id = 0,
      .ap_id = 0,
      .peu_id = 0,
      .slot_id = 1,
      .block_id = 2,
      .wave_id = 3,
      .pc = 0x20,
  };
  trace.OnEvent(MakeTraceRuntimeLaunchEvent(0, "kernel=artifact_state arch=mac500"));
  trace.OnEvent(
      MakeTraceWaveStepEvent(wave, 4, TraceSlotModelKind::ResidentFixed, "pc=0x20 op=v_add_i32"));
  trace.OnEvent(MakeTraceCommitEvent(wave, 8, TraceSlotModelKind::ResidentFixed));

  ASSERT_EQ(trace.events().size(), 3u);
  ASSERT_EQ(trace.recorder().waves().size(), 1u);
  const RecorderWave& recorded_wave = trace.recorder().waves().front();
  EXPECT_EQ(recorded_wave.dpc_id, 0u);
  EXPECT_EQ(recorded_wave.ap_id, 0u);
  EXPECT_EQ(recorded_wave.peu_id, 0u);
  EXPECT_EQ(recorded_wave.slot_id, 1u);
  EXPECT_EQ(recorded_wave.block_id, 2u);
  EXPECT_EQ(recorded_wave.wave_id, 3u);
  ASSERT_EQ(recorded_wave.entries.size(), 2u);
  EXPECT_EQ(recorded_wave.entries.front().begin_cycle, 4u);
  EXPECT_EQ(recorded_wave.entries.front().end_cycle, 8u);

  std::filesystem::remove_all(out_dir);
}

// =============================================================================
// Native Perfetto Proto Tests
// =============================================================================

TEST(TracePerfettoTest, NativePerfettoProtoContainsHierarchicalTracksAndEvents) {
  const auto out_dir = MakeUniqueTempDir("gpu_model_perfetto_proto_structure");
  const struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() { std::filesystem::remove_all(path); }
  } cleanup{out_dir};

  TraceArtifactRecorder trace(out_dir);
  ExecEngine runtime(&trace);
  runtime.SetFixedGlobalMemoryLatency(20);

  const auto kernel = BuildWaitcntTraceKernel();
  const uint64_t base_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(base_addr, 17);

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;
  request.args.PushU64(base_addr);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  trace.FlushTimeline();

  const std::string bytes =
      CycleTimelineRenderer::RenderPerfettoTraceProto(trace.recorder(), FullMarkerOptions());
  ASSERT_FALSE(bytes.empty());

  const auto descriptors = ParseTrackDescriptors(bytes);
  const auto events = ParseTrackEvents(bytes);
  ASSERT_FALSE(descriptors.empty());
  ASSERT_FALSE(events.empty());

  std::map<std::string, uint64_t> uuids_by_name;
  for (const auto& descriptor : descriptors) {
    uuids_by_name[descriptor.name] = descriptor.uuid;
  }

  ASSERT_TRUE(uuids_by_name.count("Device"));
  ASSERT_TRUE(uuids_by_name.count("DPC_00"));
  ASSERT_TRUE(uuids_by_name.count("AP_00"));
  ASSERT_TRUE(uuids_by_name.count("PEU_00"));
  ASSERT_TRUE(uuids_by_name.count("WAVE_SLOT_00"));

  std::map<uint64_t, std::optional<uint64_t>> parents_by_uuid;
  for (const auto& descriptor : descriptors) {
    parents_by_uuid[descriptor.uuid] = descriptor.parent_uuid;
  }

  EXPECT_EQ(parents_by_uuid[uuids_by_name["DPC_00"]], uuids_by_name["Device"]);
  EXPECT_EQ(parents_by_uuid[uuids_by_name["AP_00"]], uuids_by_name["DPC_00"]);
  EXPECT_EQ(parents_by_uuid[uuids_by_name["PEU_00"]], uuids_by_name["AP_00"]);
  EXPECT_EQ(parents_by_uuid[uuids_by_name["WAVE_SLOT_00"]], uuids_by_name["PEU_00"]);

  bool saw_slice_begin = false;
  bool saw_slice_end = false;
  bool saw_wave_launch = false;
  bool saw_wave_exit = false;
  bool saw_load_arrive = false;
  for (const auto& event : events) {
    if (event.track_uuid != uuids_by_name["WAVE_SLOT_00"]) {
      continue;
    }
    if (event.type == 1u && event.name == "buffer_load_dword") {
      saw_slice_begin = true;
    }
    if (event.type == 2u) {
      saw_slice_end = true;
    }
    if (event.type == 3u && event.name == "wave_launch") {
      saw_wave_launch = true;
    }
    if (event.type == 3u && event.name == "wave_exit") {
      saw_wave_exit = true;
    }
    if (event.type == 3u &&
        (event.name == "load_arrive" || event.name.starts_with("load_arrive_"))) {
      saw_load_arrive = true;
    }
  }

  EXPECT_TRUE(saw_slice_begin);
  EXPECT_TRUE(saw_slice_end);
  EXPECT_TRUE(saw_wave_launch);
  EXPECT_TRUE(saw_wave_exit);
  EXPECT_TRUE(saw_load_arrive);
}

TEST(TracePerfettoTest, NativePerfettoProtoShowsBlockAdmitGenerateDispatchAndSlotBindOrdering) {
  const auto out_dir = MakeUniqueTempDir("gpu_model_perfetto_cycle_frontend_latency");
  const struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() { std::filesystem::remove_all(path); }
  } cleanup{out_dir};

  TraceArtifactRecorder trace(out_dir, FullMarkerOptions());
  ExecEngine runtime(&trace);
  runtime.SetLaunchTimingProfile(/*kernel_launch_gap_cycles=*/8,
                                 /*kernel_launch_cycles=*/0,
                                 /*block_launch_cycles=*/0,
                                 /*wave_generation_cycles=*/128,
                                 /*wave_dispatch_cycles=*/256,
                                 /*wave_launch_cycles=*/0,
                                 /*warp_switch_cycles=*/1,
                                 /*arg_load_cycles=*/4);

  const auto kernel = BuildWaitcntTraceKernel();
  const uint64_t base_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(base_addr, 17);

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;
  request.args.PushU64(base_addr);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  trace.FlushTimeline();

  const std::string bytes =
      CycleTimelineRenderer::RenderPerfettoTraceProto(trace.recorder(), FullMarkerOptions());
  const auto events = ParseTrackEvents(bytes);
  ASSERT_FALSE(events.empty());

  std::optional<uint64_t> block_admit_cycle;
  std::optional<uint64_t> block_activate_cycle;
  std::optional<uint64_t> wave_generate_cycle;
  std::optional<uint64_t> wave_dispatch_cycle;
  std::optional<uint64_t> slot_bind_cycle;
  std::optional<uint64_t> wave_launch_cycle;
  std::optional<uint64_t> issue_select_cycle;
  std::optional<uint64_t> block_retire_cycle;

  for (const auto& event : events) {
    if (event.type != 3u) {
      continue;
    }
    if (!block_admit_cycle.has_value() && event.name == "block_admit") {
      block_admit_cycle = event.timestamp;
    } else if (!block_activate_cycle.has_value() && event.name == "block_activate") {
      block_activate_cycle = event.timestamp;
    } else if (!wave_generate_cycle.has_value() && event.name == "wave_generate") {
      wave_generate_cycle = event.timestamp;
    } else if (!wave_dispatch_cycle.has_value() && event.name == "wave_dispatch") {
      wave_dispatch_cycle = event.timestamp;
    } else if (!slot_bind_cycle.has_value() && event.name == "slot_bind") {
      slot_bind_cycle = event.timestamp;
    } else if (!wave_launch_cycle.has_value() && event.name == "wave_launch") {
      wave_launch_cycle = event.timestamp;
    } else if (!issue_select_cycle.has_value() && event.name == "issue_select") {
      issue_select_cycle = event.timestamp;
    } else if (!block_retire_cycle.has_value() && event.name == "block_retire") {
      block_retire_cycle = event.timestamp;
    }
  }

  ASSERT_TRUE(block_admit_cycle.has_value());
  ASSERT_TRUE(block_activate_cycle.has_value());
  ASSERT_TRUE(wave_generate_cycle.has_value());
  ASSERT_TRUE(wave_dispatch_cycle.has_value());
  ASSERT_TRUE(slot_bind_cycle.has_value());
  ASSERT_TRUE(wave_launch_cycle.has_value());
  ASSERT_TRUE(issue_select_cycle.has_value());
  ASSERT_TRUE(block_retire_cycle.has_value());

  EXPECT_LT(*block_admit_cycle, *wave_generate_cycle);
  EXPECT_LE(*block_admit_cycle, *block_activate_cycle);
  EXPECT_LT(*wave_generate_cycle, *wave_dispatch_cycle);
  EXPECT_LE(*wave_dispatch_cycle, *slot_bind_cycle);
  EXPECT_LE(*slot_bind_cycle, *wave_launch_cycle);
  EXPECT_LE(*wave_launch_cycle, *issue_select_cycle);
  EXPECT_GT(*block_retire_cycle, *wave_launch_cycle);
  EXPECT_EQ(*wave_generate_cycle - *block_admit_cycle, 128u);
  EXPECT_EQ(*wave_dispatch_cycle - *wave_generate_cycle, 256u);
}

TEST(TracePerfettoTest, NativePerfettoProtoShowsVisibleWaitcntBubbleInCycleMode) {
  const auto out_dir = MakeUniqueTempDir("gpu_model_perfetto_cycle_gap_visible_bubble");
  const struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() { std::filesystem::remove_all(path); }
  } cleanup{out_dir};

  TraceArtifactRecorder trace(out_dir, FullMarkerOptions());
  ExecEngine runtime(&trace);
  runtime.SetFixedGlobalMemoryLatency(20);

  const auto kernel = BuildWaitcntTraceKernel();
  const uint64_t base_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(base_addr, 17);

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;
  request.args.PushU64(base_addr);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  trace.FlushTimeline();

  const std::string bytes =
      CycleTimelineRenderer::RenderPerfettoTraceProto(trace.recorder(), FullMarkerOptions());
  const auto descriptors = ParseTrackDescriptors(bytes);
  const auto events = ParseTrackEvents(bytes);
  ASSERT_FALSE(descriptors.empty());
  ASSERT_FALSE(events.empty());

  std::optional<uint64_t> slot_uuid;
  for (const auto& descriptor : descriptors) {
    if (descriptor.name == "WAVE_SLOT_00") {
      slot_uuid = descriptor.uuid;
      break;
    }
  }
  ASSERT_TRUE(slot_uuid.has_value());

  std::optional<uint64_t> first_stall_cycle;
  std::optional<uint64_t> first_arrive_cycle;
  for (const auto& event : events) {
    if (event.track_uuid != *slot_uuid || event.type != 3u) {
      continue;
    }
    if (!first_stall_cycle.has_value() && event.name == "stall_waitcnt_global") {
      first_stall_cycle = event.timestamp;
    }
    if (!first_arrive_cycle.has_value() &&
        event.name.rfind("load_arrive", 0) == 0) {
      first_arrive_cycle = event.timestamp;
    }
  }

  ASSERT_TRUE(first_stall_cycle.has_value());
  ASSERT_TRUE(first_arrive_cycle.has_value());
  EXPECT_GT(*first_arrive_cycle, *first_stall_cycle);
  EXPECT_GE(*first_arrive_cycle - *first_stall_cycle, 16u);
}

TEST(TracePerfettoTest, NativePerfettoProtoShowsCycleSamePeuResidentSlotsAcrossPeus) {
  const auto out_dir = MakeUniqueTempDir("gpu_model_perfetto_cycle_same_peu_proto");
  const struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() { std::filesystem::remove_all(path); }
  } cleanup{out_dir};

  TraceArtifactRecorder trace(out_dir, FullMarkerOptions());
  ExecEngine runtime(&trace);
  runtime.SetFixedGlobalMemoryLatency(20);

  constexpr uint32_t kBlockDim = 64 * 16;
  const auto kernel = BuildCycleMultiWaveWaitcntKernelForTraceTest();
  const uint64_t in_addr = runtime.memory().AllocateGlobal(kBlockDim * sizeof(int32_t));
  for (uint32_t i = 0; i < kBlockDim; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(in_addr + i * sizeof(int32_t),
                                               static_cast<int32_t>(100 + i));
  }

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = kBlockDim;
  request.args.PushU64(in_addr);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  trace.FlushTimeline();

  const std::string bytes =
      CycleTimelineRenderer::RenderPerfettoTraceProto(trace.recorder(), FullMarkerOptions());
  const auto descriptors = ParseTrackDescriptors(bytes);
  const auto events = ParseTrackEvents(bytes);
  ASSERT_FALSE(descriptors.empty());
  ASSERT_FALSE(events.empty());

  std::map<std::string, uint64_t> uuid_by_name;
  std::vector<ParsedPerfettoTrackDescriptor> peu_descriptors;
  std::vector<ParsedPerfettoTrackDescriptor> slot_descriptors;
  for (const auto& descriptor : descriptors) {
    uuid_by_name[descriptor.name] = descriptor.uuid;
    if (descriptor.name == "PEU_00" || descriptor.name == "PEU_01" || descriptor.name == "PEU_02" ||
        descriptor.name == "PEU_03") {
      peu_descriptors.push_back(descriptor);
    }
    if (descriptor.name == "WAVE_SLOT_00" || descriptor.name == "WAVE_SLOT_01" ||
        descriptor.name == "WAVE_SLOT_02" || descriptor.name == "WAVE_SLOT_03") {
      slot_descriptors.push_back(descriptor);
    }
  }

  EXPECT_EQ(peu_descriptors.size(), 4u);
  EXPECT_EQ(slot_descriptors.size(), 16u);

  bool saw_wave_switch_away = false;
  for (const auto& event : events) {
    if (event.type == 3u && event.name == "wave_switch_away") {
      saw_wave_switch_away = true;
      break;
    }
  }
  EXPECT_TRUE(saw_wave_switch_away);
}

TEST(TracePerfettoTest, NativePerfettoProtoShowsSwitchAwayOnEveryCyclePeu) {
  const auto out_dir = MakeUniqueTempDir("gpu_model_perfetto_cycle_switch_every_peu");
  const struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() { std::filesystem::remove_all(path); }
  } cleanup{out_dir};

  TraceArtifactRecorder trace(out_dir, FullMarkerOptions());
  ExecEngine runtime(&trace);
  runtime.SetFixedGlobalMemoryLatency(20);

  constexpr uint32_t kBlockDim = 64 * 16;
  const auto kernel = BuildCycleMultiWaveWaitcntKernelForTraceTest();
  const uint64_t in_addr = runtime.memory().AllocateGlobal(kBlockDim * sizeof(int32_t));
  for (uint32_t i = 0; i < kBlockDim; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(in_addr + i * sizeof(int32_t),
                                               static_cast<int32_t>(100 + i));
  }

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = kBlockDim;
  request.args.PushU64(in_addr);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  trace.FlushTimeline();

  const std::string bytes =
      CycleTimelineRenderer::RenderPerfettoTraceProto(trace.recorder(), FullMarkerOptions());
  const auto descriptors = ParseTrackDescriptors(bytes);
  const auto events = ParseTrackEvents(bytes);
  ASSERT_FALSE(descriptors.empty());
  ASSERT_FALSE(events.empty());

  std::map<uint64_t, std::string> peu_name_by_uuid;
  std::map<uint64_t, uint64_t> slot_parent_uuid;
  for (const auto& descriptor : descriptors) {
    if (descriptor.name.rfind("PEU_", 0) == 0) {
      peu_name_by_uuid[descriptor.uuid] = descriptor.name;
    }
  }
  for (const auto& descriptor : descriptors) {
    if (descriptor.name.rfind("WAVE_SLOT_", 0) == 0 && descriptor.parent_uuid.has_value()) {
      slot_parent_uuid[descriptor.uuid] = *descriptor.parent_uuid;
    }
  }

  std::map<std::string, size_t> switch_counts_by_peu;
  for (const auto& event : events) {
    if (event.type != 3u || event.name != "wave_switch_away") {
      continue;
    }
    const auto slot_it = slot_parent_uuid.find(event.track_uuid);
    ASSERT_NE(slot_it, slot_parent_uuid.end());
    const auto peu_it = peu_name_by_uuid.find(slot_it->second);
    ASSERT_NE(peu_it, peu_name_by_uuid.end());
    ++switch_counts_by_peu[peu_it->second];
  }

  EXPECT_GE(switch_counts_by_peu["PEU_00"], 1u);
  EXPECT_GE(switch_counts_by_peu["PEU_01"], 1u);
  EXPECT_GE(switch_counts_by_peu["PEU_02"], 1u);
  EXPECT_GE(switch_counts_by_peu["PEU_03"], 1u);
}

// =============================================================================
// Functional Mode Slot Tests
// =============================================================================

TEST(TracePerfettoTest, NativePerfettoProtoShowsFunctionalLogicalUnboundedSlotsOnPeu0) {
  const auto out_dir = MakeUniqueTempDir("gpu_model_perfetto_st_same_peu_proto");
  const struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() { std::filesystem::remove_all(path); }
  } cleanup{out_dir};

  TraceArtifactRecorder trace(out_dir, FullMarkerOptions());
  ExecEngine runtime(&trace);
  runtime.SetFunctionalExecutionMode(FunctionalExecutionMode::SingleThreaded);
  runtime.SetFixedGlobalMemoryLatency(20);

  constexpr uint32_t kBlockDim = 64 * 33;
  const auto kernel = BuildSamePeuWaitcntSiblingKernel();
  const uint64_t in_addr = runtime.memory().AllocateGlobal(kBlockDim * sizeof(int32_t));
  const uint64_t out_addr = runtime.memory().AllocateGlobal(kBlockDim * sizeof(int32_t));
  for (uint32_t i = 0; i < kBlockDim; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(in_addr + i * sizeof(int32_t),
                                               static_cast<int32_t>(100 + i));
    runtime.memory().StoreGlobalValue<int32_t>(out_addr + i * sizeof(int32_t), -1);
  }

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Functional;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = kBlockDim;
  request.args.PushU64(in_addr);
  request.args.PushU64(out_addr);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  trace.FlushTimeline();

  const std::string timeline = ReadTextFile(out_dir / "timeline.perfetto.json");
  EXPECT_NE(timeline.find("\"slot_model\":\"logical_unbounded\""), std::string::npos);

  const std::string bytes =
      CycleTimelineRenderer::RenderPerfettoTraceProto(trace.recorder(), FullMarkerOptions());
  const auto descriptors = ParseTrackDescriptors(bytes);
  ASSERT_FALSE(descriptors.empty());

  std::optional<uint64_t> p0_uuid;
  size_t p0_count = 0;
  size_t p0_slot_count = 0;
  for (const auto& descriptor : descriptors) {
    if (descriptor.name == "PEU_00") {
      ++p0_count;
      p0_uuid = descriptor.uuid;
    }
  }
  ASSERT_TRUE(p0_uuid.has_value());
  for (const auto& descriptor : descriptors) {
    if (descriptor.name.rfind("WAVE_SLOT_", 0) == 0 && descriptor.parent_uuid == p0_uuid) {
      ++p0_slot_count;
    }
  }

  EXPECT_GE(p0_count, 1u);
  EXPECT_GE(p0_slot_count, 9u);
}

TEST(TracePerfettoTest, NativePerfettoProtoShowsMultiThreadedLogicalUnboundedSlotsOnPeu0) {
  const auto out_dir = MakeUniqueTempDir("gpu_model_perfetto_mt_same_peu_proto");
  const struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() { std::filesystem::remove_all(path); }
  } cleanup{out_dir};

  TraceArtifactRecorder trace(out_dir, FullMarkerOptions());
  ExecEngine runtime(&trace);
  runtime.SetFunctionalExecutionMode(FunctionalExecutionMode::MultiThreaded);
  runtime.SetFixedGlobalMemoryLatency(20);

  constexpr uint32_t kBlockDim = 64 * 33;
  const auto kernel = BuildSamePeuWaitcntSiblingKernel();
  const uint64_t in_addr = runtime.memory().AllocateGlobal(kBlockDim * sizeof(int32_t));
  const uint64_t out_addr = runtime.memory().AllocateGlobal(kBlockDim * sizeof(int32_t));
  for (uint32_t i = 0; i < kBlockDim; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(in_addr + i * sizeof(int32_t),
                                               static_cast<int32_t>(100 + i));
    runtime.memory().StoreGlobalValue<int32_t>(out_addr + i * sizeof(int32_t), -1);
  }

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Functional;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = kBlockDim;
  request.args.PushU64(in_addr);
  request.args.PushU64(out_addr);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  trace.FlushTimeline();

  const std::string timeline = ReadTextFile(out_dir / "timeline.perfetto.json");
  EXPECT_NE(timeline.find("\"slot_model\":\"logical_unbounded\""), std::string::npos);

  const std::string bytes =
      CycleTimelineRenderer::RenderPerfettoTraceProto(trace.recorder(), FullMarkerOptions());
  const auto descriptors = ParseTrackDescriptors(bytes);
  ASSERT_FALSE(descriptors.empty());

  std::optional<uint64_t> p0_uuid;
  size_t p0_count = 0;
  size_t p0_slot_count = 0;
  for (const auto& descriptor : descriptors) {
    if (descriptor.name == "PEU_00") {
      ++p0_count;
      p0_uuid = descriptor.uuid;
    }
  }
  ASSERT_TRUE(p0_uuid.has_value());
  for (const auto& descriptor : descriptors) {
    if (descriptor.name.rfind("WAVE_SLOT_", 0) == 0 && descriptor.parent_uuid == p0_uuid) {
      ++p0_slot_count;
    }
  }

  EXPECT_GE(p0_count, 1u);
  EXPECT_GE(p0_slot_count, 9u);
}

TEST(TracePerfettoTest, NativePerfettoProtoShowsFunctionalSamePeuSwitchAwayInSingleThreadedMode) {
  const auto out_dir = MakeUniqueTempDir("gpu_model_perfetto_st_same_peu_markers");
  const struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() { std::filesystem::remove_all(path); }
  } cleanup{out_dir};

  TraceArtifactRecorder trace(out_dir, FullMarkerOptions());
  ExecEngine runtime(&trace);
  runtime.SetFunctionalExecutionMode(FunctionalExecutionMode::SingleThreaded);
  runtime.SetFixedGlobalMemoryLatency(20);
  constexpr uint32_t kBlockDim = 64 * 33;
  const auto kernel = BuildCycleMultiWaveWaitcntKernelForTraceTest();
  const uint64_t in_addr = runtime.memory().AllocateGlobal(kBlockDim * sizeof(int32_t));
  for (uint32_t i = 0; i < kBlockDim; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(in_addr + i * sizeof(int32_t),
                                               static_cast<int32_t>(100 + i));
  }

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Functional;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = kBlockDim;
  request.args.PushU64(in_addr);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  trace.FlushTimeline();

  const std::string timeline = ReadTextFile(out_dir / "timeline.perfetto.json");
  EXPECT_NE(timeline.find("\"name\":\"wave_switch_away\""), std::string::npos);
  EXPECT_NE(timeline.find("\"slot_model\":\"logical_unbounded\""), std::string::npos);

  const std::string bytes =
      CycleTimelineRenderer::RenderPerfettoTraceProto(trace.recorder(), FullMarkerOptions());
  const auto events = ParseTrackEvents(bytes);
  bool saw_switch_away = false;
  for (const auto& event : events) {
    if (event.type != 3u) {
      continue;
    }
    saw_switch_away = saw_switch_away || event.name == "wave_switch_away";
  }
  EXPECT_TRUE(saw_switch_away);
}

TEST(TracePerfettoTest, NativePerfettoProtoShowsFunctionalSamePeuSwitchAwayInMultiThreadedMode) {
  const auto out_dir = MakeUniqueTempDir("gpu_model_perfetto_mt_same_peu_markers");
  const struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() { std::filesystem::remove_all(path); }
  } cleanup{out_dir};

  TraceArtifactRecorder trace(out_dir, FullMarkerOptions());
  ExecEngine runtime(&trace);
  runtime.SetFunctionalExecutionMode(FunctionalExecutionMode::MultiThreaded);
  constexpr uint32_t kBlockDim = 64 * 33;
  const auto kernel = BuildSamePeuWaitcntSiblingKernel();
  const uint64_t in_addr = runtime.memory().AllocateGlobal(kBlockDim * sizeof(int32_t));
  const uint64_t out_addr = runtime.memory().AllocateGlobal(kBlockDim * sizeof(int32_t));
  for (uint32_t i = 0; i < kBlockDim; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(in_addr + i * sizeof(int32_t),
                                               static_cast<int32_t>(100 + i));
    runtime.memory().StoreGlobalValue<int32_t>(out_addr + i * sizeof(int32_t), -1);
  }

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Functional;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = kBlockDim;
  request.args.PushU64(in_addr);
  request.args.PushU64(out_addr);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  trace.FlushTimeline();

  const std::string timeline = ReadTextFile(out_dir / "timeline.perfetto.json");
  EXPECT_NE(timeline.find("\"name\":\"wave_switch_away\""), std::string::npos);
  EXPECT_NE(timeline.find("\"slot_model\":\"logical_unbounded\""), std::string::npos);

  const std::string bytes =
      CycleTimelineRenderer::RenderPerfettoTraceProto(trace.recorder(), FullMarkerOptions());
  const auto events = ParseTrackEvents(bytes);
  bool saw_switch_away = false;
  for (const auto& event : events) {
    if (event.type != 3u) {
      continue;
    }
    saw_switch_away = saw_switch_away || event.name == "wave_switch_away";
  }
  EXPECT_TRUE(saw_switch_away);
}

TEST(TracePerfettoTest, NativePerfettoProtoShowsFunctionalTimelineGapWaitArriveInMultiThreadedMode) {
  const auto out_dir = MakeUniqueTempDir("gpu_model_perfetto_mt_timeline_gap_wait");
  const struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() { std::filesystem::remove_all(path); }
  } cleanup{out_dir};

  TraceArtifactRecorder trace(out_dir);
  ExecEngine runtime(&trace);
  runtime.SetFunctionalExecutionMode(FunctionalExecutionMode::MultiThreaded);

  const auto kernel = BuildWaitcntTraceKernel();
  const uint64_t base_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(base_addr, 11);

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Functional;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;
  request.args.PushU64(base_addr);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  trace.FlushTimeline();

  const std::string timeline = ReadTextFile(out_dir / "timeline.perfetto.json");
  EXPECT_NE(timeline.find("\"name\":\"stall_waitcnt_global\""), std::string::npos);
  EXPECT_NE(timeline.find("load_arrive"), std::string::npos);
  EXPECT_NE(timeline.find("\"slot_model\":\"logical_unbounded\""), std::string::npos);

  const std::string bytes = CycleTimelineRenderer::RenderPerfettoTraceProto(trace.recorder());
  const auto events = ParseTrackEvents(bytes);
  bool saw_waitcnt_stall = false;
  bool saw_load_arrive = false;
  for (const auto& event : events) {
    if (event.type != 3u) {
      continue;
    }
    saw_waitcnt_stall = saw_waitcnt_stall || event.name == "stall_waitcnt_global";
    saw_load_arrive = saw_load_arrive || event.name == "load_arrive" ||
                      event.name.starts_with("load_arrive_");
  }
  EXPECT_TRUE(saw_waitcnt_stall);
  EXPECT_TRUE(saw_load_arrive);
}

// =============================================================================
// Waitcnt Arrival Progress Tests
// =============================================================================

TEST(TracePerfettoTest, NativePerfettoProtoShowsWaitcntArrivalProgressMarkersAndBubble) {
  const auto out_dir = MakeUniqueTempDir("gpu_model_perfetto_waitcnt_arrival_progress");
  const struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() { std::filesystem::remove_all(path); }
  } cleanup{out_dir};

  TraceArtifactRecorder trace(out_dir);
  ExecEngine runtime(&trace);
  runtime.SetFunctionalExecutionMode(FunctionalExecutionMode::SingleThreaded);
  runtime.SetFixedGlobalMemoryLatency(20);

  const auto kernel = BuildWaitcntThresholdProgressKernel();
  const uint64_t base_addr = runtime.memory().AllocateGlobal(3 * sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(base_addr + 0 * sizeof(int32_t), 11);
  runtime.memory().StoreGlobalValue<int32_t>(base_addr + 1 * sizeof(int32_t), 13);
  runtime.memory().StoreGlobalValue<int32_t>(base_addr + 2 * sizeof(int32_t), 17);

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Functional;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;
  request.args.PushU64(base_addr);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  trace.FlushTimeline();

  const std::string timeline = ReadTextFile(out_dir / "timeline.perfetto.json");
  EXPECT_NE(timeline.find("\"name\":\"stall_waitcnt_global\""), std::string::npos) << timeline;
  EXPECT_NE(timeline.find("\"name\":\"load_arrive_resume\""), std::string::npos) << timeline;
  EXPECT_EQ(timeline.find("\"name\":\"s_waitcnt\""), std::string::npos) << timeline;

  const std::string bytes = CycleTimelineRenderer::RenderPerfettoTraceProto(trace.recorder());
  const auto events = ParseTrackEvents(bytes);
  size_t stall_index = std::numeric_limits<size_t>::max();
  size_t still_blocked_index = std::numeric_limits<size_t>::max();
  size_t resume_index = std::numeric_limits<size_t>::max();
  for (size_t i = 0; i < events.size(); ++i) {
    if (events[i].type != 3u) {
      continue;
    }
    if (events[i].name == "stall_waitcnt_global" &&
        stall_index == std::numeric_limits<size_t>::max()) {
      stall_index = i;
    } else if (events[i].name == "load_arrive_still_blocked" &&
               still_blocked_index == std::numeric_limits<size_t>::max()) {
      still_blocked_index = i;
    } else if (events[i].name == "load_arrive_resume" &&
               resume_index == std::numeric_limits<size_t>::max()) {
      resume_index = i;
    }
  }
  ASSERT_NE(stall_index, std::numeric_limits<size_t>::max());
  ASSERT_NE(resume_index, std::numeric_limits<size_t>::max());
  if (still_blocked_index != std::numeric_limits<size_t>::max()) {
    EXPECT_LT(stall_index, still_blocked_index);
    EXPECT_LT(still_blocked_index, resume_index);
  }
  EXPECT_LT(stall_index, resume_index);
}

}  // namespace
}  // namespace gpu_model
