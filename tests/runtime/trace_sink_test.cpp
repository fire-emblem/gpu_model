#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include "debug/trace/artifact_recorder.h"
#include "debug/trace/event_factory.h"
#include "debug/trace/sink.h"
#include "instruction/isa/instruction_builder.h"
#include "runtime/exec_engine/exec_engine.h"
#include "tests/test_utils/trace_test_support.h"
#include "gpu_arch/occupancy/occupancy_calculator.h"

namespace gpu_model {
namespace {

using test::ReadTextFile;
using test::ExpectContainsTypedSlotFields;
using test::ExpectContainsTypedSlotFieldsJson;
using test::ExpectContainsTypedStallReasonFields;
using test::ExpectContainsTypedStallReasonFieldsJson;

// =============================================================================
// FileTraceSink Tests
// =============================================================================

TEST(TraceSinkTest, WritesHumanReadableTraceFile) {
  const std::filesystem::path dir =
      std::filesystem::temp_directory_path() / "gpu_model_trace_structured";
  {
    TraceArtifactRecorder trace(dir);
    ExecEngine runtime(&trace);

    InstructionBuilder builder;
    builder.BExit();
    const auto kernel = builder.Build("file_trace_kernel");

    LaunchRequest request;
    request.kernel = &kernel;
    request.config.grid_dim_x = 1;
    request.config.block_dim_x = 64;

    const auto result = runtime.Launch(request);
    ASSERT_TRUE(result.ok);
    trace.FlushTimeline();
  }

  const std::filesystem::path text_path = dir / "trace.txt";
  std::ifstream input(text_path);
  ASSERT_TRUE(static_cast<bool>(input));
  std::ostringstream buffer;
  buffer << input.rdbuf();
  const std::string text = buffer.str();
  // Verify structured trace output
  EXPECT_NE(text.find("GPU_MODEL TRACE"), std::string::npos);
  EXPECT_NE(text.find("[RUN]"), std::string::npos);
  EXPECT_NE(text.find("[EVENTS]"), std::string::npos);
  EXPECT_NE(text.find("[SUMMARY]"), std::string::npos);
  // New format: event kind is just the name, not "kind=Launch"
  EXPECT_NE(text.find("launch"), std::string::npos);
  EXPECT_NE(text.find("wave_exit"), std::string::npos);
  std::filesystem::remove_all(dir);
}

TEST(TraceSinkTest, WritesWaveStatsEventsToTraceSinks) {
  const std::filesystem::path text_path =
      std::filesystem::temp_directory_path() / "gpu_model_wave_stats_trace.txt";
  const std::filesystem::path json_path =
      std::filesystem::temp_directory_path() / "gpu_model_wave_stats_trace.jsonl";

  {
    FileTraceSink text_trace(text_path);
    JsonTraceSink json_trace(json_path);
    TraceEvent event{
        .kind = TraceEventKind::WaveStats,
        .cycle = 7,
        .slot_model = {},
        .waitcnt_state = {},
        .has_cycle_range = false,
        .range_end_cycle = 0,
        .semantic_canonical_name = {},
        .semantic_presentation_name = {},
        .semantic_category = {},
        .display_name = {},
        .message = "launch=2 init=2 active=2 end=0",
    };
    text_trace.OnEvent(event);
    json_trace.OnEvent(event);
  }

  std::ifstream text_in(text_path);
  std::ifstream json_in(json_path);
  std::string text_line;
  std::string json_line;
  ASSERT_TRUE(static_cast<bool>(std::getline(text_in, text_line)));
  ASSERT_TRUE(static_cast<bool>(std::getline(json_in, json_line)));
  // New format: event kind is just the name
  EXPECT_NE(text_line.find("wave_stats"), std::string::npos);
  EXPECT_NE(json_line.find("\"kind\":\"WaveStats\""), std::string::npos);
  std::filesystem::remove(text_path);
  std::filesystem::remove(json_path);
}

TEST(TraceSinkTest, PrefersTypedSchemaFieldsWhenCompatibilityStringsAreEmpty) {
  const std::filesystem::path text_path =
      std::filesystem::temp_directory_path() / "gpu_model_typed_trace.txt";
  const std::filesystem::path json_path =
      std::filesystem::temp_directory_path() / "gpu_model_typed_trace.jsonl";

  {
    FileTraceSink text_trace(text_path);
    JsonTraceSink json_trace(json_path);
    TraceEvent event{
        .kind = TraceEventKind::Stall,
        .cycle = 11,
        .slot_model_kind = TraceSlotModelKind::LogicalUnbounded,
        .slot_model = {},
        .stall_reason = TraceStallReason::WaitCntGlobal,
        .waitcnt_state = {},
        .has_cycle_range = false,
        .range_end_cycle = 0,
        .semantic_canonical_name = {},
        .semantic_presentation_name = {},
        .semantic_category = {},
        .display_name = {},
        .message = {},
    };
    text_trace.OnEvent(event);
    json_trace.OnEvent(event);
  }

  std::ifstream text_in(text_path);
  std::ifstream json_in(json_path);
  std::string text_line;
  std::string json_line;
  ASSERT_TRUE(static_cast<bool>(std::getline(text_in, text_line)));
  ASSERT_TRUE(static_cast<bool>(std::getline(json_in, json_line)));
  ExpectContainsTypedSlotFields(text_line, "logical_unbounded");
  ExpectContainsTypedStallReasonFields(text_line, "waitcnt_global");
  ExpectContainsTypedSlotFieldsJson(json_line, "logical_unbounded");
  ExpectContainsTypedStallReasonFieldsJson(json_line, "waitcnt_global");
  std::filesystem::remove(text_path);
  std::filesystem::remove(json_path);
}

TEST(TraceSinkTest, FileSinkSerializesCanonicalTypedSubkinds) {
  const std::filesystem::path text_path =
      std::filesystem::temp_directory_path() / "gpu_model_trace_canonical.txt";

  {
    FileTraceSink sink(text_path);
    TraceEvent event{
        .kind = TraceEventKind::Barrier,
        .cycle = 3,
        .slot_model = {},
        .barrier_kind = TraceBarrierKind::Release,
        .waitcnt_state = {},
        .has_cycle_range = false,
        .range_end_cycle = 0,
        .semantic_canonical_name = {},
        .semantic_presentation_name = {},
        .semantic_category = {},
        .display_name = "release",
        .message = "release",
    };
    sink.OnEvent(event);
  }

  const std::string text = ReadTextFile(text_path);
  // New format: [cycle]   kind   wave   pc   details
  // barrier_release is the canonical name, release is display_name
  EXPECT_NE(text.find("barrier_release"), std::string::npos);
  EXPECT_NE(text.find("release"), std::string::npos);
  std::filesystem::remove(text_path);
}

TEST(TraceSinkTest, FileSinkSerializesFlowMetadata) {
  const auto temp_dir = test::MakeUniqueTempDir("gpu_model_flow_trace_text");
  const struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() { std::filesystem::remove_all(path); }
  } cleanup{temp_dir};
  const auto text_path = temp_dir / "trace.txt";

  {
    FileTraceSink trace(text_path);
    TraceEvent event{
        .kind = TraceEventKind::MemoryAccess,
        .cycle = 5,
        .slot_model_kind = TraceSlotModelKind::ResidentFixed,
        .slot_model = {},
        .waitcnt_state = {},
        .has_cycle_range = false,
        .range_end_cycle = 0,
        .semantic_canonical_name = {},
        .semantic_presentation_name = {},
        .semantic_category = {},
        .display_name = {},
        .message = "flow_event",
        .flow_id = 1,
        .flow_phase = TraceFlowPhase::Start,
    };
    trace.OnEvent(event);
  }

  const std::string text = ReadTextFile(text_path);
  // New format: [cycle]   kind   wave   pc   details
  // Flow metadata is in JSON, text format just shows the event
  EXPECT_NE(text.find("memory_access"), std::string::npos);
}

// =============================================================================
// JsonTraceSink Tests
// =============================================================================

TEST(TraceSinkTest, WritesJsonTraceFile) {
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / "gpu_model_trace.jsonl";
  {
    JsonTraceSink trace(path);
    ExecEngine runtime(&trace);

    InstructionBuilder builder;
    builder.BExit();
    const auto kernel = builder.Build("json_trace_kernel");

    LaunchRequest request;
    request.kernel = &kernel;
    request.config.grid_dim_x = 1;
    request.config.block_dim_x = 64;

    const auto result = runtime.Launch(request);
    ASSERT_TRUE(result.ok);
  }

  std::ifstream input(path);
  ASSERT_TRUE(static_cast<bool>(input));
  std::string line;
  ASSERT_TRUE(static_cast<bool>(std::getline(input, line)));
  EXPECT_NE(line.find("\"kind\":\"Launch\""), std::string::npos);
  EXPECT_NE(line.find("\"pc\":\"0x0\""), std::string::npos);
  EXPECT_NE(line.find("\"message\":\"kernel=json_trace_kernel arch=mac500\""), std::string::npos);
  std::filesystem::remove(path);
}

TEST(TraceSinkTest, JsonSinkSerializesCanonicalTypedSubkinds) {
  const std::filesystem::path json_path =
      std::filesystem::temp_directory_path() / "gpu_model_trace_canonical.jsonl";

  {
    JsonTraceSink sink(json_path);
    TraceEvent event{
        .kind = TraceEventKind::Arrive,
        .cycle = 4,
        .slot_model = {},
        .arrive_kind = TraceArriveKind::Shared,
        .waitcnt_state = {},
        .has_cycle_range = false,
        .range_end_cycle = 0,
        .semantic_canonical_name = {},
        .semantic_presentation_name = {},
        .semantic_category = {},
        .display_name = "shared",
        .message = "shared_arrive",
    };
    sink.OnEvent(event);
  }

  const std::string text = ReadTextFile(json_path);
  EXPECT_NE(text.find("\"arrive_kind\":\"shared\""), std::string::npos);
  EXPECT_NE(text.find("\"canonical_name\":\"shared_arrive\""), std::string::npos);
  EXPECT_NE(text.find("\"presentation_name\":\"shared_arrive\""), std::string::npos);
  EXPECT_NE(text.find("\"category\":\"memory/shared_arrive\""), std::string::npos);
  EXPECT_NE(text.find("\"display_name\":\"shared\""), std::string::npos);
  std::filesystem::remove(json_path);
}

TEST(TraceSinkTest, JsonSinkSerializesFlowMetadata) {
  const auto temp_dir = test::MakeUniqueTempDir("gpu_model_flow_trace_json");
  const struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() { std::filesystem::remove_all(path); }
  } cleanup{temp_dir};
  const auto json_path = temp_dir / "trace.jsonl";

  {
    JsonTraceSink trace(json_path);
    TraceEvent event{
        .kind = TraceEventKind::MemoryAccess,
        .cycle = 5,
        .slot_model_kind = TraceSlotModelKind::ResidentFixed,
        .slot_model = {},
        .waitcnt_state = {},
        .has_cycle_range = false,
        .range_end_cycle = 0,
        .semantic_canonical_name = {},
        .semantic_presentation_name = {},
        .semantic_category = {},
        .display_name = {},
        .message = "flow_event",
        .flow_id = 1,
        .flow_phase = TraceFlowPhase::Start,
    };
    trace.OnEvent(event);
  }

  const std::string line = ReadTextFile(json_path);
  EXPECT_NE(line.find("\"has_flow\":true"), std::string::npos);
  EXPECT_NE(line.find("\"flow_id\":\"0x1\""), std::string::npos);
  EXPECT_NE(line.find("\"flow_phase\":\"start\""), std::string::npos);
}

TEST(TraceSinkTest, JsonSinkSkipsFlowMetadataWhenNoFlow) {
  const auto temp_dir = test::MakeUniqueTempDir("gpu_model_flow_trace_json");
  const struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() { std::filesystem::remove_all(path); }
  } cleanup{temp_dir};
  const auto json_path = temp_dir / "trace.jsonl";

  {
    JsonTraceSink trace(json_path);
    TraceEvent event{
        .kind = TraceEventKind::MemoryAccess,
        .cycle = 6,
        .slot_model_kind = TraceSlotModelKind::ResidentFixed,
        .slot_model = {},
        .waitcnt_state = {},
        .has_cycle_range = false,
        .range_end_cycle = 0,
        .semantic_canonical_name = {},
        .semantic_presentation_name = {},
        .semantic_category = {},
        .display_name = {},
        .message = "flowless_event",
    };
    trace.OnEvent(event);
  }

  const std::string line = ReadTextFile(json_path);
  EXPECT_EQ(line.find("\"has_flow\""), std::string::npos);
  EXPECT_EQ(line.find("\"flow_id\""), std::string::npos);
  EXPECT_EQ(line.find("\"flow_phase\""), std::string::npos);
}

// =============================================================================
// Waitcnt Metadata Serialization
// =============================================================================

TEST(TraceSinkTest, JsonSinkSerializesWaitcntMetadataFields) {
  const TraceWaveView wave{
      .dpc_id = 0,
      .ap_id = 0,
      .peu_id = 0,
      .slot_id = 1,
      .block_id = 2,
      .wave_id = 3,
      .pc = 4,
  };
  const TraceEvent event = MakeTraceWaitStallEvent(
      wave,
      /*cycle=*/9,
      TraceStallReason::WaitCntGlobal,
      TraceSlotModelKind::LogicalUnbounded,
      std::numeric_limits<uint64_t>::max(),
      TraceWaitcntState{.valid = true,
                        .threshold_global = 0,
                        .threshold_shared = 0,
                        .threshold_private = UINT32_MAX,
                        .threshold_scalar_buffer = UINT32_MAX,
                        .pending_global = 2,
                        .pending_shared = 1,
                        .pending_private = 0,
                        .pending_scalar_buffer = 0,
                        .blocked_global = true,
                        .blocked_shared = true});

  const auto temp_dir = test::MakeUniqueTempDir("gpu_model_waitcnt_trace_json");
  const struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() { std::filesystem::remove_all(path); }
  } cleanup{temp_dir};
  const auto temp_path = temp_dir / "trace.json";

  {
    JsonTraceSink sink(temp_path);
    sink.OnEvent(event);
  }

  const std::string line = ReadTextFile(temp_path);
  EXPECT_NE(line.find("\"waitcnt_thresholds\":\"g=0 s=0 p=* sb=*\""), std::string::npos);
  EXPECT_NE(line.find("\"waitcnt_pending\":\"g=2 s=1 p=0 sb=0\""), std::string::npos);
  EXPECT_NE(line.find("\"waitcnt_blocked_domains\":\"global|shared\""), std::string::npos);
  EXPECT_NE(line.find("\"presentation_name\":\"stall_waitcnt_global_shared\""), std::string::npos);
}

TEST(TraceSinkTest, JsonSinkSerializesWaitcntArrivalProgressFields) {
  TraceEvent arrive{
      .kind = TraceEventKind::Arrive,
      .cycle = 18,
      .dpc_id = 0,
      .ap_id = 0,
      .peu_id = 0,
      .slot_id = 1,
      .slot_model_kind = TraceSlotModelKind::None,
      .slot_model = {},
      .block_id = 0,
      .wave_id = 0,
      .pc = 0x40,
      .arrive_kind = TraceArriveKind::Load,
      .arrive_progress = TraceArriveProgressKind::StillBlocked,
      .waitcnt_state =
          TraceWaitcntState{
              .valid = true,
              .threshold_global = 0,
              .threshold_shared = UINT32_MAX,
              .threshold_private = UINT32_MAX,
              .threshold_scalar_buffer = UINT32_MAX,
              .pending_global = 1,
              .pending_shared = 0,
              .pending_private = 0,
              .pending_scalar_buffer = 0,
              .blocked_global = true,
          },
      .has_cycle_range = false,
      .range_end_cycle = 0,
      .semantic_canonical_name = {},
      .semantic_presentation_name = {},
      .semantic_category = {},
      .display_name = "load",
      .message = "load_arrive",
  };
  arrive.waitcnt_state.has_pending_before = true;
  arrive.waitcnt_state.pending_before_global = 2;

  const auto temp_dir = test::MakeUniqueTempDir("gpu_model_waitcnt_arrive_progress_json");
  {
    JsonTraceSink sink(temp_dir / "trace.jsonl");
    sink.OnEvent(arrive);
  }

  const std::string line = ReadTextFile(temp_dir / "trace.jsonl");
  EXPECT_NE(line.find("\"canonical_name\":\"load_arrive_still_blocked\""), std::string::npos);
  EXPECT_NE(line.find("\"waitcnt_pending_before\":\"g=2 s=0 p=0 sb=0\""), std::string::npos);
  EXPECT_NE(line.find("\"waitcnt_pending\":\"g=1 s=0 p=0 sb=0\""), std::string::npos);
  EXPECT_NE(line.find("\"waitcnt_pending_transition\":\"g=2->1 s=0->0 p=0->0 sb=0->0\""),
            std::string::npos);
}

// =============================================================================
// Occupancy in Trace Output
// =============================================================================

TEST(TraceSinkTest, OccupancySectionAppearsWhenResourceMetadataPresent) {
  const auto dir =
      std::filesystem::temp_directory_path() / "gpu_model_trace_occupancy";
  const struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() { std::filesystem::remove_all(path); }
  } cleanup{dir};

  {
    TraceArtifactRecorder trace(dir);
    ExecEngine runtime(&trace);

    InstructionBuilder builder;
    builder.BExit();
    auto kernel = builder.Build("occ_test_kernel");
    // Add resource metadata so occupancy is computed
    auto& meta = const_cast<MetadataBlob&>(kernel.metadata());
    meta.values["vgpr_count"] = "16";
    meta.values["sgpr_count"] = "16";

    LaunchRequest request;
    request.kernel = &kernel;
    request.config.grid_dim_x = 1;
    request.config.block_dim_x = 64;

    const auto result = runtime.Launch(request);
    ASSERT_TRUE(result.ok) << "Launch failed: " << result.error_message;
    trace.FlushTimeline();
  }

  const std::string text = ReadTextFile(dir / "trace.txt");
  EXPECT_NE(text.find("[OCCUPANCY]"), std::string::npos);
  EXPECT_NE(text.find("theoretical_max_waves_per_peu="), std::string::npos);
  EXPECT_NE(text.find("theoretical_occupancy_pct="), std::string::npos);
  EXPECT_NE(text.find("occupancy_wave_limiter="), std::string::npos);

  const std::string json = ReadTextFile(dir / "trace.jsonl");
  EXPECT_NE(json.find("\"theoretical_max_waves_per_peu\":"), std::string::npos);
  EXPECT_NE(json.find("\"theoretical_occupancy_pct\":"), std::string::npos);
}

}  // namespace
}  // namespace gpu_model
