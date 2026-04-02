#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>

#include "gpu_model/debug/trace_artifact_recorder.h"
#include "gpu_model/debug/trace_sink.h"
#include "gpu_model/isa/instruction_builder.h"
#include "gpu_model/runtime/runtime_engine.h"

namespace gpu_model {
namespace {

ExecutableKernel BuildWaitcntTraceKernel() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SMov("s1", 0);
  builder.MLoadGlobal("v1", "s0", "s1", 4);
  builder.SWaitCnt(/*global_count=*/0, /*shared_count=*/UINT32_MAX,
                   /*private_count=*/UINT32_MAX, /*scalar_buffer_count=*/UINT32_MAX);
  builder.SMov("s2", 7);
  builder.BExit();
  return builder.Build("trace_waitcnt_kernel");
}

std::filesystem::path MakeUniqueTempDir(const std::string& stem) {
  const auto suffix =
      std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
  const auto path = std::filesystem::temp_directory_path() / (stem + "_" + suffix);
  std::filesystem::remove_all(path);
  std::filesystem::create_directories(path);
  return path;
}

std::string ReadTextFile(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("failed to read text file: " + path.string());
  }
  std::ostringstream buffer;
  buffer << input.rdbuf();
  return buffer.str();
}

std::string_view ExtractTraceEventsPayload(std::string_view text) {
  const auto key_pos = text.find("\"traceEvents\"");
  if (key_pos == std::string_view::npos) {
    return {};
  }
  const auto array_begin = text.find('[', key_pos);
  const auto array_end = text.rfind(']');
  if (array_begin == std::string_view::npos || array_end == std::string_view::npos ||
      array_end <= array_begin) {
    return {};
  }
  return text.substr(array_begin + 1, array_end - array_begin - 1);
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

size_t FindFirst(std::string_view text, std::string_view needle) {
  return text.find(needle);
}

TEST(TraceTest, EmitsLaunchAndBlockPlacementEvents) {
  CollectingTraceSink trace;
  RuntimeEngine runtime(&trace);

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

TEST(TraceTest, EmitsWaveLaunchEventWithInitialWaveStateSummary) {
  CollectingTraceSink trace;
  RuntimeEngine runtime(&trace);

  InstructionBuilder builder;
  builder.BExit();
  const auto kernel = builder.Build("wave_launch_trace_kernel");

  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok);

  bool saw_wave_launch = false;
  for (const auto& event : trace.events()) {
    if (event.kind != TraceEventKind::WaveLaunch) {
      continue;
    }
    saw_wave_launch = true;
    EXPECT_NE(event.message.find("lanes=0x40"), std::string::npos);
    EXPECT_NE(event.message.find("exec=0xffffffffffffffff"), std::string::npos);
    EXPECT_NE(event.message.find("sgpr={"), std::string::npos);
    EXPECT_NE(event.message.find("vgpr={"), std::string::npos);
    EXPECT_TRUE(event.message.find("s0=") != std::string::npos ||
                event.message.find("kernarg_ptr=") != std::string::npos);
    }
  EXPECT_TRUE(saw_wave_launch);
}

TEST(TraceTest, EmitsWaveStatsSnapshotsForFunctionalLaunch) {
  CollectingTraceSink trace;
  RuntimeEngine runtime(&trace);
  runtime.SetFunctionalExecutionMode(FunctionalExecutionMode::MultiThreaded);

  InstructionBuilder builder;
  builder.BExit();
  const auto kernel = builder.Build("wave_stats_trace_kernel");

  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 2;
  request.config.block_dim_x = 64;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok);

  std::vector<std::string> wave_stats_messages;
  for (const auto& event : trace.events()) {
    if (event.kind == TraceEventKind::WaveStats) {
      wave_stats_messages.push_back(event.message);
    }
  }

  constexpr const char* kInitial =
      "launch=2 init=2 active=2 runnable=2 waiting=0 end=0";
  constexpr const char* kIntermediate =
      "launch=2 init=2 active=1 runnable=1 waiting=0 end=1";
  constexpr const char* kFinal =
      "launch=2 init=2 active=0 runnable=0 waiting=0 end=2";
  ASSERT_EQ(wave_stats_messages.size(), 4u);
  EXPECT_EQ(wave_stats_messages.front(), kInitial);
  EXPECT_EQ(wave_stats_messages.back(), kFinal);

  for (size_t i = 1; i + 1 < wave_stats_messages.size(); ++i) {
    EXPECT_TRUE(wave_stats_messages[i] == kIntermediate ||
                wave_stats_messages[i] == kFinal);
  }

  const size_t final_count =
      std::count(wave_stats_messages.begin(), wave_stats_messages.end(), kFinal);
  EXPECT_GE(final_count, 2u);
  EXPECT_LE(final_count, 3u);
}

TEST(TraceTest, EmitsWaveStatsStateSplitForFunctionalLaunch) {
  CollectingTraceSink trace;
  RuntimeEngine runtime(&trace);
  runtime.SetFunctionalExecutionMode(FunctionalExecutionMode::MultiThreaded);

  InstructionBuilder builder;
  builder.BExit();
  const auto kernel = builder.Build("wave_stats_state_split_kernel");

  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 2;
  request.config.block_dim_x = 64;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok);

  std::vector<std::string> messages;
  for (const auto& event : trace.events()) {
    if (event.kind == TraceEventKind::WaveStats) {
      messages.push_back(event.message);
    }
  }

  ASSERT_FALSE(messages.empty());
  EXPECT_EQ(messages.front(), "launch=2 init=2 active=2 runnable=2 waiting=0 end=0");
  EXPECT_EQ(messages.back(), "launch=2 init=2 active=0 runnable=0 waiting=0 end=2");
}

TEST(TraceTest, EmitsUnifiedWaitStateMachineTraceForWaitcnt) {
  CollectingTraceSink trace;
  RuntimeEngine runtime(&trace);
  runtime.SetFunctionalExecutionMode(FunctionalExecutionMode::SingleThreaded);

  const auto kernel = BuildWaitcntTraceKernel();
  const uint64_t base_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(base_addr, 11);

  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;
  request.args.PushU64(base_addr);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok);

  bool saw_waiting_snapshot = false;
  bool saw_waitcnt_stall = false;
  for (const auto& event : trace.events()) {
    if (event.kind == TraceEventKind::WaveStats &&
        event.message.find("waiting=1") != std::string::npos) {
      saw_waiting_snapshot = true;
    }
    if (event.kind == TraceEventKind::Stall && event.message == "waitcnt_global") {
      saw_waitcnt_stall = true;
    }
  }

  EXPECT_TRUE(saw_waiting_snapshot);
  EXPECT_TRUE(saw_waitcnt_stall);
}

TEST(TraceTest, WritesHumanReadableTraceFile) {
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / "gpu_model_trace.txt";
  {
    FileTraceSink trace(path);
    RuntimeEngine runtime(&trace);

    InstructionBuilder builder;
    builder.BExit();
    const auto kernel = builder.Build("file_trace_kernel");

    LaunchRequest request;
    request.kernel = &kernel;
    request.config.grid_dim_x = 1;
    request.config.block_dim_x = 64;

    const auto result = runtime.Launch(request);
    ASSERT_TRUE(result.ok);
  }

  std::ifstream input(path);
  ASSERT_TRUE(static_cast<bool>(input));
  std::ostringstream buffer;
  buffer << input.rdbuf();
  const std::string text = buffer.str();
  EXPECT_NE(text.find("kind=Launch"), std::string::npos);
  EXPECT_NE(text.find("kind=WaveExit"), std::string::npos);
  EXPECT_NE(text.find("pc=0x0"), std::string::npos);
  std::filesystem::remove(path);
}

TEST(TraceTest, WritesJsonTraceFile) {
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / "gpu_model_trace.jsonl";
  {
    JsonTraceSink trace(path);
    RuntimeEngine runtime(&trace);

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
  EXPECT_NE(line.find("\"message\":\"kernel=json_trace_kernel arch=c500\""), std::string::npos);
  std::filesystem::remove(path);
}

TEST(TraceTest, WritesWaveStatsEventsToTraceSinks) {
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
  EXPECT_NE(text_line.find("kind=WaveStats"), std::string::npos);
  EXPECT_NE(text_line.find("msg=launch=2 init=2 active=2 end=0"), std::string::npos);
  EXPECT_NE(json_line.find("\"kind\":\"WaveStats\""), std::string::npos);
  EXPECT_NE(json_line.find("\"message\":\"launch=2 init=2 active=2 end=0\""), std::string::npos);
  std::filesystem::remove(text_path);
  std::filesystem::remove(json_path);
}

TEST(TraceTest, PerfettoDumpContainsTraceEventsAndRequiredFields) {
  const auto out_dir = MakeUniqueTempDir("gpu_model_perfetto_structure");
  const struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() { std::filesystem::remove_all(path); }
  } cleanup{out_dir};

  TraceArtifactRecorder trace(out_dir);
  RuntimeEngine runtime(&trace);

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
}

TEST(TraceTest, PerfettoDumpForMultiThreadedWaitKernelShowsWaitingAndResumeOrdering) {
  const auto out_dir = MakeUniqueTempDir("gpu_model_perfetto_mt_wait");
  const struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() { std::filesystem::remove_all(path); }
  } cleanup{out_dir};

  TraceArtifactRecorder trace(out_dir);
  RuntimeEngine runtime(&trace);
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
  EXPECT_NE(FindFirst(trace_events, "\"args\":{\"name\":\"B0W0\"}"), std::string::npos)
      << timeline;
  EXPECT_NE(FindFirst(trace_events, "\"args\":{\"name\":\"B1W0\"}"), std::string::npos)
      << timeline;
  EXPECT_NE(FindFirst(trace_events, "\"name\":\"stall_waitcnt_global\""), std::string::npos)
      << timeline;
  EXPECT_NE(FindFirst(trace_events, "\"name\":\"wave_exit\""), std::string::npos) << timeline;
  EXPECT_LT(FindFirst(trace_events, "\"name\":\"stall_waitcnt_global\""),
            FindFirst(trace_events, "\"name\":\"wave_exit\""))
      << timeline;
}

TEST(TraceTest, TraceArtifactRecorderWritesTraceAndPerfettoFiles) {
  const auto out_dir =
      std::filesystem::temp_directory_path() / "gpu_model_trace_artifact_recorder";
  std::filesystem::remove_all(out_dir);
  std::filesystem::create_directories(out_dir);

  {
    TraceArtifactRecorder trace(out_dir);
    trace.OnEvent(TraceEvent{
        .kind = TraceEventKind::Launch,
        .cycle = 0,
        .message = "kernel=artifact_trace arch=c500",
    });
    trace.OnEvent(TraceEvent{
        .kind = TraceEventKind::WaveLaunch,
        .cycle = 0,
        .block_id = 0,
        .wave_id = 0,
        .message = "lanes=0x40 exec=0xffffffffffffffff",
    });
    trace.OnEvent(TraceEvent{
        .kind = TraceEventKind::WaveStep,
        .cycle = 1,
        .block_id = 0,
        .wave_id = 0,
        .message = "op=v_add_i32",
    });
    trace.OnEvent(TraceEvent{
        .kind = TraceEventKind::Commit,
        .cycle = 4,
        .block_id = 0,
        .wave_id = 0,
        .message = "op=v_add_i32",
    });
    trace.OnEvent(TraceEvent{
        .kind = TraceEventKind::WaveExit,
        .cycle = 5,
        .block_id = 0,
        .wave_id = 0,
        .message = "done",
    });
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

  EXPECT_NE(text_buffer.str().find("kind=Launch"), std::string::npos);
  EXPECT_NE(json_buffer.str().find("\"kind\":\"Launch\""), std::string::npos);
  EXPECT_NE(timeline_buffer.str().find("\"traceEvents\""), std::string::npos);

  std::filesystem::remove_all(out_dir);
}

}  // namespace
}  // namespace gpu_model
