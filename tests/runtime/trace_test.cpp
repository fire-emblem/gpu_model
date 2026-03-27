#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

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

TEST(TraceTest, WritesHumanReadableTraceFile) {
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / "gpu_model_trace.txt";
  {
    FileTraceSink trace(path);
    HostRuntime runtime(&trace);

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
    HostRuntime runtime(&trace);

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

}  // namespace
}  // namespace gpu_model
