#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "gpu_model/debug/trace/event_factory.h"
#include "gpu_model/debug/trace/event_view.h"
#include "gpu_model/debug/trace/sink.h"
#include "gpu_model/isa/instruction_builder.h"
#include "gpu_model/runtime/exec_engine.h"
#include "tests/test_utils/trace_test_support.h"

namespace gpu_model {
namespace {

using test::BuildWaitcntTraceKernel;
using test::BuildDenseScalarIssueKernel;
using test::BuildSamePeuWaitcntSiblingKernel;
using test::BuildCycleMultiWaveWaitcntKernelForTraceTest;

// =============================================================================
// Wave Launch and State Summary Events
// =============================================================================

TEST(TraceWaveStatsTest, EmitsWaveLaunchEventWithInitialWaveStateSummary) {
  CollectingTraceSink trace;
  ExecEngine runtime(&trace);

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
  bool saw_state_summary = false;
  for (const auto& event : trace.events()) {
    if (event.kind == TraceEventKind::WaveLaunch) {
      saw_wave_launch = true;
    }
    if (event.kind == TraceEventKind::WaveStats) {
      saw_state_summary = true;
    }
  }

  EXPECT_TRUE(saw_wave_launch);
  EXPECT_TRUE(saw_state_summary);
}

TEST(TraceWaveStatsTest, CycleExecutionEmitsCanonicalLifecycleAndStallMessagesViaFactories) {
  CollectingTraceSink trace;
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

  bool saw_wave_start = false;
  bool saw_wave_end = false;
  bool saw_switch = false;
  for (const auto& event : trace.events()) {
    saw_wave_start = saw_wave_start || (event.kind == TraceEventKind::WaveLaunch &&
                                        event.lifecycle_stage == TraceLifecycleStage::Launch);
    saw_wave_end = saw_wave_end || (event.kind == TraceEventKind::WaveExit &&
                                    event.lifecycle_stage == TraceLifecycleStage::Exit);
    saw_switch = saw_switch || TraceHasStallReason(event, TraceStallReason::WarpSwitch);
  }

  EXPECT_TRUE(saw_wave_start);
  EXPECT_TRUE(saw_wave_end);
  EXPECT_TRUE(saw_switch);
}

// =============================================================================
// WaveStats Snapshots
// =============================================================================

TEST(TraceWaveStatsTest, EmitsWaveStatsSnapshotsForFunctionalLaunch) {
  CollectingTraceSink trace;
  ExecEngine runtime(&trace);
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

TEST(TraceWaveStatsTest, EmitsWaveStatsStateSplitForFunctionalLaunch) {
  CollectingTraceSink trace;
  ExecEngine runtime(&trace);
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

TEST(TraceWaveStatsTest, EmitsUnifiedWaitStateMachineTraceForWaitcnt) {
  CollectingTraceSink trace;
  ExecEngine runtime(&trace);
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
    if (event.kind == TraceEventKind::Stall &&
        event.stall_reason == TraceStallReason::WaitCntGlobal) {
      saw_waitcnt_stall = true;
    }
  }

  EXPECT_TRUE(saw_waiting_snapshot);
  EXPECT_TRUE(saw_waitcnt_stall);
}

TEST(TraceWaveStatsTest, WaveStatsSnapshotsForWaitcntUseExecutionCycleAnchors) {
  CollectingTraceSink trace;
  ExecEngine runtime(&trace);
  runtime.SetFunctionalExecutionMode(FunctionalExecutionMode::SingleThreaded);

  const auto kernel = BuildWaitcntTraceKernel();
  uint64_t waitcnt_pc = std::numeric_limits<uint64_t>::max();
  for (const auto& [pc, instruction] : kernel.instructions_by_pc()) {
    if (instruction.opcode == Opcode::SWaitCnt) {
      waitcnt_pc = pc;
      break;
    }
  }
  ASSERT_NE(waitcnt_pc, std::numeric_limits<uint64_t>::max());

  const uint64_t base_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(base_addr, 11);

  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;
  request.args.PushU64(base_addr);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok);

  uint64_t wait_cycle = std::numeric_limits<uint64_t>::max();
  uint64_t resume_cycle = std::numeric_limits<uint64_t>::max();
  uint64_t waiting_stats_cycle = std::numeric_limits<uint64_t>::max();
  uint64_t resumed_stats_cycle = std::numeric_limits<uint64_t>::max();
  bool saw_waiting_stats = false;
  for (const auto& event : trace.events()) {
    if (event.kind == TraceEventKind::WaveWait && event.pc == waitcnt_pc &&
        wait_cycle == std::numeric_limits<uint64_t>::max()) {
      wait_cycle = event.cycle;
    }
    if (event.kind == TraceEventKind::WaveResume &&
        resume_cycle == std::numeric_limits<uint64_t>::max()) {
      resume_cycle = event.cycle;
    }
    if (event.kind != TraceEventKind::WaveStats) {
      continue;
    }
    if (!saw_waiting_stats && event.message.find("waiting=1") != std::string::npos) {
      waiting_stats_cycle = event.cycle;
      saw_waiting_stats = true;
      continue;
    }
    if (saw_waiting_stats &&
        event.message.find("active=1 runnable=1 waiting=0 end=0") != std::string::npos &&
        resumed_stats_cycle == std::numeric_limits<uint64_t>::max()) {
      resumed_stats_cycle = event.cycle;
    }
  }

  ASSERT_NE(wait_cycle, std::numeric_limits<uint64_t>::max());
  ASSERT_NE(resume_cycle, std::numeric_limits<uint64_t>::max());
  ASSERT_NE(waiting_stats_cycle, std::numeric_limits<uint64_t>::max());
  ASSERT_NE(resumed_stats_cycle, std::numeric_limits<uint64_t>::max());
  EXPECT_EQ(waiting_stats_cycle, wait_cycle);
  EXPECT_EQ(resumed_stats_cycle, resume_cycle);
}

}  // namespace
}  // namespace gpu_model
