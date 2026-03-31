#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string_view>
#include <vector>

#include "gpu_model/debug/trace_sink.h"
#include "gpu_model/isa/instruction_builder.h"
#include "gpu_model/isa/opcode.h"
#include "gpu_model/runtime/runtime_engine.h"

namespace gpu_model {
namespace {

ExecutableKernel BuildPendingMemoryBeforeExplicitWaitcntKernel() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SMov("s1", 0);
  builder.MLoadGlobal("v1", "s0", "s1", 4);
  builder.SMov("s2", 7);
  builder.SWaitCnt(/*global_count=*/0, /*shared_count=*/UINT32_MAX,
                   /*private_count=*/UINT32_MAX, /*scalar_buffer_count=*/UINT32_MAX);
  builder.BExit();
  return builder.Build("pending_memory_before_explicit_waitcnt");
}

ExecutableKernel BuildWaitcntThresholdResumeKernel() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SMov("s1", 0);
  builder.MLoadGlobal("v1", "s0", "s1", 4);
  builder.SMov("s2", 1);
  builder.MLoadGlobal("v2", "s0", "s2", 4);
  builder.SWaitCnt(/*global_count=*/1, /*shared_count=*/UINT32_MAX,
                   /*private_count=*/UINT32_MAX, /*scalar_buffer_count=*/UINT32_MAX);
  builder.SMov("s3", 7);
  builder.SWaitCnt(/*global_count=*/0, /*shared_count=*/UINT32_MAX,
                   /*private_count=*/UINT32_MAX, /*scalar_buffer_count=*/UINT32_MAX);
  builder.SMov("s4", 9);
  builder.BExit();
  return builder.Build("waitcnt_threshold_resume");
}

ExecutableKernel BuildParallelWaitcntZeroResumeKernel() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SMov("s1", 0);
  for (int i = 0; i < 16; ++i) {
    builder.MLoadGlobal("v1", "s0", "s1", 4);
  }
  builder.SWaitCnt(/*global_count=*/0, /*shared_count=*/UINT32_MAX,
                   /*private_count=*/UINT32_MAX, /*scalar_buffer_count=*/UINT32_MAX);
  builder.SMov("s2", 9);
  builder.BExit();
  return builder.Build("parallel_waitcnt_zero_resume");
}

uint64_t NthInstructionPcWithOpcode(const ExecutableKernel& kernel, Opcode opcode, size_t ordinal) {
  size_t seen = 0;
  for (uint64_t pc = 0; pc < kernel.instructions().size(); ++pc) {
    if (kernel.instructions()[pc].opcode != opcode) {
      continue;
    }
    if (seen == ordinal) {
      return pc;
    }
    ++seen;
  }
  return std::numeric_limits<uint64_t>::max();
}

size_t FirstEventIndex(const std::vector<TraceEvent>& events,
                       TraceEventKind kind,
                       uint64_t pc,
                       std::optional<std::string_view> message = std::nullopt) {
  for (size_t i = 0; i < events.size(); ++i) {
    if (events[i].kind != kind || events[i].pc != pc) {
      continue;
    }
    if (message.has_value() && events[i].message != *message) {
      continue;
    }
    return i;
  }
  return std::numeric_limits<size_t>::max();
}

size_t FirstEventIndexAfter(const std::vector<TraceEvent>& events,
                            size_t start,
                            TraceEventKind kind,
                            uint64_t pc,
                            std::optional<std::string_view> message = std::nullopt) {
  for (size_t i = start + 1; i < events.size(); ++i) {
    if (events[i].kind != kind || events[i].pc != pc) {
      continue;
    }
    if (message.has_value() && events[i].message != *message) {
      continue;
    }
    return i;
  }
  return std::numeric_limits<size_t>::max();
}

size_t FirstEventIndexForBlockWave(const std::vector<TraceEvent>& events,
                              uint32_t block_id,
                              uint32_t wave_id,
                              TraceEventKind kind,
                              uint64_t pc,
                              std::optional<std::string_view> message = std::nullopt) {
  for (size_t i = 0; i < events.size(); ++i) {
    if (events[i].block_id != block_id || events[i].wave_id != wave_id ||
        events[i].kind != kind || events[i].pc != pc) {
      continue;
    }
    if (message.has_value() && events[i].message != *message) {
      continue;
    }
    return i;
  }
  return std::numeric_limits<size_t>::max();
}

const TraceEvent* FirstEventForBlockWave(const std::vector<TraceEvent>& events,
                                    uint32_t block_id,
                                    uint32_t wave_id,
                                    TraceEventKind kind,
                                    uint64_t pc) {
  for (const auto& event : events) {
    if (event.block_id == block_id && event.wave_id == wave_id &&
        event.kind == kind && event.pc == pc) {
      return &event;
    }
  }
  return nullptr;
}

TEST(WaitcntFunctionalTest, PendingMemoryDoesNotStallBeforeExplicitWaitcnt) {
  CollectingTraceSink trace;
  RuntimeEngine runtime(&trace);

  const uint64_t base_addr = runtime.memory().AllocateGlobal(2 * sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(base_addr + 0 * sizeof(int32_t), 11);
  runtime.memory().StoreGlobalValue<int32_t>(base_addr + 1 * sizeof(int32_t), 13);

  const auto kernel = BuildPendingMemoryBeforeExplicitWaitcntKernel();
  const uint64_t load_pc = NthInstructionPcWithOpcode(kernel, Opcode::MLoadGlobal, 0);
  const uint64_t marker_pc = NthInstructionPcWithOpcode(kernel, Opcode::SMov, 1);
  const uint64_t waitcnt_pc = NthInstructionPcWithOpcode(kernel, Opcode::SWaitCnt, 0);
  ASSERT_NE(load_pc, std::numeric_limits<uint64_t>::max());
  ASSERT_NE(marker_pc, std::numeric_limits<uint64_t>::max());
  ASSERT_NE(waitcnt_pc, std::numeric_limits<uint64_t>::max());

  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;
  request.args.PushU64(base_addr);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  const auto& events = trace.events();
  const size_t load_index = FirstEventIndex(events, TraceEventKind::WaveStep, load_pc);
  const size_t marker_index = FirstEventIndex(events, TraceEventKind::WaveStep, marker_pc);
  const size_t waitcnt_index = FirstEventIndex(events, TraceEventKind::WaveStep, waitcnt_pc);
  const size_t stall_index =
      FirstEventIndex(events, TraceEventKind::Stall, waitcnt_pc, "waitcnt_global");

  ASSERT_NE(load_index, std::numeric_limits<size_t>::max());
  ASSERT_NE(marker_index, std::numeric_limits<size_t>::max());
  ASSERT_NE(waitcnt_index, std::numeric_limits<size_t>::max());
  ASSERT_NE(stall_index, std::numeric_limits<size_t>::max());
  EXPECT_LT(load_index, marker_index);
  EXPECT_LT(marker_index, waitcnt_index);
  EXPECT_LT(waitcnt_index, stall_index);
}

TEST(WaitcntFunctionalTest, WaitcntResumesWhenThresholdBecomesSatisfiedNotOnlyAtZero) {
  CollectingTraceSink trace;
  RuntimeEngine runtime(&trace);

  const uint64_t base_addr = runtime.memory().AllocateGlobal(2 * sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(base_addr + 0 * sizeof(int32_t), 11);
  runtime.memory().StoreGlobalValue<int32_t>(base_addr + 1 * sizeof(int32_t), 13);

  const auto kernel = BuildWaitcntThresholdResumeKernel();
  const uint64_t waitcnt_one_pc = NthInstructionPcWithOpcode(kernel, Opcode::SWaitCnt, 0);
  const uint64_t waitcnt_zero_pc = NthInstructionPcWithOpcode(kernel, Opcode::SWaitCnt, 1);
  const uint64_t marker_after_threshold_pc = NthInstructionPcWithOpcode(kernel, Opcode::SMov, 2);
  const uint64_t marker_after_zero_pc = NthInstructionPcWithOpcode(kernel, Opcode::SMov, 3);
  ASSERT_NE(waitcnt_one_pc, std::numeric_limits<uint64_t>::max());
  ASSERT_NE(waitcnt_zero_pc, std::numeric_limits<uint64_t>::max());
  ASSERT_NE(marker_after_threshold_pc, std::numeric_limits<uint64_t>::max());
  ASSERT_NE(marker_after_zero_pc, std::numeric_limits<uint64_t>::max());

  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;
  request.args.PushU64(base_addr);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  const auto& events = trace.events();
  const size_t first_waitcnt_index =
      FirstEventIndex(events, TraceEventKind::WaveStep, waitcnt_one_pc);
  const size_t first_waitcnt_stall_index =
      FirstEventIndex(events, TraceEventKind::Stall, waitcnt_one_pc, "waitcnt_global");
  const size_t threshold_resume_marker_index =
      FirstEventIndex(events, TraceEventKind::WaveStep, marker_after_threshold_pc);
  const size_t second_waitcnt_index =
      FirstEventIndex(events, TraceEventKind::WaveStep, waitcnt_zero_pc);
  const size_t second_waitcnt_stall_index =
      FirstEventIndexAfter(events, threshold_resume_marker_index, TraceEventKind::Stall,
                           waitcnt_zero_pc, "waitcnt_global");
  const size_t zero_resume_marker_index =
      FirstEventIndex(events, TraceEventKind::WaveStep, marker_after_zero_pc);

  ASSERT_NE(first_waitcnt_index, std::numeric_limits<size_t>::max());
  ASSERT_NE(first_waitcnt_stall_index, std::numeric_limits<size_t>::max());
  ASSERT_NE(threshold_resume_marker_index, std::numeric_limits<size_t>::max());
  ASSERT_NE(second_waitcnt_index, std::numeric_limits<size_t>::max());
  ASSERT_NE(second_waitcnt_stall_index, std::numeric_limits<size_t>::max());
  ASSERT_NE(zero_resume_marker_index, std::numeric_limits<size_t>::max());
  EXPECT_LT(first_waitcnt_index, first_waitcnt_stall_index);
  EXPECT_LT(first_waitcnt_stall_index, threshold_resume_marker_index);
  EXPECT_LT(threshold_resume_marker_index, second_waitcnt_index);
  EXPECT_LT(second_waitcnt_index, second_waitcnt_stall_index);
  EXPECT_LT(second_waitcnt_stall_index, zero_resume_marker_index);
}

TEST(WaitcntFunctionalTest, EmitsWaveStatsDuringWaitcntProgress) {
  CollectingTraceSink trace;
  RuntimeEngine runtime(&trace);
  runtime.SetFunctionalExecutionMode(FunctionalExecutionMode::SingleThreaded);

  const uint64_t base_addr = runtime.memory().AllocateGlobal(2 * sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(base_addr + 0 * sizeof(int32_t), 11);
  runtime.memory().StoreGlobalValue<int32_t>(base_addr + 1 * sizeof(int32_t), 13);

  const auto kernel = BuildPendingMemoryBeforeExplicitWaitcntKernel();

  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;
  request.args.PushU64(base_addr);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  bool saw_waiting_stats = false;
  for (const auto& event : trace.events()) {
    if (event.kind != TraceEventKind::WaveStats) {
      continue;
    }
    if (event.message.find("active=1 runnable=0 waiting=1 end=0") != std::string::npos) {
      saw_waiting_stats = true;
      break;
    }
  }

  EXPECT_TRUE(saw_waiting_stats);
}

TEST(WaitcntFunctionalTest, MarlParallelWaitcntResumeIsConsistentAcrossTwoBlocks) {
  constexpr uint32_t kBlockDim = 64;
  constexpr std::array<uint32_t, 2> kTargetBlockIds{0, 1};
  const auto kernel = BuildParallelWaitcntZeroResumeKernel();
  const uint64_t waitcnt_pc = NthInstructionPcWithOpcode(kernel, Opcode::SWaitCnt, 0);
  const uint64_t resume_marker_pc = NthInstructionPcWithOpcode(kernel, Opcode::SMov, 1);
  ASSERT_NE(waitcnt_pc, std::numeric_limits<uint64_t>::max());
  ASSERT_NE(resume_marker_pc, std::numeric_limits<uint64_t>::max());

  for (int iteration = 0; iteration < 5; ++iteration) {
    SCOPED_TRACE(iteration);
    CollectingTraceSink trace;
    RuntimeEngine runtime(&trace);
    runtime.SetFunctionalExecutionConfig(
        FunctionalExecutionConfig{
            .mode = FunctionalExecutionMode::MarlParallel,
            .worker_threads = 2,
        });

    const uint64_t base_addr = runtime.memory().AllocateGlobal(2 * sizeof(int32_t));
    runtime.memory().StoreGlobalValue<int32_t>(base_addr + 0 * sizeof(int32_t), 11);
    runtime.memory().StoreGlobalValue<int32_t>(base_addr + 1 * sizeof(int32_t), 13);

    LaunchRequest request;
    request.kernel = &kernel;
    request.config.grid_dim_x = 2;
    request.config.block_dim_x = kBlockDim;
    request.args.PushU64(base_addr);

    const auto result = runtime.Launch(request);
    ASSERT_TRUE(result.ok) << result.error_message;

    const auto& events = trace.events();
    for (uint32_t block_id : kTargetBlockIds) {
      const auto* waitcnt_event =
          FirstEventForBlockWave(events, block_id, 0, TraceEventKind::WaveStep, waitcnt_pc);
      const size_t waitcnt_index =
          FirstEventIndexForBlockWave(events, block_id, 0, TraceEventKind::WaveStep, waitcnt_pc);
      const size_t waitcnt_stall_index =
          FirstEventIndexForBlockWave(events, block_id, 0, TraceEventKind::Stall,
                                      waitcnt_pc, "waitcnt_global");
      const size_t resume_marker_index =
          FirstEventIndexForBlockWave(events, block_id, 0, TraceEventKind::WaveStep, resume_marker_pc);

      ASSERT_NE(waitcnt_event, nullptr);
      ASSERT_NE(waitcnt_index, std::numeric_limits<size_t>::max());
      ASSERT_NE(waitcnt_stall_index, std::numeric_limits<size_t>::max());
      ASSERT_NE(resume_marker_index, std::numeric_limits<size_t>::max());
      EXPECT_NE(waitcnt_event->message.find("op=s_waitcnt"), std::string::npos);
      EXPECT_LT(waitcnt_index, waitcnt_stall_index);
      EXPECT_LT(waitcnt_stall_index, resume_marker_index);
    }
  }
}

}  // namespace
}  // namespace gpu_model
