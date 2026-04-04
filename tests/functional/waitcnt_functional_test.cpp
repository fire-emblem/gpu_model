#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <set>
#include <string_view>
#include <vector>

#include "gpu_model/debug/cycle_timeline.h"
#include "gpu_model/debug/trace_event_builder.h"
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

ExecutableKernel BuildSamePeuWaitcntSiblingKernel() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SLoadArg("s1", 1);
  builder.SysGlobalIdX("v0");

  builder.SMov("s2", 64);
  builder.VCmpLtCmask("v0", "s2");
  builder.MaskSaveExec("s10");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("after_wave0");
  builder.MLoadGlobal("v1", "s0", "v0", 4);
  builder.SWaitCnt(/*global_count=*/0, /*shared_count=*/UINT32_MAX,
                   /*private_count=*/UINT32_MAX, /*scalar_buffer_count=*/UINT32_MAX);
  builder.Label("after_wave0");
  builder.MaskRestoreExec("s10");

  builder.VMov("v4", 21);
  builder.VAdd("v5", "v4", "v4");
  builder.VAdd("v6", "v5", "v4");
  builder.MStoreGlobal("s1", "v0", "v6", 4);
  builder.BExit();
  return builder.Build("same_peu_waitcnt_sibling");
}

ExecutableKernel BuildTimelineWaitcntBubbleKernel() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SMov("s1", 0);
  builder.MLoadGlobal("v1", "s0", "s1", 4);
  builder.SWaitCnt(/*global_count=*/0, /*shared_count=*/UINT32_MAX,
                   /*private_count=*/UINT32_MAX, /*scalar_buffer_count=*/UINT32_MAX);
  builder.SMov("s2", 7);
  builder.BExit();
  return builder.Build("timeline_waitcnt_bubble");
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
    if (kind == TraceEventKind::Stall && message.has_value() &&
        TraceHasStallReason(events[i], TraceStallReasonFromMessage(*message))) {
      return i;
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

size_t CountOccurrences(std::string_view text, std::string_view needle) {
  size_t count = 0;
  size_t pos = 0;
  while ((pos = text.find(needle, pos)) != std::string_view::npos) {
    ++count;
    pos += needle.size();
  }
  return count;
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

TEST(WaitcntFunctionalTest, MultiThreadedWaitcntResumeIsConsistentAcrossTwoBlocks) {
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
            .mode = FunctionalExecutionMode::MultiThreaded,
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
    bool saw_any_waitcnt_stall = false;
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
      ASSERT_NE(resume_marker_index, std::numeric_limits<size_t>::max());
      EXPECT_NE(waitcnt_event->message.find("op=s_waitcnt"), std::string::npos);
      if (waitcnt_stall_index != std::numeric_limits<size_t>::max()) {
        saw_any_waitcnt_stall = true;
        EXPECT_LT(waitcnt_index, waitcnt_stall_index);
        EXPECT_LT(waitcnt_stall_index, resume_marker_index);
      } else {
        EXPECT_LT(waitcnt_index, resume_marker_index);
      }
    }
    EXPECT_TRUE(saw_any_waitcnt_stall);
  }
}

TEST(WaitcntFunctionalTest, WaitingWaveDoesNotBlockReadySiblingOnSamePeu) {
  CollectingTraceSink trace;
  RuntimeEngine runtime(&trace);
  runtime.SetFunctionalExecutionMode(FunctionalExecutionMode::SingleThreaded);

  constexpr uint32_t kBlockDim = 320;
  constexpr uint32_t kElementCount = kBlockDim;
  const auto kernel = BuildSamePeuWaitcntSiblingKernel();
  const uint64_t waitcnt_pc = NthInstructionPcWithOpcode(kernel, Opcode::SWaitCnt, 0);
  const uint64_t resume_marker_pc = NthInstructionPcWithOpcode(kernel, Opcode::VMov, 0);
  const uint64_t sibling_second_add_pc = NthInstructionPcWithOpcode(kernel, Opcode::VAdd, 1);
  ASSERT_NE(waitcnt_pc, std::numeric_limits<uint64_t>::max());
  ASSERT_NE(resume_marker_pc, std::numeric_limits<uint64_t>::max());
  ASSERT_NE(sibling_second_add_pc, std::numeric_limits<uint64_t>::max());

  const uint64_t in_addr = runtime.memory().AllocateGlobal(kElementCount * sizeof(int32_t));
  const uint64_t out_addr = runtime.memory().AllocateGlobal(kElementCount * sizeof(int32_t));
  for (uint32_t i = 0; i < kElementCount; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(in_addr + i * sizeof(int32_t),
                                               static_cast<int32_t>(100 + i));
    runtime.memory().StoreGlobalValue<int32_t>(out_addr + i * sizeof(int32_t), -1);
  }

  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = kBlockDim;
  request.args.PushU64(in_addr);
  request.args.PushU64(out_addr);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  for (uint32_t i = 0; i < kElementCount; ++i) {
    EXPECT_EQ(runtime.memory().LoadGlobalValue<int32_t>(out_addr + i * sizeof(int32_t)), 63);
  }

  const size_t waitcnt_stall_index = FirstEventIndexForBlockWave(
      trace.events(), 0, 0, TraceEventKind::Stall, waitcnt_pc, "waitcnt_global");
  const size_t resume_marker_index =
      FirstEventIndexForBlockWave(trace.events(), 0, 0, TraceEventKind::WaveStep, resume_marker_pc);
  const size_t sibling_second_add_index = FirstEventIndexForBlockWave(
      trace.events(), 0, 4, TraceEventKind::WaveStep, sibling_second_add_pc);

  ASSERT_NE(waitcnt_stall_index, std::numeric_limits<size_t>::max());
  ASSERT_NE(resume_marker_index, std::numeric_limits<size_t>::max());
  ASSERT_NE(sibling_second_add_index, std::numeric_limits<size_t>::max());
  EXPECT_LT(waitcnt_stall_index, sibling_second_add_index);
  EXPECT_LT(sibling_second_add_index, resume_marker_index);
}

TEST(WaitcntFunctionalTest, SingleAndMultiThreadedTraceUseUnboundedLogicalLaneIdsPerPeu) {
  constexpr uint32_t kBlockDim = 64 * 33;
  constexpr uint32_t kExpectedLogicalSlotsOnPeu0 = 9;
  constexpr uint32_t kElementCount = kBlockDim;
  const auto kernel = BuildSamePeuWaitcntSiblingKernel();

  const auto collect_slots = [&](FunctionalExecutionConfig config) {
    CollectingTraceSink trace;
    RuntimeEngine runtime(&trace);
    runtime.SetFunctionalExecutionConfig(config);

    const uint64_t in_addr = runtime.memory().AllocateGlobal(kElementCount * sizeof(int32_t));
    const uint64_t out_addr = runtime.memory().AllocateGlobal(kElementCount * sizeof(int32_t));
    for (uint32_t i = 0; i < kElementCount; ++i) {
      runtime.memory().StoreGlobalValue<int32_t>(in_addr + i * sizeof(int32_t),
                                                 static_cast<int32_t>(100 + i));
      runtime.memory().StoreGlobalValue<int32_t>(out_addr + i * sizeof(int32_t), -1);
    }

    LaunchRequest request;
    request.kernel = &kernel;
    request.config.grid_dim_x = 1;
    request.config.block_dim_x = kBlockDim;
    request.args.PushU64(in_addr);
    request.args.PushU64(out_addr);

    const auto result = runtime.Launch(request);
    EXPECT_TRUE(result.ok) << result.error_message;

    std::set<uint32_t> seen_slots;
    for (const auto& event : trace.events()) {
      if (event.block_id != 0 || event.peu_id != 0) {
        continue;
      }
      if (event.kind != TraceEventKind::WaveLaunch &&
          event.kind != TraceEventKind::WaveStep &&
          event.kind != TraceEventKind::Stall &&
          event.kind != TraceEventKind::WaveExit) {
        continue;
      }
      seen_slots.insert(event.slot_id);
    }
    return seen_slots;
  };

  const auto st_slots = collect_slots(FunctionalExecutionConfig{
      .mode = FunctionalExecutionMode::SingleThreaded,
  });
  const auto mt_slots = collect_slots(FunctionalExecutionConfig{
      .mode = FunctionalExecutionMode::MultiThreaded,
      .worker_threads = 2,
  });

  ASSERT_EQ(st_slots.size(), kExpectedLogicalSlotsOnPeu0);
  ASSERT_EQ(mt_slots.size(), kExpectedLogicalSlotsOnPeu0);
  EXPECT_EQ(*st_slots.begin(), 0u);
  EXPECT_EQ(*mt_slots.begin(), 0u);
  EXPECT_GE(*st_slots.rbegin(), kExpectedLogicalSlotsOnPeu0 - 1);
  EXPECT_GE(*mt_slots.rbegin(), kExpectedLogicalSlotsOnPeu0 - 1);
}

TEST(FunctionalWaitcntTest, TimelineShowsBlankBubbleWithWaitcntStallAndArrive) {
  CollectingTraceSink trace;
  RuntimeEngine runtime(&trace);
  runtime.SetFunctionalExecutionMode(FunctionalExecutionMode::SingleThreaded);
  runtime.SetFixedGlobalMemoryLatency(20);

  const auto kernel = BuildTimelineWaitcntBubbleKernel();
  const uint64_t load_pc = NthInstructionPcWithOpcode(kernel, Opcode::MLoadGlobal, 0);
  const uint64_t waitcnt_pc = NthInstructionPcWithOpcode(kernel, Opcode::SWaitCnt, 0);
  const uint64_t resume_pc = NthInstructionPcWithOpcode(kernel, Opcode::SMov, 1);
  ASSERT_NE(load_pc, std::numeric_limits<uint64_t>::max());
  ASSERT_NE(waitcnt_pc, std::numeric_limits<uint64_t>::max());
  ASSERT_NE(resume_pc, std::numeric_limits<uint64_t>::max());
  const uint64_t base_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(base_addr, 11);

  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;
  request.args.PushU64(base_addr);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  const auto& events = trace.events();
  const size_t load_step_index = FirstEventIndex(events, TraceEventKind::WaveStep, load_pc);
  const size_t waitcnt_stall_index =
      FirstEventIndex(events, TraceEventKind::Stall, waitcnt_pc, "waitcnt_global");
  size_t arrive_index = std::numeric_limits<size_t>::max();
  for (size_t i = 0; i < events.size(); ++i) {
    if (events[i].kind == TraceEventKind::Arrive && events[i].pc == waitcnt_pc &&
        events[i].arrive_kind == TraceArriveKind::Load) {
      arrive_index = i;
      break;
    }
  }
  const size_t resume_step_index = FirstEventIndex(events, TraceEventKind::WaveStep, resume_pc);
  ASSERT_NE(load_step_index, std::numeric_limits<size_t>::max());
  ASSERT_NE(waitcnt_stall_index, std::numeric_limits<size_t>::max());
  ASSERT_NE(arrive_index, std::numeric_limits<size_t>::max());
  ASSERT_NE(resume_step_index, std::numeric_limits<size_t>::max());
  EXPECT_EQ(events[waitcnt_stall_index].message,
            MakeTraceStallReasonMessage(kTraceStallReasonWaitCntGlobal));
  EXPECT_EQ(events[arrive_index].arrive_kind, TraceArriveKind::Load);
  EXPECT_TRUE(TraceHasSlotModel(events[waitcnt_stall_index], TraceSlotModelKind::LogicalUnbounded));
  EXPECT_TRUE(TraceHasSlotModel(events[arrive_index], TraceSlotModelKind::LogicalUnbounded));
  bool saw_wave_end = false;
  for (const auto& event : events) {
    if (event.kind == TraceEventKind::WaveExit &&
        event.lifecycle_stage == TraceLifecycleStage::Exit) {
      saw_wave_end = true;
      EXPECT_TRUE(TraceHasSlotModel(event, TraceSlotModelKind::LogicalUnbounded));
    }
  }
  EXPECT_TRUE(saw_wave_end);
  EXPECT_LT(events[load_step_index].cycle, events[arrive_index].cycle);
  EXPECT_LT(events[waitcnt_stall_index].cycle, events[arrive_index].cycle);
  EXPECT_LT(events[arrive_index].cycle, events[resume_step_index].cycle);

  const std::string timeline = CycleTimelineRenderer::RenderGoogleTrace(trace.events());
  EXPECT_NE(timeline.find("\"name\":\"buffer_load_dword\""), std::string::npos) << timeline;
  EXPECT_NE(timeline.find("\"name\":\"load_arrive\""), std::string::npos) << timeline;
  EXPECT_NE(timeline.find("\"name\":\"s_mov_b32\""), std::string::npos) << timeline;
  EXPECT_NE(timeline.find("\"name\":\"stall_waitcnt_global\""), std::string::npos) << timeline;
  EXPECT_EQ(timeline.find("\"name\":\"s_waitcnt\""), std::string::npos) << timeline;
  EXPECT_GE(CountOccurrences(timeline, "\"ph\":\"X\""), 2u) << timeline;
  EXPECT_EQ(CountOccurrences(timeline, "\"name\":\"bubble\""), 0u) << timeline;
}

}  // namespace
}  // namespace gpu_model
