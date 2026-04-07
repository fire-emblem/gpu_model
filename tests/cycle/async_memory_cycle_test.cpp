#include <gtest/gtest.h>

#include <cstring>
#include <cstdint>
#include <limits>
#include <set>
#include <string_view>
#include <vector>

#include "gpu_model/debug/trace/sink.h"
#include "gpu_model/isa/instruction_builder.h"
#include "gpu_model/runtime/exec_engine.h"

namespace gpu_model {
namespace {

void ConfigureZeroFrontendTiming(ExecEngine& runtime) {
  runtime.SetLaunchTimingProfile(/*kernel_launch_gap_cycles=*/8,
                                 /*kernel_launch_cycles=*/0,
                                 /*block_launch_cycles=*/0,
                                 /*wave_generation_cycles=*/0,
                                 /*wave_dispatch_cycles=*/0,
                                 /*wave_launch_cycles=*/0,
                                 /*warp_switch_cycles=*/1,
                                 /*arg_load_cycles=*/4);
}

uint64_t NthWaveStepCycle(const std::vector<TraceEvent>& events,
                          std::string_view opcode,
                          size_t ordinal) {
  size_t seen = 0;
  for (const auto& event : events) {
    if (event.kind == TraceEventKind::WaveStep &&
        event.message.find(std::string(opcode)) != std::string::npos) {
      if (seen == ordinal) {
        return event.cycle;
      }
      ++seen;
    }
  }
  return std::numeric_limits<uint64_t>::max();
}

uint64_t NthWaveStepCycleForPeu(const std::vector<TraceEvent>& events,
                                std::string_view opcode,
                                uint32_t peu_id,
                                size_t ordinal) {
  size_t seen = 0;
  for (const auto& event : events) {
    if (event.kind != TraceEventKind::WaveStep || event.peu_id != peu_id ||
        event.message.find(std::string(opcode)) == std::string::npos) {
      continue;
    }
    if (seen == ordinal) {
      return event.cycle;
    }
    ++seen;
  }
  return std::numeric_limits<uint64_t>::max();
}

uint64_t FirstWaveStepCycle(const std::vector<TraceEvent>& events, std::string_view opcode) {
  return NthWaveStepCycle(events, opcode, 0);
}

std::set<uint32_t> WaveStepSlotIdsForPeu(const std::vector<TraceEvent>& events,
                                         std::string_view opcode,
                                         uint32_t peu_id) {
  std::set<uint32_t> slot_ids;
  for (const auto& event : events) {
    if (event.kind != TraceEventKind::WaveStep || event.peu_id != peu_id) {
      continue;
    }
    if (event.message.find(std::string(opcode)) == std::string::npos) {
      continue;
    }
    slot_ids.insert(event.slot_id);
  }
  return slot_ids;
}

std::set<uint32_t> WaveStepSlotIdsForCycleAndPeu(const std::vector<TraceEvent>& events,
                                                 std::string_view opcode,
                                                 uint64_t cycle,
                                                 uint32_t peu_id) {
  std::set<uint32_t> slot_ids;
  for (const auto& event : events) {
    if (event.kind != TraceEventKind::WaveStep || event.cycle != cycle ||
        event.peu_id != peu_id) {
      continue;
    }
    if (event.message.find(std::string(opcode)) == std::string::npos) {
      continue;
    }
    slot_ids.insert(event.slot_id);
  }
  return slot_ids;
}

uint64_t FirstCycleForSlotEvent(const std::vector<TraceEvent>& events,
                                TraceEventKind kind,
                                uint32_t slot_id,
                                std::string_view message = {}) {
  for (const auto& event : events) {
    if (event.kind != kind || event.slot_id != slot_id) {
      continue;
    }
    if (!message.empty() &&
        event.message.find(std::string(message)) == std::string::npos) {
      continue;
    }
    return event.cycle;
  }
  return std::numeric_limits<uint64_t>::max();
}

uint64_t FirstCycleForSlotEventOnPeu(const std::vector<TraceEvent>& events,
                                     TraceEventKind kind,
                                     uint32_t peu_id,
                                     uint32_t slot_id,
                                     std::string_view message = {}) {
  for (const auto& event : events) {
    if (event.kind != kind || event.peu_id != peu_id || event.slot_id != slot_id) {
      continue;
    }
    if (!message.empty() &&
        event.message.find(std::string(message)) == std::string::npos) {
      continue;
    }
    return event.cycle;
  }
  return std::numeric_limits<uint64_t>::max();
}

bool HasStallReason(const std::vector<TraceEvent>& events, TraceStallReason reason) {
  for (const auto& event : events) {
    if (TraceHasStallReason(event, reason)) {
      return true;
    }
  }
  return false;
}

size_t FirstEventIndex(const std::vector<TraceEvent>& events,
                       TraceEventKind kind,
                       std::string_view message = {}) {
  for (size_t i = 0; i < events.size(); ++i) {
    if (events[i].kind != kind) {
      continue;
    }
    if (!message.empty() &&
        events[i].message.find(std::string(message)) == std::string::npos) {
      continue;
    }
    return i;
  }
  return std::numeric_limits<size_t>::max();
}

size_t FirstEventIndexAfter(const std::vector<TraceEvent>& events,
                            size_t start,
                            TraceEventKind kind,
                            std::string_view message = {}) {
  for (size_t i = start + 1; i < events.size(); ++i) {
    if (events[i].kind != kind) {
      continue;
    }
    if (!message.empty() &&
        events[i].message.find(std::string(message)) == std::string::npos) {
      continue;
    }
    return i;
  }
  return std::numeric_limits<size_t>::max();
}

size_t FirstArriveIndexWithProgress(const std::vector<TraceEvent>& events,
                                    TraceArriveProgressKind progress) {
  for (size_t i = 0; i < events.size(); ++i) {
    if (events[i].kind != TraceEventKind::Arrive) {
      continue;
    }
    if (events[i].arrive_progress != progress) {
      continue;
    }
    return i;
  }
  return std::numeric_limits<size_t>::max();
}

TEST(AsyncMemoryCycleTest, LoadUsesIssuePlusArriveLatency) {
  CollectingTraceSink trace;
  ExecEngine runtime(&trace);
  ConfigureZeroFrontendTiming(runtime);
  runtime.SetFixedGlobalMemoryLatency(20);

  const uint64_t base_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(base_addr, 9);

  InstructionBuilder builder;
  builder.SMov("s0", base_addr);
  builder.SMov("s1", 0);
  builder.MLoadGlobal("v1", "s0", "s1", 4);
  builder.SWaitCnt(/*global_count=*/0, /*shared_count=*/UINT32_MAX,
                   /*private_count=*/UINT32_MAX, /*scalar_buffer_count=*/UINT32_MAX);
  builder.VAdd("v2", "v1", "v1");
  builder.BExit();
  const auto kernel = builder.Build("one_load");

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  EXPECT_EQ(result.total_cycles, 40u);
  EXPECT_EQ(FirstWaveStepCycle(trace.events(), "v_add_i32"), 32u);
}

TEST(AsyncMemoryCycleTest, ResidentSlotsIssueIndependentlyWithinSamePeu) {
  CollectingTraceSink trace;
  ExecEngine runtime(&trace);
  ConfigureZeroFrontendTiming(runtime);
  ArchitecturalIssueLimits widened_limits = DefaultArchitecturalIssueLimits();
  widened_limits.scalar_alu_or_memory = 2;
  runtime.SetCycleIssueLimits(widened_limits);

  InstructionBuilder builder;
  builder.SMov("s0", 1);
  builder.SMov("s1", 2);
  builder.BExit();
  const auto kernel = builder.Build("resident_slot_independent_issue");

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 320;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  const auto cycle0_slots = WaveStepSlotIdsForCycleAndPeu(trace.events(), "s_mov_b32", 0, 0);
  EXPECT_GE(cycle0_slots.size(), 2u);
  EXPECT_EQ(cycle0_slots.count(0u), 1u);
  EXPECT_EQ(cycle0_slots.count(1u), 1u);
}

TEST(AsyncMemoryCycleTest, ResidentSlotsStillHonorIssueLimitsPerPeu) {
  CollectingTraceSink trace;
  ExecEngine runtime(&trace);
  ConfigureZeroFrontendTiming(runtime);

  ArchitecturalIssueLimits scalar_limited = DefaultArchitecturalIssueLimits();
  scalar_limited.scalar_alu_or_memory = 2;
  runtime.SetCycleIssueLimits(scalar_limited);

  InstructionBuilder builder;
  builder.SMov("s0", 1);
  builder.BExit();
  const auto kernel = builder.Build("resident_slot_issue_limited");

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 320;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  const auto cycle0_slots = WaveStepSlotIdsForCycleAndPeu(trace.events(), "s_mov_b32", 0, 0);
  EXPECT_EQ(cycle0_slots.size(), 2u);
}

TEST(AsyncMemoryCycleTest, ResidentSlotsDoNotBypassPeuIssueTiming) {
  CollectingTraceSink trace;
  ExecEngine runtime(&trace);
  ConfigureZeroFrontendTiming(runtime);

  InstructionBuilder builder;
  builder.SMov("s0", 1);
  builder.BExit();
  const auto kernel = builder.Build("resident_slot_peu_timing");

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 320;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  const uint64_t slot0_step_cycle =
      FirstCycleForSlotEvent(trace.events(), TraceEventKind::WaveStep, 0u, "s_mov_b32");
  const uint64_t slot1_step_cycle =
      FirstCycleForSlotEvent(trace.events(), TraceEventKind::WaveStep, 1u, "s_mov_b32");

  ASSERT_NE(slot0_step_cycle, std::numeric_limits<uint64_t>::max());
  ASSERT_NE(slot1_step_cycle, std::numeric_limits<uint64_t>::max());
  EXPECT_EQ(slot0_step_cycle, 0u);
  EXPECT_EQ(slot1_step_cycle, 4u);
}

TEST(AsyncMemoryCycleTest, ResidentSlotBundlesUsePeuWideMaxTiming) {
  CollectingTraceSink trace;
  ExecEngine runtime(&trace);
  runtime.SetLaunchTimingProfile(/*kernel_launch_gap_cycles=*/8,
                                 /*kernel_launch_cycles=*/0,
                                 /*block_launch_cycles=*/0,
                                 /*wave_generation_cycles=*/0,
                                 /*wave_dispatch_cycles=*/0,
                                 /*wave_launch_cycles=*/4,
                                 /*warp_switch_cycles=*/1,
                                 /*arg_load_cycles=*/4);

  IssueCycleClassOverridesSpec class_overrides;
  class_overrides.scalar_alu = 8;
  class_overrides.vector_alu = 4;
  runtime.SetIssueCycleClassOverrides(class_overrides);

  InstructionBuilder builder;
  builder.SMov("s0", 1);
  builder.VMov("v0", "s0");
  builder.BExit();
  const auto kernel = builder.Build("resident_slot_bundle_max_timing");

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 576;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  const uint64_t third_peu0_scalar_cycle =
      NthWaveStepCycleForPeu(trace.events(), "s_mov_b32", 0u, 2u);

  ASSERT_NE(third_peu0_scalar_cycle, std::numeric_limits<uint64_t>::max());
  EXPECT_EQ(third_peu0_scalar_cycle, 17u);
}

TEST(AsyncMemoryCycleTest, ResidentSlotDoesNotIssueBeforeDelayedWaveLaunch) {
  CollectingTraceSink trace;
  ExecEngine runtime(&trace);
  runtime.SetLaunchTimingProfile(/*kernel_launch_gap_cycles=*/8,
                                 /*kernel_launch_cycles=*/0,
                                 /*block_launch_cycles=*/0,
                                 /*wave_generation_cycles=*/0,
                                 /*wave_dispatch_cycles=*/0,
                                 /*wave_launch_cycles=*/4,
                                 /*warp_switch_cycles=*/1,
                                 /*arg_load_cycles=*/4);

  InstructionBuilder builder;
  builder.SMov("s0", 1);
  builder.BExit();
  const auto kernel = builder.Build("resident_slot_delayed_launch_gate");

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 320;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  const uint64_t slot1_launch_cycle =
      FirstCycleForSlotEventOnPeu(trace.events(), TraceEventKind::WaveLaunch, 0u, 1u, "wave_start");
  const uint64_t slot1_step_cycle =
      FirstCycleForSlotEventOnPeu(trace.events(), TraceEventKind::WaveStep, 0u, 1u, "s_mov_b32");

  ASSERT_NE(slot1_launch_cycle, std::numeric_limits<uint64_t>::max());
  ASSERT_NE(slot1_step_cycle, std::numeric_limits<uint64_t>::max());
  EXPECT_GE(slot1_step_cycle, slot1_launch_cycle);
}

TEST(AsyncMemoryCycleTest, LoadAllowsIndependentScalarIssueBeforeArrive) {
  CollectingTraceSink trace;
  ExecEngine runtime(&trace);
  ConfigureZeroFrontendTiming(runtime);
  runtime.SetFixedGlobalMemoryLatency(20);

  const uint64_t base_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(base_addr, 9);

  InstructionBuilder builder;
  builder.SMov("s0", base_addr);
  builder.SMov("s1", 0);
  builder.MLoadGlobal("v1", "s0", "s1", 4);
  builder.SMov("s2", 7);
  builder.BExit();
  const auto kernel = builder.Build("load_blocks_wave");

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  EXPECT_EQ(NthWaveStepCycle(trace.events(), "s_mov_b32", 0), 0u);
  EXPECT_EQ(NthWaveStepCycle(trace.events(), "s_mov_b32", 1), 4u);
  EXPECT_EQ(FirstWaveStepCycle(trace.events(), "buffer_load_dword"), 8u);
  EXPECT_EQ(NthWaveStepCycle(trace.events(), "s_mov_b32", 2), 12u);
  EXPECT_EQ(result.total_cycles, 32u);
}

TEST(AsyncMemoryCycleTest, IndependentGlobalLoadsIssueBeforeExplicitWaitcnt) {
  CollectingTraceSink trace;
  ExecEngine runtime(&trace);
  ConfigureZeroFrontendTiming(runtime);
  runtime.SetFixedGlobalMemoryLatency(20);

  const uint64_t base_addr = runtime.memory().AllocateGlobal(2 * sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(base_addr + 0 * sizeof(int32_t), 9);
  runtime.memory().StoreGlobalValue<int32_t>(base_addr + 1 * sizeof(int32_t), 13);

  InstructionBuilder builder;
  builder.SMov("s0", base_addr);
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
  const auto kernel = builder.Build("independent_global_loads_before_waitcnt");

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  EXPECT_EQ(FirstWaveStepCycle(trace.events(), "buffer_load_dword"), 8u);
  EXPECT_EQ(NthWaveStepCycle(trace.events(), "buffer_load_dword", 1), 16u);
  EXPECT_EQ(FirstWaveStepCycle(trace.events(), "s_waitcnt"), 20u);
  EXPECT_EQ(NthWaveStepCycle(trace.events(), "s_mov_b32", 3), 32u);
  EXPECT_EQ(NthWaveStepCycle(trace.events(), "s_waitcnt", 1), 36u);
  EXPECT_EQ(NthWaveStepCycle(trace.events(), "s_mov_b32", 4), 40u);
  EXPECT_TRUE(HasStallReason(trace.events(), TraceStallReason::WaitCntGlobal));
  EXPECT_EQ(result.total_cycles, 48u);
}

TEST(AsyncMemoryCycleTest, WaitCntZeroKeepsWaveBlockedUntilFinalArriveThenResumesIssue) {
  CollectingTraceSink trace;
  ExecEngine runtime(&trace);
  ConfigureZeroFrontendTiming(runtime);
  runtime.SetFixedGlobalMemoryLatency(20);

  const uint64_t base_addr = runtime.memory().AllocateGlobal(2 * sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(base_addr + 0 * sizeof(int32_t), 9);
  runtime.memory().StoreGlobalValue<int32_t>(base_addr + 1 * sizeof(int32_t), 13);

  InstructionBuilder builder;
  builder.SMov("s0", base_addr);
  builder.SMov("s1", 0);
  builder.MLoadGlobal("v1", "s0", "s1", 4);
  builder.SMov("s2", 1);
  builder.MLoadGlobal("v2", "s0", "s2", 4);
  builder.SWaitCnt(/*global_count=*/0, /*shared_count=*/UINT32_MAX,
                   /*private_count=*/UINT32_MAX, /*scalar_buffer_count=*/UINT32_MAX);
  builder.SMov("s3", 7);
  builder.BExit();
  const auto kernel = builder.Build("waitcnt_zero_blocked_until_final_arrive");

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  const auto& events = trace.events();
  const size_t wave_wait_index = FirstEventIndex(events, TraceEventKind::WaveWait);
  const size_t still_blocked_arrive_index =
      FirstArriveIndexWithProgress(events, TraceArriveProgressKind::StillBlocked);
  const size_t resume_arrive_index =
      FirstArriveIndexWithProgress(events, TraceArriveProgressKind::Resume);
  const size_t wave_resume_index = FirstEventIndex(events, TraceEventKind::WaveResume);
  const size_t resumed_step_index =
      FirstEventIndexAfter(events, wave_resume_index, TraceEventKind::WaveStep, "s_mov_b32");

  ASSERT_NE(wave_wait_index, std::numeric_limits<size_t>::max());
  ASSERT_NE(still_blocked_arrive_index, std::numeric_limits<size_t>::max());
  ASSERT_NE(resume_arrive_index, std::numeric_limits<size_t>::max());
  ASSERT_NE(wave_resume_index, std::numeric_limits<size_t>::max());
  ASSERT_NE(resumed_step_index, std::numeric_limits<size_t>::max());
  EXPECT_LT(wave_wait_index, still_blocked_arrive_index);
  EXPECT_LT(still_blocked_arrive_index, resume_arrive_index);
  EXPECT_LT(resume_arrive_index, wave_resume_index);
  EXPECT_LT(wave_resume_index, resumed_step_index);
}

TEST(AsyncMemoryCycleTest, WaitCntCanWaitForGlobalMemoryOnly) {
  CollectingTraceSink trace;
  ExecEngine runtime(&trace);
  ConfigureZeroFrontendTiming(runtime);
  runtime.SetFixedGlobalMemoryLatency(20);

  const uint64_t base_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(base_addr, 9);

  InstructionBuilder builder;
  builder.SMov("s0", base_addr);
  builder.SMov("s1", 0);
  builder.MLoadGlobal("v1", "s0", "s1", 4);
  builder.SMov("s2", 7);
  builder.SWaitCnt(/*global_count=*/0, /*shared_count=*/UINT32_MAX,
                   /*private_count=*/UINT32_MAX, /*scalar_buffer_count=*/UINT32_MAX);
  builder.SMov("s3", 9);
  builder.BExit();
  const auto kernel = builder.Build("waitcnt_global_only");

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 320;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  const auto waitcnt_slot_ids = WaveStepSlotIdsForPeu(trace.events(), "s_waitcnt", 0);
  EXPECT_GE(waitcnt_slot_ids.size(), 2u);
  EXPECT_TRUE(HasStallReason(trace.events(), TraceStallReason::WaitCntGlobal));
}

TEST(AsyncMemoryCycleTest, WaitCntIgnoresGlobalWhenWaitingSharedOnly) {
  CollectingTraceSink trace;
  ExecEngine runtime(&trace);
  ConfigureZeroFrontendTiming(runtime);
  runtime.SetFixedGlobalMemoryLatency(20);

  const uint64_t base_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(base_addr, 9);

  InstructionBuilder builder;
  builder.SMov("s0", base_addr);
  builder.SMov("s1", 0);
  builder.MLoadGlobal("v1", "s0", "s1", 4);
  builder.SMov("s2", 7);
  builder.SWaitCnt(/*global_count=*/UINT32_MAX, /*shared_count=*/0,
                   /*private_count=*/UINT32_MAX, /*scalar_buffer_count=*/UINT32_MAX);
  builder.SMov("s3", 9);
  builder.BExit();
  const auto kernel = builder.Build("waitcnt_shared_only");

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  EXPECT_EQ(FirstWaveStepCycle(trace.events(), "buffer_load_dword"), 8u);
  EXPECT_EQ(FirstWaveStepCycle(trace.events(), "s_waitcnt"), 16u);
  EXPECT_EQ(NthWaveStepCycle(trace.events(), "s_mov_b32", 3), 20u);
  EXPECT_EQ(result.total_cycles, 32u);
  EXPECT_FALSE(HasStallReason(trace.events(), TraceStallReason::WaitCntGlobal));
}

TEST(AsyncMemoryCycleTest, WaitCntCanWaitForScalarBufferOnly) {
  CollectingTraceSink trace;
  ExecEngine runtime(&trace);
  ConfigureZeroFrontendTiming(runtime);

  ConstSegment const_segment;
  const_segment.bytes.resize(sizeof(int32_t));
  const int32_t value = 9;
  std::memcpy(const_segment.bytes.data(), &value, sizeof(value));

  InstructionBuilder builder;
  builder.VMov("v0", 0);
  builder.MLoadConst("v1", "v0", 4);
  builder.SWaitCnt(/*global_count=*/UINT32_MAX, /*shared_count=*/UINT32_MAX,
                   /*private_count=*/UINT32_MAX, /*scalar_buffer_count=*/0);
  builder.SMov("s0", 1);
  builder.BExit();
  const auto kernel = builder.Build("waitcnt_scalar_buffer", {}, std::move(const_segment));

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  EXPECT_EQ(FirstWaveStepCycle(trace.events(), "scalar_buffer_load_dword"), 4u);
  EXPECT_EQ(FirstWaveStepCycle(trace.events(), "s_waitcnt"), 8u);
  EXPECT_EQ(FirstWaveStepCycle(trace.events(), "s_mov_b32"), 12u);
  EXPECT_EQ(result.total_cycles, 20u);
  // This test validates scalar-buffer waitcnt behavior via timing; no explicit stall is emitted.
}

TEST(AsyncMemoryCycleTest, WaitCntCanWaitForScalarBufferScalarLoadOnly) {
  CollectingTraceSink trace;
  ExecEngine runtime(&trace);
  ConfigureZeroFrontendTiming(runtime);

  ConstSegment const_segment;
  const_segment.bytes.resize(sizeof(int32_t));
  const int32_t value = 11;
  std::memcpy(const_segment.bytes.data(), &value, sizeof(value));

  InstructionBuilder builder;
  builder.SMov("s0", 0);
  builder.SBufferLoadDword("s1", "s0", 4);
  builder.SWaitCnt(/*global_count=*/UINT32_MAX, /*shared_count=*/UINT32_MAX,
                   /*private_count=*/UINT32_MAX, /*scalar_buffer_count=*/0);
  builder.SMov("s2", 1);
  builder.BExit();
  const auto kernel = builder.Build("waitcnt_scalar_buffer_scalar", {}, std::move(const_segment));

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  EXPECT_EQ(FirstWaveStepCycle(trace.events(), "s_buffer_load_dword"), 4u);
  EXPECT_EQ(FirstWaveStepCycle(trace.events(), "s_waitcnt"), 8u);
  EXPECT_EQ(NthWaveStepCycle(trace.events(), "s_mov_b32", 1), 12u);
  EXPECT_EQ(result.total_cycles, 20u);
  // Scalar-buffer-only waitcnt is validated by ordering, not by an emitted stall in current cycles.
}

TEST(AsyncMemoryCycleTest, BufferLoadUsesImmediateOffset) {
  CollectingTraceSink trace;
  ExecEngine runtime(&trace);
  ConfigureZeroFrontendTiming(runtime);
  runtime.SetFixedGlobalMemoryLatency(20);

  const uint64_t base_addr = runtime.memory().AllocateGlobal(2 * sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(base_addr + 0 * sizeof(int32_t), 5);
  runtime.memory().StoreGlobalValue<int32_t>(base_addr + 1 * sizeof(int32_t), 17);

  InstructionBuilder builder;
  builder.SMov("s0", base_addr);
  builder.SMov("s1", 0);
  builder.MLoadGlobal("v1", "s0", "s1", 4, 4);
  builder.SWaitCnt(/*global_count=*/0, /*shared_count=*/UINT32_MAX,
                   /*private_count=*/UINT32_MAX, /*scalar_buffer_count=*/UINT32_MAX);
  builder.VAdd("v2", "v1", "v1");
  builder.BExit();
  const auto kernel = builder.Build("buffer_offset_load");

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  EXPECT_EQ(FirstWaveStepCycle(trace.events(), "buffer_load_dword"), 8u);
  EXPECT_EQ(FirstWaveStepCycle(trace.events(), "v_add_i32"), 32u);
  EXPECT_EQ(result.total_cycles, 40u);
}

TEST(AsyncMemoryCycleTest, DenseGlobalLoadsIssueEveryFourCyclesAndOverlapLatency) {
  CollectingTraceSink trace;
  ExecEngine runtime(&trace);
  runtime.SetFixedGlobalMemoryLatency(20);

  constexpr uint32_t kLoadCount = 100;
  const uint64_t base_addr = runtime.memory().AllocateGlobal(kLoadCount * sizeof(int32_t));
  for (uint32_t i = 0; i < kLoadCount; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(base_addr + i * sizeof(int32_t),
                                               static_cast<int32_t>(100 + i));
  }

  InstructionBuilder builder;
  builder.SMov("s0", base_addr);
  builder.SMov("s1", 0);
  for (uint32_t i = 0; i < kLoadCount; ++i) {
    builder.MLoadGlobal("v1", "s0", "s1", 4, i * 4);
  }
  builder.SWaitCnt(/*global_count=*/0, /*shared_count=*/UINT32_MAX,
                   /*private_count=*/UINT32_MAX, /*scalar_buffer_count=*/UINT32_MAX);
  builder.BExit();
  const auto kernel = builder.Build("dense_global_load_overlap_waitcnt");

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  const uint64_t first_load_cycle = FirstWaveStepCycle(trace.events(), "buffer_load_dword");
  ASSERT_NE(first_load_cycle, std::numeric_limits<uint64_t>::max());
  for (size_t i = 1; i < kLoadCount; ++i) {
    const uint64_t prior = NthWaveStepCycle(trace.events(), "buffer_load_dword", i - 1);
    const uint64_t current = NthWaveStepCycle(trace.events(), "buffer_load_dword", i);
    ASSERT_NE(prior, std::numeric_limits<uint64_t>::max());
    ASSERT_NE(current, std::numeric_limits<uint64_t>::max());
    EXPECT_EQ(current - prior, 4u) << i;
  }

  const uint64_t last_load_cycle = NthWaveStepCycle(trace.events(), "buffer_load_dword", kLoadCount - 1);
  ASSERT_NE(last_load_cycle, std::numeric_limits<uint64_t>::max());
  EXPECT_LT(result.total_cycles, static_cast<uint64_t>(kLoadCount) * 20u);
  EXPECT_GE(result.total_cycles, last_load_cycle + 24u);
  EXPECT_LE(result.total_cycles, last_load_cycle + 40u);
}

TEST(AsyncMemoryCycleTest, EndKernelImplicitlyDrainsOutstandingGlobalLoads) {
  auto run_dense_load_kernel = [](bool explicit_waitcnt) {
    CollectingTraceSink trace;
    ExecEngine runtime(&trace);
    ConfigureZeroFrontendTiming(runtime);
    runtime.SetFixedGlobalMemoryLatency(20);

    constexpr uint32_t kLoadCount = 100;
    const uint64_t base_addr = runtime.memory().AllocateGlobal(kLoadCount * sizeof(int32_t));
    for (uint32_t i = 0; i < kLoadCount; ++i) {
      runtime.memory().StoreGlobalValue<int32_t>(base_addr + i * sizeof(int32_t),
                                                 static_cast<int32_t>(100 + i));
    }

    InstructionBuilder builder;
    builder.SMov("s0", base_addr);
    builder.SMov("s1", 0);
    for (uint32_t i = 0; i < kLoadCount; ++i) {
      builder.MLoadGlobal("v1", "s0", "s1", 4, i * 4);
    }
    if (explicit_waitcnt) {
      builder.SWaitCnt(/*global_count=*/0, /*shared_count=*/UINT32_MAX,
                       /*private_count=*/UINT32_MAX, /*scalar_buffer_count=*/UINT32_MAX);
    }
    builder.BExit();
    const auto kernel = builder.Build(explicit_waitcnt ? "dense_global_load_overlap_waitcnt_end"
                                                       : "dense_global_load_overlap_end_only");

    LaunchRequest request;
    request.kernel = &kernel;
    request.mode = ExecutionMode::Cycle;
    request.config.grid_dim_x = 1;
    request.config.block_dim_x = 64;

    const auto result = runtime.Launch(request);
    EXPECT_TRUE(result.ok) << result.error_message;

    const uint64_t last_load_cycle =
        NthWaveStepCycle(trace.events(), "buffer_load_dword", kLoadCount - 1);
    return std::make_pair(result, last_load_cycle);
  };

  const auto [waitcnt_result, waitcnt_last_load_cycle] = run_dense_load_kernel(true);
  const auto [end_only_result, end_only_last_load_cycle] = run_dense_load_kernel(false);

  ASSERT_NE(waitcnt_last_load_cycle, std::numeric_limits<uint64_t>::max());
  ASSERT_NE(end_only_last_load_cycle, std::numeric_limits<uint64_t>::max());
  EXPECT_EQ(waitcnt_last_load_cycle, end_only_last_load_cycle);

  EXPECT_GE(end_only_result.total_cycles, end_only_last_load_cycle + 24u);
  EXPECT_GE(waitcnt_result.total_cycles, end_only_result.total_cycles);
  EXPECT_LE(waitcnt_result.total_cycles - end_only_result.total_cycles, 4u);
}

}  // namespace
}  // namespace gpu_model
