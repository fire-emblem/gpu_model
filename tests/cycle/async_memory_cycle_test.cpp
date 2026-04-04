#include <gtest/gtest.h>

#include <cstring>
#include <cstdint>
#include <limits>
#include <set>
#include <string_view>
#include <vector>

#include "gpu_model/debug/trace/sink.h"
#include "gpu_model/isa/instruction_builder.h"
#include "gpu_model/runtime/runtime_engine.h"

namespace gpu_model {
namespace {

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

bool HasStallReason(const std::vector<TraceEvent>& events, TraceStallReason reason) {
  for (const auto& event : events) {
    if (TraceHasStallReason(event, reason)) {
      return true;
    }
  }
  return false;
}

TEST(AsyncMemoryCycleTest, LoadUsesIssuePlusArriveLatency) {
  CollectingTraceSink trace;
  RuntimeEngine runtime(&trace);
  runtime.SetFixedGlobalMemoryLatency(20);

  const uint64_t base_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(base_addr, 9);

  InstructionBuilder builder;
  builder.SMov("s0", base_addr);
  builder.SMov("s1", 0);
  builder.MLoadGlobal("v1", "s0", "s1", 4);
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
  RuntimeEngine runtime(&trace);
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
  RuntimeEngine runtime(&trace);

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
  RuntimeEngine runtime(&trace);

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
  RuntimeEngine runtime(&trace);
  runtime.SetLaunchTimingProfile(/*kernel_launch_gap_cycles=*/8,
                                 /*kernel_launch_cycles=*/0,
                                 /*block_launch_cycles=*/0,
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
  RuntimeEngine runtime(&trace);
  runtime.SetLaunchTimingProfile(/*kernel_launch_gap_cycles=*/8,
                                 /*kernel_launch_cycles=*/0,
                                 /*block_launch_cycles=*/0,
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
      FirstCycleForSlotEvent(trace.events(), TraceEventKind::WaveLaunch, 1u, "wave_start");
  const uint64_t slot1_step_cycle =
      FirstCycleForSlotEvent(trace.events(), TraceEventKind::WaveStep, 1u, "s_mov_b32");

  ASSERT_NE(slot1_launch_cycle, std::numeric_limits<uint64_t>::max());
  ASSERT_NE(slot1_step_cycle, std::numeric_limits<uint64_t>::max());
  EXPECT_EQ(slot1_launch_cycle, 4u);
  EXPECT_GE(slot1_step_cycle, slot1_launch_cycle);
}

TEST(AsyncMemoryCycleTest, LoadAllowsIndependentScalarIssueBeforeArrive) {
  CollectingTraceSink trace;
  RuntimeEngine runtime(&trace);
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
  RuntimeEngine runtime(&trace);
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

TEST(AsyncMemoryCycleTest, WaitCntCanWaitForGlobalMemoryOnly) {
  CollectingTraceSink trace;
  RuntimeEngine runtime(&trace);
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
  RuntimeEngine runtime(&trace);
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
  RuntimeEngine runtime(&trace);

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
  RuntimeEngine runtime(&trace);

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
  RuntimeEngine runtime(&trace);
  runtime.SetFixedGlobalMemoryLatency(20);

  const uint64_t base_addr = runtime.memory().AllocateGlobal(2 * sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(base_addr + 0 * sizeof(int32_t), 5);
  runtime.memory().StoreGlobalValue<int32_t>(base_addr + 1 * sizeof(int32_t), 17);

  InstructionBuilder builder;
  builder.SMov("s0", base_addr);
  builder.SMov("s1", 0);
  builder.MLoadGlobal("v1", "s0", "s1", 4, 4);
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

}  // namespace
}  // namespace gpu_model
