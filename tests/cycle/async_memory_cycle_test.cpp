#include <gtest/gtest.h>

#include <cstring>
#include <cstdint>
#include <limits>
#include <set>
#include <string_view>
#include <vector>

#include "gpu_model/debug/trace_sink.h"
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
  bool saw_waitcnt_global_stall = false;
  for (const auto& event : trace.events()) {
    if (event.kind == TraceEventKind::Stall &&
        event.message.find("reason=waitcnt_global") != std::string::npos) {
      saw_waitcnt_global_stall = true;
      break;
    }
  }
  EXPECT_TRUE(saw_waitcnt_global_stall);
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
  bool saw_waitcnt_global_stall = false;
  // Ensure this shared-only waitcnt never gets mislabeled as global.
  for (const auto& event : trace.events()) {
    if (event.kind != TraceEventKind::Stall) {
      continue;
    }
    if (event.message.find("reason=waitcnt_global") != std::string::npos) {
      saw_waitcnt_global_stall = true;
    }
  }
  EXPECT_FALSE(saw_waitcnt_global_stall);
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
