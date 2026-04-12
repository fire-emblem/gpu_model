#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include "debug/trace/event_factory.h"
#include "debug/trace/sink.h"
#include "instruction/isa/instruction_builder.h"
#include "runtime/exec_engine.h"

namespace gpu_model {
namespace {

ExecutableKernel BuildSharedBarrierCycleKernel() {
  InstructionBuilder builder;
  builder.SysGlobalIdX("v0");
  builder.SysBlockIdxX("s0");
  builder.SysBlockDimX("s1");
  builder.SMul("s2", "s0", "s1");
  builder.SMov("s3", static_cast<uint64_t>(-1));
  builder.SMul("s4", "s2", "s3");
  builder.VAdd("v1", "v0", "s4");
  builder.VMov("v2", 1);
  builder.MStoreShared("v1", "v2", 4);
  builder.SMov("s5", 64);
  builder.VCmpLtCmask("v1", "s5");
  builder.MaskSaveExec("s10");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("after_extra");
  builder.VMov("v3", 7);
  builder.Label("after_extra");
  builder.MaskRestoreExec("s10");
  builder.SyncBarrier();
  builder.MLoadShared("v4", "v1", 4);
  builder.SLoadArg("s6", 0);
  builder.SMov("s7", 0);
  builder.MLoadGlobal("v5", "s6", "s7", 4);
  builder.SWaitCnt(/*global_count=*/0, /*shared_count=*/UINT32_MAX,
                   /*private_count=*/UINT32_MAX, /*scalar_buffer_count=*/UINT32_MAX);
  builder.BExit();
  return builder.Build("shared_barrier_cycle");
}

uint64_t FirstCycle(const std::vector<TraceEvent>& events,
                    TraceEventKind kind,
                    std::string_view message = {}) {
  for (const auto& event : events) {
    if (event.kind == kind &&
        (message.empty() || event.message.find(std::string(message)) != std::string::npos)) {
      return event.cycle;
    }
  }
  return std::numeric_limits<uint64_t>::max();
}

uint64_t FirstBarrierCycle(const std::vector<TraceEvent>& events, TraceBarrierKind barrier_kind) {
  for (const auto& event : events) {
    if (event.kind == TraceEventKind::Barrier && event.barrier_kind == barrier_kind) {
      return event.cycle;
    }
  }
  return std::numeric_limits<uint64_t>::max();
}

std::set<uint32_t> SlotIdsForPeu(const std::vector<TraceEvent>& events,
                                 TraceEventKind kind,
                                 std::string_view message,
                                 uint32_t peu_id) {
  std::set<uint32_t> slot_ids;
  for (const auto& event : events) {
    if (event.kind != kind || event.peu_id != peu_id) {
      continue;
    }
    if (!message.empty() &&
        event.message.find(std::string(message)) == std::string::npos) {
      continue;
    }
    slot_ids.insert(event.slot_id);
  }
  return slot_ids;
}

std::set<uint32_t> BarrierSlotIdsForPeu(const std::vector<TraceEvent>& events,
                                        TraceBarrierKind barrier_kind,
                                        uint32_t peu_id) {
  std::set<uint32_t> slot_ids;
  for (const auto& event : events) {
    if (event.kind != TraceEventKind::Barrier || event.peu_id != peu_id ||
        event.barrier_kind != barrier_kind) {
      continue;
    }
    slot_ids.insert(event.slot_id);
  }
  return slot_ids;
}

std::string StallMessages(const std::vector<TraceEvent>& events) {
  std::string messages;
  for (const auto& event : events) {
    if (event.kind != TraceEventKind::Stall) {
      continue;
    }
    if (!messages.empty()) {
      messages += "; ";
    }
    messages += event.message;
  }
  return messages;
}

bool HasStallReason(const std::vector<TraceEvent>& events, TraceStallReason reason) {
  for (const auto& event : events) {
    if (TraceHasStallReason(event, reason)) {
      return true;
    }
  }
  return false;
}

uint64_t FirstWaveEventCycle(const std::vector<TraceEvent>& events, TraceEventKind kind) {
  for (const auto& event : events) {
    if (event.kind == kind) {
      return event.cycle;
    }
  }
  return std::numeric_limits<uint64_t>::max();
}

TEST(SharedBarrierCycleTest, BarrierReleaseAllowsWaitingWaveToResume) {
  CollectingTraceSink trace;
  ExecEngine runtime(&trace);
  runtime.SetFixedGlobalMemoryLatency(20);

  const auto kernel = BuildSharedBarrierCycleKernel();
  const uint64_t base_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(base_addr, 9);
  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 320;
  request.config.shared_memory_bytes = 320 * sizeof(int32_t);
  request.args.PushU64(base_addr);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  const uint64_t first_arrive = FirstBarrierCycle(trace.events(), TraceBarrierKind::Arrive);
  const uint64_t release = FirstBarrierCycle(trace.events(), TraceBarrierKind::Release);
  const uint64_t shared_load_issue =
      FirstCycle(trace.events(), TraceEventKind::WaveStep, "ds_read_b32");

  ASSERT_NE(first_arrive, std::numeric_limits<uint64_t>::max());
  ASSERT_NE(release, std::numeric_limits<uint64_t>::max());
  ASSERT_NE(shared_load_issue, std::numeric_limits<uint64_t>::max());
  EXPECT_LT(first_arrive, release);
  EXPECT_GE(shared_load_issue, release);
  EXPECT_TRUE(HasStallReason(trace.events(), TraceStallReason::WaitCntGlobal))
      << StallMessages(trace.events());

  const auto arrive_slot_ids =
      BarrierSlotIdsForPeu(trace.events(), TraceBarrierKind::Arrive, 0);
  const auto exit_slot_ids = SlotIdsForPeu(trace.events(), TraceEventKind::WaveExit, "", 0);
  EXPECT_GE(arrive_slot_ids.size(), 2u);
  EXPECT_GE(exit_slot_ids.size(), 2u);
  for (uint32_t slot_id : arrive_slot_ids) {
    EXPECT_LT(slot_id, 8u);
  }
  for (uint32_t slot_id : exit_slot_ids) {
    EXPECT_LT(slot_id, 8u);
  }
}

TEST(SharedBarrierCycleTest, BarrierLifecycleEmitsWaveWaitAndWaveResumeMarkers) {
  CollectingTraceSink trace;
  ExecEngine runtime(&trace);
  runtime.SetFixedGlobalMemoryLatency(20);

  const auto kernel = BuildSharedBarrierCycleKernel();
  const uint64_t base_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(base_addr, 9);
  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 320;
  request.config.shared_memory_bytes = 320 * sizeof(int32_t);
  request.args.PushU64(base_addr);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  const uint64_t wave_wait_cycle = FirstWaveEventCycle(trace.events(), TraceEventKind::WaveWait);
  const uint64_t barrier_release_cycle =
      FirstBarrierCycle(trace.events(), TraceBarrierKind::Release);
  const uint64_t wave_resume_cycle = FirstWaveEventCycle(trace.events(), TraceEventKind::WaveResume);

  ASSERT_NE(wave_wait_cycle, std::numeric_limits<uint64_t>::max());
  ASSERT_NE(barrier_release_cycle, std::numeric_limits<uint64_t>::max());
  ASSERT_NE(wave_resume_cycle, std::numeric_limits<uint64_t>::max());
  EXPECT_LE(wave_wait_cycle, barrier_release_cycle);
  EXPECT_GE(wave_resume_cycle, barrier_release_cycle);
}

}  // namespace
}  // namespace gpu_model
