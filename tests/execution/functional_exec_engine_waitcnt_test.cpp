#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "gpu_arch/chip_config/arch_registry.h"
#include "debug/trace/event_factory.h"
#include "debug/trace/sink.h"
#include "execution/functional/functional_exec_engine.h"
#include "instruction/isa/instruction_builder.h"
#include "instruction/isa/opcode.h"
#include "runtime/model_runtime/core/mapper.h"

namespace gpu_model {
namespace {

ConstSegment MakeConstSegment(std::initializer_list<int32_t> values) {
  ConstSegment segment;
  segment.bytes.resize(values.size() * sizeof(int32_t));
  std::memcpy(segment.bytes.data(), values.begin(), segment.bytes.size());
  return segment;
}

ExecutableKernel BuildPendingMemoryBeforeExplicitWaitcntKernel() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SMov("s1", 0);
  builder.MLoadGlobal("v1", "s0", "s1", 4);
  builder.SMov("s2", 7);
  builder.SWaitCnt(/*global_count=*/0, /*shared_count=*/UINT32_MAX,
                   /*private_count=*/UINT32_MAX, /*scalar_buffer_count=*/UINT32_MAX);
  builder.BExit();
  return builder.Build("exec_pending_memory_before_explicit_waitcnt");
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
  return builder.Build("exec_waitcnt_threshold_resume");
}

ExecutableKernel BuildGlobalWaitcntLifecycleKernel() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SMov("s1", 0);
  builder.MLoadGlobal("v1", "s0", "s1", 4);
  builder.SWaitCnt(/*global_count=*/0, /*shared_count=*/UINT32_MAX,
                   /*private_count=*/UINT32_MAX, /*scalar_buffer_count=*/UINT32_MAX);
  builder.SMov("s2", 7);
  builder.BExit();
  return builder.Build("exec_waitcnt_global_lifecycle");
}

ExecutableKernel BuildSharedWaitcntLifecycleKernel() {
  InstructionBuilder builder;
  builder.VMov("v0", 0);
  builder.VMov("v1", 11);
  builder.MStoreShared("v0", "v1", 4);
  builder.SWaitCnt(/*global_count=*/UINT32_MAX, /*shared_count=*/0,
                   /*private_count=*/UINT32_MAX, /*scalar_buffer_count=*/UINT32_MAX);
  builder.SMov("s0", 7);
  builder.BExit();
  return builder.Build("exec_waitcnt_shared_lifecycle");
}

ExecutableKernel BuildPrivateWaitcntLifecycleKernel() {
  InstructionBuilder builder;
  builder.VMov("v0", 0);
  builder.VMov("v1", 11);
  builder.MStorePrivate("v0", "v1", 4);
  builder.SWaitCnt(/*global_count=*/UINT32_MAX, /*shared_count=*/UINT32_MAX,
                   /*private_count=*/0, /*scalar_buffer_count=*/UINT32_MAX);
  builder.SMov("s0", 7);
  builder.BExit();
  return builder.Build("exec_waitcnt_private_lifecycle");
}

ExecutableKernel BuildScalarBufferWaitcntLifecycleKernel() {
  InstructionBuilder builder;
  builder.SMov("s0", 0);
  builder.SBufferLoadDword("s1", "s0", 4);
  builder.SWaitCnt(/*global_count=*/UINT32_MAX, /*shared_count=*/UINT32_MAX,
                   /*private_count=*/UINT32_MAX, /*scalar_buffer_count=*/0);
  builder.SMov("s2", 7);
  builder.BExit();
  return builder.Build("exec_waitcnt_scalar_buffer_lifecycle", {},
                       MakeConstSegment({321}));
}

ExecutableKernel BuildMultiDomainWaitcntResumeKernel() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SMov("s1", 0);
  builder.MLoadGlobal("v1", "s0", "s1", 4);
  builder.VMov("v0", 0);
  builder.VMov("v2", 11);
  builder.MStoreShared("v0", "v2", 4);
  builder.SWaitCnt(/*global_count=*/0, /*shared_count=*/0,
                   /*private_count=*/UINT32_MAX, /*scalar_buffer_count=*/UINT32_MAX);
  builder.SMov("s2", 17);
  builder.SWaitCnt(/*global_count=*/UINT32_MAX, /*shared_count=*/0,
                   /*private_count=*/UINT32_MAX, /*scalar_buffer_count=*/UINT32_MAX);
  builder.SMov("s3", 19);
  builder.BExit();
  return builder.Build("exec_waitcnt_multi_domain_resume");
}

ExecutableKernel BuildBarrierLifecycleKernel() {
  InstructionBuilder builder;
  builder.SysGlobalIdX("v0");
  builder.SyncBarrier();
  builder.SMov("s0", 7);
  builder.BExit();
  return builder.Build("exec_barrier_lifecycle");
}

ExecutableKernel BuildSemanticTraceCoverageKernel() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SMov("s1", 0);
  builder.MLoadGlobal("v1", "s0", "s1", 4);
  builder.SWaitCnt(/*global_count=*/0, /*shared_count=*/UINT32_MAX,
                   /*private_count=*/UINT32_MAX, /*scalar_buffer_count=*/UINT32_MAX);
  builder.SyncWaveBarrier();
  builder.SyncBarrier();
  builder.BExit();
  return builder.Build("exec_semantic_trace_coverage");
}

uint64_t NthInstructionPcWithOpcode(const ExecutableKernel& kernel, Opcode opcode, size_t ordinal) {
  size_t seen = 0;
  for (const auto& [pc, instruction] : kernel.instructions_by_pc()) {
    if (instruction.opcode != opcode) {
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

struct FunctionalExecHarness {
  std::shared_ptr<const GpuArchSpec> spec;
  ExecutableKernel kernel;
  LaunchConfig launch_config;
  KernelArgPack args;
  PlacementMap placement;
  MemorySystem memory;
  CollectingTraceSink trace;
  ExecutionStats stats;
  ExecutionContext context;

  FunctionalExecHarness(ExecutableKernel kernel_in, LaunchConfig launch_config_in)
      : spec(ArchRegistry::Get("mac500")),
        kernel(std::move(kernel_in)),
        launch_config(launch_config_in),
        placement(Mapper::Place(*spec, launch_config)),
        context{
            .spec = *spec,
            .kernel = kernel,
            .launch_config = launch_config,
            .args = args,
            .placement = placement,
            .memory = memory,
            .trace = trace,
            .stats = &stats,
            .cycle = 0,
            .global_memory_latency_cycles = 20,
            .arg_load_cycles = 4,
            .issue_cycle_class_overrides = {},
            .issue_cycle_op_overrides = {},
        } {}
};

FunctionalExecHarness MakeWaitcntHarness(ExecutableKernel kernel,
                                         uint32_t shared_memory_bytes = 0) {
  return FunctionalExecHarness(std::move(kernel),
                               LaunchConfig{.grid_dim_x = 1,
                                            .block_dim_x = 64,
                                            .shared_memory_bytes = shared_memory_bytes});
}

std::vector<TraceEvent> RunHarnessAndCollectTrace(FunctionalExecHarness& harness) {
  FunctionalExecEngine engine(harness.context);
  EXPECT_GT(engine.RunSequential(), 0u);
  return harness.trace.events();
}

bool ContainsWaveStatsMessage(const std::vector<TraceEvent>& events, std::string_view message) {
  return std::any_of(events.begin(), events.end(), [message](const TraceEvent& event) {
    return event.kind == TraceEventKind::WaveStats && event.message == message;
  });
}

bool ContainsStallMessage(const std::vector<TraceEvent>& events, std::string_view message) {
  const TraceStallReason reason = TraceStallReasonFromMessage(message);
  return std::any_of(events.begin(), events.end(), [reason](const TraceEvent& event) {
    return TraceHasStallReason(event, reason);
  });
}

size_t FirstArriveIndexAfter(const std::vector<TraceEvent>& events,
                             size_t start,
                             TraceArriveKind arrive_kind,
                             std::optional<uint64_t> pc = std::nullopt) {
  for (size_t i = start + 1; i < events.size(); ++i) {
    if (events[i].kind != TraceEventKind::Arrive || events[i].arrive_kind != arrive_kind) {
      continue;
    }
    if (pc.has_value() && events[i].pc != *pc) {
      continue;
    }
    return i;
  }
  return std::numeric_limits<size_t>::max();
}

size_t FirstWaveStatsIndexContainingAfter(const std::vector<TraceEvent>& events,
                                          size_t start,
                                          std::string_view snippet) {
  for (size_t i = start + 1; i < events.size(); ++i) {
    if (events[i].kind != TraceEventKind::WaveStats) {
      continue;
    }
    if (events[i].message.find(snippet) == std::string::npos) {
      continue;
    }
    return i;
  }
  return std::numeric_limits<size_t>::max();
}

bool HasResumeOrdering(const std::vector<TraceEvent>& events,
                       uint64_t waitcnt_pc,
                       uint64_t resume_marker_pc,
                       std::string_view stall_reason) {
  const size_t waitcnt_index = FirstEventIndex(events, TraceEventKind::WaveStep, waitcnt_pc);
  const size_t stall_index =
      FirstEventIndex(events, TraceEventKind::Stall, waitcnt_pc, stall_reason);
  const size_t waiting_stats_index = FirstWaveStatsIndexContainingAfter(events, stall_index, "waiting=1");
  const size_t runnable_stats_index =
      FirstWaveStatsIndexContainingAfter(events, waiting_stats_index, "waiting=0");
  const size_t resume_marker_index =
      FirstEventIndexAfter(events, runnable_stats_index, TraceEventKind::WaveStep, resume_marker_pc);

  return waitcnt_index != std::numeric_limits<size_t>::max() &&
         stall_index != std::numeric_limits<size_t>::max() &&
         waiting_stats_index != std::numeric_limits<size_t>::max() &&
         runnable_stats_index != std::numeric_limits<size_t>::max() &&
         resume_marker_index != std::numeric_limits<size_t>::max() &&
         waitcnt_index < stall_index &&
         stall_index < waiting_stats_index &&
         waiting_stats_index < runnable_stats_index &&
         runnable_stats_index < resume_marker_index;
}

bool HasArriveBeforeResumeOrdering(const std::vector<TraceEvent>& events,
                                   uint64_t waitcnt_pc,
                                   uint64_t resume_marker_pc,
                                   std::string_view stall_reason,
                                   TraceArriveKind arrive_kind) {
  const size_t stall_index =
      FirstEventIndex(events, TraceEventKind::Stall, waitcnt_pc, stall_reason);
  const size_t arrive_index = FirstArriveIndexAfter(events, stall_index, arrive_kind, waitcnt_pc);
  const size_t waiting_stats_index =
      FirstWaveStatsIndexContainingAfter(events, stall_index, "waiting=1");
  const size_t runnable_stats_index =
      FirstWaveStatsIndexContainingAfter(events, waiting_stats_index, "waiting=0");
  const size_t resume_marker_index =
      FirstEventIndexAfter(events, runnable_stats_index, TraceEventKind::WaveStep, resume_marker_pc);
  return stall_index != std::numeric_limits<size_t>::max() &&
         arrive_index != std::numeric_limits<size_t>::max() &&
         waiting_stats_index != std::numeric_limits<size_t>::max() &&
         runnable_stats_index != std::numeric_limits<size_t>::max() &&
         resume_marker_index != std::numeric_limits<size_t>::max() &&
         stall_index < waiting_stats_index &&
         waiting_stats_index <= arrive_index &&
         arrive_index < runnable_stats_index &&
         runnable_stats_index < resume_marker_index;
}

void ExpectWaveStateEdgeOrdering(const std::vector<TraceEvent>& events,
                                 uint64_t waitcnt_pc,
                                 uint64_t resume_pc) {
  const size_t wave_wait_index = FirstEventIndex(events, TraceEventKind::WaveWait, waitcnt_pc);
  const size_t wave_arrive_index = FirstEventIndex(events, TraceEventKind::WaveArrive, waitcnt_pc);
  const size_t wave_resume_index = FirstEventIndex(events, TraceEventKind::WaveResume, resume_pc);
  const size_t resumed_step_index = FirstEventIndex(events, TraceEventKind::WaveStep, resume_pc);

  ASSERT_NE(wave_wait_index, std::numeric_limits<size_t>::max());
  ASSERT_NE(wave_arrive_index, std::numeric_limits<size_t>::max());
  ASSERT_NE(wave_resume_index, std::numeric_limits<size_t>::max());
  ASSERT_NE(resumed_step_index, std::numeric_limits<size_t>::max());
  EXPECT_LT(wave_wait_index, wave_arrive_index);
  EXPECT_LT(wave_arrive_index, wave_resume_index);
  EXPECT_LT(wave_resume_index, resumed_step_index);
}

TEST(FunctionalExecEngineWaitcntTest, PendingMemoryDoesNotStallBeforeExplicitWaitcnt) {
  FunctionalExecHarness harness(
      BuildPendingMemoryBeforeExplicitWaitcntKernel(),
      LaunchConfig{.grid_dim_x = 1, .block_dim_x = 64});

  const uint64_t base_addr = harness.memory.AllocateGlobal(2 * sizeof(int32_t));
  harness.memory.StoreGlobalValue<int32_t>(base_addr + 0 * sizeof(int32_t), 11);
  harness.memory.StoreGlobalValue<int32_t>(base_addr + 1 * sizeof(int32_t), 13);
  harness.args.PushU64(base_addr);

  const uint64_t load_pc = NthInstructionPcWithOpcode(harness.kernel, Opcode::MLoadGlobal, 0);
  const uint64_t marker_pc = NthInstructionPcWithOpcode(harness.kernel, Opcode::SMov, 1);
  const uint64_t waitcnt_pc = NthInstructionPcWithOpcode(harness.kernel, Opcode::SWaitCnt, 0);
  ASSERT_NE(load_pc, std::numeric_limits<uint64_t>::max());
  ASSERT_NE(marker_pc, std::numeric_limits<uint64_t>::max());
  ASSERT_NE(waitcnt_pc, std::numeric_limits<uint64_t>::max());

  FunctionalExecEngine engine(harness.context);
  EXPECT_GT(engine.RunSequential(), 0u);

  const auto& events = harness.trace.events();
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

TEST(FunctionalExecEngineWaitcntTest, ResumesWhenStoredThresholdBecomesSatisfied) {
  FunctionalExecHarness harness(
      BuildWaitcntThresholdResumeKernel(),
      LaunchConfig{.grid_dim_x = 1, .block_dim_x = 64});

  const uint64_t base_addr = harness.memory.AllocateGlobal(2 * sizeof(int32_t));
  harness.memory.StoreGlobalValue<int32_t>(base_addr + 0 * sizeof(int32_t), 11);
  harness.memory.StoreGlobalValue<int32_t>(base_addr + 1 * sizeof(int32_t), 13);
  harness.args.PushU64(base_addr);

  const uint64_t waitcnt_one_pc = NthInstructionPcWithOpcode(harness.kernel, Opcode::SWaitCnt, 0);
  const uint64_t waitcnt_zero_pc = NthInstructionPcWithOpcode(harness.kernel, Opcode::SWaitCnt, 1);
  const uint64_t marker_after_threshold_pc =
      NthInstructionPcWithOpcode(harness.kernel, Opcode::SMov, 2);
  const uint64_t marker_after_zero_pc =
      NthInstructionPcWithOpcode(harness.kernel, Opcode::SMov, 3);
  ASSERT_NE(waitcnt_one_pc, std::numeric_limits<uint64_t>::max());
  ASSERT_NE(waitcnt_zero_pc, std::numeric_limits<uint64_t>::max());
  ASSERT_NE(marker_after_threshold_pc, std::numeric_limits<uint64_t>::max());
  ASSERT_NE(marker_after_zero_pc, std::numeric_limits<uint64_t>::max());

  FunctionalExecEngine engine(harness.context);
  EXPECT_GT(engine.RunSequential(), 0u);

  const auto& events = harness.trace.events();
  const size_t first_waitcnt_index =
      FirstEventIndex(events, TraceEventKind::WaveStep, waitcnt_one_pc);
  const size_t first_waitcnt_stall_index =
      FirstEventIndex(events, TraceEventKind::Stall, waitcnt_one_pc, "waitcnt_global");
  const size_t first_arrive_index =
      FirstArriveIndexAfter(events, first_waitcnt_stall_index, TraceArriveKind::Load, waitcnt_one_pc);
  const size_t threshold_resume_marker_index =
      FirstEventIndex(events, TraceEventKind::WaveStep, marker_after_threshold_pc);
  const size_t second_waitcnt_index =
      FirstEventIndex(events, TraceEventKind::WaveStep, waitcnt_zero_pc);
  const size_t second_waitcnt_stall_index =
      FirstEventIndexAfter(events, threshold_resume_marker_index, TraceEventKind::Stall,
                           waitcnt_zero_pc, "waitcnt_global");
  const size_t second_arrive_index =
      FirstArriveIndexAfter(events, second_waitcnt_stall_index, TraceArriveKind::Load, waitcnt_zero_pc);
  const size_t zero_resume_marker_index =
      FirstEventIndex(events, TraceEventKind::WaveStep, marker_after_zero_pc);

  ASSERT_NE(first_waitcnt_index, std::numeric_limits<size_t>::max());
  ASSERT_NE(threshold_resume_marker_index, std::numeric_limits<size_t>::max());
  ASSERT_NE(second_waitcnt_index, std::numeric_limits<size_t>::max());
  ASSERT_NE(zero_resume_marker_index, std::numeric_limits<size_t>::max());
  EXPECT_LT(first_waitcnt_index, threshold_resume_marker_index);
  EXPECT_LT(threshold_resume_marker_index, second_waitcnt_index);
  EXPECT_LT(second_waitcnt_index, zero_resume_marker_index);
  if (first_waitcnt_stall_index != std::numeric_limits<size_t>::max()) {
    ASSERT_NE(first_arrive_index, std::numeric_limits<size_t>::max());
    EXPECT_LT(first_waitcnt_index, first_waitcnt_stall_index);
    EXPECT_LT(first_waitcnt_stall_index, first_arrive_index);
    EXPECT_LT(first_arrive_index, threshold_resume_marker_index);
  }
  if (second_waitcnt_stall_index != std::numeric_limits<size_t>::max()) {
    ASSERT_NE(second_arrive_index, std::numeric_limits<size_t>::max());
    EXPECT_LT(second_waitcnt_index, second_waitcnt_stall_index);
    EXPECT_LT(second_waitcnt_stall_index, second_arrive_index);
    EXPECT_LT(second_arrive_index, zero_resume_marker_index);
  }
}

TEST(FunctionalExecEngineWaitcntTest, GlobalWaitcntTransitionsThroughWaitingAndResume) {
  auto harness = MakeWaitcntHarness(BuildGlobalWaitcntLifecycleKernel());
  const uint64_t base_addr = harness.memory.AllocateGlobal(sizeof(int32_t));
  harness.memory.StoreGlobalValue<int32_t>(base_addr, 11);
  harness.args.PushU64(base_addr);

  const auto events = RunHarnessAndCollectTrace(harness);
  const uint64_t waitcnt_pc = NthInstructionPcWithOpcode(harness.kernel, Opcode::SWaitCnt, 0);
  ASSERT_NE(waitcnt_pc, std::numeric_limits<uint64_t>::max());
  ASSERT_TRUE(harness.kernel.NextPc(waitcnt_pc).has_value());
  const uint64_t resume_marker_pc = *harness.kernel.NextPc(waitcnt_pc);

  EXPECT_TRUE(
      ContainsWaveStatsMessage(events, "launch=1 init=1 active=1 runnable=0 waiting=1 end=0"));
  EXPECT_TRUE(ContainsStallMessage(events, "waitcnt_global"));
  EXPECT_TRUE(HasResumeOrdering(events, waitcnt_pc, resume_marker_pc, "waitcnt_global"));
  EXPECT_TRUE(HasArriveBeforeResumeOrdering(
      events, waitcnt_pc, resume_marker_pc, "waitcnt_global", TraceArriveKind::Load));
}

TEST(FunctionalExecEngineWaitcntTest, GlobalWaitcntEmitsWaveWaitArriveAndResumeMarkers) {
  auto harness = MakeWaitcntHarness(BuildGlobalWaitcntLifecycleKernel());
  const uint64_t base_addr = harness.memory.AllocateGlobal(sizeof(int32_t));
  harness.memory.StoreGlobalValue<int32_t>(base_addr, 11);
  harness.args.PushU64(base_addr);

  const auto events = RunHarnessAndCollectTrace(harness);
  const uint64_t waitcnt_pc = NthInstructionPcWithOpcode(harness.kernel, Opcode::SWaitCnt, 0);
  ASSERT_NE(waitcnt_pc, std::numeric_limits<uint64_t>::max());
  ASSERT_TRUE(harness.kernel.NextPc(waitcnt_pc).has_value());
  const uint64_t resume_pc = *harness.kernel.NextPc(waitcnt_pc);

  const size_t wave_wait_index = FirstEventIndex(events, TraceEventKind::WaveWait, waitcnt_pc);
  const size_t wave_arrive_index = FirstEventIndex(events, TraceEventKind::WaveArrive, waitcnt_pc);
  const size_t wave_resume_index = FirstEventIndex(events, TraceEventKind::WaveResume, resume_pc);
  const size_t resumed_step_index = FirstEventIndex(events, TraceEventKind::WaveStep, resume_pc);

  ASSERT_NE(wave_wait_index, std::numeric_limits<size_t>::max());
  ASSERT_NE(wave_arrive_index, std::numeric_limits<size_t>::max());
  ASSERT_NE(wave_resume_index, std::numeric_limits<size_t>::max());
  ASSERT_NE(resumed_step_index, std::numeric_limits<size_t>::max());
  EXPECT_LT(wave_wait_index, wave_arrive_index);
  EXPECT_LT(wave_arrive_index, wave_resume_index);
  EXPECT_LT(wave_resume_index, resumed_step_index);
}

TEST(FunctionalExecEngineWaitcntTest, SharedWaitcntTransitionsThroughWaitingAndResume) {
  auto harness = MakeWaitcntHarness(BuildSharedWaitcntLifecycleKernel(), /*shared_memory_bytes=*/4);
  const auto events = RunHarnessAndCollectTrace(harness);
  const uint64_t waitcnt_pc = NthInstructionPcWithOpcode(harness.kernel, Opcode::SWaitCnt, 0);
  ASSERT_NE(waitcnt_pc, std::numeric_limits<uint64_t>::max());
  ASSERT_TRUE(harness.kernel.NextPc(waitcnt_pc).has_value());
  const uint64_t resume_marker_pc = *harness.kernel.NextPc(waitcnt_pc);

  EXPECT_TRUE(
      ContainsWaveStatsMessage(events, "launch=1 init=1 active=1 runnable=0 waiting=1 end=0"));
  EXPECT_TRUE(ContainsStallMessage(events, "waitcnt_shared"));
  EXPECT_TRUE(HasResumeOrdering(events, waitcnt_pc, resume_marker_pc, "waitcnt_shared"));
  EXPECT_TRUE(HasArriveBeforeResumeOrdering(
      events, waitcnt_pc, resume_marker_pc, "waitcnt_shared", TraceArriveKind::Shared));
}

TEST(FunctionalExecEngineWaitcntTest, SharedWaitcntEmitsWaveWaitArriveAndResumeMarkers) {
  auto harness = MakeWaitcntHarness(BuildSharedWaitcntLifecycleKernel(), /*shared_memory_bytes=*/4);
  const auto events = RunHarnessAndCollectTrace(harness);
  const uint64_t waitcnt_pc = NthInstructionPcWithOpcode(harness.kernel, Opcode::SWaitCnt, 0);
  ASSERT_NE(waitcnt_pc, std::numeric_limits<uint64_t>::max());
  ASSERT_TRUE(harness.kernel.NextPc(waitcnt_pc).has_value());

  ExpectWaveStateEdgeOrdering(events, waitcnt_pc, *harness.kernel.NextPc(waitcnt_pc));
}

TEST(FunctionalExecEngineWaitcntTest,
     FunctionalTraceUsesCanonicalBarrierArriveReleaseAndExitMessages) {
  FunctionalExecHarness harness(BuildBarrierLifecycleKernel(),
                                LaunchConfig{.grid_dim_x = 1, .block_dim_x = 128});
  const auto events = RunHarnessAndCollectTrace(harness);

  bool saw_arrive = false;
  bool saw_release = false;
  bool saw_exit = false;
  for (const auto& event : events) {
    saw_arrive = saw_arrive || (event.kind == TraceEventKind::Barrier &&
                                event.barrier_kind == TraceBarrierKind::Arrive);
    saw_release = saw_release || (event.kind == TraceEventKind::Barrier &&
                                  event.barrier_kind == TraceBarrierKind::Release);
    saw_exit = saw_exit || (event.kind == TraceEventKind::WaveExit &&
                            event.lifecycle_stage == TraceLifecycleStage::Exit);
  }

  EXPECT_TRUE(saw_arrive);
  EXPECT_TRUE(saw_release);
  EXPECT_TRUE(saw_exit);
}

TEST(FunctionalExecEngineWaitcntTest, PrivateWaitcntTransitionsThroughWaitingAndResume) {
  auto harness = MakeWaitcntHarness(BuildPrivateWaitcntLifecycleKernel());
  const auto events = RunHarnessAndCollectTrace(harness);
  const uint64_t waitcnt_pc = NthInstructionPcWithOpcode(harness.kernel, Opcode::SWaitCnt, 0);
  ASSERT_NE(waitcnt_pc, std::numeric_limits<uint64_t>::max());
  ASSERT_TRUE(harness.kernel.NextPc(waitcnt_pc).has_value());
  const uint64_t resume_marker_pc = *harness.kernel.NextPc(waitcnt_pc);

  EXPECT_TRUE(
      ContainsWaveStatsMessage(events, "launch=1 init=1 active=1 runnable=0 waiting=1 end=0"));
  EXPECT_TRUE(ContainsStallMessage(events, "waitcnt_private"));
  EXPECT_TRUE(HasResumeOrdering(events, waitcnt_pc, resume_marker_pc, "waitcnt_private"));
  EXPECT_TRUE(HasArriveBeforeResumeOrdering(
      events, waitcnt_pc, resume_marker_pc, "waitcnt_private", TraceArriveKind::Private));
}

TEST(FunctionalExecEngineWaitcntTest, PrivateWaitcntEmitsWaveWaitArriveAndResumeMarkers) {
  auto harness = MakeWaitcntHarness(BuildPrivateWaitcntLifecycleKernel());
  const auto events = RunHarnessAndCollectTrace(harness);
  const uint64_t waitcnt_pc = NthInstructionPcWithOpcode(harness.kernel, Opcode::SWaitCnt, 0);
  ASSERT_NE(waitcnt_pc, std::numeric_limits<uint64_t>::max());
  ASSERT_TRUE(harness.kernel.NextPc(waitcnt_pc).has_value());

  ExpectWaveStateEdgeOrdering(events, waitcnt_pc, *harness.kernel.NextPc(waitcnt_pc));
}

TEST(FunctionalExecEngineWaitcntTest, ScalarBufferWaitcntTransitionsThroughWaitingAndResume) {
  auto harness = MakeWaitcntHarness(BuildScalarBufferWaitcntLifecycleKernel());
  const auto events = RunHarnessAndCollectTrace(harness);
  const uint64_t waitcnt_pc = NthInstructionPcWithOpcode(harness.kernel, Opcode::SWaitCnt, 0);
  ASSERT_NE(waitcnt_pc, std::numeric_limits<uint64_t>::max());
  ASSERT_TRUE(harness.kernel.NextPc(waitcnt_pc).has_value());
  const uint64_t resume_marker_pc = *harness.kernel.NextPc(waitcnt_pc);

  EXPECT_TRUE(
      ContainsWaveStatsMessage(events, "launch=1 init=1 active=1 runnable=0 waiting=1 end=0"));
  EXPECT_TRUE(ContainsStallMessage(events, "waitcnt_scalar_buffer"));
  EXPECT_TRUE(
      HasResumeOrdering(events, waitcnt_pc, resume_marker_pc, "waitcnt_scalar_buffer"));
  EXPECT_TRUE(HasArriveBeforeResumeOrdering(events,
                                            waitcnt_pc,
                                            resume_marker_pc,
                                            "waitcnt_scalar_buffer",
                                            TraceArriveKind::ScalarBuffer));
}

TEST(FunctionalExecEngineWaitcntTest, ScalarBufferWaitcntEmitsWaveWaitArriveAndResumeMarkers) {
  auto harness = MakeWaitcntHarness(BuildScalarBufferWaitcntLifecycleKernel());
  const auto events = RunHarnessAndCollectTrace(harness);
  const uint64_t waitcnt_pc = NthInstructionPcWithOpcode(harness.kernel, Opcode::SWaitCnt, 0);
  ASSERT_NE(waitcnt_pc, std::numeric_limits<uint64_t>::max());
  ASSERT_TRUE(harness.kernel.NextPc(waitcnt_pc).has_value());

  ExpectWaveStateEdgeOrdering(events, waitcnt_pc, *harness.kernel.NextPc(waitcnt_pc));
}

TEST(FunctionalExecEngineWaitcntTest, WaitcntResumeRequiresAllStoredThresholdDomains) {
  auto harness =
      MakeWaitcntHarness(BuildMultiDomainWaitcntResumeKernel(), /*shared_memory_bytes=*/4);
  const uint64_t base_addr = harness.memory.AllocateGlobal(sizeof(int32_t));
  harness.memory.StoreGlobalValue<int32_t>(base_addr, 11);
  harness.args.PushU64(base_addr);

  const auto events = RunHarnessAndCollectTrace(harness);
  const uint64_t combined_waitcnt_pc = NthInstructionPcWithOpcode(harness.kernel, Opcode::SWaitCnt, 0);
  const uint64_t shared_only_waitcnt_pc = NthInstructionPcWithOpcode(harness.kernel, Opcode::SWaitCnt, 1);
  const uint64_t marker_after_combined_pc = NthInstructionPcWithOpcode(harness.kernel, Opcode::SMov, 1);
  const uint64_t marker_after_shared_only_pc = NthInstructionPcWithOpcode(harness.kernel, Opcode::SMov, 2);
  ASSERT_NE(combined_waitcnt_pc, std::numeric_limits<uint64_t>::max());
  ASSERT_NE(shared_only_waitcnt_pc, std::numeric_limits<uint64_t>::max());
  ASSERT_NE(marker_after_combined_pc, std::numeric_limits<uint64_t>::max());
  ASSERT_NE(marker_after_shared_only_pc, std::numeric_limits<uint64_t>::max());

  const bool saw_combined_stall = ContainsStallMessage(events, "waitcnt_global");
  EXPECT_EQ(FirstEventIndex(events, TraceEventKind::Stall, shared_only_waitcnt_pc, "waitcnt_shared"),
            std::numeric_limits<size_t>::max());
  if (saw_combined_stall) {
    EXPECT_TRUE(
        HasResumeOrdering(events, combined_waitcnt_pc, marker_after_combined_pc, "waitcnt_global"));
  } else {
    EXPECT_LT(FirstEventIndex(events, TraceEventKind::WaveStep, combined_waitcnt_pc),
              FirstEventIndex(events, TraceEventKind::WaveStep, marker_after_combined_pc));
  }
  EXPECT_EQ(
      FirstEventIndexAfter(events,
                           FirstEventIndex(events, TraceEventKind::WaveStep, marker_after_combined_pc),
                           TraceEventKind::Stall,
                           shared_only_waitcnt_pc,
                           "waitcnt_shared"),
      std::numeric_limits<size_t>::max());
  EXPECT_NE(FirstEventIndex(events, TraceEventKind::WaveStep, marker_after_shared_only_pc),
            std::numeric_limits<size_t>::max());

  const size_t combined_waitcnt_stall_index =
      FirstEventIndex(events, TraceEventKind::Stall, combined_waitcnt_pc, "waitcnt_global");
  if (combined_waitcnt_stall_index != std::numeric_limits<size_t>::max()) {
    const TraceEvent& combined_waitcnt_stall = events.at(combined_waitcnt_stall_index);
    EXPECT_TRUE(TraceHasWaitcntState(combined_waitcnt_stall));
    EXPECT_TRUE(combined_waitcnt_stall.waitcnt_state.blocked_global);
    EXPECT_TRUE(combined_waitcnt_stall.waitcnt_state.blocked_shared);
    EXPECT_FALSE(combined_waitcnt_stall.waitcnt_state.blocked_private);
    EXPECT_FALSE(combined_waitcnt_stall.waitcnt_state.blocked_scalar_buffer);
    EXPECT_EQ(combined_waitcnt_stall.waitcnt_state.threshold_global, 0u);
    EXPECT_EQ(combined_waitcnt_stall.waitcnt_state.threshold_shared, 0u);
    EXPECT_EQ(combined_waitcnt_stall.waitcnt_state.pending_global, 1u);
    EXPECT_EQ(combined_waitcnt_stall.waitcnt_state.pending_shared, 1u);
  }
}

TEST(FunctionalExecEngineWaitcntTest,
     FunctionalTraceUsesCanonicalSemanticFactoryMessagesAndLogicalSlots) {
  auto harness = MakeWaitcntHarness(BuildSemanticTraceCoverageKernel());
  const uint64_t base_addr = harness.memory.AllocateGlobal(sizeof(int32_t));
  harness.memory.StoreGlobalValue<int32_t>(base_addr, 11);
  harness.args.PushU64(base_addr);

  const auto events = RunHarnessAndCollectTrace(harness);

  bool saw_launch = false;
  bool saw_commit = false;
  bool saw_barrier_wave = false;
  bool saw_barrier_arrive = false;
  bool saw_barrier_release = false;
  bool saw_wait_stall = false;
  bool saw_memory_arrive = false;
  bool saw_wave_exit = false;

  for (const auto& event : events) {
    if (event.kind == TraceEventKind::WaveLaunch &&
        event.lifecycle_stage == TraceLifecycleStage::Launch) {
      saw_launch = true;
      EXPECT_TRUE(TraceHasSlotModel(event, TraceSlotModelKind::LogicalUnbounded));
    }
    if (event.kind == TraceEventKind::Commit && event.display_name == "commit") {
      saw_commit = true;
      EXPECT_TRUE(TraceHasSlotModel(event, TraceSlotModelKind::LogicalUnbounded));
    }
    if (event.kind == TraceEventKind::Barrier && event.barrier_kind == TraceBarrierKind::Wave) {
      saw_barrier_wave = true;
      EXPECT_TRUE(TraceHasSlotModel(event, TraceSlotModelKind::LogicalUnbounded));
    }
    if (event.kind == TraceEventKind::Barrier && event.barrier_kind == TraceBarrierKind::Arrive) {
      saw_barrier_arrive = true;
      EXPECT_TRUE(TraceHasSlotModel(event, TraceSlotModelKind::LogicalUnbounded));
    }
    if (event.kind == TraceEventKind::Barrier && event.barrier_kind == TraceBarrierKind::Release) {
      saw_barrier_release = true;
    }
    if (event.kind == TraceEventKind::Stall &&
        event.stall_reason == TraceStallReason::WaitCntGlobal) {
      saw_wait_stall = true;
      EXPECT_TRUE(TraceHasSlotModel(event, TraceSlotModelKind::LogicalUnbounded));
    }
    if (event.kind == TraceEventKind::Arrive && event.arrive_kind == TraceArriveKind::Load) {
      saw_memory_arrive = true;
      EXPECT_TRUE(TraceHasSlotModel(event, TraceSlotModelKind::LogicalUnbounded));
    }
    if (event.kind == TraceEventKind::WaveExit &&
        event.lifecycle_stage == TraceLifecycleStage::Exit) {
      saw_wave_exit = true;
      EXPECT_TRUE(TraceHasSlotModel(event, TraceSlotModelKind::LogicalUnbounded));
    }
    EXPECT_NE(event.message, kTraceExitMessage);
  }

  EXPECT_TRUE(saw_launch);
  EXPECT_TRUE(saw_commit);
  EXPECT_TRUE(saw_barrier_wave);
  EXPECT_TRUE(saw_barrier_arrive);
  EXPECT_TRUE(saw_barrier_release);
  EXPECT_TRUE(saw_wait_stall);
  EXPECT_TRUE(saw_memory_arrive);
  EXPECT_TRUE(saw_wave_exit);
}

}  // namespace
}  // namespace gpu_model
