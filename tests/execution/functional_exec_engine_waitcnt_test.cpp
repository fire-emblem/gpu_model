#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string_view>
#include <vector>

#include "gpu_model/arch/arch_registry.h"
#include "gpu_model/debug/trace_sink.h"
#include "gpu_model/execution/functional_exec_engine.h"
#include "gpu_model/isa/instruction_builder.h"
#include "gpu_model/isa/opcode.h"
#include "gpu_model/runtime/mapper.h"

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
      : spec(ArchRegistry::Get("c500")),
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
            .arg_load_cycles = 4,
            .issue_cycle_class_overrides = {},
            .issue_cycle_op_overrides = {},
        } {}
};

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
  EXPECT_EQ(engine.RunSequential(), 0u);

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
  EXPECT_EQ(engine.RunSequential(), 0u);

  const auto& events = harness.trace.events();
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

}  // namespace
}  // namespace gpu_model
