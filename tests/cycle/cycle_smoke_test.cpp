#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <optional>

#include "gpu_model/arch/arch_registry.h"
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

bool HasStallReason(const std::vector<TraceEvent>& events, TraceStallReason reason) {
  return std::any_of(events.begin(), events.end(), [reason](const TraceEvent& event) {
    return TraceHasStallReason(event, reason);
  });
}

size_t FirstEventIndexForWave(const std::vector<TraceEvent>& events,
                              uint32_t wave_id,
                              uint32_t peu_id,
                              TraceEventKind kind,
                              std::optional<uint64_t> pc = std::nullopt) {
  for (size_t i = 0; i < events.size(); ++i) {
    if (events[i].wave_id != wave_id || events[i].peu_id != peu_id || events[i].kind != kind) {
      continue;
    }
    if (pc.has_value() && events[i].pc != *pc) {
      continue;
    }
    return i;
  }
  return std::numeric_limits<size_t>::max();
}

size_t FirstEventIndexForWaveAfter(const std::vector<TraceEvent>& events,
                                   size_t start,
                                   uint32_t wave_id,
                                   uint32_t peu_id,
                                   TraceEventKind kind,
                                   std::optional<uint64_t> pc = std::nullopt) {
  for (size_t i = start + 1; i < events.size(); ++i) {
    if (events[i].wave_id != wave_id || events[i].peu_id != peu_id || events[i].kind != kind) {
      continue;
    }
    if (pc.has_value() && events[i].pc != *pc) {
      continue;
    }
    return i;
  }
  return std::numeric_limits<size_t>::max();
}

ExecutableKernel BuildSamePeuReadyNotSelectedKernel() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SLoadArg("s1", 1);
  builder.SysGlobalIdX("v0");

  builder.SMov("s2", 64);
  builder.VCmpLtCmask("v0", "s2");
  builder.MaskSaveExec("s10");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("after_wait_wave");
  builder.MLoadGlobal("v1", "s0", "v0", 4);
  builder.SWaitCnt(/*global_count=*/0, /*shared_count=*/UINT32_MAX,
                   /*private_count=*/UINT32_MAX, /*scalar_buffer_count=*/UINT32_MAX);
  builder.Label("after_wait_wave");
  builder.MaskRestoreExec("s10");

  builder.VMov("v4", 21);
  builder.VAdd("v5", "v4", "v4");
  builder.VAdd("v6", "v5", "v4");
  builder.MStoreGlobal("s1", "v0", "v6", 4);
  builder.BExit();
  return builder.Build("cycle_same_peu_ready_not_selected");
}

TEST(CycleSmokeTest, ScalarAndVectorOpsConsumeFourCyclesEach) {
  InstructionBuilder builder;
  builder.SMov("s0", 7);
  builder.VMov("v0", "s0");
  builder.BExit();
  const auto kernel = builder.Build("tiny_cycle_kernel");

  ExecEngine runtime;
  ConfigureZeroFrontendTiming(runtime);
  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  EXPECT_EQ(result.total_cycles, 12u);
}

TEST(CycleSmokeTest, ConsecutiveKernelLaunchesIncludeDeviceGap) {
  InstructionBuilder builder;
  builder.BExit();
  const auto kernel = builder.Build("launch_gap_kernel");

  ExecEngine runtime;
  ConfigureZeroFrontendTiming(runtime);
  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;

  const auto first = runtime.Launch(request);
  ASSERT_TRUE(first.ok) << first.error_message;
  EXPECT_EQ(first.submit_cycle, 0u);
  EXPECT_EQ(first.begin_cycle, 0u);
  EXPECT_EQ(first.end_cycle, 4u);
  EXPECT_EQ(first.total_cycles, 4u);

  const auto second = runtime.Launch(request);
  ASSERT_TRUE(second.ok) << second.error_message;
  EXPECT_EQ(second.submit_cycle, first.end_cycle + 8u);
  EXPECT_EQ(second.begin_cycle, second.submit_cycle);
  EXPECT_EQ(second.end_cycle, second.begin_cycle + 4u);
  EXPECT_EQ(second.total_cycles, 4u);
  EXPECT_EQ(runtime.device_cycle(), second.end_cycle);
}

TEST(CycleSmokeTest, FrontendLatenciesAdvanceCycleWithoutTraceSink) {
  InstructionBuilder builder;
  builder.BExit();
  const auto kernel = builder.Build("frontend_latency_kernel");

  ExecEngine runtime;
  runtime.SetLaunchTimingProfile(/*kernel_launch_gap_cycles=*/8,
                                 /*kernel_launch_cycles=*/0,
                                 /*block_launch_cycles=*/0,
                                 /*wave_generation_cycles=*/128,
                                 /*wave_dispatch_cycles=*/256,
                                 /*wave_launch_cycles=*/0,
                                 /*warp_switch_cycles=*/1,
                                 /*arg_load_cycles=*/4);

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  EXPECT_EQ(result.begin_cycle, 0u);
  EXPECT_EQ(result.total_cycles, 388u);
  EXPECT_EQ(result.end_cycle, 388u);
}

TEST(CycleSmokeTest, QueuesBlocksWhenGridExceedsPhysicalApCount) {
  const auto spec = ArchRegistry::Get("c500");
  ASSERT_NE(spec, nullptr);

  CollectingTraceSink trace;
  ExecEngine runtime(&trace);
  ConfigureZeroFrontendTiming(runtime);

  InstructionBuilder builder;
  builder.BExit();
  const auto kernel = builder.Build("queued_blocks_kernel");

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = spec->total_ap_count() + 1;
  request.config.block_dim_x = 64;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  EXPECT_EQ(result.total_cycles, 8u);

  uint32_t block_launches = 0;
  uint32_t wave_launches = 0;
  uint64_t wrapped_block_launch_cycle = 0;
  for (const auto& event : trace.events()) {
    if (event.kind == TraceEventKind::BlockLaunch) {
      ++block_launches;
      if (event.block_id == spec->total_ap_count()) {
        wrapped_block_launch_cycle = event.cycle;
      }
    } else if (event.kind == TraceEventKind::WaveLaunch) {
      ++wave_launches;
    }
  }

  EXPECT_EQ(block_launches, spec->total_ap_count() + 1);
  EXPECT_EQ(wave_launches, spec->total_ap_count() + 1);
  EXPECT_EQ(wrapped_block_launch_cycle, 0u);
  EXPECT_FALSE(HasStallReason(trace.events(), TraceStallReason::WarpSwitch));
}

TEST(CycleSmokeTest, AsyncLoadDoesNotPromoteOverflowResidentWavesPerPeu) {
  CollectingTraceSink trace;
  ExecEngine runtime(&trace);
  runtime.SetFixedGlobalMemoryLatency(40);
  runtime.SetLaunchTimingProfile(/*kernel_launch_gap_cycles=*/8,
                                 /*kernel_launch_cycles=*/0,
                                 /*block_launch_cycles=*/0,
                                 /*wave_launch_cycles=*/0,
                                 /*warp_switch_cycles=*/1,
                                 /*arg_load_cycles=*/4);

  const uint64_t base_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(base_addr, 7);

  InstructionBuilder builder;
  builder.SMov("s0", base_addr);
  builder.SMov("s1", 0);
  builder.MLoadGlobal("v0", "s0", "s1", 4);
  builder.BExit();
  const auto kernel = builder.Build("resident_overflow_kernel");

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 1280;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  uint32_t wave_launches_at_0 = 0;
  uint32_t wave_launches_at_1 = 0;
  for (const auto& event : trace.events()) {
    if (event.kind != TraceEventKind::WaveLaunch) {
      continue;
    }
    if (event.cycle == 0u) {
      ++wave_launches_at_0;
    } else if (event.cycle == 1u) {
      ++wave_launches_at_1;
    }
  }

  EXPECT_EQ(wave_launches_at_0, 16u);
  EXPECT_EQ(wave_launches_at_1, 0u);
  EXPECT_GT(result.total_cycles, 0u);
}

TEST(CycleSmokeTest, ReadyWavesIssueRoundRobinWithinPeu) {
  CollectingTraceSink trace;
  ExecEngine runtime(&trace);
  runtime.SetLaunchTimingProfile(/*kernel_launch_gap_cycles=*/8,
                                 /*kernel_launch_cycles=*/0,
                                 /*block_launch_cycles=*/0,
                                 /*wave_launch_cycles=*/0,
                                 /*warp_switch_cycles=*/1,
                                 /*arg_load_cycles=*/4);

  InstructionBuilder builder;
  builder.SMov("s0", 1);
  builder.VMov("v0", "s0");
  builder.VAdd("v1", "v0", "s0");
  builder.BExit();
  const auto kernel = builder.Build("round_robin_issue_kernel");

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 128;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  std::vector<uint32_t> issued_waves;
  for (const auto& event : trace.events()) {
    if (event.kind == TraceEventKind::WaveStep) {
      issued_waves.push_back(event.wave_id);
    }
  }

  ASSERT_GE(issued_waves.size(), 6u);
  EXPECT_EQ(issued_waves[0], 0u);
  EXPECT_EQ(issued_waves[1], 1u);
  EXPECT_EQ(issued_waves[2], 0u);
  EXPECT_EQ(issued_waves[3], 1u);
  EXPECT_EQ(issued_waves[4], 0u);
  EXPECT_EQ(issued_waves[5], 1u);
}

TEST(CycleSmokeTest, ReadyDoesNotGuaranteeImmediateConsumerIssue) {
  CollectingTraceSink trace;
  ExecEngine runtime(&trace);
  runtime.SetFixedGlobalMemoryLatency(20);

  const auto kernel = BuildSamePeuReadyNotSelectedKernel();
  constexpr uint32_t kBlockDim = 320;
  constexpr uint32_t kElementCount = kBlockDim;
  const uint64_t in_addr = runtime.memory().AllocateGlobal(kElementCount * sizeof(int32_t));
  const uint64_t out_addr = runtime.memory().AllocateGlobal(kElementCount * sizeof(int32_t));
  for (uint32_t i = 0; i < kElementCount; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(in_addr + i * sizeof(int32_t),
                                               static_cast<int32_t>(100 + i));
    runtime.memory().StoreGlobalValue<int32_t>(out_addr + i * sizeof(int32_t), -1);
  }

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = kBlockDim;
  request.args.PushU64(in_addr);
  request.args.PushU64(out_addr);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  uint64_t resumed_wave0_cycle = std::numeric_limits<uint64_t>::max();
  uint64_t resumed_wave0_consumer_cycle = std::numeric_limits<uint64_t>::max();
  for (const auto& event : trace.events()) {
    if (event.kind == TraceEventKind::Arrive && event.wave_id == 0 && event.peu_id == 0 &&
        event.arrive_progress == TraceArriveProgressKind::Resume) {
      resumed_wave0_cycle = std::min(resumed_wave0_cycle, event.cycle);
    }
    if (event.kind == TraceEventKind::WaveStep && event.wave_id == 0 && event.peu_id == 0 &&
        event.message.find("v_add_i32") != std::string::npos) {
      resumed_wave0_consumer_cycle = std::min(resumed_wave0_consumer_cycle, event.cycle);
    }
  }

  ASSERT_NE(resumed_wave0_cycle, std::numeric_limits<uint64_t>::max());
  ASSERT_NE(resumed_wave0_consumer_cycle, std::numeric_limits<uint64_t>::max());
  EXPECT_GT(resumed_wave0_consumer_cycle, resumed_wave0_cycle);
}

TEST(CycleSmokeTest, ResumeSelectionAndIssueOrderingStayObservable) {
  CollectingTraceSink trace;
  ExecEngine runtime(&trace);
  runtime.SetFixedGlobalMemoryLatency(20);

  const auto kernel = BuildSamePeuReadyNotSelectedKernel();
  const uint64_t resume_pc = [&]() {
    for (const auto& [pc, instruction] : kernel.instructions_by_pc()) {
      if (instruction.opcode == Opcode::VAdd) {
        return pc;
      }
    }
    return std::numeric_limits<uint64_t>::max();
  }();
  ASSERT_NE(resume_pc, std::numeric_limits<uint64_t>::max());

  constexpr uint32_t kBlockDim = 320;
  constexpr uint32_t kElementCount = kBlockDim;
  const uint64_t in_addr = runtime.memory().AllocateGlobal(kElementCount * sizeof(int32_t));
  const uint64_t out_addr = runtime.memory().AllocateGlobal(kElementCount * sizeof(int32_t));
  for (uint32_t i = 0; i < kElementCount; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(in_addr + i * sizeof(int32_t),
                                               static_cast<int32_t>(100 + i));
    runtime.memory().StoreGlobalValue<int32_t>(out_addr + i * sizeof(int32_t), -1);
  }

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = kBlockDim;
  request.args.PushU64(in_addr);
  request.args.PushU64(out_addr);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  const auto& events = trace.events();
  const size_t wave_resume_index = FirstEventIndexForWave(
      events, /*wave_id=*/0, /*peu_id=*/0, TraceEventKind::WaveResume);
  const size_t issue_select_index = FirstEventIndexForWaveAfter(
      events, wave_resume_index, /*wave_id=*/0, /*peu_id=*/0, TraceEventKind::IssueSelect, resume_pc);
  const size_t resumed_step_index = FirstEventIndexForWave(
      events, /*wave_id=*/0, /*peu_id=*/0, TraceEventKind::WaveStep, resume_pc);
  const size_t switch_away_index = FirstEventIndexForWaveAfter(
      events, wave_resume_index, /*wave_id=*/0, /*peu_id=*/0, TraceEventKind::WaveSwitchAway, resume_pc);

  ASSERT_NE(wave_resume_index, std::numeric_limits<size_t>::max());
  ASSERT_NE(issue_select_index, std::numeric_limits<size_t>::max());
  ASSERT_NE(resumed_step_index, std::numeric_limits<size_t>::max());
  EXPECT_TRUE(switch_away_index == std::numeric_limits<size_t>::max() ||
              wave_resume_index < switch_away_index);
  EXPECT_LT(wave_resume_index, issue_select_index);
  EXPECT_LT(issue_select_index, resumed_step_index);
}

TEST(CycleSmokeTest, SamePeuIssueCyclesStayOnFourCycleGrid) {
  CollectingTraceSink trace;
  ExecEngine runtime(&trace);
  runtime.SetFixedGlobalMemoryLatency(20);

  const auto kernel = BuildSamePeuReadyNotSelectedKernel();

  constexpr uint32_t kBlockDim = 320;
  constexpr uint32_t kElementCount = kBlockDim;
  const uint64_t in_addr = runtime.memory().AllocateGlobal(kElementCount * sizeof(int32_t));
  const uint64_t out_addr = runtime.memory().AllocateGlobal(kElementCount * sizeof(int32_t));
  for (uint32_t i = 0; i < kElementCount; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(in_addr + i * sizeof(int32_t),
                                               static_cast<int32_t>(100 + i));
    runtime.memory().StoreGlobalValue<int32_t>(out_addr + i * sizeof(int32_t), -1);
  }

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = kBlockDim;
  request.args.PushU64(in_addr);
  request.args.PushU64(out_addr);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  bool saw_wave_step = false;
  for (const auto& event : trace.events()) {
    if (event.kind != TraceEventKind::WaveStep || event.block_id != 0 || event.peu_id != 0) {
      continue;
    }
    saw_wave_step = true;
    EXPECT_EQ(event.cycle % 4u, 0u) << event.wave_id << " pc=0x" << std::hex << event.pc;
  }
  EXPECT_TRUE(saw_wave_step);
}

TEST(CycleSmokeTest, VectorIssueLimitOverrideAllowsTwoWaveBundleIssue) {
  const auto spec = ArchRegistry::Get("c500");
  ASSERT_NE(spec, nullptr);

  CollectingTraceSink baseline_trace;
  ExecEngine runtime(&baseline_trace);
  ConfigureZeroFrontendTiming(runtime);

  InstructionBuilder builder;
  builder.VMov("v0", 1);
  builder.BExit();
  const auto kernel = builder.Build("vector_bundle_issue_kernel");

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = spec->peu_per_ap * 64 + 64;

  const auto baseline = runtime.Launch(request);
  ASSERT_TRUE(baseline.ok) << baseline.error_message;

  runtime.ResetDeviceCycle();
  CollectingTraceSink widened_trace;
  ExecEngine widened_runtime(&widened_trace);
  widened_runtime.SetLaunchTimingProfile(/*kernel_launch_gap_cycles=*/8,
                                         /*kernel_launch_cycles=*/0,
                                         /*block_launch_cycles=*/0,
                                         /*wave_launch_cycles=*/0,
                                         /*warp_switch_cycles=*/1,
                                         /*arg_load_cycles=*/4);
  ArchitecturalIssueLimits widened_limits = DefaultArchitecturalIssueLimits();
  widened_limits.vector_alu = 2;
  widened_runtime.SetCycleIssueLimits(widened_limits);

  const auto widened = widened_runtime.Launch(request);
  ASSERT_TRUE(widened.ok) << widened.error_message;

  uint32_t baseline_wave_steps_at_0 = 0;
  for (const auto& event : baseline_trace.events()) {
    if (event.kind == TraceEventKind::WaveStep && event.cycle == 0u) {
      ++baseline_wave_steps_at_0;
    }
  }

  uint32_t widened_wave_steps_at_0 = 0;
  for (const auto& event : widened_trace.events()) {
    if (event.kind == TraceEventKind::WaveStep && event.cycle == 0u) {
      ++widened_wave_steps_at_0;
    }
  }

  EXPECT_EQ(baseline_wave_steps_at_0, spec->peu_per_ap);
  EXPECT_GT(widened_wave_steps_at_0, baseline_wave_steps_at_0);
}

TEST(CycleSmokeTest, IssuePolicyOverrideCanWidenVectorBundleWithoutSeparateLimitOverride) {
  const auto spec = ArchRegistry::Get("c500");
  ASSERT_NE(spec, nullptr);

  InstructionBuilder builder;
  builder.VMov("v0", 1);
  builder.BExit();
  const auto kernel = builder.Build("vector_policy_bundle_issue_kernel");

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = spec->peu_per_ap * 64 + 64;

  CollectingTraceSink baseline_trace;
  ExecEngine baseline_runtime(&baseline_trace);
  ConfigureZeroFrontendTiming(baseline_runtime);
  const auto baseline = baseline_runtime.Launch(request);
  ASSERT_TRUE(baseline.ok) << baseline.error_message;

  CollectingTraceSink grouped_trace;
  ExecEngine grouped_runtime(&grouped_trace);
  ConfigureZeroFrontendTiming(grouped_runtime);
  ArchitecturalIssueLimits widened_limits = DefaultArchitecturalIssueLimits();
  widened_limits.vector_alu = 2;
  auto grouped_policy = ArchitecturalIssuePolicyFromLimits(widened_limits);
  grouped_runtime.SetCycleIssuePolicy(grouped_policy);

  const auto grouped = grouped_runtime.Launch(request);
  ASSERT_TRUE(grouped.ok) << grouped.error_message;

  uint32_t baseline_wave_steps_at_0 = 0;
  for (const auto& event : baseline_trace.events()) {
    if (event.kind == TraceEventKind::WaveStep && event.cycle == 0u) {
      ++baseline_wave_steps_at_0;
    }
  }

  uint32_t grouped_wave_steps_at_0 = 0;
  for (const auto& event : grouped_trace.events()) {
    if (event.kind == TraceEventKind::WaveStep && event.cycle == 0u) {
      ++grouped_wave_steps_at_0;
    }
  }

  EXPECT_EQ(baseline_wave_steps_at_0, spec->peu_per_ap);
  EXPECT_GT(grouped_wave_steps_at_0, baseline_wave_steps_at_0);
}

TEST(CycleSmokeTest, IssueLimitOverrideTakesPriorityOverIssuePolicyTypeLimits) {
  const auto spec = ArchRegistry::Get("c500");
  ASSERT_NE(spec, nullptr);

  InstructionBuilder builder;
  builder.VMov("v0", 1);
  builder.BExit();
  const auto kernel = builder.Build("vector_policy_limit_precedence_kernel");

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = spec->peu_per_ap * 64 + 64;

  CollectingTraceSink baseline_trace;
  ExecEngine baseline_runtime(&baseline_trace);
  ConfigureZeroFrontendTiming(baseline_runtime);
  const auto baseline = baseline_runtime.Launch(request);
  ASSERT_TRUE(baseline.ok) << baseline.error_message;

  CollectingTraceSink limited_trace;
  ExecEngine limited_runtime(&limited_trace);
  ConfigureZeroFrontendTiming(limited_runtime);
  ArchitecturalIssueLimits widened_limits = DefaultArchitecturalIssueLimits();
  widened_limits.vector_alu = 2;
  limited_runtime.SetCycleIssuePolicy(ArchitecturalIssuePolicyFromLimits(widened_limits));
  limited_runtime.SetCycleIssueLimits(DefaultArchitecturalIssueLimits());

  const auto limited = limited_runtime.Launch(request);
  ASSERT_TRUE(limited.ok) << limited.error_message;

  uint32_t baseline_wave_steps_at_0 = 0;
  for (const auto& event : baseline_trace.events()) {
    if (event.kind == TraceEventKind::WaveStep && event.cycle == 0u) {
      ++baseline_wave_steps_at_0;
    }
  }

  uint32_t limited_wave_steps_at_0 = 0;
  for (const auto& event : limited_trace.events()) {
    if (event.kind == TraceEventKind::WaveStep && event.cycle == 0u) {
      ++limited_wave_steps_at_0;
    }
  }

  EXPECT_EQ(baseline_wave_steps_at_0, spec->peu_per_ap);
  EXPECT_EQ(limited_wave_steps_at_0, baseline_wave_steps_at_0);
  EXPECT_EQ(limited.total_cycles, baseline.total_cycles);
}

TEST(CycleSmokeTest, IssueCycleClassOverrideChangesSelectedInstructionCategory) {
  ConstSegment const_segment;
  const int32_t value = 7;
  const_segment.bytes.resize(sizeof(value));
  std::memcpy(const_segment.bytes.data(), &value, sizeof(value));

  InstructionBuilder builder;
  builder.SMov("s0", 0);
  builder.SBufferLoadDword("s1", "s0", 4);
  builder.BExit();
  const auto kernel = builder.Build("class_override_kernel", {}, std::move(const_segment));

  ExecEngine runtime;
  ConfigureZeroFrontendTiming(runtime);
  IssueCycleClassOverridesSpec class_overrides;
  class_overrides.scalar_memory = 6;
  runtime.SetIssueCycleClassOverrides(class_overrides);

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  EXPECT_EQ(result.total_cycles, 14u);
}

TEST(CycleSmokeTest, IssueCycleOpOverrideTakesPriorityOverClassOverride) {
  ConstSegment const_segment;
  const int32_t value = 7;
  const_segment.bytes.resize(sizeof(value));
  std::memcpy(const_segment.bytes.data(), &value, sizeof(value));

  InstructionBuilder builder;
  builder.SMov("s0", 0);
  builder.SBufferLoadDword("s1", "s0", 4);
  builder.BExit();
  const auto kernel = builder.Build("op_override_kernel", {}, std::move(const_segment));

  ExecEngine runtime;
  ConfigureZeroFrontendTiming(runtime);
  IssueCycleClassOverridesSpec class_overrides;
  class_overrides.scalar_memory = 6;
  runtime.SetIssueCycleClassOverrides(class_overrides);
  IssueCycleOpOverridesSpec op_overrides;
  op_overrides.s_buffer_load_dword = 9;
  runtime.SetIssueCycleOpOverrides(op_overrides);

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  EXPECT_EQ(result.total_cycles, 18u);
}

}  // namespace
}  // namespace gpu_model
