#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>

#include "gpu_model/arch/arch_registry.h"
#include "gpu_model/debug/trace_sink.h"
#include "gpu_model/isa/instruction_builder.h"
#include "gpu_model/runtime/runtime_engine.h"

namespace gpu_model {
namespace {

TEST(CycleSmokeTest, ScalarAndVectorOpsConsumeFourCyclesEach) {
  InstructionBuilder builder;
  builder.SMov("s0", 7);
  builder.VMov("v0", "s0");
  builder.BExit();
  const auto kernel = builder.Build("tiny_cycle_kernel");

  RuntimeEngine runtime;
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

  RuntimeEngine runtime;
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

TEST(CycleSmokeTest, QueuesBlocksWhenGridExceedsPhysicalApCount) {
  const auto spec = ArchRegistry::Get("c500");
  ASSERT_NE(spec, nullptr);

  CollectingTraceSink trace;
  RuntimeEngine runtime(&trace);

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
  EXPECT_EQ(result.total_cycles, 9u);

  uint32_t block_launches = 0;
  uint32_t wave_launches = 0;
  uint64_t wrapped_block_launch_cycle = 0;
  bool saw_warp_switch = false;
  for (const auto& event : trace.events()) {
    if (event.kind == TraceEventKind::BlockLaunch) {
      ++block_launches;
      if (event.block_id == spec->total_ap_count()) {
        wrapped_block_launch_cycle = event.cycle;
      }
    } else if (event.kind == TraceEventKind::WaveLaunch) {
      ++wave_launches;
    } else if (event.kind == TraceEventKind::Stall &&
               event.message.find("reason=warp_switch") != std::string::npos) {
      saw_warp_switch = true;
    }
  }

  EXPECT_EQ(block_launches, spec->total_ap_count() + 1);
  EXPECT_EQ(wave_launches, spec->total_ap_count() + 1);
  EXPECT_EQ(wrapped_block_launch_cycle, 0u);
  EXPECT_TRUE(saw_warp_switch);
}

TEST(CycleSmokeTest, AsyncLoadDoesNotPromoteOverflowResidentWavesPerPeu) {
  CollectingTraceSink trace;
  RuntimeEngine runtime(&trace);
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
  RuntimeEngine runtime(&trace);
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

TEST(CycleSmokeTest, VectorIssueLimitOverrideAllowsTwoWaveBundleIssue) {
  const auto spec = ArchRegistry::Get("c500");
  ASSERT_NE(spec, nullptr);

  CollectingTraceSink baseline_trace;
  RuntimeEngine runtime(&baseline_trace);
  runtime.SetLaunchTimingProfile(/*kernel_launch_gap_cycles=*/8,
                                 /*kernel_launch_cycles=*/0,
                                 /*block_launch_cycles=*/0,
                                 /*wave_launch_cycles=*/0,
                                 /*warp_switch_cycles=*/1,
                                 /*arg_load_cycles=*/4);

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
  RuntimeEngine widened_runtime(&widened_trace);
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
  RuntimeEngine baseline_runtime(&baseline_trace);
  baseline_runtime.SetLaunchTimingProfile(/*kernel_launch_gap_cycles=*/8,
                                          /*kernel_launch_cycles=*/0,
                                          /*block_launch_cycles=*/0,
                                          /*wave_launch_cycles=*/0,
                                          /*warp_switch_cycles=*/1,
                                          /*arg_load_cycles=*/4);
  const auto baseline = baseline_runtime.Launch(request);
  ASSERT_TRUE(baseline.ok) << baseline.error_message;

  CollectingTraceSink grouped_trace;
  RuntimeEngine grouped_runtime(&grouped_trace);
  grouped_runtime.SetLaunchTimingProfile(/*kernel_launch_gap_cycles=*/8,
                                         /*kernel_launch_cycles=*/0,
                                         /*block_launch_cycles=*/0,
                                         /*wave_launch_cycles=*/0,
                                         /*warp_switch_cycles=*/1,
                                         /*arg_load_cycles=*/4);
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
  RuntimeEngine baseline_runtime(&baseline_trace);
  baseline_runtime.SetLaunchTimingProfile(/*kernel_launch_gap_cycles=*/8,
                                          /*kernel_launch_cycles=*/0,
                                          /*block_launch_cycles=*/0,
                                          /*wave_launch_cycles=*/0,
                                          /*warp_switch_cycles=*/1,
                                          /*arg_load_cycles=*/4);
  const auto baseline = baseline_runtime.Launch(request);
  ASSERT_TRUE(baseline.ok) << baseline.error_message;

  CollectingTraceSink limited_trace;
  RuntimeEngine limited_runtime(&limited_trace);
  limited_runtime.SetLaunchTimingProfile(/*kernel_launch_gap_cycles=*/8,
                                         /*kernel_launch_cycles=*/0,
                                         /*block_launch_cycles=*/0,
                                         /*wave_launch_cycles=*/0,
                                         /*warp_switch_cycles=*/1,
                                         /*arg_load_cycles=*/4);
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

  RuntimeEngine runtime;
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

  RuntimeEngine runtime;
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
  EXPECT_EQ(result.total_cycles, 17u);
}

}  // namespace
}  // namespace gpu_model
