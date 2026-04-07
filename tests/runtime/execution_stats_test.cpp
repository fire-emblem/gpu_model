#include <gtest/gtest.h>

#include <cstdint>
#include <tuple>
#include <vector>

#include "gpu_model/debug/trace/sink.h"
#include "gpu_model/isa/instruction_builder.h"
#include "gpu_model/runtime/exec_engine.h"

namespace gpu_model {
namespace {

ExecutableKernel BuildStatsFunctionalKernel() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SLoadArg("s1", 1);
  builder.SysGlobalIdX("v0");
  builder.VCmpLtCmask("v0", "s1");
  builder.MaskSaveExec("s10");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("exit");
  builder.VMov("v1", 1);
  builder.MStoreShared("v0", "v1", 4);
  builder.SyncBarrier();
  builder.MLoadShared("v2", "v0", 4);
  builder.MStoreGlobal("s0", "v0", "v2", 4);
  builder.Label("exit");
  builder.MaskRestoreExec("s10");
  builder.BExit();
  return builder.Build("stats_functional");
}

ExecutableKernel BuildStatsCycleKernel() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SMov("s1", 0);
  builder.MLoadGlobal("v1", "s0", "s1", 4);
  builder.SWaitCnt(/*global_count=*/0, /*shared_count=*/UINT32_MAX,
                   /*private_count=*/UINT32_MAX, /*scalar_buffer_count=*/UINT32_MAX);
  builder.VAdd("v5", "v1", "v1");
  builder.MLoadGlobal("v2", "s0", "s1", 4);
  builder.VMov("v3", 0);
  builder.MLoadShared("v4", "v3", 4);
  builder.SWaitCnt(/*global_count=*/0, /*shared_count=*/0,
                   /*private_count=*/UINT32_MAX, /*scalar_buffer_count=*/UINT32_MAX);
  builder.VMov("v6", 1);
  builder.BExit();
  return builder.Build("stats_cycle");
}

ExecutableKernel BuildTraceParityFunctionalWaitcntKernel() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SMov("s1", 0);
  builder.MLoadGlobal("v1", "s0", "s1", 4);
  builder.SWaitCnt(/*global_count=*/0, /*shared_count=*/UINT32_MAX,
                   /*private_count=*/UINT32_MAX, /*scalar_buffer_count=*/UINT32_MAX);
  builder.VAdd("v2", "v1", "v1");
  builder.MStoreGlobal("s0", "s1", "v2", 4);
  builder.BExit();
  return builder.Build("trace_parity_functional_waitcnt");
}

ExecutableKernel BuildTraceParityCycleWaitcntKernel() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SMov("s1", 0);
  builder.MLoadGlobal("v1", "s0", "s1", 4);
  builder.SMov("s2", 1);
  builder.MLoadGlobal("v2", "s0", "s2", 4);
  builder.SWaitCnt(/*global_count=*/0, /*shared_count=*/UINT32_MAX,
                   /*private_count=*/UINT32_MAX, /*scalar_buffer_count=*/UINT32_MAX);
  builder.VAdd("v3", "v1", "v2");
  builder.BExit();
  return builder.Build("trace_parity_cycle_waitcnt");
}

TEST(ExecutionStatsTest, FunctionalLaunchReportsMemoryAndBarrierCounts) {
  ExecEngine runtime;
  const auto kernel = BuildStatsFunctionalKernel();

  constexpr uint32_t n = 128;
  const uint64_t out_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  for (uint32_t i = 0; i < n; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(out_addr + i * sizeof(int32_t), -1);
  }

  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 128;
  request.config.shared_memory_bytes = 128 * sizeof(int32_t);
  request.args.PushU64(out_addr);
  request.args.PushU32(n);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  EXPECT_EQ(result.stats.shared_stores, 2u);
  EXPECT_EQ(result.stats.shared_loads, 2u);
  EXPECT_EQ(result.stats.global_stores, 2u);
  EXPECT_EQ(result.stats.barriers, 2u);
  EXPECT_EQ(result.stats.wave_exits, 2u);
}

TEST(ExecutionStatsTest, FunctionalLaunchReportsProgramCycleStats) {
  ExecEngine runtime;
  runtime.SetFunctionalExecutionMode(FunctionalExecutionMode::SingleThreaded);

  const auto kernel = BuildStatsFunctionalKernel();
  constexpr uint32_t n = 64;
  const uint64_t out_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  for (uint32_t i = 0; i < n; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(out_addr + i * sizeof(int32_t), -1);
  }

  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;
  request.config.shared_memory_bytes = n * sizeof(int32_t);
  request.args.PushU64(out_addr);
  request.args.PushU32(n);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  ASSERT_TRUE(result.program_cycle_stats.has_value());
  EXPECT_GT(result.program_cycle_stats->total_cycles, 0u);
  EXPECT_EQ(result.total_cycles, result.program_cycle_stats->total_cycles);
  EXPECT_GE(result.program_cycle_stats->total_issued_work_cycles,
            result.program_cycle_stats->total_cycles);
}

TEST(ExecutionStatsTest, GlobalDisableTraceEnvForcesNullTraceSinkWithoutBreakingCycles) {
  setenv("GPU_MODEL_DISABLE_TRACE", "1", 1);

  CollectingTraceSink trace;
  ExecEngine runtime(&trace);
  runtime.SetFunctionalExecutionMode(FunctionalExecutionMode::SingleThreaded);

  InstructionBuilder builder;
  builder.SMov("s0", 1);
  builder.SMov("s1", 2);
  builder.BExit();
  const auto kernel = builder.Build("stats_disable_trace_env");

  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;

  const auto result = runtime.Launch(request);
  unsetenv("GPU_MODEL_DISABLE_TRACE");

  ASSERT_TRUE(result.ok) << result.error_message;
  ASSERT_TRUE(result.program_cycle_stats.has_value());
  EXPECT_GT(result.total_cycles, 0u);
  EXPECT_EQ(result.total_cycles, result.program_cycle_stats->total_cycles);
  EXPECT_TRUE(trace.events().empty());
}

TEST(ExecutionStatsTest, FunctionalWaitcntKeepsCyclesAndResultsWhenTraceIsDisabled) {
  const auto kernel = BuildTraceParityFunctionalWaitcntKernel();
  auto launch_once = [&](bool disable_trace) {
    if (disable_trace) {
      setenv("GPU_MODEL_DISABLE_TRACE", "1", 1);
    } else {
      unsetenv("GPU_MODEL_DISABLE_TRACE");
    }

    CollectingTraceSink trace;
    ExecEngine runtime(&trace);
    runtime.SetFunctionalExecutionMode(FunctionalExecutionMode::SingleThreaded);
    runtime.SetFixedGlobalMemoryLatency(20);

    const uint64_t base_addr = runtime.memory().AllocateGlobal(2 * sizeof(int32_t));
    runtime.memory().StoreGlobalValue<int32_t>(base_addr, 11);
    runtime.memory().StoreGlobalValue<int32_t>(base_addr + sizeof(int32_t), -1);

    LaunchRequest request;
    request.kernel = &kernel;
    request.config.grid_dim_x = 1;
    request.config.block_dim_x = 64;
    request.args.PushU64(base_addr);

    const auto result = runtime.Launch(request);
    const int32_t out_value =
        runtime.memory().LoadGlobalValue<int32_t>(base_addr + sizeof(int32_t));
    return std::make_tuple(result, out_value, trace.events().size());
  };

  const auto [enabled_result, enabled_value, enabled_trace_events] = launch_once(false);
  const auto [disabled_result, disabled_value, disabled_trace_events] = launch_once(true);
  unsetenv("GPU_MODEL_DISABLE_TRACE");

  ASSERT_TRUE(enabled_result.ok) << enabled_result.error_message;
  ASSERT_TRUE(disabled_result.ok) << disabled_result.error_message;
  EXPECT_EQ(enabled_result.total_cycles, disabled_result.total_cycles);
  EXPECT_EQ(enabled_value, disabled_value);
  EXPECT_GT(enabled_trace_events, 0u);
  EXPECT_EQ(disabled_trace_events, 0u);
}

TEST(ExecutionStatsTest, CycleWaitcntKeepsCyclesWhenTraceIsDisabled) {
  const auto kernel = BuildTraceParityCycleWaitcntKernel();
  auto launch_once = [&](bool disable_trace) {
    if (disable_trace) {
      setenv("GPU_MODEL_DISABLE_TRACE", "1", 1);
    } else {
      unsetenv("GPU_MODEL_DISABLE_TRACE");
    }

    CollectingTraceSink trace;
    ExecEngine runtime(&trace);
    runtime.SetFixedGlobalMemoryLatency(20);

    const uint64_t base_addr = runtime.memory().AllocateGlobal(2 * sizeof(int32_t));
    runtime.memory().StoreGlobalValue<int32_t>(base_addr, 11);
    runtime.memory().StoreGlobalValue<int32_t>(base_addr + sizeof(int32_t), 13);

    LaunchRequest request;
    request.kernel = &kernel;
    request.mode = ExecutionMode::Cycle;
    request.config.grid_dim_x = 1;
    request.config.block_dim_x = 64;
    request.args.PushU64(base_addr);

    const auto result = runtime.Launch(request);
    return std::make_pair(result, trace.events().size());
  };

  const auto [enabled_result, enabled_trace_events] = launch_once(false);
  const auto [disabled_result, disabled_trace_events] = launch_once(true);
  unsetenv("GPU_MODEL_DISABLE_TRACE");

  ASSERT_TRUE(enabled_result.ok) << enabled_result.error_message;
  ASSERT_TRUE(disabled_result.ok) << disabled_result.error_message;
  EXPECT_EQ(enabled_result.total_cycles, disabled_result.total_cycles);
  EXPECT_EQ(enabled_result.end_cycle, disabled_result.end_cycle);
  EXPECT_GT(enabled_trace_events, 0u);
  EXPECT_EQ(disabled_trace_events, 0u);
}

TEST(ExecutionStatsTest, CycleLaunchReportsCacheAndBankPenaltyCounts) {
  ExecEngine runtime;
  runtime.SetGlobalMemoryLatencyProfile(/*dram=*/40, /*l2=*/20, /*l1=*/8);
  runtime.SetSharedBankConflictModel(/*bank_count=*/32, /*bank_width_bytes=*/4);

  const auto kernel = BuildStatsCycleKernel();
  const uint64_t base_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(base_addr, 7);

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;
  request.config.shared_memory_bytes = 4;
  request.args.PushU64(base_addr);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  EXPECT_EQ(result.stats.global_loads, 2u);
  EXPECT_EQ(result.stats.shared_loads, 2u);
  EXPECT_EQ(result.stats.cache_misses, 1u);
  EXPECT_EQ(result.stats.l1_hits, 1u);
  EXPECT_EQ(result.stats.shared_bank_conflict_penalty_cycles, 126u);
}

}  // namespace
}  // namespace gpu_model
