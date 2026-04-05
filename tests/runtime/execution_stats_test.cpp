#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

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
