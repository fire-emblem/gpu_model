#include <gtest/gtest.h>

#include <cstdint>

#include "instruction/isa/instruction_builder.h"
#include "runtime/exec_engine/exec_engine.h"
#include "execution/stats/program_cycle_stats.h"

namespace gpu_model {
namespace {

ExecutableKernel BuildSimpleVectorAluKernel() {
  InstructionBuilder builder;
  builder.VAdd("v0", "v0", "v0");
  builder.VAdd("v1", "v1", "v1");
  builder.VAdd("v2", "v2", "v2");
  builder.BExit();
  return builder.Build("simple_vector_alu");
}

ExecutableKernel BuildGlobalMemoryKernel() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SMov("s1", 0);
  builder.MLoadGlobal("v0", "s0", "s1", 4);
  builder.SWaitCnt(/*global_count=*/0, /*shared_count=*/UINT32_MAX,
                   /*private_count=*/UINT32_MAX, /*scalar_buffer_count=*/UINT32_MAX);
  builder.MStoreGlobal("s0", "s1", "v0", 4);
  builder.BExit();
  return builder.Build("global_memory");
}

ExecutableKernel BuildSharedMemoryKernel() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SMov("s1", 0);
  builder.MLoadShared("v0", "s1", 4);
  builder.VAdd("v1", "v0", "v0");
  builder.MStoreShared("s1", "v1", 4);
  builder.BExit();
  return builder.Build("shared_memory");
}

ExecutableKernel BuildBarrierKernel() {
  InstructionBuilder builder;
  builder.VAdd("v0", "v0", "v0");
  builder.SyncBarrier();
  builder.VAdd("v1", "v1", "v1");
  builder.BExit();
  return builder.Build("barrier_test");
}

TEST(ProgramCycleStatsTest, TracksInstructionsAndWaves) {
  ExecEngine runtime;

  auto kernel = BuildSimpleVectorAluKernel();
  constexpr uint32_t n = 64;
  const uint64_t out_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));

  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;
  request.mode = ExecutionMode::Cycle;
  request.args.PushU64(out_addr);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  auto stats = result.program_cycle_stats;
  ASSERT_TRUE(stats.has_value());

  // Should have launched 1 wave
  EXPECT_EQ(stats->waves_launched, 1);
  EXPECT_EQ(stats->waves_completed, 1);

  // Should have executed instructions
  EXPECT_GE(stats->instructions_executed, 4);

  // Should have vector ALU instructions
  EXPECT_GE(stats->vector_alu_insts, 3);

  // IPC should be calculable
  double ipc = stats->IPC();
  EXPECT_GE(ipc, 0.0);
}

TEST(ProgramCycleStatsTest, TracksMemoryOperations) {
  ExecEngine runtime;

  auto kernel = BuildGlobalMemoryKernel();
  constexpr uint32_t n = 64;
  const uint64_t out_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  for (uint32_t i = 0; i < n; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(out_addr + i * sizeof(int32_t), i);
  }

  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;
  request.mode = ExecutionMode::Cycle;
  request.args.PushU64(out_addr);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  auto stats = result.program_cycle_stats;
  ASSERT_TRUE(stats.has_value());

  // Should track memory operations
  EXPECT_GE(stats->global_loads, 1);
  EXPECT_GE(stats->global_stores, 1);
}

TEST(ProgramCycleStatsTest, TracksActiveAndIdleCycles) {
  ExecEngine runtime;

  auto kernel = BuildSimpleVectorAluKernel();
  constexpr uint32_t n = 64;
  const uint64_t out_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));

  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;
  request.mode = ExecutionMode::Cycle;
  request.args.PushU64(out_addr);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  auto stats = result.program_cycle_stats;
  ASSERT_TRUE(stats.has_value());

  // Should have active and idle cycles
  EXPECT_GT(stats->total_cycles, 0u);
  EXPECT_GT(stats->active_cycles, 0u);

  // active + idle should equal total
  EXPECT_EQ(stats->active_cycles + stats->idle_cycles, stats->total_cycles);

  // Most cycles should be active for a simple kernel
  EXPECT_GT(stats->active_cycles, stats->idle_cycles);
}

TEST(ProgramCycleStatsTest, TracksSharedMemoryOperations) {
  ExecEngine runtime;

  auto kernel = BuildSharedMemoryKernel();
  const uint64_t shared_addr = 0;  // Shared memory offset

  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;
  request.config.shared_memory_bytes = 256;
  request.mode = ExecutionMode::Cycle;
  request.args.PushU64(shared_addr);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  auto stats = result.program_cycle_stats;
  ASSERT_TRUE(stats.has_value());

  // Should track shared memory operations
  EXPECT_GE(stats->shared_loads, 1);
  EXPECT_GE(stats->shared_stores, 1);
}

TEST(ProgramCycleStatsTest, TracksBarrierInstructions) {
  ExecEngine runtime;

  auto kernel = BuildBarrierKernel();

  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;
  request.mode = ExecutionMode::Cycle;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  auto stats = result.program_cycle_stats;
  ASSERT_TRUE(stats.has_value());

  // Should track sync instructions (barrier, waitcnt)
  EXPECT_GE(stats->sync_insts, 1);
}

TEST(ProgramCycleStatsTest, TracksMultipleWaves) {
  ExecEngine runtime;

  auto kernel = BuildSimpleVectorAluKernel();

  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 2;  // 2 blocks
  request.config.block_dim_x = 64;
  request.mode = ExecutionMode::Cycle;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  auto stats = result.program_cycle_stats;
  ASSERT_TRUE(stats.has_value());

  // Should track multiple waves
  EXPECT_EQ(stats->waves_launched, 2);
  EXPECT_EQ(stats->waves_completed, 2);
}

TEST(ProgramCycleStatsTest, DerivedMetricsWork) {
  ProgramCycleStats stats;
  stats.total_cycles = 1000;
  stats.active_cycles = 800;
  stats.instructions_executed = 400;
  stats.waves_launched = 10;
  stats.waves_completed = 10;

  // Test IPC
  EXPECT_DOUBLE_EQ(stats.IPC(), 0.5);

  // Test ActiveUtilization
  EXPECT_DOUBLE_EQ(stats.ActiveUtilization(), 0.8);

  // Test WaveOccupancy
  EXPECT_DOUBLE_EQ(stats.WaveOccupancy(), 1.0);
}

TEST(ProgramCycleStatsTest, HandlesZeroCycles) {
  ProgramCycleStats stats;
  stats.total_cycles = 0;
  stats.active_cycles = 0;
  stats.instructions_executed = 0;

  // Should not divide by zero
  EXPECT_DOUBLE_EQ(stats.IPC(), 0.0);
  EXPECT_DOUBLE_EQ(stats.ActiveUtilization(), 0.0);
  EXPECT_DOUBLE_EQ(stats.WaveOccupancy(), 0.0);
}

TEST(ProgramCycleStatsTest, MemoryOpFractionCorrect) {
  ProgramCycleStats stats;
  stats.instructions_executed = 100;
  stats.global_loads = 20;
  stats.global_stores = 10;
  stats.shared_loads = 5;
  stats.shared_stores = 5;

  // Memory ops = 20 + 10 + 5 + 5 = 40
  // Fraction = 40 / 100 = 0.4
  EXPECT_DOUBLE_EQ(stats.MemoryOpFraction(), 0.4);
}

TEST(ProgramCycleStatsTest, StallFractionCorrect) {
  ProgramCycleStats stats;
  stats.total_cycles = 1000;
  stats.stall_barrier = 100;
  stats.stall_waitcnt = 200;
  stats.stall_resource = 50;
  stats.stall_dependency = 30;
  stats.stall_switch_away = 20;

  // Total stalls = 100 + 200 + 50 + 30 + 20 = 400
  // Fraction = 400 / 1000 = 0.4
  EXPECT_DOUBLE_EQ(stats.StallFraction(), 0.4);
}

}  // namespace
}  // namespace gpu_model
