#include <gtest/gtest.h>

#include <cstdint>

#include "gpu_model/isa/instruction_builder.h"
#include "gpu_model/runtime/exec_engine.h"
#include "gpu_model/runtime/program_cycle_stats.h"

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

}  // namespace
}  // namespace gpu_model
