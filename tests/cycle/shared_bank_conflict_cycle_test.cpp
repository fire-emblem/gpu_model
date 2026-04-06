#include <gtest/gtest.h>

#include <cstdint>

#include "gpu_model/isa/instruction_builder.h"
#include "gpu_model/runtime/exec_engine.h"

namespace gpu_model {
namespace {

ExecutableKernel BuildSharedBankConflictKernel() {
  InstructionBuilder builder;
  builder.SysLaneId("v0");
  builder.MLoadShared("v2", "v0", 4);
  builder.BExit();
  return builder.Build("shared_bank_conflict");
}

TEST(SharedBankConflictCycleTest, SharedLoadPenaltyReflectsBankConflicts) {
  ExecEngine runtime;
  runtime.SetSharedBankConflictModel(/*bank_count=*/32, /*bank_width_bytes=*/4);

  const auto kernel = BuildSharedBankConflictKernel();
  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;
  request.config.shared_memory_bytes = 64 * sizeof(int32_t);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  EXPECT_EQ(result.total_cycles, 397u);
}

}  // namespace
}  // namespace gpu_model
