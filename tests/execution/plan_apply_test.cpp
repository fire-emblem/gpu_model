#include <gtest/gtest.h>

#include "gpu_model/execution/plan_apply.h"

namespace gpu_model {
namespace {

TEST(PlanApplyTest, AppliesRegisterWritesAndFormatsExecUpdate) {
  WaveContext wave;
  wave.thread_count = 64;
  wave.ResetInitialExec();

  OpPlan plan;
  plan.scalar_writes.push_back(ScalarWrite{.reg_index = 2, .value = 17});
  VectorWrite vector_write;
  vector_write.reg_index = 3;
  vector_write.mask.set(0);
  vector_write.mask.set(7);
  vector_write.values[0] = 11;
  vector_write.values[7] = 19;
  plan.vector_writes.push_back(vector_write);
  plan.smask_write = 5;
  plan.cmask_write = std::bitset<64>(0x3);
  plan.exec_write = std::bitset<64>(0x81);

  ApplyExecutionPlanRegisterWrites(plan, wave);

  EXPECT_EQ(wave.sgpr.Read(2), 17u);
  EXPECT_EQ(wave.vgpr.Read(3, 0), 11u);
  EXPECT_EQ(wave.vgpr.Read(3, 7), 19u);
  EXPECT_EQ(wave.smask, 5u);
  EXPECT_TRUE(wave.cmask.test(0));
  EXPECT_TRUE(wave.cmask.test(1));
  EXPECT_TRUE(wave.exec.test(0));
  EXPECT_TRUE(wave.exec.test(7));

  const auto mask_text = MaybeFormatExecutionMaskUpdate(plan, wave);
  ASSERT_TRUE(mask_text.has_value());
  EXPECT_FALSE(mask_text->empty());
}

TEST(PlanApplyTest, AppliesControlFlowAndExitSemantics) {
  WaveContext branch_wave;
  branch_wave.thread_count = 64;
  branch_wave.ResetInitialExec();
  branch_wave.pc = 10;
  branch_wave.branch_pending = true;
  branch_wave.valid_entry = false;

  OpPlan branch_plan;
  branch_plan.branch_target = 42;
  ApplyExecutionPlanControlFlow(branch_plan, branch_wave, true, true);
  EXPECT_EQ(branch_wave.pc, 42u);
  EXPECT_FALSE(branch_wave.branch_pending);
  EXPECT_TRUE(branch_wave.valid_entry);
  EXPECT_EQ(branch_wave.status, WaveStatus::Active);

  WaveContext exit_wave;
  exit_wave.thread_count = 64;
  exit_wave.ResetInitialExec();
  exit_wave.pc = 9;

  OpPlan exit_plan;
  exit_plan.exit_wave = true;
  ApplyExecutionPlanControlFlow(exit_plan, exit_wave, true, true);
  EXPECT_EQ(exit_wave.status, WaveStatus::Exited);
  EXPECT_EQ(exit_wave.pc, 9u);
}

}  // namespace
}  // namespace gpu_model
