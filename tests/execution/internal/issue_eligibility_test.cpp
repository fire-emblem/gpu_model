#include <gtest/gtest.h>

#include "execution/internal/issue_eligibility.h"

namespace gpu_model {
namespace {

Instruction MakeWaitCntInstruction(uint32_t global_count,
                                   uint32_t shared_count,
                                   uint32_t private_count,
                                   uint32_t scalar_buffer_count) {
  Instruction instruction;
  instruction.opcode = Opcode::SWaitCnt;
  instruction.operands = {Operand::ImmediateU64(global_count),
                          Operand::ImmediateU64(shared_count),
                          Operand::ImmediateU64(private_count),
                          Operand::ImmediateU64(scalar_buffer_count)};
  return instruction;
}

TEST(IssueEligibilityTest, MapsMemoryOpcodesToDomains) {
  EXPECT_EQ(MemoryDomainForOpcode(Opcode::MLoadGlobal), MemoryWaitDomain::Global);
  EXPECT_EQ(MemoryDomainForOpcode(Opcode::MLoadShared), MemoryWaitDomain::Shared);
  EXPECT_EQ(MemoryDomainForOpcode(Opcode::MLoadPrivate), MemoryWaitDomain::Private);
  EXPECT_EQ(MemoryDomainForOpcode(Opcode::MLoadConst), MemoryWaitDomain::ScalarBuffer);
  EXPECT_EQ(MemoryDomainForOpcode(Opcode::SBufferLoadDword), MemoryWaitDomain::ScalarBuffer);
}

TEST(IssueEligibilityTest, WaitCntSatisfiedChecksPerDomainCounters) {
  WaveContext wave;
  wave.pending_global_mem_ops = 1;
  wave.pending_shared_mem_ops = 2;
  wave.pending_private_mem_ops = 3;
  wave.pending_scalar_buffer_mem_ops = 4;

  EXPECT_FALSE(WaitCntSatisfied(wave, MakeWaitCntInstruction(0, 2, 3, 4)));
  EXPECT_TRUE(WaitCntSatisfied(wave, MakeWaitCntInstruction(1, 2, 3, 4)));
  EXPECT_FALSE(WaitCntSatisfied(wave, MakeWaitCntInstruction(1, 1, 3, 4)));
  EXPECT_FALSE(WaitCntSatisfied(wave, MakeWaitCntInstruction(1, 2, 2, 4)));
  EXPECT_FALSE(WaitCntSatisfied(wave, MakeWaitCntInstruction(1, 2, 3, 3)));
}

TEST(IssueEligibilityTest, ReportsGlobalWaitcntReasonWhenGlobalOpsExceedThreshold) {
  WaveContext wave;
  wave.pending_global_mem_ops = 1;

  const auto wait_reason = WaitCntBlockReason(wave, MakeWaitCntInstruction(0, 0, 0, 0));

  ASSERT_TRUE(wait_reason.has_value());
  EXPECT_EQ(*wait_reason, "waitcnt_global");
}

TEST(IssueEligibilityTest, ReportsSharedWaitcntReasonWhenSharedOpsExceedThreshold) {
  WaveContext wave;
  wave.pending_shared_mem_ops = 1;

  const auto wait_reason = WaitCntBlockReason(wave, MakeWaitCntInstruction(0, 0, 0, 0));

  ASSERT_TRUE(wait_reason.has_value());
  EXPECT_EQ(*wait_reason, "waitcnt_shared");
}

TEST(IssueEligibilityTest, ReportsFirstExceededWaitcntReasonByDomainPrecedence) {
  WaveContext wave;
  wave.pending_global_mem_ops = 1;
  wave.pending_shared_mem_ops = 1;
  wave.pending_private_mem_ops = 1;
  wave.pending_scalar_buffer_mem_ops = 1;

  const auto wait_reason = WaitCntBlockReason(wave, MakeWaitCntInstruction(0, 0, 0, 0));

  ASSERT_TRUE(wait_reason.has_value());
  EXPECT_EQ(*wait_reason, "waitcnt_global");
}

TEST(IssueEligibilityTest, IssueBlockReasonReportsWaitcntReason) {
  WaveContext wave;
  wave.status = WaveStatus::Active;
  wave.valid_entry = true;
  wave.pending_global_mem_ops = 1;

  const auto wait_reason =
      IssueBlockReason(true, wave, MakeWaitCntInstruction(0, UINT32_MAX, UINT32_MAX, UINT32_MAX));
  ASSERT_TRUE(wait_reason.has_value());
  EXPECT_EQ(*wait_reason, "waitcnt_global");
}

TEST(IssueEligibilityTest, WaitingWaveCannotIssue) {
  WaveContext wave;
  wave.status = WaveStatus::Active;
  wave.valid_entry = true;
  wave.run_state = WaveRunState::Waiting;
  wave.wait_reason = WaveWaitReason::PendingGlobalMemory;

  Instruction vector_instr;
  vector_instr.opcode = Opcode::VAdd;

  EXPECT_FALSE(CanIssueInstruction(true, wave, vector_instr));
}

TEST(IssueEligibilityTest, WaitingWaveReportsExplicitWaitReasonBeforeOtherChecks) {
  WaveContext wave;
  wave.status = WaveStatus::Active;
  wave.valid_entry = true;
  wave.run_state = WaveRunState::Waiting;
  wave.wait_reason = WaveWaitReason::PendingGlobalMemory;

  Instruction vector_instr;
  vector_instr.opcode = Opcode::VAdd;

  const auto reason = IssueBlockReason(true, wave, vector_instr);
  ASSERT_TRUE(reason.has_value());
  EXPECT_EQ(*reason, "waitcnt_global");
}

TEST(IssueEligibilityTest, BarrierWaitingWaveReportsBarrierWait) {
  WaveContext wave;
  wave.status = WaveStatus::Active;
  wave.valid_entry = true;
  wave.run_state = WaveRunState::Waiting;
  wave.wait_reason = WaveWaitReason::BlockBarrier;
  wave.waiting_at_barrier = true;

  Instruction vector_instr;
  vector_instr.opcode = Opcode::VAdd;

  const auto reason = IssueBlockReason(true, wave, vector_instr);
  ASSERT_TRUE(reason.has_value());
  EXPECT_EQ(*reason, "barrier_wait");
}

TEST(IssueEligibilityTest, FrontEndBlockReasonReportsBranchWait) {
  WaveContext wave;
  wave.status = WaveStatus::Active;
  wave.valid_entry = true;
  wave.branch_pending = true;

  const auto reason = FrontEndBlockReason(true, wave);

  ASSERT_TRUE(reason.has_value());
  EXPECT_EQ(*reason, "branch_wait");
}

TEST(IssueEligibilityTest, OptionalTraceWaitcntStateIsInvalidWhenThresholdsMissing) {
  WaveContext wave;

  const TraceWaitcntState state = MakeOptionalTraceWaitcntState(wave, std::nullopt);

  EXPECT_FALSE(state.valid);
}

}  // namespace
}  // namespace gpu_model
