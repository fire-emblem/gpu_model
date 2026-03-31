#include <gtest/gtest.h>

#include "gpu_model/execution/internal/functional_wait_reason.h"

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

TEST(FunctionalWaitReasonTest, DoesNotEnterWaitingBeforeExplicitWaitcnt) {
  WaveContext wave;
  wave.pending_global_mem_ops = 1;
  wave.run_state = WaveRunState::Runnable;

  Instruction instruction;
  instruction.opcode = Opcode::VAdd;

  EXPECT_FALSE(EnterWaitStateFromInstruction(instruction, wave));
  EXPECT_EQ(wave.run_state, WaveRunState::Runnable);
  EXPECT_EQ(wave.wait_reason, WaveWaitReason::None);
}

TEST(FunctionalWaitReasonTest, EntersWaitingOnExplicitWaitcntWithPendingMemory) {
  WaveContext wave;
  wave.pending_global_mem_ops = 1;

  EXPECT_TRUE(EnterWaitStateFromInstruction(MakeWaitCntInstruction(0, 0, 0, 0), wave));
  EXPECT_EQ(wave.run_state, WaveRunState::Waiting);
  EXPECT_EQ(wave.wait_reason, WaveWaitReason::PendingGlobalMemory);
}

TEST(FunctionalWaitReasonTest, ResumesWhenMemoryWaitReasonIsSatisfied) {
  WaveContext wave;
  wave.run_state = WaveRunState::Waiting;
  wave.wait_reason = WaveWaitReason::PendingSharedMemory;
  wave.pending_shared_mem_ops = 0;

  EXPECT_TRUE(ResumeWaveIfWaitReasonSatisfied(wave));
  EXPECT_EQ(wave.run_state, WaveRunState::Runnable);
  EXPECT_EQ(wave.wait_reason, WaveWaitReason::None);
}

TEST(FunctionalWaitReasonTest, DoesNotResumeWhenPendingMemoryRemains) {
  WaveContext wave;
  wave.run_state = WaveRunState::Waiting;
  wave.wait_reason = WaveWaitReason::PendingPrivateMemory;
  wave.pending_private_mem_ops = 1;

  EXPECT_FALSE(ResumeWaveIfWaitReasonSatisfied(wave));
  EXPECT_EQ(wave.run_state, WaveRunState::Waiting);
  EXPECT_EQ(wave.wait_reason, WaveWaitReason::PendingPrivateMemory);
}

}  // namespace
}  // namespace gpu_model
