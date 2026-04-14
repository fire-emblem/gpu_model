#include <gtest/gtest.h>

#include "execution/internal/issue_logic/encoded_issue_candidate.h"

namespace gpu_model {
namespace {

TEST(EncodedIssueCandidateTest, MarksRunnableScalarAndVectorInputsReady) {
  WaveContext scalar_wave;
  scalar_wave.status = WaveStatus::Active;
  scalar_wave.run_state = WaveRunState::Runnable;

  WaveContext vector_wave = scalar_wave;

  const auto scalar_inst = DecodedInstruction{.mnemonic = "s_add_i32"};
  const auto vector_inst = DecodedInstruction{.mnemonic = "v_add_f32_e32"};
  const auto scalar_desc =
      EncodedInstructionDescriptor{.category = EncodedInstructionCategory::Scalar};
  const auto vector_desc =
      EncodedInstructionDescriptor{.category = EncodedInstructionCategory::Vector};

  const auto candidates = BuildEncodedIssueCandidates({
      EncodedIssueCandidateInput{
          .candidate_index = 0,
          .wave_id = 0,
          .dispatch_enabled = true,
          .wave = &scalar_wave,
          .instruction = &scalar_inst,
          .descriptor = scalar_desc,
          .has_descriptor = true,
      },
      EncodedIssueCandidateInput{
          .candidate_index = 1,
          .wave_id = 1,
          .dispatch_enabled = true,
          .wave = &vector_wave,
          .instruction = &vector_inst,
          .descriptor = vector_desc,
          .has_descriptor = true,
      },
  });

  ASSERT_EQ(candidates.size(), 2u);
  EXPECT_TRUE(candidates[0].ready);
  EXPECT_EQ(candidates[0].issue_type, ArchitecturalIssueType::ScalarAluOrMemory);
  EXPECT_TRUE(candidates[1].ready);
  EXPECT_EQ(candidates[1].issue_type, ArchitecturalIssueType::VectorAlu);
}

TEST(EncodedIssueCandidateTest, MarksBarrierNotReadyWhenBarrierSlotsAreFull) {
  WaveContext wave;
  wave.status = WaveStatus::Active;
  wave.run_state = WaveRunState::Runnable;

  const auto inst = DecodedInstruction{.mnemonic = "s_barrier"};
  const auto desc = EncodedInstructionDescriptor{.category = EncodedInstructionCategory::Vector};

  const auto candidates = BuildEncodedIssueCandidates({
      EncodedIssueCandidateInput{
          .candidate_index = 0,
          .wave_id = 0,
          .dispatch_enabled = true,
          .wave = &wave,
          .instruction = &inst,
          .descriptor = desc,
          .has_descriptor = true,
          .barrier_slots_in_use = 16,
          .barrier_slot_capacity = 16,
          .barrier_slot_acquired = false,
      },
  });

  ASSERT_EQ(candidates.size(), 1u);
  EXPECT_FALSE(candidates[0].ready);
  EXPECT_EQ(candidates[0].issue_type, ArchitecturalIssueType::Special);
}

TEST(EncodedIssueCandidateTest, MarksWaitingWaveNotReady) {
  WaveContext wave;
  wave.status = WaveStatus::Active;
  wave.run_state = WaveRunState::Waiting;
  wave.wait_reason = WaveWaitReason::PendingGlobalMemory;

  const auto inst = DecodedInstruction{.mnemonic = "buffer_load_dword"};
  const auto desc = EncodedInstructionDescriptor{.category = EncodedInstructionCategory::Memory};

  const auto candidates = BuildEncodedIssueCandidates({
      EncodedIssueCandidateInput{
          .candidate_index = 0,
          .wave_id = 7,
          .dispatch_enabled = true,
          .wave = &wave,
          .instruction = &inst,
          .descriptor = desc,
          .has_descriptor = true,
      },
  });

  ASSERT_EQ(candidates.size(), 1u);
  EXPECT_FALSE(candidates[0].ready);
  EXPECT_EQ(candidates[0].issue_type, ArchitecturalIssueType::VectorMemory);
}

}  // namespace
}  // namespace gpu_model
