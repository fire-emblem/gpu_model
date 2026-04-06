#pragma once

#include <vector>

#include "gpu_model/execution/internal/barrier_resource_pool.h"
#include "gpu_model/execution/internal/encoded_issue_type.h"
#include "gpu_model/execution/internal/issue_scheduler.h"
#include "gpu_model/execution/wave_context.h"

namespace gpu_model {

struct EncodedIssueCandidateInput {
  size_t candidate_index = 0;
  uint32_t wave_id = 0;
  bool dispatch_enabled = false;
  const WaveContext* wave = nullptr;
  const DecodedInstruction* instruction = nullptr;
  EncodedInstructionDescriptor descriptor{};
  bool has_descriptor = false;
  uint32_t barrier_slots_in_use = 0;
  uint32_t barrier_slot_capacity = 0;
  bool barrier_slot_acquired = false;
};

inline std::vector<IssueSchedulerCandidate> BuildEncodedIssueCandidates(
    const std::vector<EncodedIssueCandidateInput>& inputs) {
  std::vector<IssueSchedulerCandidate> candidates;
  candidates.reserve(inputs.size());
  for (const auto& input : inputs) {
    bool ready = false;
    auto issue_type = ArchitecturalIssueType::Special;
    if (input.wave != nullptr && input.instruction != nullptr && input.has_descriptor) {
      issue_type = ArchitecturalIssueTypeForEncodedInstruction(*input.instruction, input.descriptor);
      ready = input.dispatch_enabled &&
              input.wave->status == WaveStatus::Active &&
              input.wave->run_state == WaveRunState::Runnable &&
              !input.wave->waiting_at_barrier;
      if (ready && input.instruction->mnemonic == "s_barrier") {
        uint32_t slots_in_use = input.barrier_slots_in_use;
        bool acquired = input.barrier_slot_acquired;
        ready = TryAcquireBarrierSlot(input.barrier_slot_capacity, slots_in_use, acquired);
      }
    }
    candidates.push_back(IssueSchedulerCandidate{
        .candidate_index = input.candidate_index,
        .wave_id = input.wave_id,
        .issue_type = issue_type,
        .ready = ready,
    });
  }
  return candidates;
}

}  // namespace gpu_model
