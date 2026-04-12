#include "execution/internal/cycle_issue_schedule.h"

#include <string>

#include "execution/internal/barrier_resource_pool.h"
#include "instruction/isa/opcode.h"
#include "program/executable/executable_kernel.h"

namespace gpu_model {
namespace cycle_internal {

bool ResidentSlotReadyToIssue(const ResidentIssueSlot& resident_slot, uint64_t cycle) {
  return resident_slot.active && resident_slot.resident_wave != nullptr &&
         !resident_slot.resident_wave->launch_scheduled &&
         resident_slot.resident_wave->launch_cycle <= cycle &&
         resident_slot.resident_wave->dispatch_enabled &&
         resident_slot.resident_wave->wave.status == WaveStatus::Active;
}

std::optional<std::pair<ScheduledWave*, std::string>> BlockedResidentWave(
    ResidentIssueSlot& resident_slot,
    const ExecutableKernel& kernel,
    uint64_t cycle,
    const std::map<uint32_t, ApResidentState>& ap_states) {
  ScheduledWave* scheduled_wave = resident_slot.resident_wave;
  if (!ResidentSlotReadyToIssue(resident_slot, cycle)) {
    return std::nullopt;
  }
  if (!kernel.ContainsPc(scheduled_wave->wave.pc)) {
    return std::nullopt;
  }
  const auto& instruction = kernel.InstructionAtPc(scheduled_wave->wave.pc);
  if (instruction.opcode == Opcode::SyncBarrier) {
    const auto ap_state_it = ap_states.find(scheduled_wave->block->global_ap_id);
    if (ap_state_it != ap_states.end()) {
      uint32_t slots_in_use = ap_state_it->second.barrier_slots_in_use;
      bool acquired = scheduled_wave->block->barrier_slot_acquired;
      if (!TryAcquireBarrierSlot(ap_state_it->second.barrier_slot_capacity,
                                 slots_in_use,
                                 acquired)) {
        return std::make_pair(scheduled_wave, std::string(kStallReasonBarrierSlotUnavailable));
      }
    }
  }
  if (const auto reason =
          IssueBlockReason(scheduled_wave->dispatch_enabled,
                           scheduled_wave->wave,
                           instruction)) {
    return std::make_pair(scheduled_wave, *reason);
  }
  return std::nullopt;
}

std::optional<std::pair<ScheduledWave*, std::string>> PickFirstBlockedResidentWave(
    PeuSlot& slot,
    const ExecutableKernel& kernel,
    uint64_t cycle,
    const std::map<uint32_t, ApResidentState>& ap_states) {
  if (slot.resident_slots.empty()) {
    return std::nullopt;
  }

  const size_t count = slot.resident_slots.size();
  const size_t start = slot.issue_round_robin_index % count;
  for (size_t offset = 0; offset < count; ++offset) {
    ResidentIssueSlot& resident_slot = slot.resident_slots[(start + offset) % count];
    if (!resident_slot.active || resident_slot.resident_wave == nullptr ||
        !ResidentSlotReadyToIssue(resident_slot, cycle)) {
      continue;
    }
    if (const auto blocked =
            BlockedResidentWave(resident_slot, kernel, cycle, ap_states)) {
      return blocked;
    }
  }
  return std::nullopt;
}

std::optional<std::pair<ScheduledWave*, std::string>> PickFirstReadyUnselectedResidentWave(
    const std::vector<IssueSchedulerCandidate>& candidates,
    const IssueSchedulerResult& bundle,
    const std::vector<ResidentIssueSlot*>& ordered_resident_slots) {
  if (bundle.selected_candidate_indices.empty()) {
    return std::nullopt;
  }

  std::vector<bool> selected(ordered_resident_slots.size(), false);
  for (const size_t candidate_index : bundle.selected_candidate_indices) {
    if (candidate_index < selected.size()) {
      selected[candidate_index] = true;
    }
  }

  for (const auto& candidate : candidates) {
    if (!candidate.ready || candidate.candidate_index >= ordered_resident_slots.size()) {
      continue;
    }
    if (selected[candidate.candidate_index]) {
      continue;
    }
    ResidentIssueSlot* resident_slot = ordered_resident_slots[candidate.candidate_index];
    if (resident_slot == nullptr || resident_slot->resident_wave == nullptr) {
      continue;
    }
    return std::make_pair(resident_slot->resident_wave,
                          std::string(kStallReasonIssueGroupConflict));
  }

  return std::nullopt;
}

std::vector<IssueSchedulerCandidate> BuildResidentIssueCandidates(
    PeuSlot& slot,
    const ExecutableKernel& kernel,
    uint64_t cycle,
    const std::map<uint32_t, ApResidentState>& ap_states,
    std::vector<ResidentIssueSlot*>& ordered_resident_slots) {
  ordered_resident_slots.clear();
  std::vector<IssueSchedulerCandidate> candidates;
  if (slot.resident_slots.empty()) {
    return candidates;
  }

  const size_t count = slot.resident_slots.size();
  for (size_t index = 0; index < count; ++index) {
    ResidentIssueSlot& resident_slot = slot.resident_slots[index];
    ScheduledWave* scheduled_wave = resident_slot.resident_wave;
    if (!resident_slot.active || scheduled_wave == nullptr ||
        !ResidentSlotReadyToIssue(resident_slot, cycle)) {
      continue;
    }

    ordered_resident_slots.push_back(&resident_slot);
    auto& wave = scheduled_wave->wave;

    // Check next_issue_cycle timing constraint
    const bool timing_ready = scheduled_wave->next_issue_cycle <= cycle;

    // Track eligible_since_cycle for dynamic age ordering
    if (timing_ready && !scheduled_wave->eligible_since_valid) {
      scheduled_wave->eligible_since_cycle = cycle;
      scheduled_wave->eligible_since_valid = true;
    }

    bool ready = false;
    auto issue_type = ArchitecturalIssueType::Special;
    if (kernel.ContainsPc(wave.pc)) {
      const auto& instruction = kernel.InstructionAtPc(wave.pc);
      if (instruction.opcode == Opcode::SyncBarrier) {
        const auto ap_state_it = ap_states.find(scheduled_wave->block->global_ap_id);
        if (ap_state_it != ap_states.end()) {
          uint32_t slots_in_use = ap_state_it->second.barrier_slots_in_use;
          bool acquired = scheduled_wave->block->barrier_slot_acquired;
          if (!TryAcquireBarrierSlot(ap_state_it->second.barrier_slot_capacity,
                                     slots_in_use,
                                     acquired)) {
            candidates.push_back(IssueSchedulerCandidate{
                .candidate_index = ordered_resident_slots.size() - 1,
                .wave_id = wave.wave_id,
                .age_order_key = WaveAgeOrderKey(*scheduled_wave, cycle),
                .issue_type = ArchitecturalIssueType::Special,
                .ready = false,
                .eligible_since_cycle = scheduled_wave->eligible_since_valid ? scheduled_wave->eligible_since_cycle : 0,
                .next_issue_earliest_global_cycle = scheduled_wave->next_issue_cycle,
                .blocked_reason = IssueBlockedReason::WaveWaiting,
            });
            continue;
          }
        }
      }
      ready = timing_ready && CanIssueInstruction(scheduled_wave->dispatch_enabled, wave, instruction);
      issue_type = ArchitecturalIssueTypeForOpcode(instruction.opcode)
                       .value_or(ArchitecturalIssueType::Special);
    }
    candidates.push_back(IssueSchedulerCandidate{
        .candidate_index = ordered_resident_slots.size() - 1,
        .wave_id = wave.wave_id,
        .age_order_key = WaveAgeOrderKey(*scheduled_wave, cycle),
        .issue_type = issue_type,
        .ready = ready,
        .eligible_since_cycle = scheduled_wave->eligible_since_valid ? scheduled_wave->eligible_since_cycle : 0,
        .next_issue_earliest_global_cycle = scheduled_wave->next_issue_cycle,
        .blocked_reason = ready ? IssueBlockedReason::None
                                 : (timing_ready ? IssueBlockedReason::WaveWaiting
                                                 : IssueBlockedReason::NotYetEligible),
    });
  }

  return candidates;
}

}  // namespace cycle_internal
}  // namespace gpu_model
