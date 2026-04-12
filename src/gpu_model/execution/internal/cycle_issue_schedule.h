#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "gpu_model/execution/internal/cycle_types.h"
#include "gpu_model/execution/internal/cycle_wave_schedule.h"
#include "gpu_model/execution/internal/issue_eligibility.h"
#include "gpu_model/execution/internal/issue_scheduler.h"
#include "gpu_model/gpu_arch/issue_config/issue_config.h"

namespace gpu_model {

class ExecutableKernel;

namespace cycle_internal {

bool ResidentSlotReadyToIssue(const ResidentIssueSlot& resident_slot, uint64_t cycle);

std::optional<std::pair<ScheduledWave*, std::string>> BlockedResidentWave(
    ResidentIssueSlot& resident_slot,
    const ExecutableKernel& kernel,
    uint64_t cycle,
    const std::map<uint32_t, ApResidentState>& ap_states);

std::optional<std::pair<ScheduledWave*, std::string>> PickFirstBlockedResidentWave(
    PeuSlot& slot,
    const ExecutableKernel& kernel,
    uint64_t cycle,
    const std::map<uint32_t, ApResidentState>& ap_states);

std::optional<std::pair<ScheduledWave*, std::string>> PickFirstReadyUnselectedResidentWave(
    const std::vector<IssueSchedulerCandidate>& candidates,
    const IssueSchedulerResult& bundle,
    const std::vector<ResidentIssueSlot*>& ordered_resident_slots);

std::vector<IssueSchedulerCandidate> BuildResidentIssueCandidates(
    PeuSlot& slot,
    const ExecutableKernel& kernel,
    uint64_t cycle,
    const std::map<uint32_t, ApResidentState>& ap_states,
    std::vector<ResidentIssueSlot*>& ordered_resident_slots);

}  // namespace cycle_internal
}  // namespace gpu_model
