#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <string_view>
#include <vector>

#include "debug/trace/event_factory.h"  // TraceWaveView, MakeTrace*Event functions
#include "debug/trace/instruction_trace.h"  // FormatWaveStepMessage
#include "debug/trace/wave_launch_trace.h"  // FormatWaveLaunchTraceMessage
#include "execution/internal/cost_model/cycle_types.h"  // ScheduledWave, ExecutableBlock, PeuSlot, ApResidentState, etc.
#include "execution/internal/wave_schedule/event_queue.h"  // EventQueue, TimedEvent
#include "execution/internal/block_schedule/wave_context_builder.h"  // BuildWaveContextBlocks
#include "gpu_arch/device/gpu_arch_spec.h"  // GpuArchSpec
#include "runtime/config/launch_config.h"  // LaunchConfig
#include "runtime/model_runtime/mapper.h"  // PlacementMap

namespace gpu_model {
class TraceSink;

namespace cycle_internal {

// ---------------------------------------------------------------------------
// Block management
// ---------------------------------------------------------------------------

std::vector<ExecutableBlock> MaterializeBlocks(const PlacementMap& placement,
                                               const LaunchConfig& launch_config);

uint32_t TraceSlotId(const ScheduledWave& scheduled_wave);

std::vector<PeuSlot> BuildPeuSlots(std::vector<ExecutableBlock>& blocks,
                                   uint32_t resident_wave_slots_per_peu);

bool AllWavesExited(const std::vector<ExecutableBlock>& blocks);
bool AllWavesExited(const ExecutableBlock& block);

std::vector<uint64_t> ActiveAddresses(const MemoryRequest& request);

// ---------------------------------------------------------------------------
// Wave helpers
// ---------------------------------------------------------------------------

uint64_t WaveTag(const WaveContext& wave);

uint64_t WaveAgeOrderKey(const ScheduledWave& scheduled_wave, uint64_t current_cycle);

constexpr std::string_view kStallReasonBarrierSlotUnavailable = "barrier_slot_unavailable";
constexpr std::string_view kStallReasonIssueGroupConflict = "issue_group_conflict";

TraceWaveView MakeTraceWaveView(const ScheduledWave& wave, uint32_t slot_id);

// ---------------------------------------------------------------------------
// Wave scheduling
// ---------------------------------------------------------------------------

void ScheduleWaveLaunch(ScheduledWave& scheduled_wave,
                        uint64_t cycle,
                        EventQueue& events,
                        TraceSink& trace,
                        bool immediate = false);

void ScheduleWaveGenerate(ScheduledWave& scheduled_wave,
                          uint64_t cycle,
                          EventQueue& events,
                          TraceSink& trace);

template <typename Container>
void RemoveWaveFromList(Container& waves, const ScheduledWave* target) {
  waves.erase(std::remove(waves.begin(), waves.end(), target), waves.end());
}

template <typename Container>
bool ContainsWave(const Container& waves, const ScheduledWave* target) {
  return std::find(waves.begin(), waves.end(), target) != waves.end();
}

bool ContainsSlotId(const std::deque<size_t>& slot_ids, size_t target);

uint32_t ResidentWaveSlotCapacityPerPeu(const GpuArchSpec& spec);

bool CanAdmitBlockToResidentWaveSlots(const ExecutableBlock& block,
                                      const std::vector<PeuSlot>& slots,
                                      uint32_t resident_wave_slots_per_peu);

ResidentIssueSlot& ResidentSlotForWave(PeuSlot& slot, const ScheduledWave& scheduled_wave);

void RegisterResidentWave(PeuSlot& slot, ScheduledWave& scheduled_wave);

void RefillActiveWindow(PeuSlot& slot,
                        uint64_t cycle,
                        uint32_t max_issuable_waves,
                        uint64_t wave_launch_cycles,
                        EventQueue& events,
                        TraceSink& trace,
                        bool immediate_launch = false);

void ScheduleWaveDispatch(ScheduledWave& scheduled_wave,
                          PeuSlot& slot,
                          uint64_t cycle,
                          uint32_t max_issuable_waves,
                          uint64_t wave_launch_cycles,
                          EventQueue& events,
                          TraceSink& trace);

void RemoveResidentSlotFromStandby(PeuSlot& slot, size_t resident_slot_id);

bool ResidentSlotEligibleForActivation(const ResidentIssueSlot& resident_slot);

void DeactivateResidentSlot(ResidentIssueSlot& resident_slot);

void QueueResidentWaveForRefill(PeuSlot& slot, ScheduledWave& scheduled_wave);

void RemoveResidentWave(PeuSlot& slot, ScheduledWave& scheduled_wave);

void ActivateBlock(ExecutableBlock& block,
                   uint64_t cycle,
                   std::vector<PeuSlot>& slots,
                   uint32_t max_issuable_waves,
                   uint64_t wave_generation_cycles,
                   uint64_t wave_dispatch_cycles,
                   uint64_t wave_launch_cycles,
                   EventQueue& events,
                   TraceSink& trace);

bool AdmitOneResidentBlock(ApResidentState& ap_state,
                           uint64_t cycle,
                           std::vector<PeuSlot>& slots,
                           uint32_t resident_wave_slots_per_peu,
                           uint32_t max_issuable_waves,
                           uint64_t wave_generation_cycles,
                           uint64_t wave_dispatch_cycles,
                           uint64_t wave_launch_cycles,
                           EventQueue& events,
                           TraceSink& trace);

void AdmitResidentBlocks(ApResidentState& ap_state,
                         uint64_t cycle,
                         std::vector<PeuSlot>& slots,
                         uint32_t resident_wave_slots_per_peu,
                         uint32_t max_issuable_waves,
                         uint64_t wave_generation_cycles,
                         uint64_t wave_dispatch_cycles,
                         uint64_t wave_launch_cycles,
                         EventQueue& events,
                         TraceSink& trace);

bool RetireResidentBlock(ApResidentState& ap_state, ExecutableBlock* block);

bool CanScheduleDelayedReadmit(const ApResidentState& ap_state);

void FillDispatchWindow(PeuSlot& slot,
                        uint64_t cycle,
                        uint32_t max_issuable_waves,
                        uint64_t wave_launch_cycles,
                        EventQueue& events,
                        TraceSink& trace);

}  // namespace cycle_internal
}  // namespace gpu_model
