#include "gpu_model/execution/internal/cycle_wave_schedule.h"

#include <algorithm>
#include <cstdint>
#include <deque>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "gpu_model/debug/trace/sink.h"  // TraceSink
#include "gpu_model/execution/wave_context.h"  // WaveContext, WaveStatus
#include "gpu_model/gpu_arch/wave/wave_def.h"  // kWaveSize
#include "gpu_model/memory/memory_request.h"  // MemoryRequest

namespace gpu_model {
namespace cycle_internal {

std::vector<ExecutableBlock> MaterializeBlocks(const PlacementMap& placement,
                                               const LaunchConfig& launch_config) {
  const auto shared_blocks = BuildWaveContextBlocks(placement, launch_config);
  std::vector<ExecutableBlock> blocks;
  blocks.reserve(shared_blocks.size());

  for (const auto& shared_block : shared_blocks) {
    ExecutableBlock block{
        .block_id = shared_block.block_id,
        .dpc_id = shared_block.dpc_id,
        .ap_id = shared_block.ap_id,
        .global_ap_id = shared_block.global_ap_id,
        .barrier_generation = shared_block.barrier_generation,
        .barrier_arrivals = shared_block.barrier_arrivals,
        .shared_memory = shared_block.shared_memory,
        .waves = {},
    };
    block.waves.reserve(shared_block.waves.size());
    for (const auto& base_wave : shared_block.waves) {
      ScheduledWave scheduled;
      scheduled.dpc_id = shared_block.dpc_id;
      scheduled.wave = base_wave;
      scheduled.wave.status = WaveStatus::Stalled;
      block.waves.push_back(std::move(scheduled));
    }
    blocks.push_back(std::move(block));
  }

  for (auto& block : blocks) {
    for (auto& scheduled_wave : block.waves) {
      scheduled_wave.block = &block;
    }
  }

  return blocks;
}

uint32_t TraceSlotId(const ScheduledWave& scheduled_wave) {
  return scheduled_wave.resident_slot_id == std::numeric_limits<size_t>::max()
             ? 0u
             : static_cast<uint32_t>(scheduled_wave.resident_slot_id);
}

std::vector<PeuSlot> BuildPeuSlots(std::vector<ExecutableBlock>& blocks,
                                   uint32_t resident_wave_slots_per_peu) {
  std::vector<PeuSlot> slots;
  std::map<std::tuple<uint32_t, uint32_t, uint32_t>, size_t> slot_indices;

  for (auto& block : blocks) {
    for (auto& scheduled_wave : block.waves) {
      const auto key = std::make_tuple(block.dpc_id, block.ap_id, scheduled_wave.wave.peu_id);
      auto [it, inserted] = slot_indices.emplace(key, slots.size());
      if (inserted) {
        std::vector<ResidentIssueSlot> resident_slots;
        resident_slots.reserve(resident_wave_slots_per_peu);
        for (uint32_t slot_id = 0; slot_id < resident_wave_slots_per_peu; ++slot_id) {
          resident_slots.push_back(ResidentIssueSlot{
              .dpc_id = block.dpc_id,
              .ap_id = block.ap_id,
              .peu_id = scheduled_wave.wave.peu_id,
              .slot_id = slot_id,
          });
        }
        slots.push_back(PeuSlot{
            .dpc_id = block.dpc_id,
            .ap_id = block.ap_id,
            .peu_id = scheduled_wave.wave.peu_id,
            .last_wave_trace = std::nullopt,
            .last_wave_pc = 0,
            .waves = {},
            .resident_waves = {},
            .resident_slots = std::move(resident_slots),
            .standby_slot_ids = {},
        });
        it->second = slots.size() - 1;
      }
      scheduled_wave.peu_slot_index = it->second;
      slots[it->second].waves.push_back(&scheduled_wave);
    }
  }

  return slots;
}

bool AllWavesExited(const std::vector<ExecutableBlock>& blocks) {
  for (const auto& block : blocks) {
    for (const auto& scheduled_wave : block.waves) {
      if (scheduled_wave.wave.status != WaveStatus::Exited) {
        return false;
      }
    }
  }
  return true;
}

bool AllWavesExited(const ExecutableBlock& block) {
  return std::all_of(block.waves.begin(), block.waves.end(), [](const ScheduledWave& wave) {
    return wave.wave.status == WaveStatus::Exited;
  });
}

std::vector<uint64_t> ActiveAddresses(const MemoryRequest& request) {
  std::vector<uint64_t> addrs;
  addrs.reserve(kWaveSize);
  for (const auto& lane : request.lanes) {
    if (lane.active) {
      addrs.push_back(lane.addr);
    }
  }
  return addrs;
}

// ---------------------------------------------------------------------------
// Wave helpers
// ---------------------------------------------------------------------------

uint64_t WaveTag(const WaveContext& wave) {
  return (static_cast<uint64_t>(wave.block_id) << 32) | wave.wave_id;
}

uint64_t WaveAgeOrderKey(const ScheduledWave& scheduled_wave, uint64_t current_cycle) {
  // Use eligible_since_cycle for dynamic ready age (oldest-first semantics)
  // Fall back to WaveTag for static ordering if not yet eligible
  if (scheduled_wave.eligible_since_valid && scheduled_wave.eligible_since_cycle > 0) {
    // Lower value = earlier eligible = higher priority
    return scheduled_wave.eligible_since_cycle;
  }
  // Not yet eligible: use current_cycle + WaveTag to sort after eligible waves
  return current_cycle + WaveTag(scheduled_wave.wave);
}

TraceWaveView MakeTraceWaveView(const ScheduledWave& wave, uint32_t slot_id) {
  return TraceWaveView{
      .dpc_id = wave.wave.dpc_id,
      .ap_id = wave.wave.ap_id,
      .peu_id = wave.wave.peu_id,
      .slot_id = slot_id,
      .block_id = wave.wave.block_id,
      .wave_id = wave.wave.wave_id,
      .pc = wave.wave.pc,
  };
}

// ---------------------------------------------------------------------------
// Wave scheduling
// ---------------------------------------------------------------------------

void ScheduleWaveLaunch(ScheduledWave& scheduled_wave,
                        uint64_t cycle,
                        EventQueue& events,
                        TraceSink& trace,
                        bool immediate) {
  scheduled_wave.launch_cycle = cycle;
  if (scheduled_wave.launch_scheduled || scheduled_wave.launch_completed ||
      scheduled_wave.wave.status == WaveStatus::Exited) {
    return;
  }
  auto activate_launch = [&trace](ScheduledWave& wave) {
    wave.launch_scheduled = false;
    if (wave.wave.status == WaveStatus::Exited) {
      return;
    }
    wave.launch_completed = true;
    wave.dispatch_enabled = true;
    wave.wave.status = WaveStatus::Active;
    wave.wave.valid_entry = true;
    trace.OnEvent(MakeTraceWaveLaunchEvent(MakeTraceWaveView(wave, TraceSlotId(wave)),
                                           wave.launch_cycle,
                                           FormatWaveLaunchTraceMessage(wave.wave),
                                           TraceSlotModelKind::ResidentFixed));
  };
  if (immediate) {
    activate_launch(scheduled_wave);
    return;
  }
  scheduled_wave.launch_scheduled = true;
  ScheduledWave* wave_ptr = &scheduled_wave;
  events.Schedule(TimedEvent{
      .cycle = cycle,
      .action = [wave_ptr, &trace, activate_launch]() { activate_launch(*wave_ptr); },
  });
}

void ScheduleWaveGenerate(ScheduledWave& scheduled_wave,
                          uint64_t cycle,
                          EventQueue& events,
                          TraceSink& trace) {
  scheduled_wave.generate_cycle = cycle;
  if (scheduled_wave.generate_scheduled || scheduled_wave.generate_completed ||
      scheduled_wave.wave.status == WaveStatus::Exited) {
    return;
  }
  scheduled_wave.generate_scheduled = true;
  ScheduledWave* wave_ptr = &scheduled_wave;
  events.Schedule(TimedEvent{
      .cycle = cycle,
      .action = [wave_ptr, &trace]() {
        wave_ptr->generate_scheduled = false;
        if (wave_ptr->wave.status == WaveStatus::Exited) {
          return;
        }
        wave_ptr->generate_completed = true;
        trace.OnEvent(MakeTraceWaveGenerateEvent(
            MakeTraceWaveView(*wave_ptr, TraceSlotId(*wave_ptr)),
            wave_ptr->generate_cycle,
            TraceSlotModelKind::ResidentFixed));
      },
  });
}

bool ContainsSlotId(const std::deque<size_t>& slot_ids, size_t target) {
  return std::find(slot_ids.begin(), slot_ids.end(), target) != slot_ids.end();
}

uint32_t ResidentWaveSlotCapacityPerPeu(const GpuArchSpec& spec) {
  return spec.cycle_resources.resident_wave_slots_per_peu > 0
             ? spec.cycle_resources.resident_wave_slots_per_peu
             : spec.max_resident_waves;
}

bool CanAdmitBlockToResidentWaveSlots(const ExecutableBlock& block,
                                      const std::vector<PeuSlot>& slots,
                                      uint32_t resident_wave_slots_per_peu) {
  std::map<size_t, uint32_t> required_per_slot;
  for (const auto& scheduled_wave : block.waves) {
    ++required_per_slot[scheduled_wave.peu_slot_index];
  }
  for (const auto& [slot_index, required] : required_per_slot) {
    const auto& slot = slots.at(slot_index);
    const auto free_slots = static_cast<uint32_t>(std::count_if(
        slot.resident_slots.begin(),
        slot.resident_slots.end(),
        [](const ResidentIssueSlot& resident_slot) {
          return resident_slot.resident_wave == nullptr;
        }));
    if (required > free_slots || slot.resident_waves.size() + required > resident_wave_slots_per_peu) {
      return false;
    }
  }
  return true;
}

ResidentIssueSlot& ResidentSlotForWave(PeuSlot& slot, const ScheduledWave& scheduled_wave) {
  if (scheduled_wave.resident_slot_id >= slot.resident_slots.size()) {
    throw std::out_of_range("resident slot id out of range");
  }
  return slot.resident_slots.at(scheduled_wave.resident_slot_id);
}

void RegisterResidentWave(PeuSlot& slot, ScheduledWave& scheduled_wave) {
  RemoveWaveFromList(slot.resident_waves, &scheduled_wave);
  const auto resident_slot_it =
      std::find_if(slot.resident_slots.begin(),
                   slot.resident_slots.end(),
                   [](const ResidentIssueSlot& resident_slot) {
                     return resident_slot.resident_wave == nullptr;
                   });
  if (resident_slot_it == slot.resident_slots.end()) {
    throw std::runtime_error("no free resident slot available for wave admission");
  }
  scheduled_wave.resident_slot_id = resident_slot_it->slot_id;
  scheduled_wave.launch_completed = false;
  resident_slot_it->resident_wave = &scheduled_wave;
  resident_slot_it->active = false;
  scheduled_wave.dispatch_enabled = false;
  slot.resident_waves.push_back(&scheduled_wave);
  slot.standby_slot_ids.push_back(scheduled_wave.resident_slot_id);
}

void ScheduleWaveDispatch(ScheduledWave& scheduled_wave,
                          PeuSlot& slot,
                          uint64_t cycle,
                          uint32_t max_issuable_waves,
                          uint64_t wave_launch_cycles,
                          EventQueue& events,
                          TraceSink& trace) {
  scheduled_wave.dispatch_cycle = cycle;
  if (scheduled_wave.dispatch_scheduled || scheduled_wave.dispatch_completed ||
      scheduled_wave.wave.status == WaveStatus::Exited) {
    return;
  }
  scheduled_wave.dispatch_scheduled = true;
  ScheduledWave* wave_ptr = &scheduled_wave;
  PeuSlot* slot_ptr = &slot;
  events.Schedule(TimedEvent{
      .cycle = cycle,
      .action = [wave_ptr, slot_ptr, cycle, max_issuable_waves, wave_launch_cycles, &events, &trace]() {
        wave_ptr->dispatch_scheduled = false;
        if (wave_ptr->wave.status == WaveStatus::Exited) {
          return;
        }
        RegisterResidentWave(*slot_ptr, *wave_ptr);
        wave_ptr->dispatch_completed = true;
        trace.OnEvent(MakeTraceWaveDispatchEvent(
            MakeTraceWaveView(*wave_ptr, TraceSlotId(*wave_ptr)),
            wave_ptr->dispatch_cycle,
            TraceSlotModelKind::ResidentFixed));
        trace.OnEvent(MakeTraceSlotBindEvent(
            MakeTraceWaveView(*wave_ptr, TraceSlotId(*wave_ptr)),
            wave_ptr->dispatch_cycle,
            TraceSlotModelKind::ResidentFixed));
        RefillActiveWindow(*slot_ptr,
                           cycle,
                           max_issuable_waves,
                           wave_launch_cycles,
                           events,
                           trace,
                           false);
      },
  });
}

void RemoveResidentSlotFromStandby(PeuSlot& slot, size_t resident_slot_id) {
  slot.standby_slot_ids.erase(
      std::remove(slot.standby_slot_ids.begin(), slot.standby_slot_ids.end(), resident_slot_id),
      slot.standby_slot_ids.end());
}

bool ResidentSlotEligibleForActivation(const ResidentIssueSlot& resident_slot) {
  return resident_slot.resident_wave != nullptr &&
         resident_slot.resident_wave->resident_slot_id != std::numeric_limits<size_t>::max() &&
         resident_slot.resident_wave->wave.status != WaveStatus::Exited &&
         !resident_slot.resident_wave->wave.waiting_at_barrier;
}

void DeactivateResidentSlot(ResidentIssueSlot& resident_slot) {
  resident_slot.active = false;
  if (resident_slot.resident_wave != nullptr) {
    resident_slot.resident_wave->dispatch_enabled = false;
  }
}

void QueueResidentWaveForRefill(PeuSlot& slot, ScheduledWave& scheduled_wave) {
  if (!ContainsWave(slot.resident_waves, &scheduled_wave) ||
      scheduled_wave.resident_slot_id == std::numeric_limits<size_t>::max() ||
      scheduled_wave.wave.status == WaveStatus::Exited ||
      scheduled_wave.wave.waiting_at_barrier) {
    return;
  }
  ResidentIssueSlot& resident_slot = ResidentSlotForWave(slot, scheduled_wave);
  if (resident_slot.active || ContainsSlotId(slot.standby_slot_ids, resident_slot.slot_id)) {
    return;
  }
  scheduled_wave.dispatch_enabled = false;
  slot.standby_slot_ids.push_back(resident_slot.slot_id);
}

void RemoveResidentWave(PeuSlot& slot, ScheduledWave& scheduled_wave) {
  RemoveWaveFromList(slot.resident_waves, &scheduled_wave);
  if (scheduled_wave.resident_slot_id != std::numeric_limits<size_t>::max() &&
      scheduled_wave.resident_slot_id < slot.resident_slots.size()) {
    ResidentIssueSlot& resident_slot = ResidentSlotForWave(slot, scheduled_wave);
    RemoveResidentSlotFromStandby(slot, resident_slot.slot_id);
    DeactivateResidentSlot(resident_slot);
    if (resident_slot.resident_wave == &scheduled_wave) {
      resident_slot.resident_wave = nullptr;
    }
  }
  scheduled_wave.resident_slot_id = std::numeric_limits<size_t>::max();
  scheduled_wave.dispatch_enabled = false;
}

void RefillActiveWindow(PeuSlot& slot,
                        uint64_t cycle,
                        uint32_t max_issuable_waves,
                        uint64_t wave_launch_cycles,
                        EventQueue& events,
                        TraceSink& trace,
                        bool immediate_launch) {
  uint32_t active_count = static_cast<uint32_t>(std::count_if(
      slot.resident_slots.begin(),
      slot.resident_slots.end(),
      [](const ResidentIssueSlot& resident_slot) {
        return resident_slot.active && resident_slot.resident_wave != nullptr;
      }));
  uint32_t launch_order = 0;
  while (active_count < max_issuable_waves && !slot.standby_slot_ids.empty()) {
    const size_t resident_slot_id = slot.standby_slot_ids.front();
    slot.standby_slot_ids.pop_front();
    if (resident_slot_id >= slot.resident_slots.size()) {
      continue;
    }
    ResidentIssueSlot& resident_slot = slot.resident_slots.at(resident_slot_id);
    if (resident_slot.active || !ResidentSlotEligibleForActivation(resident_slot)) {
      continue;
    }
    resident_slot.active = true;
    ScheduledWave& scheduled_wave = *resident_slot.resident_wave;
    trace.OnEvent(MakeTraceActivePromoteEvent(
        MakeTraceWaveView(scheduled_wave, TraceSlotId(scheduled_wave)),
        cycle,
        TraceSlotModelKind::ResidentFixed));
    if (scheduled_wave.launch_completed) {
      scheduled_wave.launch_scheduled = false;
      scheduled_wave.dispatch_enabled = true;
      scheduled_wave.wave.valid_entry = true;
      if (scheduled_wave.wave.status != WaveStatus::Exited &&
          !scheduled_wave.wave.waiting_at_barrier) {
        scheduled_wave.wave.status = WaveStatus::Active;
      }
    } else {
      const uint64_t launch_cycle =
          cycle + static_cast<uint64_t>(launch_order) * wave_launch_cycles;
      ScheduleWaveLaunch(scheduled_wave,
                         launch_cycle,
                         events,
                         trace,
                         immediate_launch && launch_cycle == cycle);
    }
    ++active_count;
    ++launch_order;
  }
}

void ActivateBlock(ExecutableBlock& block,
                   uint64_t cycle,
                   std::vector<PeuSlot>& slots,
                   uint32_t max_issuable_waves,
                   uint64_t wave_generation_cycles,
                   uint64_t wave_dispatch_cycles,
                   uint64_t wave_launch_cycles,
                   EventQueue& events,
                   TraceSink& trace) {
  block.active = true;
  trace.OnEvent(MakeTraceBlockAdmitEvent(block.dpc_id,
                                         block.ap_id,
                                         block.block_id,
                                         cycle,
                                         "ap=" + std::to_string(block.ap_id)));
  trace.OnEvent(MakeTraceBlockEvent(block.dpc_id,
                                    block.ap_id,
                                    block.block_id,
                                    TraceEventKind::BlockLaunch,
                                    cycle,
                                    "ap=" + std::to_string(block.ap_id)));
  trace.OnEvent(MakeTraceBlockActivateEvent(block.dpc_id,
                                            block.ap_id,
                                            block.block_id,
                                            cycle,
                                            "ap=" + std::to_string(block.ap_id)));
  for (auto& scheduled_wave : block.waves) {
    scheduled_wave.wave.status = WaveStatus::Stalled;
    scheduled_wave.dispatch_enabled = false;
    scheduled_wave.launch_scheduled = false;
    scheduled_wave.generate_completed = false;
    scheduled_wave.generate_scheduled = false;
    scheduled_wave.dispatch_completed = false;
    scheduled_wave.dispatch_scheduled = false;
    scheduled_wave.resident_slot_id = std::numeric_limits<size_t>::max();
    PeuSlot& slot = slots.at(scheduled_wave.peu_slot_index);
    const uint64_t generate_cycle = cycle + wave_generation_cycles;
    const uint64_t dispatch_cycle = generate_cycle + wave_dispatch_cycles;
    ScheduleWaveGenerate(scheduled_wave, generate_cycle, events, trace);
    ScheduleWaveDispatch(scheduled_wave,
                         slot,
                         dispatch_cycle,
                         max_issuable_waves,
                         wave_launch_cycles,
                         events,
                         trace);
  }
}

bool AdmitOneResidentBlock(ApResidentState& ap_state,
                           uint64_t cycle,
                           std::vector<PeuSlot>& slots,
                           uint32_t resident_wave_slots_per_peu,
                           uint32_t max_issuable_waves,
                           uint64_t wave_generation_cycles,
                           uint64_t wave_dispatch_cycles,
                           uint64_t wave_launch_cycles,
                           EventQueue& events,
                           TraceSink& trace) {
  while (ap_state.resident_blocks.size() < ap_state.resident_block_limit &&
         !ap_state.pending_blocks.empty()) {
    ExecutableBlock* next_block = ap_state.pending_blocks.front();
    if (next_block == nullptr || next_block->active || next_block->completed) {
      ap_state.pending_blocks.pop_front();
      continue;
    }
    if (!CanAdmitBlockToResidentWaveSlots(*next_block, slots, resident_wave_slots_per_peu)) {
      return false;
    }
    ap_state.pending_blocks.pop_front();
    ap_state.resident_blocks.push_back(next_block);
    ActivateBlock(*next_block,
                  cycle,
                  slots,
                  max_issuable_waves,
                  wave_generation_cycles,
                  wave_dispatch_cycles,
                  wave_launch_cycles,
                  events,
                  trace);
    return true;
  }
  return false;
}

void AdmitResidentBlocks(ApResidentState& ap_state,
                         uint64_t cycle,
                         std::vector<PeuSlot>& slots,
                         uint32_t resident_wave_slots_per_peu,
                         uint32_t max_issuable_waves,
                         uint64_t wave_generation_cycles,
                         uint64_t wave_dispatch_cycles,
                         uint64_t wave_launch_cycles,
                         EventQueue& events,
                         TraceSink& trace) {
  while (AdmitOneResidentBlock(ap_state,
                               cycle,
                               slots,
                               resident_wave_slots_per_peu,
                               max_issuable_waves,
                               wave_generation_cycles,
                               wave_dispatch_cycles,
                               wave_launch_cycles,
                               events,
                               trace)) {
  }
}

bool RetireResidentBlock(ApResidentState& ap_state, ExecutableBlock* block) {
  if (block == nullptr) {
    return false;
  }
  auto it = std::find(ap_state.resident_blocks.begin(), ap_state.resident_blocks.end(), block);
  if (it != ap_state.resident_blocks.end()) {
    ap_state.resident_blocks.erase(it);
    return true;
  }
  return false;
}

bool CanScheduleDelayedReadmit(const ApResidentState& ap_state) {
  return ap_state.pending_blocks.size() >
         static_cast<size_t>(ap_state.scheduled_readmit_count);
}

void FillDispatchWindow(PeuSlot& slot,
                        uint64_t cycle,
                        uint32_t max_issuable_waves,
                        uint64_t wave_launch_cycles,
                        EventQueue& events,
                        TraceSink& trace) {
  RefillActiveWindow(slot, cycle, max_issuable_waves, wave_launch_cycles, events, trace, false);
}

}  // namespace cycle_internal
}  // namespace gpu_model
