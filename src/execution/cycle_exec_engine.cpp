#include "gpu_model/execution/cycle_exec_engine.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <vector>

#include "gpu_model/debug/trace/event.h"
#include "gpu_model/debug/trace/event_factory.h"
#include "gpu_model/debug/trace/instruction_trace.h"
#include "gpu_model/debug/trace/wave_launch_trace.h"
#include "gpu_model/execution/internal/barrier_resource_pool.h"
#include "gpu_model/execution/internal/cycle_issue_policy.h"
#include "gpu_model/execution/internal/event_queue.h"
#include "gpu_model/execution/internal/issue_model.h"
#include "gpu_model/execution/internal/issue_scheduler.h"
#include "gpu_model/execution/internal/opcode_execution_info.h"
#include "gpu_model/execution/internal/async_scoreboard.h"
#include "gpu_model/execution/memory_ops.h"
#include "gpu_model/execution/plan_apply.h"
#include "gpu_model/execution/sync_ops.h"
#include "gpu_model/execution/wave_context_builder.h"
#include "gpu_model/execution/wave_stats.h"
#include "gpu_model/execution/internal/issue_eligibility.h"
#include "gpu_model/isa/opcode.h"
#include "gpu_model/loader/device_image_loader.h"
#include "gpu_model/memory/cache_model.h"
#include "gpu_model/memory/shared_bank_model.h"
#include "gpu_model/runtime/program_cycle_tracker.h"
#include "gpu_model/runtime/program_cycle_stats.h"

namespace gpu_model {

namespace {

constexpr uint64_t kIssueTimelineQuantumCycles = 4;

uint64_t QuantizeIssueDuration(uint64_t cycles) {
  const uint64_t clamped = std::max<uint64_t>(kIssueTimelineQuantumCycles, cycles);
  const uint64_t remainder = clamped % kIssueTimelineQuantumCycles;
  if (remainder == 0) {
    return clamped;
  }
  return clamped + (kIssueTimelineQuantumCycles - remainder);
}

std::optional<ExecutedStepClass> ClassifyCycleInstruction(const Instruction& instruction,
                                                          const OpPlan& plan) {
  // Sync instructions: barrier, waitcnt
  if (plan.sync_barrier || plan.sync_wave_barrier || plan.wait_cnt) {
    return ExecutedStepClass::Sync;
  }

  // Vector memory instructions: global, shared, private
  if (plan.memory.has_value()) {
    return ExecutedStepClass::VectorMem;
  }

  // Classify by semantic family (hardware execution unit)
  switch (GetOpcodeExecutionInfo(instruction.opcode).family) {
    case SemanticFamily::ScalarAlu:
    case SemanticFamily::ScalarCompare:
      return ExecutedStepClass::ScalarAlu;
    case SemanticFamily::VectorAluInt:
    case SemanticFamily::VectorAluFloat:
    case SemanticFamily::VectorCompare:
      return ExecutedStepClass::VectorAlu;
    case SemanticFamily::ScalarMemory:
      return ExecutedStepClass::ScalarMem;
    case SemanticFamily::VectorMemory:
    case SemanticFamily::LocalDataShare:
      return ExecutedStepClass::VectorMem;
    case SemanticFamily::Branch:
      return ExecutedStepClass::Branch;
    case SemanticFamily::Builtin:
    case SemanticFamily::Mask:
    case SemanticFamily::Sync:
    case SemanticFamily::Special:
      return ExecutedStepClass::Other;
  }
  return ExecutedStepClass::Other;
}

uint64_t CostForCycleStep(const OpPlan& plan,
                          ExecutedStepClass step_class,
                          const ProgramCycleStatsConfig& config) {
  switch (step_class) {
    case ExecutedStepClass::ScalarAlu:
    case ExecutedStepClass::VectorAlu:
    case ExecutedStepClass::Branch:
    case ExecutedStepClass::Sync:
      return plan.issue_cycles;
    case ExecutedStepClass::Tensor:
      return config.tensor_cycles;
    case ExecutedStepClass::ScalarMem:
      return config.scalar_mem_cycles;
    case ExecutedStepClass::VectorMem:
      // VectorMem includes global, shared, private memory
      // Use global_mem_cycles as default (dominant case)
      return config.global_mem_cycles;
    case ExecutedStepClass::Other:
      return plan.issue_cycles == 0 ? config.default_issue_cycles : plan.issue_cycles;
  }
  return config.default_issue_cycles;
}

void AccumulateProgramCycleStep(ProgramCycleStats& stats,
                                ExecutedStepClass step_class,
                                uint64_t cost_cycles,
                                uint64_t work_weight) {
  const uint64_t weighted_cycles = cost_cycles * work_weight;
  stats.total_issued_work_cycles += weighted_cycles;
  switch (step_class) {
    case ExecutedStepClass::ScalarAlu:
      stats.scalar_alu_cycles += weighted_cycles;
      stats.scalar_alu_insts += 1;
      return;
    case ExecutedStepClass::ScalarMem:
      stats.scalar_mem_cycles += weighted_cycles;
      stats.scalar_mem_insts += 1;
      return;
    case ExecutedStepClass::VectorAlu:
      stats.vector_alu_cycles += weighted_cycles;
      stats.vector_alu_insts += 1;
      return;
    case ExecutedStepClass::VectorMem:
      stats.global_mem_cycles += weighted_cycles;
      stats.vector_mem_insts += 1;
      return;
    case ExecutedStepClass::Branch:
      stats.branch_insts += 1;
      return;
    case ExecutedStepClass::Sync:
      stats.barrier_cycles += weighted_cycles;
      stats.sync_insts += 1;
      return;
    case ExecutedStepClass::Tensor:
      stats.tensor_cycles += weighted_cycles;
      stats.tensor_insts += 1;
      return;
    case ExecutedStepClass::Other:
      stats.other_insts += 1;
      return;
  }
}

struct ExecutableBlock;

struct ScheduledWave {
  uint32_t dpc_id = 0;
  ExecutableBlock* block = nullptr;
  WaveContext wave;
  uint64_t generate_cycle = 0;
  uint64_t dispatch_cycle = 0;
  uint64_t launch_cycle = 0;
  bool generate_completed = false;
  bool generate_scheduled = false;
  bool dispatch_completed = false;
  bool dispatch_scheduled = false;
  bool launch_completed = false;
  bool dispatch_enabled = false;
  bool launch_scheduled = false;
  size_t peu_slot_index = std::numeric_limits<size_t>::max();
  size_t resident_slot_id = std::numeric_limits<size_t>::max();
  // Issue timing state (aligned with WaveExecutionState)
  uint64_t last_issue_cycle = 0;
  uint64_t next_issue_cycle = 0;
  uint64_t eligible_since_cycle = 0;
  bool eligible_since_valid = false;
};

struct ExecutableBlock {
  uint32_t block_id = 0;
  uint32_t dpc_id = 0;
  uint32_t ap_id = 0;
  uint32_t global_ap_id = 0;
  uint32_t ap_queue_index = 0;
  uint64_t barrier_generation = 0;
  uint32_t barrier_arrivals = 0;
  bool barrier_slot_acquired = false;
  bool active = false;
  bool completed = false;
  std::vector<std::byte> shared_memory;
  std::vector<ScheduledWave> waves;
};

struct ResidentIssueSlot {
  uint32_t dpc_id = 0;
  uint32_t ap_id = 0;
  uint32_t peu_id = 0;
  uint32_t slot_id = 0;
  ScheduledWave* resident_wave = nullptr;
  bool active = false;
};

struct PeuSlot {
  uint32_t dpc_id = 0;
  uint32_t ap_id = 0;
  uint32_t peu_id = 0;
  uint64_t busy_until = 0;
  uint64_t last_wave_tag = std::numeric_limits<uint64_t>::max();
  std::optional<TraceWaveView> last_wave_trace;
  uint64_t last_wave_pc = 0;
  size_t issue_round_robin_index = 0;
  uint64_t switch_ready_cycle = 0;  // Earliest cycle when wave switch is ready
  std::vector<ScheduledWave*> waves;
  std::vector<ScheduledWave*> resident_waves;
  std::vector<ResidentIssueSlot> resident_slots;
  std::deque<size_t> standby_slot_ids;
};

struct ApResidentState {
  uint32_t global_ap_id = 0;
  std::deque<ExecutableBlock*> pending_blocks;
  std::vector<ExecutableBlock*> resident_blocks;
  uint32_t resident_block_limit = 2;
  uint32_t scheduled_readmit_count = 0;
  uint32_t barrier_slot_capacity = 0;
  uint32_t barrier_slots_in_use = 0;
};

struct L1Key {
  uint32_t dpc_id = 0;
  uint32_t ap_id = 0;

  bool operator<(const L1Key& other) const {
    return std::tie(dpc_id, ap_id) < std::tie(other.dpc_id, other.ap_id);
  }
};

uint64_t LoadLaneValue(const MemorySystem& memory, MemoryPoolKind pool, const LaneAccess& lane) {
  return memory_ops::LoadPoolLaneValue(memory, pool, lane);
}

uint64_t LoadLaneValue(const MemorySystem& memory, const LaneAccess& lane) {
  return memory_ops::LoadGlobalLaneValue(memory, lane);
}

uint64_t ConstantPoolBase(const ExecutionContext& context) {
  if (context.device_load == nullptr) {
    return 0;
  }
  for (const auto& segment : context.device_load->segments) {
    if (segment.segment.kind == DeviceSegmentKind::ConstantData) {
      return segment.allocation.range.base;
    }
  }
  return 0;
}

bool ResumeWaitcntWaveIfReady(const ExecutableKernel& kernel, WaveContext& wave) {
  if (wave.run_state != WaveRunState::Waiting) {
    return false;
  }
  if (!kernel.ContainsPc(wave.pc)) {
    return false;
  }
  const Instruction& instruction = kernel.InstructionAtPc(wave.pc);
  if (instruction.opcode != Opcode::SWaitCnt ||
      !ResumeMemoryWaitStateIfSatisfied(WaitCntThresholdsForInstruction(instruction), wave)) {
    return false;
  }
  const auto next_pc = kernel.NextPc(wave.pc);
  if (!next_pc.has_value()) {
    throw std::out_of_range("next instruction pc not found");
  }
  wave.pc = *next_pc;
  wave.valid_entry = true;
  if (wave.status != WaveStatus::Exited && !wave.waiting_at_barrier) {
    wave.status = WaveStatus::Active;
  }
  return true;
}

AsyncArriveResult MakeCycleArriveResult(const ExecutableKernel& kernel,
                                        const WaveContext& wave,
                                        MemoryWaitDomain arrive_domain) {
  if (wave.run_state != WaveRunState::Waiting) {
    return {};
  }
  if (!kernel.ContainsPc(wave.pc)) {
    return {};
  }
  const Instruction& instruction = kernel.InstructionAtPc(wave.pc);
  if (instruction.opcode != Opcode::SWaitCnt) {
    return {};
  }
  return MakeAsyncArriveResult(wave, arrive_domain, WaitCntThresholdsForInstruction(instruction));
}

bool IssueLimitsUnset(const ArchitecturalIssueLimits& limits) {
  return limits.branch == 0 && limits.scalar_alu_or_memory == 0 && limits.vector_alu == 0 &&
         limits.vector_memory == 0 && limits.local_data_share == 0 &&
         limits.global_data_share_or_export == 0 && limits.special == 0;
}

ArchitecturalIssuePolicy ResolveIssuePolicy(const CycleTimingConfig& timing_config,
                                            const GpuArchSpec& spec) {
  if (timing_config.issue_policy.has_value()) {
    return *timing_config.issue_policy;
  }
  if (IssueLimitsUnset(timing_config.issue_limits)) {
    return CycleIssuePolicyForSpec(spec);
  }
  return ArchitecturalIssuePolicyFromLimits(timing_config.issue_limits);
}

uint64_t ModeledAsyncCompletionDelay(uint32_t issue_cycles, uint32_t default_issue_cycles) {
  if (issue_cycles <= default_issue_cycles) {
    return 0;
  }
  return static_cast<uint64_t>(issue_cycles - default_issue_cycles);
}

void StoreLaneValue(MemorySystem& memory, const LaneAccess& lane) {
  memory_ops::StoreGlobalLaneValue(memory, lane);
}

uint64_t LoadLaneValue(const std::vector<std::byte>& memory, const LaneAccess& lane) {
  return memory_ops::LoadByteLaneValue(memory, lane);
}

void StoreLaneValue(std::vector<std::byte>& memory, const LaneAccess& lane) {
  memory_ops::StoreByteLaneValue(memory, lane);
}

uint64_t LoadLaneValue(const std::array<std::vector<std::byte>, kWaveSize>& memory,
                       uint32_t lane_id,
                       const LaneAccess& lane) {
  return memory_ops::LoadPrivateLaneValue(memory, lane_id, lane);
}

void StoreLaneValue(std::array<std::vector<std::byte>, kWaveSize>& memory,
                    uint32_t lane_id,
                    const LaneAccess& lane) {
  memory_ops::StorePrivateLaneValue(memory, lane_id, lane);
}

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

constexpr std::string_view kStallReasonBarrierSlotUnavailable = "barrier_slot_unavailable";
constexpr std::string_view kStallReasonIssueGroupConflict = "issue_group_conflict";

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

void ScheduleWaveLaunch(ScheduledWave& scheduled_wave,
                        uint64_t cycle,
                        EventQueue& events,
                        TraceSink& trace,
                        bool immediate = false) {
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

template <typename Container>
void RemoveWaveFromList(Container& waves, const ScheduledWave* target) {
  waves.erase(std::remove(waves.begin(), waves.end(), target), waves.end());
}

template <typename Container>
bool ContainsWave(const Container& waves, const ScheduledWave* target) {
  return std::find(waves.begin(), waves.end(), target) != waves.end();
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
    });
  }

  return candidates;
}

}  // namespace

uint64_t CycleExecEngine::Run(ExecutionContext& context) {
  const uint64_t run_begin_cycle = context.cycle;
  ProgramCycleStatsConfig cycle_stats_config;
  cycle_stats_config.default_issue_cycles = context.spec.default_issue_cycles;
  program_cycle_stats_ = ProgramCycleStats{};

  auto blocks = MaterializeBlocks(context.placement, context.launch_config);

  // Count waves launched
  if (program_cycle_stats_.has_value()) {
    for (const auto& block : blocks) {
      program_cycle_stats_->waves_launched += static_cast<uint32_t>(block.waves.size());
    }
  }

  const uint32_t resident_wave_slots_per_peu = ResidentWaveSlotCapacityPerPeu(context.spec);
  auto slots = BuildPeuSlots(blocks, resident_wave_slots_per_peu);
  EventQueue events;
  std::map<L1Key, CacheModel> l1_caches;
  CacheModel l2_cache(timing_config_.cache_model);
  SharedBankModel shared_bank_model(timing_config_.shared_bank_model);
  std::map<uint32_t, ApResidentState> ap_states;

  for (auto& slot : slots) {
    slot.busy_until = 0;
    l1_caches.emplace(L1Key{.dpc_id = slot.dpc_id, .ap_id = slot.ap_id},
                      CacheModel(timing_config_.cache_model));
  }

  for (auto& block : blocks) {
    auto& ap_state = ap_states[block.global_ap_id];
    ap_state.global_ap_id = block.global_ap_id;
    ap_state.barrier_slot_capacity = context.spec.cycle_resources.barrier_slots_per_ap;
    ap_state.pending_blocks.push_back(&block);
  }
  for (auto& [global_ap_id, ap_state] : ap_states) {
    (void)global_ap_id;
    AdmitResidentBlocks(ap_state,
                        context.cycle,
                        slots,
                        resident_wave_slots_per_peu,
                        context.spec.max_issuable_waves,
                        timing_config_.launch_timing.wave_generation_cycles,
                        timing_config_.launch_timing.wave_dispatch_cycles,
                        timing_config_.launch_timing.wave_launch_cycles,
                        events,
                        context.trace);
  }

  uint64_t cycle = context.cycle;
  while (true) {
    context.cycle = cycle;
    events.RunReady(cycle);
    for (auto& slot : slots) {
      FillDispatchWindow(slot,
                         cycle,
                         context.spec.max_issuable_waves,
                         timing_config_.launch_timing.wave_launch_cycles,
                         events,
                         context.trace);
    }
    events.RunReady(cycle);

    if (AllWavesExited(blocks) && events.empty()) {
      if (program_cycle_stats_.has_value()) {
        program_cycle_stats_->total_cycles = cycle - run_begin_cycle;
      }
      return cycle;
    }

    // Track active/idle cycles
    if (program_cycle_stats_.has_value()) {
      bool has_active_wave = false;
      for (const auto& block : blocks) {
        for (const auto& scheduled_wave : block.waves) {
          if (scheduled_wave.wave.status != WaveStatus::Exited) {
            has_active_wave = true;
            break;
          }
        }
        if (has_active_wave) break;
      }
      if (has_active_wave) {
        program_cycle_stats_->active_cycles += 1;
      } else {
        program_cycle_stats_->idle_cycles += 1;
      }
    }

    bool issued_any = false;
    for (auto& slot : slots) {
      if (slot.busy_until > cycle) {
        continue;
      }

      std::vector<ResidentIssueSlot*> ordered_resident_slots;
      const auto candidates =
          BuildResidentIssueCandidates(slot, context.kernel, cycle, ap_states, ordered_resident_slots);
      const auto bundle = IssueScheduler::SelectIssueBundle(
          candidates,
          slot.issue_round_robin_index,
          timing_config_.eligible_wave_selection_policy,
          ResolveIssuePolicy(timing_config_, context.spec));
      slot.issue_round_robin_index = bundle.next_round_robin_index;

      if (bundle.selected_candidate_indices.empty()) {
        if (const auto blocked =
                PickFirstBlockedResidentWave(slot, context.kernel, cycle, ap_states)) {
          std::optional<WaitCntThresholds> waitcnt_thresholds;
          const WaveContext& blocked_wave = blocked->first->wave;
          if (context.kernel.ContainsPc(blocked_wave.pc)) {
            const Instruction& blocked_instruction = context.kernel.InstructionAtPc(blocked_wave.pc);
            if (blocked_instruction.opcode == Opcode::SWaitCnt) {
              waitcnt_thresholds = WaitCntThresholdsForInstruction(blocked_instruction);
            }
          }
          context.trace.OnEvent(MakeTraceBlockedStallEvent(
              MakeTraceWaveView(*blocked->first, TraceSlotId(*blocked->first)),
              cycle,
              blocked->second,
              TraceSlotModelKind::ResidentFixed,
              std::numeric_limits<uint64_t>::max(),
              MakeOptionalTraceWaitcntState(blocked_wave, waitcnt_thresholds)));
        }
        continue;
      }

      if (const auto ready_unselected =
              PickFirstReadyUnselectedResidentWave(candidates, bundle, ordered_resident_slots)) {
        context.trace.OnEvent(MakeTraceBlockedStallEvent(
            MakeTraceWaveView(*ready_unselected->first, TraceSlotId(*ready_unselected->first)),
            cycle,
            ready_unselected->second,
            TraceSlotModelKind::ResidentFixed));
      }

      uint64_t bundle_commit_cycle = cycle;
      uint64_t bundle_last_wave_tag = slot.last_wave_tag;
      for (const size_t selected_candidate_index : bundle.selected_candidate_indices) {
        ResidentIssueSlot& resident_slot = *ordered_resident_slots.at(selected_candidate_index);
        ScheduledWave* candidate = resident_slot.resident_wave;
        if (candidate == nullptr || !context.kernel.ContainsPc(candidate->wave.pc)) {
          throw std::out_of_range("wave pc out of range");
        }

        WaveContext& wave = candidate->wave;
        const Instruction instruction = context.kernel.InstructionAtPc(wave.pc);
        const uint32_t slot_id = TraceSlotId(*candidate);
        const uint64_t wave_tag = WaveTag(wave);

        // Calculate actual_issue_cycle considering switch penalty and wave timing
        const bool wave_switched = slot.last_wave_tag != std::numeric_limits<uint64_t>::max() &&
                                    slot.last_wave_tag != wave_tag;
        const uint64_t switch_penalty = wave_switched 
            ? timing_config_.launch_timing.warp_switch_cycles 
            : 0;
        
        // Update switch_ready_cycle if wave switched
        if (wave_switched && switch_penalty > 0) {
          slot.switch_ready_cycle = cycle + switch_penalty;
        }
        
        // Calculate actual issue cycle
        const uint64_t actual_issue_cycle = std::max({cycle, candidate->next_issue_cycle, slot.switch_ready_cycle});
        
        // Emit switch events if penalty applies
        if (wave_switched && switch_penalty > 0 && slot.last_wave_trace.has_value()) {
          context.trace.OnEvent(MakeTraceWaveSwitchAwayEvent(
              *slot.last_wave_trace, cycle, TraceSlotModelKind::ResidentFixed, slot.last_wave_pc));
          context.trace.OnEvent(MakeTraceWaveSwitchStallEvent(
              *slot.last_wave_trace, cycle, TraceSlotModelKind::ResidentFixed, slot.last_wave_pc));
        }
        
                context.trace.OnEvent(MakeTraceIssueSelectEvent(
            MakeTraceWaveView(*candidate, slot_id), cycle, TraceSlotModelKind::ResidentFixed));
        if (context.stats != nullptr) {
          ++context.stats->wave_steps;
          ++context.stats->instructions_issued;
        }
        const OpPlan plan = semantics_.BuildPlan(instruction, wave, context);
        if (program_cycle_stats_.has_value()) {
          // Track total instructions
          program_cycle_stats_->instructions_executed += 1;

          if (const auto step_class = ClassifyCycleInstruction(instruction, plan); step_class.has_value()) {
            AccumulateProgramCycleStep(*program_cycle_stats_,
                                       *step_class,
                                       CostForCycleStep(plan, *step_class, cycle_stats_config),
                                       wave.exec.count());
          }
        }
        context.trace.OnEvent(MakeTraceWaveStepEvent(MakeTraceWaveView(*candidate, slot_id),
                                                     actual_issue_cycle,
                                                     TraceSlotModelKind::ResidentFixed,
                                                     FormatWaveStepMessage(instruction, wave),
                                                     std::numeric_limits<uint64_t>::max(),
                                                     QuantizeIssueDuration(plan.issue_cycles)));
        const uint64_t commit_cycle = actual_issue_cycle + plan.issue_cycles;
        bundle_commit_cycle = std::max(bundle_commit_cycle, commit_cycle);
        bundle_last_wave_tag = wave_tag;
        slot.last_wave_trace = MakeTraceWaveView(*candidate, slot_id);
        slot.last_wave_pc = wave.pc;
        // Update issue timing state
        candidate->last_issue_cycle = actual_issue_cycle;
        candidate->next_issue_cycle = commit_cycle;
        candidate->eligible_since_valid = false;
        wave.status = WaveStatus::Stalled;
        wave.valid_entry = false;
        uint64_t flow_id = 0;
        if (plan.memory.has_value()) {
          IncrementPendingMemoryOps(wave, MemoryDomainForOpcode(instruction.opcode));
        }
        if (instruction.opcode == Opcode::BBranch || instruction.opcode == Opcode::BIfSmask ||
            instruction.opcode == Opcode::BIfNoexec) {
          wave.branch_pending = true;
        }

        if (plan.memory.has_value()) {
          if (context.stats != nullptr) {
            ++context.stats->memory_ops;
            const auto& request = *plan.memory;
            if (request.space == MemorySpace::Global) {
              if (request.kind == AccessKind::Load) {
                ++context.stats->global_loads;
              } else if (request.kind == AccessKind::Store || request.kind == AccessKind::Atomic) {
                ++context.stats->global_stores;
              }
            } else if (request.space == MemorySpace::Shared) {
              if (request.kind == AccessKind::Load) {
                ++context.stats->shared_loads;
              } else if (request.kind == AccessKind::Store || request.kind == AccessKind::Atomic) {
                ++context.stats->shared_stores;
              }
            } else if (request.space == MemorySpace::Private) {
              if (request.kind == AccessKind::Load) {
                ++context.stats->private_loads;
              } else if (request.kind == AccessKind::Store) {
                ++context.stats->private_stores;
              }
            } else if (request.space == MemorySpace::Constant && request.kind == AccessKind::Load) {
              ++context.stats->constant_loads;
            }
          }
          // Track memory ops in program_cycle_stats
          if (program_cycle_stats_.has_value()) {
            const auto& request = *plan.memory;
            if (request.space == MemorySpace::Global) {
              if (request.kind == AccessKind::Load) {
                program_cycle_stats_->global_loads += 1;
              } else if (request.kind == AccessKind::Store || request.kind == AccessKind::Atomic) {
                program_cycle_stats_->global_stores += 1;
              }
            } else if (request.space == MemorySpace::Shared) {
              if (request.kind == AccessKind::Load) {
                program_cycle_stats_->shared_loads += 1;
              } else if (request.kind == AccessKind::Store || request.kind == AccessKind::Atomic) {
                program_cycle_stats_->shared_stores += 1;
              }
            } else if (request.space == MemorySpace::Private) {
              if (request.kind == AccessKind::Load) {
                program_cycle_stats_->private_loads += 1;
              } else if (request.kind == AccessKind::Store) {
                program_cycle_stats_->private_stores += 1;
              }
            } else if (request.space == MemorySpace::Constant && request.kind == AccessKind::Load) {
              program_cycle_stats_->scalar_loads += 1;
            }
          }
          flow_id = context.AllocateTraceFlowId();
          TraceEvent issue_event = MakeTraceWaveEvent(
              MakeTraceWaveView(*candidate, slot_id),
              TraceEventKind::MemoryAccess,
              cycle,
              TraceSlotModelKind::ResidentFixed,
              plan.memory->kind == AccessKind::Load ? "load_issue" : "store_issue");
          issue_event.flow_id = flow_id;
          issue_event.flow_phase = TraceFlowPhase::Start;
          context.trace.OnEvent(std::move(issue_event));
        }

        events.Schedule(TimedEvent{
            .cycle = commit_cycle,
            .action =
                [&, candidate, instruction, plan, commit_cycle, slot_id, flow_id]() {
                context.cycle = commit_cycle;

                ApplyExecutionPlanRegisterWrites(plan, candidate->wave);
                if (const auto mask_text = MaybeFormatExecutionMaskUpdate(plan, candidate->wave);
                    mask_text.has_value()) {
                  context.trace.OnEvent(MakeTraceWaveEvent(MakeTraceWaveView(*candidate, slot_id),
                                                           TraceEventKind::ExecMaskUpdate,
                                                           commit_cycle,
                                                           TraceSlotModelKind::ResidentFixed,
                                                           *mask_text));
                }

                context.trace.OnEvent(MakeTraceCommitEvent(
                    MakeTraceWaveView(*candidate, slot_id),
                    commit_cycle,
                    TraceSlotModelKind::ResidentFixed));

                if (plan.memory.has_value()) {
                  const MemoryRequest request = *plan.memory;
                  if (request.space == MemorySpace::Global) {
                    auto& l1_cache =
                        l1_caches.at(L1Key{.dpc_id = candidate->block->dpc_id, .ap_id = candidate->block->ap_id});
                    const std::vector<uint64_t> addrs = ActiveAddresses(request);
                    const CacheProbeResult l1_probe = l1_cache.Probe(addrs);
                    const CacheProbeResult l2_probe = l2_cache.Probe(addrs);
                    uint64_t arrive_latency = std::min(l1_probe.latency, l2_probe.latency);
                    // gem5: store uses 2x bus latency
                    if (request.kind == AccessKind::Store || request.kind == AccessKind::Atomic) {
                      arrive_latency *= cycle_stats_config.store_latency_multiplier;
                    }
                    if (context.stats != nullptr) {
                      if (l1_probe.l1_hits > 0) {
                        context.stats->l1_hits += l1_probe.l1_hits;
                      } else if (l2_probe.l2_hits > 0) {
                        context.stats->l2_hits += l2_probe.l2_hits;
                      } else {
                        context.stats->cache_misses += std::max<uint64_t>(1, l2_probe.misses);
                      }
                    }
                    const uint64_t arrive_cycle = commit_cycle + arrive_latency;
                    const uint64_t completion_flow_id = flow_id;
                    events.Schedule(TimedEvent{
                        .cycle = arrive_cycle,
                        .action =
                            [&, candidate, request, addrs, arrive_cycle, slot_id, completion_flow_id]() {
                              context.cycle = arrive_cycle;
                              if (request.kind == AccessKind::Load) {
                                if (!request.dst.has_value()) {
                                  throw std::invalid_argument("load request missing destination");
                                }
                                for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
                                  if (request.lanes[lane].active) {
                                    const uint64_t value =
                                        LoadLaneValue(context.memory, request.lanes[lane]);
                                    candidate->wave.vgpr.Write(request.dst->index, lane, value);
                                  }
                                }
                                l2_cache.Promote(addrs);
                                l1_caches
                                    .at(L1Key{.dpc_id = candidate->block->dpc_id, .ap_id = candidate->block->ap_id})
                                    .Promote(addrs);
                              } else if (request.kind == AccessKind::Store) {
                                for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
                                  if (request.lanes[lane].active) {
                                    StoreLaneValue(context.memory, request.lanes[lane]);
                                  }
                                }
                              } else if (request.kind == AccessKind::Atomic) {
                                for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
                                  if (!request.lanes[lane].active) {
                                    continue;
                                  }
                                  const int32_t prior = static_cast<int32_t>(
                                      LoadLaneValue(context.memory, request.lanes[lane]));
                                  const int32_t updated =
                                      prior + static_cast<int32_t>(request.lanes[lane].value);
                                  LaneAccess writeback = request.lanes[lane];
                                  writeback.value = static_cast<uint64_t>(static_cast<int64_t>(updated));
                                  StoreLaneValue(context.memory, writeback);
                                }
                              }

                              TraceEvent arrive_event = MakeTraceMemoryArriveEvent(
                                  MakeTraceWaveView(*candidate, slot_id),
                                  arrive_cycle,
                                  request.kind == AccessKind::Load ? TraceMemoryArriveKind::Load
                                                                   : TraceMemoryArriveKind::Store,
                                  TraceSlotModelKind::ResidentFixed);
                              arrive_event.flow_id = completion_flow_id;
                              arrive_event.flow_phase = TraceFlowPhase::Finish;
                              DecrementPendingMemoryOps(candidate->wave, MemoryWaitDomain::Global);
                              const AsyncArriveResult arrive_result = MakeCycleArriveResult(
                                  context.kernel, candidate->wave, MemoryWaitDomain::Global);
                              arrive_event.waitcnt_state = arrive_result.waitcnt_state;
                              arrive_event.arrive_progress = arrive_result.arrive_progress;
                              context.trace.OnEvent(std::move(arrive_event));
                              context.trace.OnEvent(MakeTraceWaveArriveEvent(
                                  MakeTraceWaveView(*candidate, slot_id),
                                  arrive_cycle,
                                  request.kind == AccessKind::Load ? TraceMemoryArriveKind::Load
                                                                   : TraceMemoryArriveKind::Store,
                                  TraceSlotModelKind::ResidentFixed,
                                  arrive_result.arrive_progress,
                                  std::numeric_limits<uint64_t>::max(),
                                  arrive_result.waitcnt_state));
                              if (!ResumeWaitcntWaveIfReady(context.kernel, candidate->wave)) {
                                candidate->wave.valid_entry = true;
                                if (candidate->wave.status != WaveStatus::Exited &&
                                    !candidate->wave.waiting_at_barrier) {
                                  candidate->wave.status = WaveStatus::Active;
                                }
                              } else {
                                candidate->eligible_since_cycle = arrive_cycle;
                                candidate->eligible_since_valid = true;
                                context.trace.OnEvent(MakeTraceWaveResumeEvent(
                                    MakeTraceWaveView(*candidate, slot_id),
                                    arrive_cycle,
                                    TraceSlotModelKind::ResidentFixed));
                              }
                            },
                    });
                  } else if (request.space == MemorySpace::Shared) {
                    const uint64_t penalty = shared_bank_model.ConflictPenalty(request);
                    uint64_t async_delay = ModeledAsyncCompletionDelay(
                        plan.issue_cycles, context.spec.default_issue_cycles);
                    // gem5: store uses 2x bus latency
                    if (request.kind == AccessKind::Store || request.kind == AccessKind::Atomic) {
                      async_delay *= cycle_stats_config.store_latency_multiplier;
                    }
                    if (context.stats != nullptr) {
                      context.stats->shared_bank_conflict_penalty_cycles += penalty;
                    }
                    const uint64_t ready_cycle = commit_cycle + penalty + async_delay;
                    const uint64_t completion_flow_id = flow_id;
                    events.Schedule(TimedEvent{
                        .cycle = ready_cycle,
                        .action =
                            [&, candidate, request, ready_cycle, plan, slot_id, completion_flow_id]() {
                              context.cycle = ready_cycle;
                              if (request.kind == AccessKind::Load) {
                                if (!request.dst.has_value()) {
                                  throw std::invalid_argument("load request missing destination");
                                }
                                for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
                                  if (request.lanes[lane].active) {
                                    const uint64_t value =
                                        LoadLaneValue(candidate->block->shared_memory, request.lanes[lane]);
                                    candidate->wave.vgpr.Write(request.dst->index, lane, value);
                                  }
                                }
                              } else if (request.kind == AccessKind::Store) {
                                for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
                                  if (request.lanes[lane].active) {
                                    StoreLaneValue(candidate->block->shared_memory, request.lanes[lane]);
                                  }
                                }
                              } else if (request.kind == AccessKind::Atomic) {
                                for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
                                  if (!request.lanes[lane].active) {
                                    continue;
                                  }
                                  const int32_t prior = static_cast<int32_t>(
                                      LoadLaneValue(candidate->block->shared_memory, request.lanes[lane]));
                                  const int32_t updated =
                                      prior + static_cast<int32_t>(request.lanes[lane].value);
                                  LaneAccess writeback = request.lanes[lane];
                                  writeback.value = static_cast<uint64_t>(static_cast<int64_t>(updated));
                                  StoreLaneValue(candidate->block->shared_memory, writeback);
                                }
                              }

                              TraceEvent arrive_event = MakeTraceMemoryArriveEvent(
                                  MakeTraceWaveView(*candidate, slot_id),
                                  ready_cycle,
                                  request.kind == AccessKind::Load ? TraceMemoryArriveKind::Shared
                                                                   : TraceMemoryArriveKind::Store,
                                  TraceSlotModelKind::ResidentFixed);
                              arrive_event.flow_id = completion_flow_id;
                              arrive_event.flow_phase = TraceFlowPhase::Finish;
                              DecrementPendingMemoryOps(candidate->wave, MemoryWaitDomain::Shared);
                              const AsyncArriveResult arrive_result = MakeCycleArriveResult(
                                  context.kernel, candidate->wave, MemoryWaitDomain::Shared);
                              arrive_event.waitcnt_state = arrive_result.waitcnt_state;
                              arrive_event.arrive_progress = arrive_result.arrive_progress;
                              context.trace.OnEvent(std::move(arrive_event));
                              context.trace.OnEvent(MakeTraceWaveArriveEvent(
                                  MakeTraceWaveView(*candidate, slot_id),
                                  ready_cycle,
                                  TraceMemoryArriveKind::Shared,
                                  TraceSlotModelKind::ResidentFixed,
                                  arrive_result.arrive_progress,
                                  std::numeric_limits<uint64_t>::max(),
                                  arrive_result.waitcnt_state));
                              if (!ResumeWaitcntWaveIfReady(context.kernel, candidate->wave)) {
                                candidate->wave.status = WaveStatus::Active;
                              } else {
                                candidate->eligible_since_cycle = ready_cycle;
                                candidate->eligible_since_valid = true;
                                context.trace.OnEvent(MakeTraceWaveResumeEvent(
                                    MakeTraceWaveView(*candidate, slot_id),
                                    ready_cycle,
                                    TraceSlotModelKind::ResidentFixed));
                              }
                            },
                    });
                  } else if (request.space == MemorySpace::Private) {
                    uint64_t async_delay = ModeledAsyncCompletionDelay(plan.issue_cycles, context.spec.default_issue_cycles);
                    // gem5: store uses 2x bus latency
                    if (request.kind == AccessKind::Store) {
                      async_delay *= cycle_stats_config.store_latency_multiplier;
                    }
                    const uint64_t arrive_cycle = commit_cycle + async_delay;
                    const uint64_t completion_flow_id = flow_id;
                    events.Schedule(TimedEvent{
                        .cycle = arrive_cycle,
                        .action =
                            [&, candidate, request, arrive_cycle, slot_id, completion_flow_id]() {
                              context.cycle = arrive_cycle;
                              if (request.kind == AccessKind::Load) {
                                if (!request.dst.has_value()) {
                                  throw std::invalid_argument("load request missing destination");
                                }
                                for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
                                  if (request.lanes[lane].active) {
                                    const uint64_t value = LoadLaneValue(candidate->wave.private_memory,
                                                                         lane,
                                                                         request.lanes[lane]);
                                    candidate->wave.vgpr.Write(request.dst->index, lane, value);
                                  }
                                }
                              } else if (request.kind == AccessKind::Store) {
                                for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
                                  if (request.lanes[lane].active) {
                                    StoreLaneValue(candidate->wave.private_memory,
                                                   lane,
                                                   request.lanes[lane]);
                                  }
                                }
                              }
                              TraceEvent arrive_event = MakeTraceMemoryArriveEvent(
                                  MakeTraceWaveView(*candidate, slot_id),
                                  arrive_cycle,
                                  request.kind == AccessKind::Load ? TraceMemoryArriveKind::Private
                                                                   : TraceMemoryArriveKind::Store,
                                  TraceSlotModelKind::ResidentFixed);
                              arrive_event.flow_id = completion_flow_id;
                              arrive_event.flow_phase = TraceFlowPhase::Finish;
                              DecrementPendingMemoryOps(candidate->wave, MemoryWaitDomain::Private);
                              const AsyncArriveResult arrive_result = MakeCycleArriveResult(
                                  context.kernel, candidate->wave, MemoryWaitDomain::Private);
                              arrive_event.waitcnt_state = arrive_result.waitcnt_state;
                              arrive_event.arrive_progress = arrive_result.arrive_progress;
                              context.trace.OnEvent(std::move(arrive_event));
                              context.trace.OnEvent(MakeTraceWaveArriveEvent(
                                  MakeTraceWaveView(*candidate, slot_id),
                                  arrive_cycle,
                                  TraceMemoryArriveKind::Private,
                                  TraceSlotModelKind::ResidentFixed,
                                  arrive_result.arrive_progress,
                                  std::numeric_limits<uint64_t>::max(),
                                  arrive_result.waitcnt_state));
                              if (!ResumeWaitcntWaveIfReady(context.kernel, candidate->wave)) {
                                candidate->wave.valid_entry = true;
                              } else {
                                candidate->eligible_since_cycle = arrive_cycle;
                                candidate->eligible_since_valid = true;
                                context.trace.OnEvent(MakeTraceWaveResumeEvent(
                                    MakeTraceWaveView(*candidate, slot_id),
                                    arrive_cycle,
                                    TraceSlotModelKind::ResidentFixed));
                              }
                            },
                    });
                  } else if (request.space == MemorySpace::Constant) {
                    const uint64_t arrive_cycle =
                        commit_cycle +
                        ModeledAsyncCompletionDelay(plan.issue_cycles, context.spec.default_issue_cycles);
                    const uint64_t completion_flow_id = flow_id;
                    events.Schedule(TimedEvent{
                        .cycle = arrive_cycle,
                        .action =
                            [&, candidate, request, arrive_cycle, slot_id, completion_flow_id]() {
                              context.cycle = arrive_cycle;
                              if (!request.dst.has_value()) {
                                throw std::invalid_argument("load request missing destination");
                              }
                              if (request.dst->file == RegisterFile::Scalar) {
                                uint64_t loaded_value = 0;
                                for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
                                  if (!request.lanes[lane].active) {
                                    continue;
                                  }
                                  const LaneAccess pool_lane{
                                      .active = request.lanes[lane].active,
                                      .addr = ConstantPoolBase(context) + request.lanes[lane].addr,
                                      .bytes = request.lanes[lane].bytes,
                                      .value = request.lanes[lane].value,
                                  };
                                  if (context.memory.HasRange(MemoryPoolKind::Constant,
                                                              pool_lane.addr,
                                                              request.lanes[lane].bytes)) {
                                    loaded_value =
                                        LoadLaneValue(context.memory, MemoryPoolKind::Constant, pool_lane);
                                  } else {
                                    loaded_value =
                                        LoadLaneValue(context.kernel.const_segment().bytes,
                                                      request.lanes[lane]);
                                  }
                                  break;
                                }
                                candidate->wave.sgpr.Write(request.dst->index, loaded_value);
                              } else {
                                for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
                                  if (request.lanes[lane].active) {
                                    const LaneAccess pool_lane{
                                        .active = request.lanes[lane].active,
                                        .addr = ConstantPoolBase(context) + request.lanes[lane].addr,
                                        .bytes = request.lanes[lane].bytes,
                                        .value = request.lanes[lane].value,
                                    };
                                    const uint64_t value =
                                        context.memory.HasRange(MemoryPoolKind::Constant,
                                                                pool_lane.addr,
                                                                request.lanes[lane].bytes)
                                            ? LoadLaneValue(context.memory,
                                                            MemoryPoolKind::Constant,
                                                            pool_lane)
                                            : LoadLaneValue(context.kernel.const_segment().bytes,
                                                            request.lanes[lane]);
                                    candidate->wave.vgpr.Write(request.dst->index, lane, value);
                                  }
                                }
                              }
                              TraceEvent arrive_event = MakeTraceMemoryArriveEvent(
                                  MakeTraceWaveView(*candidate, slot_id),
                                  arrive_cycle,
                                  TraceMemoryArriveKind::ScalarBuffer,
                                  TraceSlotModelKind::ResidentFixed);
                              arrive_event.flow_id = completion_flow_id;
                              arrive_event.flow_phase = TraceFlowPhase::Finish;
                              DecrementPendingMemoryOps(candidate->wave,
                                                        MemoryWaitDomain::ScalarBuffer);
                              const AsyncArriveResult arrive_result = MakeCycleArriveResult(
                                  context.kernel, candidate->wave, MemoryWaitDomain::ScalarBuffer);
                              arrive_event.waitcnt_state = arrive_result.waitcnt_state;
                              arrive_event.arrive_progress = arrive_result.arrive_progress;
                              context.trace.OnEvent(std::move(arrive_event));
                              context.trace.OnEvent(MakeTraceWaveArriveEvent(
                                  MakeTraceWaveView(*candidate, slot_id),
                                  arrive_cycle,
                                  TraceMemoryArriveKind::ScalarBuffer,
                                  TraceSlotModelKind::ResidentFixed,
                                  arrive_result.arrive_progress,
                                  std::numeric_limits<uint64_t>::max(),
                                  arrive_result.waitcnt_state));
                              if (!ResumeWaitcntWaveIfReady(context.kernel, candidate->wave)) {
                                candidate->wave.valid_entry = true;
                              } else {
                                candidate->eligible_since_cycle = arrive_cycle;
                                candidate->eligible_since_valid = true;
                                context.trace.OnEvent(MakeTraceWaveResumeEvent(
                                    MakeTraceWaveView(*candidate, slot_id),
                                    arrive_cycle,
                                    TraceSlotModelKind::ResidentFixed));
                              }
                            },
                    });
                  } else {
                    throw std::invalid_argument("unsupported memory space in cycle executor");
                  }
                }

                if (plan.sync_wave_barrier) {
                  if (context.stats != nullptr) {
                    ++context.stats->barriers;
                  }
                  context.trace.OnEvent(MakeTraceBarrierWaveEvent(
                      MakeTraceWaveView(*candidate, slot_id),
                      commit_cycle,
                      TraceSlotModelKind::ResidentFixed));
                  ApplyExecutionPlanControlFlow(context.kernel, plan, candidate->wave, true, true);
                  candidate->wave.status = WaveStatus::Active;
                  return;
                }

                if (plan.sync_barrier) {
                  if (context.stats != nullptr) {
                    ++context.stats->barriers;
                  }
                  auto& ap_state = ap_states[candidate->block->global_ap_id];
                  const bool acquired = TryAcquireBarrierSlot(ap_state.barrier_slot_capacity,
                                                              ap_state.barrier_slots_in_use,
                                                              candidate->block->barrier_slot_acquired);
                  if (!acquired) {
                    candidate->wave.status = WaveStatus::Active;
                    candidate->wave.valid_entry = true;
                    return;
                  }
                  sync_ops::MarkWaveAtBarrier(candidate->wave,
                                              candidate->block->barrier_generation,
                                              candidate->block->barrier_arrivals,
                                              true);
                  PeuSlot& peu_slot = slots.at(candidate->peu_slot_index);
                  DeactivateResidentSlot(ResidentSlotForWave(peu_slot, *candidate));
                  context.trace.OnEvent(MakeTraceBarrierArriveEvent(
                      MakeTraceWaveView(*candidate, slot_id),
                      commit_cycle,
                      TraceSlotModelKind::ResidentFixed));
                  context.trace.OnEvent(MakeTraceWaveWaitEvent(
                      MakeTraceWaveView(*candidate, slot_id),
                      commit_cycle,
                      TraceSlotModelKind::ResidentFixed));
                  context.trace.OnEvent(MakeTraceWaveSwitchAwayEvent(
                      MakeTraceWaveView(*candidate, slot_id),
                      commit_cycle,
                      TraceSlotModelKind::ResidentFixed,
                      wave.pc));
                  context.trace.OnEvent(MakeTraceWaveSwitchStallEvent(
                      MakeTraceWaveView(*candidate, slot_id),
                      commit_cycle,
                      TraceSlotModelKind::ResidentFixed,
                      wave.pc));
                  RefillActiveWindow(peu_slot,
                                     commit_cycle,
                                     context.spec.max_issuable_waves,
                                     timing_config_.launch_timing.wave_launch_cycles,
                                     events,
                                     context.trace,
                                     true);
                  std::vector<ScheduledWave*> barrier_generation_waves;
                  barrier_generation_waves.reserve(candidate->block->waves.size());
                  for (auto& waiting_wave : candidate->block->waves) {
                    if (waiting_wave.wave.waiting_at_barrier &&
                        waiting_wave.wave.barrier_generation ==
                            candidate->block->barrier_generation) {
                      barrier_generation_waves.push_back(&waiting_wave);
                    }
                  }
                  std::vector<WaveContext*> waiting_waves;
                  waiting_waves.reserve(candidate->block->waves.size());
                  for (auto& waiting_wave : candidate->block->waves) {
                    waiting_waves.push_back(&waiting_wave.wave);
                  }
                  if (sync_ops::ReleaseBarrierIfReady(waiting_waves,
                                                      context.kernel,
                                                      candidate->block->barrier_generation,
                                                      candidate->block->barrier_arrivals,
                                                      true)) {
                    ReleaseBarrierSlot(ap_state.barrier_slots_in_use,
                                       candidate->block->barrier_slot_acquired);
                    context.trace.OnEvent(MakeTraceBarrierReleaseEvent(
                        candidate->block->dpc_id,
                        candidate->block->ap_id,
                        candidate->block->block_id,
                        commit_cycle));
                    std::vector<size_t> refill_slots;
                    refill_slots.reserve(barrier_generation_waves.size());
                    for (ScheduledWave* released_wave : barrier_generation_waves) {
                      if (released_wave == nullptr || released_wave->wave.waiting_at_barrier) {
                        continue;
                      }
                      released_wave->eligible_since_cycle = commit_cycle;
                      released_wave->eligible_since_valid = true;
                      context.trace.OnEvent(MakeTraceWaveResumeEvent(
                          MakeTraceWaveView(*released_wave, TraceSlotId(*released_wave)),
                          commit_cycle,
                          TraceSlotModelKind::ResidentFixed));
                      PeuSlot& released_slot = slots.at(released_wave->peu_slot_index);
                      QueueResidentWaveForRefill(released_slot, *released_wave);
                      if (std::find(refill_slots.begin(),
                                    refill_slots.end(),
                                    released_wave->peu_slot_index) == refill_slots.end()) {
                        refill_slots.push_back(released_wave->peu_slot_index);
                      }
                    }
                    for (size_t slot_index : refill_slots) {
                      RefillActiveWindow(slots.at(slot_index),
                                         commit_cycle,
                                         context.spec.max_issuable_waves,
                                         timing_config_.launch_timing.wave_launch_cycles,
                                         events,
                                         context.trace,
                                         false);
                    }
                  }
                  return;
                }

                if (plan.exit_wave) {
                  if (context.stats != nullptr) {
                    ++context.stats->wave_exits;
                  }
                  // Track wave completion in program_cycle_stats
                  if (program_cycle_stats_.has_value()) {
                    program_cycle_stats_->waves_completed += 1;
                  }
                  ApplyExecutionPlanControlFlow(context.kernel, plan, candidate->wave, false, false);
                  PeuSlot& peu_slot = slots.at(candidate->peu_slot_index);
                  if (peu_slot.last_wave_tag == WaveTag(candidate->wave)) {
                    peu_slot.last_wave_tag = std::numeric_limits<uint64_t>::max();
                    peu_slot.last_wave_trace.reset();
                    peu_slot.last_wave_pc = 0;
                  }
                  RemoveResidentWave(peu_slot, *candidate);
                  context.trace.OnEvent(MakeTraceWaveExitEvent(
                      MakeTraceWaveView(*candidate, slot_id),
                      commit_cycle,
                      TraceSlotModelKind::ResidentFixed));
                  if (candidate->block->active && !candidate->block->completed &&
                      AllWavesExited(*candidate->block)) {
                    candidate->block->active = false;
                    candidate->block->completed = true;
                    const uint32_t global_ap_id = candidate->block->global_ap_id;
                    auto ap_state_it = ap_states.find(global_ap_id);
                    if (ap_state_it != ap_states.end()) {
                      const bool removed =
                          RetireResidentBlock(ap_state_it->second, candidate->block);
                      if (removed) {
                        context.trace.OnEvent(MakeTraceBlockRetireEvent(
                            candidate->block->dpc_id,
                            candidate->block->ap_id,
                            candidate->block->block_id,
                            commit_cycle,
                            "ap=" + std::to_string(candidate->block->ap_id)));
                      }
                      if (removed && CanScheduleDelayedReadmit(ap_state_it->second)) {
                        ++ap_state_it->second.scheduled_readmit_count;
                      events.Schedule(TimedEvent{
                            .cycle = commit_cycle + timing_config_.launch_timing.block_launch_cycles,
                            .action =
                                [&, global_ap_id, commit_cycle]() {
                                  auto state_it = ap_states.find(global_ap_id);
                                  if (state_it == ap_states.end()) {
                                    return;
                                  }
                                  if (state_it->second.scheduled_readmit_count > 0) {
                                    --state_it->second.scheduled_readmit_count;
                                  }
                                  AdmitOneResidentBlock(
                                      state_it->second,
                                      commit_cycle + timing_config_.launch_timing.block_launch_cycles,
                                      slots,
                                      resident_wave_slots_per_peu,
                                      context.spec.max_issuable_waves,
                                      timing_config_.launch_timing.wave_generation_cycles,
                                      timing_config_.launch_timing.wave_dispatch_cycles,
                                      timing_config_.launch_timing.wave_launch_cycles,
                                      events,
                                      context.trace);
                                },
                        });
                      }
                    }
                  }
                  RefillActiveWindow(peu_slot,
                                     commit_cycle,
                                     context.spec.max_issuable_waves,
                                     timing_config_.launch_timing.wave_launch_cycles,
                                     events,
                                     context.trace,
                                     true);
                  return;
                }

                if (instruction.opcode == Opcode::SWaitCnt) {
                  const WaitCntThresholds thresholds = WaitCntThresholdsForInstruction(instruction);
                  if (EnterMemoryWaitState(thresholds, candidate->wave)) {
                    candidate->wave.valid_entry = true;
                    candidate->wave.status = WaveStatus::Active;
                    context.trace.OnEvent(MakeTraceWaveWaitEvent(
                        MakeTraceWaveView(*candidate, slot_id),
                        commit_cycle,
                        TraceSlotModelKind::ResidentFixed,
                        TraceStallReasonForWaitReason(candidate->wave.wait_reason),
                        std::numeric_limits<uint64_t>::max(),
                        MakeTraceWaitcntState(candidate->wave, thresholds)));
                    context.trace.OnEvent(MakeTraceWaitStallEvent(
                        MakeTraceWaveView(*candidate, slot_id),
                        commit_cycle,
                        TraceStallReasonForWaitReason(candidate->wave.wait_reason),
                        TraceSlotModelKind::ResidentFixed,
                        std::numeric_limits<uint64_t>::max(),
                        MakeTraceWaitcntState(candidate->wave, thresholds)));
                    context.trace.OnEvent(MakeTraceWaveSwitchAwayEvent(
                        MakeTraceWaveView(*candidate, slot_id),
                        commit_cycle,
                        TraceSlotModelKind::ResidentFixed,
                        candidate->wave.pc));
                    context.trace.OnEvent(MakeTraceWaveSwitchStallEvent(
                        MakeTraceWaveView(*candidate, slot_id),
                        commit_cycle,
                        TraceSlotModelKind::ResidentFixed,
                        candidate->wave.pc));
                    return;
                  }
                }

                candidate->wave.status = WaveStatus::Active;
                ApplyExecutionPlanControlFlow(context.kernel, plan, candidate->wave, true, true);
                },
        });
        issued_any = true;
      }
      slot.busy_until = bundle_commit_cycle;
      slot.last_wave_tag = bundle_last_wave_tag;
    }

    if (!issued_any && events.empty()) {
      throw std::runtime_error("cycle execution stalled without pending events");
    }

    ++cycle;
  }
}

}  // namespace gpu_model
