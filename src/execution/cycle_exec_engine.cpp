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

#include "gpu_model/debug/instruction_trace.h"
#include "gpu_model/debug/trace_event.h"
#include "gpu_model/debug/wave_launch_trace.h"
#include "gpu_model/execution/internal/barrier_resource_pool.h"
#include "gpu_model/execution/internal/event_queue.h"
#include "gpu_model/execution/internal/issue_model.h"
#include "gpu_model/execution/memory_ops.h"
#include "gpu_model/execution/plan_apply.h"
#include "gpu_model/execution/sync_ops.h"
#include "gpu_model/execution/wave_context_builder.h"
#include "gpu_model/execution/internal/issue_eligibility.h"
#include "gpu_model/execution/internal/scoreboard.h"
#include "gpu_model/isa/opcode.h"
#include "gpu_model/loader/device_image_loader.h"
#include "gpu_model/memory/cache_model.h"
#include "gpu_model/memory/shared_bank_model.h"

namespace gpu_model {

namespace {

struct ExecutableBlock;

struct ScheduledWave {
  uint32_t dpc_id = 0;
  ExecutableBlock* block = nullptr;
  WaveContext wave;
  Scoreboard scoreboard;
  uint64_t launch_cycle = 0;
  bool launch_completed = false;
  bool dispatch_enabled = false;
  bool launch_scheduled = false;
  size_t peu_slot_index = std::numeric_limits<size_t>::max();
  size_t resident_slot_id = std::numeric_limits<size_t>::max();
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
  uint64_t busy_until = 0;
  uint64_t last_wave_tag = std::numeric_limits<uint64_t>::max();
  ScheduledWave* resident_wave = nullptr;
  bool active = false;
};

struct PeuSlot {
  uint32_t dpc_id = 0;
  uint32_t ap_id = 0;
  uint32_t peu_id = 0;
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

ReadyRef ScalarRef(uint32_t index) {
  return ReadyRef{.kind = ReadyKind::ScalarReg, .index = index};
}

ReadyRef VectorRef(uint32_t index) {
  return ReadyRef{.kind = ReadyKind::VectorReg, .index = index};
}

ReadyRef ExecRef() {
  return ReadyRef{.kind = ReadyKind::Exec};
}

ReadyRef CmaskRef() {
  return ReadyRef{.kind = ReadyKind::Cmask};
}

ReadyRef SmaskRef() {
  return ReadyRef{.kind = ReadyKind::Smask};
}

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

void AddOperandDependency(const Operand& operand, std::vector<ReadyRef>& refs) {
  if (operand.kind != OperandKind::Register) {
    return;
  }
  refs.push_back(operand.reg.file == RegisterFile::Scalar ? ScalarRef(operand.reg.index)
                                                          : VectorRef(operand.reg.index));
}

std::vector<ReadyRef> CollectReadRefs(const Instruction& instruction) {
  std::vector<ReadyRef> refs;

  switch (instruction.opcode) {
    case Opcode::SysLoadArg:
    case Opcode::SysGlobalIdX:
    case Opcode::SysGlobalIdY:
    case Opcode::SysGlobalIdZ:
    case Opcode::SysLocalIdX:
    case Opcode::SysLocalIdY:
    case Opcode::SysLocalIdZ:
    case Opcode::SysBlockOffsetX:
    case Opcode::SysBlockIdxX:
    case Opcode::SysBlockIdxY:
    case Opcode::SysBlockIdxZ:
    case Opcode::SysBlockDimX:
    case Opcode::SysBlockDimY:
    case Opcode::SysBlockDimZ:
    case Opcode::SysGridDimX:
    case Opcode::SysGridDimY:
    case Opcode::SysGridDimZ:
    case Opcode::SysLaneId:
    case Opcode::BBranch:
    case Opcode::BExit:
      break;
    case Opcode::SMov:
      AddOperandDependency(instruction.operands.at(1), refs);
      break;
    case Opcode::SAdd:
    case Opcode::SSub:
    case Opcode::SMul:
    case Opcode::SDiv:
    case Opcode::SRem:
    case Opcode::SAnd:
    case Opcode::SOr:
    case Opcode::SXor:
    case Opcode::SShl:
    case Opcode::SShr:
      AddOperandDependency(instruction.operands.at(1), refs);
      AddOperandDependency(instruction.operands.at(2), refs);
      break;
    case Opcode::SWaitCnt:
      break;
    case Opcode::SCmpLt:
    case Opcode::SCmpEq:
    case Opcode::SCmpGt:
    case Opcode::SCmpGe:
      AddOperandDependency(instruction.operands.at(0), refs);
      AddOperandDependency(instruction.operands.at(1), refs);
      break;
    case Opcode::VMov:
      refs.push_back(ExecRef());
      AddOperandDependency(instruction.operands.at(1), refs);
      break;
    case Opcode::VAdd:
    case Opcode::VAnd:
    case Opcode::VOr:
    case Opcode::VXor:
    case Opcode::VShl:
    case Opcode::VShr:
    case Opcode::VSub:
    case Opcode::VDiv:
    case Opcode::VRem:
    case Opcode::VMul:
    case Opcode::VAddF32:
    case Opcode::VMin:
    case Opcode::VMax:
    case Opcode::VFma:
      refs.push_back(ExecRef());
      AddOperandDependency(instruction.operands.at(1), refs);
      AddOperandDependency(instruction.operands.at(2), refs);
      if (instruction.opcode == Opcode::VFma) {
        AddOperandDependency(instruction.operands.at(3), refs);
      }
      break;
    case Opcode::VCmpLtCmask:
    case Opcode::VCmpEqCmask:
    case Opcode::VCmpGeCmask:
    case Opcode::VCmpGtCmask:
      refs.push_back(ExecRef());
      AddOperandDependency(instruction.operands.at(0), refs);
      AddOperandDependency(instruction.operands.at(1), refs);
      break;
    case Opcode::VSelectCmask:
      refs.push_back(ExecRef());
      refs.push_back(CmaskRef());
      AddOperandDependency(instruction.operands.at(1), refs);
      AddOperandDependency(instruction.operands.at(2), refs);
      break;
    case Opcode::MLoadGlobal:
    case Opcode::MLoadGlobalAddr:
    case Opcode::MLoadShared:
    case Opcode::MLoadPrivate:
    case Opcode::MLoadConst:
      refs.push_back(ExecRef());
      AddOperandDependency(instruction.operands.at(1), refs);
      if (instruction.opcode == Opcode::MLoadGlobal) {
        AddOperandDependency(instruction.operands.at(2), refs);
      } else if (instruction.opcode == Opcode::MLoadGlobalAddr) {
        AddOperandDependency(instruction.operands.at(2), refs);
      }
      break;
    case Opcode::SBufferLoadDword:
      AddOperandDependency(instruction.operands.at(1), refs);
      break;
    case Opcode::MStoreGlobal:
    case Opcode::MStoreGlobalAddr:
    case Opcode::MAtomicAddGlobal:
    case Opcode::MStoreShared:
    case Opcode::MStorePrivate:
    case Opcode::MAtomicAddShared:
      refs.push_back(ExecRef());
      AddOperandDependency(instruction.operands.at(0), refs);
      AddOperandDependency(instruction.operands.at(1), refs);
      if (instruction.opcode == Opcode::MStoreGlobal || instruction.opcode == Opcode::MAtomicAddGlobal ||
          instruction.opcode == Opcode::MStoreGlobalAddr) {
        AddOperandDependency(instruction.operands.at(2), refs);
      }
      break;
    case Opcode::MaskSaveExec:
      refs.push_back(ExecRef());
      break;
    case Opcode::MaskRestoreExec:
      AddOperandDependency(instruction.operands.at(0), refs);
      break;
    case Opcode::MaskAndExecCmask:
      refs.push_back(ExecRef());
      refs.push_back(CmaskRef());
      break;
    case Opcode::BIfSmask:
      refs.push_back(SmaskRef());
      break;
    case Opcode::BIfNoexec:
      refs.push_back(ExecRef());
      break;
    case Opcode::SyncWaveBarrier:
    case Opcode::SyncBarrier:
      break;
  }

  return refs;
}

bool DependenciesReady(const Instruction& instruction,
                       const Scoreboard& scoreboard,
                       uint64_t cycle) {
  for (const auto& ref : CollectReadRefs(instruction)) {
    if (!scoreboard.IsReady(ref, cycle)) {
      return false;
    }
  }
  return true;
}

void MarkPlanWritesPending(const Instruction& instruction,
                           const OpPlan& plan,
                           Scoreboard& scoreboard,
                           uint64_t ready_cycle) {
  for (const auto& write : plan.scalar_writes) {
    scoreboard.MarkReady(ScalarRef(write.reg_index), ready_cycle);
  }
  for (const auto& write : plan.vector_writes) {
    scoreboard.MarkReady(VectorRef(write.reg_index), ready_cycle);
  }
  if (plan.cmask_write.has_value()) {
    scoreboard.MarkReady(CmaskRef(), ready_cycle);
  }
  if (plan.smask_write.has_value()) {
    scoreboard.MarkReady(SmaskRef(), ready_cycle);
  }
  if (plan.exec_write.has_value()) {
    scoreboard.MarkReady(ExecRef(), ready_cycle);
  }
  if (plan.memory.has_value() && plan.memory->kind == AccessKind::Load &&
      plan.memory->dst.has_value() && plan.memory->space == MemorySpace::Global) {
    scoreboard.MarkNotReady(VectorRef(plan.memory->dst->index));
  }

  (void)instruction;
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

constexpr std::string_view kStallReasonBarrierSlotUnavailable = "barrier_slot_unavailable";
constexpr std::string_view kStallReasonWarpSwitch = "warp_switch";

std::string FormatCycleStallMessage(std::string_view reason) {
  constexpr std::string_view kPrefix = "reason=";
  if (reason.starts_with(kPrefix)) {
    return std::string(reason);
  }
  return std::string(kPrefix) + std::string(reason);
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
    trace.OnEvent(TraceEvent{
        .kind = TraceEventKind::WaveLaunch,
        .cycle = wave.launch_cycle,
        .dpc_id = wave.wave.dpc_id,
        .ap_id = wave.wave.ap_id,
        .peu_id = wave.wave.peu_id,
        .slot_id = TraceSlotId(wave),
        .block_id = wave.wave.block_id,
        .wave_id = wave.wave.wave_id,
        .pc = wave.wave.pc,
        .message = "wave_start " + FormatWaveLaunchTraceMessage(wave.wave),
    });
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
                        bool immediate_launch = false) {
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
                   uint64_t wave_launch_cycles,
                   EventQueue& events,
                   TraceSink& trace) {
  block.active = true;
  trace.OnEvent(TraceEvent{
      .kind = TraceEventKind::BlockLaunch,
      .cycle = cycle,
      .dpc_id = block.dpc_id,
      .ap_id = block.ap_id,
      .block_id = block.block_id,
      .message = "ap=" + std::to_string(block.ap_id),
  });
  std::vector<size_t> touched_slots;
  for (auto& scheduled_wave : block.waves) {
    scheduled_wave.wave.status = WaveStatus::Stalled;
    scheduled_wave.dispatch_enabled = false;
    scheduled_wave.launch_scheduled = false;
    PeuSlot& slot = slots.at(scheduled_wave.peu_slot_index);
    RegisterResidentWave(slot, scheduled_wave);
    if (std::find(touched_slots.begin(), touched_slots.end(), scheduled_wave.peu_slot_index) ==
        touched_slots.end()) {
      touched_slots.push_back(scheduled_wave.peu_slot_index);
    }
  }
  for (size_t slot_index : touched_slots) {
    RefillActiveWindow(slots.at(slot_index),
                       cycle,
                       max_issuable_waves,
                       wave_launch_cycles,
                       events,
                       trace,
                       false);
  }
}

bool AdmitOneResidentBlock(ApResidentState& ap_state,
                           uint64_t cycle,
                           std::vector<PeuSlot>& slots,
                           uint32_t resident_wave_slots_per_peu,
                           uint32_t max_issuable_waves,
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
                         uint64_t wave_launch_cycles,
                         EventQueue& events,
                         TraceSink& trace) {
  while (AdmitOneResidentBlock(ap_state,
                               cycle,
                               slots,
                               resident_wave_slots_per_peu,
                               max_issuable_waves,
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
  if (scheduled_wave->wave.pc >= kernel.instructions().size()) {
    return std::nullopt;
  }
  const auto& instruction = kernel.instructions().at(scheduled_wave->wave.pc);
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
                           instruction,
                           DependenciesReady(instruction, scheduled_wave->scoreboard, cycle))) {
    return std::make_pair(scheduled_wave, *reason);
  }
  return std::nullopt;
}

}  // namespace

uint64_t CycleExecEngine::Run(ExecutionContext& context) {
  auto blocks = MaterializeBlocks(context.placement, context.launch_config);
  const uint32_t resident_wave_slots_per_peu = ResidentWaveSlotCapacityPerPeu(context.spec);
  auto slots = BuildPeuSlots(blocks, resident_wave_slots_per_peu);
  EventQueue events;
  std::map<L1Key, CacheModel> l1_caches;
  CacheModel l2_cache(timing_config_.cache_model);
  SharedBankModel shared_bank_model(timing_config_.shared_bank_model);
  std::map<uint32_t, ApResidentState> ap_states;

  for (auto& slot : slots) {
    for (auto& resident_slot : slot.resident_slots) {
      resident_slot.busy_until = 0;
    }
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
      return cycle;
    }

    bool issued_any = false;
    for (auto& slot : slots) {
      for (auto& resident_slot : slot.resident_slots) {
        if (!resident_slot.active || resident_slot.resident_wave == nullptr ||
            resident_slot.busy_until > cycle || !ResidentSlotReadyToIssue(resident_slot, cycle)) {
          continue;
        }

        ScheduledWave* candidate = resident_slot.resident_wave;
        if (candidate->wave.pc >= context.kernel.instructions().size()) {
          throw std::out_of_range("wave pc out of range");
        }

        if (const auto blocked =
                BlockedResidentWave(resident_slot, context.kernel, cycle, ap_states)) {
          context.trace.OnEvent(TraceEvent{
              .kind = TraceEventKind::Stall,
              .cycle = cycle,
              .dpc_id = blocked->first->wave.dpc_id,
              .ap_id = blocked->first->wave.ap_id,
              .peu_id = blocked->first->wave.peu_id,
              .slot_id = TraceSlotId(*blocked->first),
              .block_id = blocked->first->wave.block_id,
              .wave_id = blocked->first->wave.wave_id,
              .pc = blocked->first->wave.pc,
              .message = FormatCycleStallMessage(blocked->second),
          });
          continue;
        }

        WaveContext& wave = candidate->wave;
        const Instruction instruction = context.kernel.instructions().at(wave.pc);
        const uint32_t slot_id = TraceSlotId(*candidate);
        if (context.stats != nullptr) {
          ++context.stats->wave_steps;
          ++context.stats->instructions_issued;
        }
        context.trace.OnEvent(TraceEvent{
            .kind = TraceEventKind::WaveStep,
            .cycle = cycle,
            .dpc_id = wave.dpc_id,
            .ap_id = wave.ap_id,
            .peu_id = wave.peu_id,
            .slot_id = slot_id,
            .block_id = wave.block_id,
            .wave_id = wave.wave_id,
            .pc = wave.pc,
            .message = FormatWaveStepMessage(instruction, wave),
        });

        const OpPlan plan = semantics_.BuildPlan(instruction, wave, context);
        const uint64_t wave_tag = WaveTag(wave);
        const uint64_t switch_penalty =
            resident_slot.last_wave_tag != std::numeric_limits<uint64_t>::max() &&
                    resident_slot.last_wave_tag != wave_tag
                ? timing_config_.launch_timing.warp_switch_cycles
                : 0;
        if (switch_penalty > 0) {
          context.trace.OnEvent(TraceEvent{
              .kind = TraceEventKind::Stall,
              .cycle = cycle,
              .dpc_id = wave.dpc_id,
              .ap_id = wave.ap_id,
              .peu_id = wave.peu_id,
              .slot_id = slot_id,
              .block_id = wave.block_id,
              .wave_id = wave.wave_id,
              .pc = wave.pc,
              .message = FormatCycleStallMessage(kStallReasonWarpSwitch),
          });
        }
        const uint64_t commit_cycle = cycle + switch_penalty + plan.issue_cycles;
        resident_slot.busy_until = commit_cycle;
        resident_slot.last_wave_tag = wave_tag;
        wave.status = WaveStatus::Stalled;
        wave.valid_entry = false;
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
          context.trace.OnEvent(TraceEvent{
              .kind = TraceEventKind::MemoryAccess,
              .cycle = cycle,
              .dpc_id = wave.dpc_id,
              .ap_id = wave.ap_id,
              .peu_id = wave.peu_id,
              .slot_id = slot_id,
              .block_id = wave.block_id,
              .wave_id = wave.wave_id,
              .pc = wave.pc,
              .message = plan.memory->kind == AccessKind::Load ? "load_issue" : "store_issue",
          });
        }

        MarkPlanWritesPending(instruction, plan, candidate->scoreboard, commit_cycle);

        events.Schedule(TimedEvent{
            .cycle = commit_cycle,
            .action =
                [&, candidate, instruction, plan, commit_cycle, slot_id]() {
                context.cycle = commit_cycle;

                ApplyExecutionPlanRegisterWrites(plan, candidate->wave);
                if (const auto mask_text = MaybeFormatExecutionMaskUpdate(plan, candidate->wave);
                    mask_text.has_value()) {
                  context.trace.OnEvent(TraceEvent{
                      .kind = TraceEventKind::ExecMaskUpdate,
                      .cycle = commit_cycle,
                      .dpc_id = candidate->wave.dpc_id,
                      .ap_id = candidate->wave.ap_id,
                      .peu_id = candidate->wave.peu_id,
                      .slot_id = slot_id,
                      .block_id = candidate->wave.block_id,
                      .wave_id = candidate->wave.wave_id,
                      .pc = candidate->wave.pc,
                      .message = *mask_text,
                  });
                }

                context.trace.OnEvent(TraceEvent{
                    .kind = TraceEventKind::Commit,
                    .cycle = commit_cycle,
                    .dpc_id = candidate->wave.dpc_id,
                    .ap_id = candidate->wave.ap_id,
                    .peu_id = candidate->wave.peu_id,
                    .slot_id = slot_id,
                    .block_id = candidate->wave.block_id,
                    .wave_id = candidate->wave.wave_id,
                    .pc = candidate->wave.pc,
                    .message = std::string(ToString(instruction.opcode)),
                });

                if (plan.memory.has_value()) {
                  const MemoryRequest request = *plan.memory;
                  if (request.space == MemorySpace::Global) {
                    auto& l1_cache =
                        l1_caches.at(L1Key{.dpc_id = candidate->block->dpc_id, .ap_id = candidate->block->ap_id});
                    const std::vector<uint64_t> addrs = ActiveAddresses(request);
                    const CacheProbeResult l1_probe = l1_cache.Probe(addrs);
                    const CacheProbeResult l2_probe = l2_cache.Probe(addrs);
                    const uint64_t arrive_latency = std::min(l1_probe.latency, l2_probe.latency);
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
                    events.Schedule(TimedEvent{
                        .cycle = arrive_cycle,
                        .action =
                            [&, candidate, request, addrs, arrive_cycle, slot_id]() {
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
                                candidate->scoreboard.MarkReady(
                                    VectorRef(request.dst->index), arrive_cycle);
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

                              context.trace.OnEvent(TraceEvent{
                                  .kind = TraceEventKind::Arrive,
                                  .cycle = arrive_cycle,
                                  .dpc_id = candidate->wave.dpc_id,
                                  .ap_id = candidate->wave.ap_id,
                                  .peu_id = candidate->wave.peu_id,
                                  .slot_id = slot_id,
                                  .block_id = candidate->wave.block_id,
                                  .wave_id = candidate->wave.wave_id,
                                  .pc = candidate->wave.pc,
                                  .message = request.kind == AccessKind::Load ? "load_arrive"
                                                                              : "store_arrive",
                              });
                              DecrementPendingMemoryOps(candidate->wave, MemoryWaitDomain::Global);
                              candidate->wave.valid_entry = true;
                              if (candidate->wave.status != WaveStatus::Exited &&
                                  !candidate->wave.waiting_at_barrier) {
                                candidate->wave.status = WaveStatus::Active;
                              }
                            },
                    });
                  } else if (request.space == MemorySpace::Shared) {
                    const uint64_t penalty = shared_bank_model.ConflictPenalty(request);
                    if (context.stats != nullptr) {
                      context.stats->shared_bank_conflict_penalty_cycles += penalty;
                    }
                    const uint64_t ready_cycle = commit_cycle + penalty;
                    const bool advance_pc = plan.advance_pc;
                    const std::optional<uint64_t> branch_target = plan.branch_target;
                    events.Schedule(TimedEvent{
                        .cycle = ready_cycle,
                        .action =
                            [&, candidate, request, ready_cycle, advance_pc, branch_target]() {
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
                                candidate->scoreboard.MarkReady(
                                    VectorRef(request.dst->index), ready_cycle);
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

                              if (branch_target.has_value()) {
                                candidate->wave.pc = *branch_target;
                              } else if (advance_pc) {
                                ++candidate->wave.pc;
                              }
                              DecrementPendingMemoryOps(candidate->wave, MemoryWaitDomain::Shared);
                              candidate->wave.valid_entry = true;
                              candidate->wave.status = WaveStatus::Active;
                            },
                    });
                    if (request.kind == AccessKind::Load && request.dst.has_value()) {
                      candidate->scoreboard.MarkNotReady(VectorRef(request.dst->index));
                    }
                    return;
                  } else if (request.space == MemorySpace::Private) {
                    if (request.kind == AccessKind::Load) {
                      if (!request.dst.has_value()) {
                        throw std::invalid_argument("load request missing destination");
                      }
                    for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
                      if (request.lanes[lane].active) {
                        const uint64_t value =
                            LoadLaneValue(candidate->wave.private_memory, lane, request.lanes[lane]);
                          candidate->wave.vgpr.Write(request.dst->index, lane, value);
                        }
                      }
                      candidate->scoreboard.MarkReady(
                          VectorRef(request.dst->index), commit_cycle);
                    } else if (request.kind == AccessKind::Store) {
                      for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
                        if (request.lanes[lane].active) {
                          StoreLaneValue(candidate->wave.private_memory, lane, request.lanes[lane]);
                        }
                      }
                    }
                    DecrementPendingMemoryOps(candidate->wave, MemoryWaitDomain::Private);
                    candidate->wave.valid_entry = true;
                  } else if (request.space == MemorySpace::Constant) {
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
                        if (context.memory.HasRange(MemoryPoolKind::Constant, pool_lane.addr,
                                                    request.lanes[lane].bytes)) {
                          loaded_value =
                              LoadLaneValue(context.memory, MemoryPoolKind::Constant, pool_lane);
                        } else {
                          loaded_value =
                              LoadLaneValue(context.kernel.const_segment().bytes, request.lanes[lane]);
                        }
                        break;
                      }
                      candidate->wave.sgpr.Write(request.dst->index, loaded_value);
                      candidate->scoreboard.MarkReady(
                          ScalarRef(request.dst->index), commit_cycle);
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
                                  ? LoadLaneValue(context.memory, MemoryPoolKind::Constant, pool_lane)
                                  : LoadLaneValue(context.kernel.const_segment().bytes,
                                                  request.lanes[lane]);
                          candidate->wave.vgpr.Write(request.dst->index, lane, value);
                        }
                      }
                      candidate->scoreboard.MarkReady(
                          VectorRef(request.dst->index), commit_cycle);
                    }
                    DecrementPendingMemoryOps(candidate->wave, MemoryWaitDomain::ScalarBuffer);
                    candidate->wave.valid_entry = true;
                  } else {
                    throw std::invalid_argument("unsupported memory space in cycle executor");
                  }
                }

                if (plan.sync_wave_barrier) {
                  if (context.stats != nullptr) {
                    ++context.stats->barriers;
                  }
                  context.trace.OnEvent(TraceEvent{
                      .kind = TraceEventKind::Barrier,
                      .cycle = commit_cycle,
                      .dpc_id = candidate->wave.dpc_id,
                      .ap_id = candidate->wave.ap_id,
                      .peu_id = candidate->wave.peu_id,
                      .slot_id = slot_id,
                      .block_id = candidate->wave.block_id,
                      .wave_id = candidate->wave.wave_id,
                      .pc = candidate->wave.pc,
                      .message = "wave",
                  });
                  if (plan.branch_target.has_value()) {
                    candidate->wave.pc = *plan.branch_target;
                  } else if (plan.advance_pc) {
                    ++candidate->wave.pc;
                  }
                  candidate->wave.valid_entry = true;
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
                  context.trace.OnEvent(TraceEvent{
                      .kind = TraceEventKind::Barrier,
                      .cycle = commit_cycle,
                      .dpc_id = candidate->wave.dpc_id,
                      .ap_id = candidate->wave.ap_id,
                      .peu_id = candidate->wave.peu_id,
                      .slot_id = slot_id,
                      .block_id = candidate->wave.block_id,
                      .wave_id = candidate->wave.wave_id,
                      .pc = candidate->wave.pc,
                      .message = "arrive",
                  });
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
                                                      candidate->block->barrier_generation,
                                                      candidate->block->barrier_arrivals,
                                                      1,
                                                      true)) {
                    ReleaseBarrierSlot(ap_state.barrier_slots_in_use,
                                       candidate->block->barrier_slot_acquired);
                    context.trace.OnEvent(TraceEvent{
                        .kind = TraceEventKind::Barrier,
                        .cycle = commit_cycle,
                        .dpc_id = candidate->wave.dpc_id,
                        .ap_id = candidate->wave.ap_id,
                        .peu_id = candidate->wave.peu_id,
                        .slot_id = slot_id,
                        .block_id = candidate->wave.block_id,
                        .wave_id = candidate->wave.wave_id,
                        .pc = candidate->wave.pc,
                        .message = "release",
                    });
                    std::vector<size_t> refill_slots;
                    refill_slots.reserve(barrier_generation_waves.size());
                    for (ScheduledWave* released_wave : barrier_generation_waves) {
                      if (released_wave == nullptr || released_wave->wave.waiting_at_barrier) {
                        continue;
                      }
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
                  ApplyExecutionPlanControlFlow(plan, candidate->wave, false, false);
                  PeuSlot& peu_slot = slots.at(candidate->peu_slot_index);
                  RemoveResidentWave(peu_slot, *candidate);
                  context.trace.OnEvent(TraceEvent{
                      .kind = TraceEventKind::WaveExit,
                      .cycle = commit_cycle,
                      .dpc_id = candidate->wave.dpc_id,
                      .ap_id = candidate->wave.ap_id,
                      .peu_id = candidate->wave.peu_id,
                      .slot_id = slot_id,
                      .block_id = candidate->wave.block_id,
                      .wave_id = candidate->wave.wave_id,
                      .pc = candidate->wave.pc,
                      .message = "wave_end",
                  });
                  if (candidate->block->active && !candidate->block->completed &&
                      AllWavesExited(*candidate->block)) {
                    candidate->block->active = false;
                    candidate->block->completed = true;
                    const uint32_t global_ap_id = candidate->block->global_ap_id;
                    auto ap_state_it = ap_states.find(global_ap_id);
                    if (ap_state_it != ap_states.end()) {
                      const bool removed =
                          RetireResidentBlock(ap_state_it->second, candidate->block);
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

                candidate->wave.status = WaveStatus::Active;
                ApplyExecutionPlanControlFlow(plan, candidate->wave, true, true);
                },
        });
        issued_any = true;
      }
    }

    if (!issued_any && events.empty()) {
      throw std::runtime_error("cycle execution stalled without pending events");
    }

    ++cycle;
  }
}

}  // namespace gpu_model
