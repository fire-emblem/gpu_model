#include "gpu_model/exec/cycle_executor.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "gpu_model/debug/instruction_trace.h"
#include "gpu_model/debug/trace_event.h"
#include "gpu_model/exec/event_queue.h"
#include "gpu_model/exec/scoreboard.h"
#include "gpu_model/isa/opcode.h"
#include "gpu_model/memory/cache_model.h"
#include "gpu_model/memory/shared_bank_model.h"

namespace gpu_model {

namespace {

struct ScheduledWave {
  uint32_t dpc_id = 0;
  struct ExecutableBlock* block = nullptr;
  WaveState wave;
  Scoreboard scoreboard;
  uint64_t launch_cycle = 0;
  bool dispatch_enabled = false;
  bool launch_scheduled = false;
};

struct ExecutableBlock {
  uint32_t block_id = 0;
  uint32_t dpc_id = 0;
  uint32_t ap_id = 0;
  uint32_t global_ap_id = 0;
  uint32_t ap_queue_index = 0;
  uint64_t barrier_generation = 0;
  uint32_t barrier_arrivals = 0;
  bool active = false;
  bool completed = false;
  std::vector<std::byte> shared_memory;
  std::vector<ScheduledWave> waves;
};

struct PeuSlot {
  uint32_t dpc_id = 0;
  uint32_t ap_id = 0;
  uint32_t peu_id = 0;
  uint64_t busy_until = 0;
  uint64_t last_wave_tag = std::numeric_limits<uint64_t>::max();
  size_t last_issue_index = std::numeric_limits<size_t>::max();
  std::vector<ScheduledWave*> waves;
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

uint64_t LoadLaneValue(const MemorySystem& memory, const LaneAccess& lane) {
  switch (lane.bytes) {
    case 4: {
      const int32_t value = memory.LoadGlobalValue<int32_t>(lane.addr);
      return static_cast<uint64_t>(static_cast<int64_t>(value));
    }
    case 8:
      return memory.LoadGlobalValue<uint64_t>(lane.addr);
    default:
      throw std::invalid_argument("unsupported load width");
  }
}

void StoreLaneValue(MemorySystem& memory, const LaneAccess& lane) {
  switch (lane.bytes) {
    case 4:
      memory.StoreGlobalValue<int32_t>(lane.addr, static_cast<int32_t>(lane.value));
      return;
    case 8:
      memory.StoreGlobalValue<uint64_t>(lane.addr, lane.value);
      return;
    default:
      throw std::invalid_argument("unsupported store width");
  }
}

uint64_t LoadLaneValue(const std::vector<std::byte>& memory, const LaneAccess& lane) {
  const size_t end = static_cast<size_t>(lane.addr) + lane.bytes;
  if (end > memory.size()) {
    throw std::out_of_range("byte-addressable memory load out of range");
  }
  switch (lane.bytes) {
    case 4: {
      int32_t value = 0;
      std::memcpy(&value, memory.data() + lane.addr, sizeof(value));
      return static_cast<uint64_t>(static_cast<int64_t>(value));
    }
    case 8: {
      uint64_t value = 0;
      std::memcpy(&value, memory.data() + lane.addr, sizeof(value));
      return value;
    }
    default:
      throw std::invalid_argument("unsupported load width");
  }
}

void StoreLaneValue(std::vector<std::byte>& memory, const LaneAccess& lane) {
  switch (lane.bytes) {
    case 4: {
      const int32_t value = static_cast<int32_t>(lane.value);
      std::memcpy(memory.data() + lane.addr, &value, sizeof(value));
      return;
    }
    case 8:
      std::memcpy(memory.data() + lane.addr, &lane.value, sizeof(lane.value));
      return;
    default:
      throw std::invalid_argument("unsupported store width");
  }
}

uint64_t LoadLaneValue(const std::array<std::vector<std::byte>, kWaveSize>& memory,
                       uint32_t lane_id,
                       const LaneAccess& lane) {
  auto& lane_memory = memory.at(lane_id);
  const size_t end = static_cast<size_t>(lane.addr) + lane.bytes;
  if (lane_memory.size() < end) {
    return 0;
  }
  return LoadLaneValue(lane_memory, lane);
}

void StoreLaneValue(std::array<std::vector<std::byte>, kWaveSize>& memory,
                    uint32_t lane_id,
                    const LaneAccess& lane) {
  auto& lane_memory = memory.at(lane_id);
  const size_t end = static_cast<size_t>(lane.addr) + lane.bytes;
  if (lane_memory.size() < end) {
    lane_memory.resize(end, std::byte{0});
  }
  StoreLaneValue(lane_memory, lane);
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
    case Opcode::SysLocalIdX:
    case Opcode::SysLocalIdY:
    case Opcode::SysBlockOffsetX:
    case Opcode::SysBlockIdxX:
    case Opcode::SysBlockIdxY:
    case Opcode::SysBlockDimX:
    case Opcode::SysBlockDimY:
    case Opcode::SysGridDimX:
    case Opcode::SysGridDimY:
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
    case Opcode::MLoadShared:
    case Opcode::MLoadPrivate:
    case Opcode::MLoadConst:
      refs.push_back(ExecRef());
      AddOperandDependency(instruction.operands.at(1), refs);
      if (instruction.opcode == Opcode::MLoadGlobal) {
        AddOperandDependency(instruction.operands.at(2), refs);
      }
      break;
    case Opcode::SBufferLoadDword:
      AddOperandDependency(instruction.operands.at(1), refs);
      break;
    case Opcode::MStoreGlobal:
    case Opcode::MAtomicAddGlobal:
    case Opcode::MStoreShared:
    case Opcode::MStorePrivate:
    case Opcode::MAtomicAddShared:
      refs.push_back(ExecRef());
      AddOperandDependency(instruction.operands.at(0), refs);
      AddOperandDependency(instruction.operands.at(1), refs);
      if (instruction.opcode == Opcode::MStoreGlobal || instruction.opcode == Opcode::MAtomicAddGlobal) {
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
  std::vector<ExecutableBlock> blocks;
  blocks.reserve(placement.blocks.size());

  for (const auto& block_placement : placement.blocks) {
    ExecutableBlock block{
        .block_id = block_placement.block_id,
        .dpc_id = block_placement.dpc_id,
        .ap_id = block_placement.ap_id,
        .global_ap_id = block_placement.global_ap_id,
        .barrier_generation = 0,
        .barrier_arrivals = 0,
        .shared_memory = std::vector<std::byte>(launch_config.shared_memory_bytes),
        .waves = {},
    };
    block.waves.reserve(block_placement.waves.size());
    for (const auto& wave_placement : block_placement.waves) {
      ScheduledWave scheduled;
      scheduled.dpc_id = block_placement.dpc_id;
      scheduled.wave.block_id = block_placement.block_id;
      scheduled.wave.block_idx_x = block_placement.block_idx_x;
      scheduled.wave.block_idx_y = block_placement.block_idx_y;
      scheduled.wave.dpc_id = block_placement.dpc_id;
      scheduled.wave.wave_id = wave_placement.wave_id;
      scheduled.wave.peu_id = wave_placement.peu_id;
      scheduled.wave.ap_id = block_placement.ap_id;
      scheduled.wave.thread_count = wave_placement.lane_count;
      scheduled.wave.ResetInitialExec();
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

std::vector<PeuSlot> BuildPeuSlots(std::vector<ExecutableBlock>& blocks) {
  std::vector<PeuSlot> slots;
  std::map<std::tuple<uint32_t, uint32_t, uint32_t>, size_t> slot_indices;

  for (auto& block : blocks) {
    for (auto& scheduled_wave : block.waves) {
      const auto key = std::make_tuple(block.dpc_id, block.ap_id, scheduled_wave.wave.peu_id);
      auto [it, inserted] = slot_indices.emplace(key, slots.size());
      if (inserted) {
        slots.push_back(PeuSlot{
            .dpc_id = block.dpc_id,
            .ap_id = block.ap_id,
            .peu_id = scheduled_wave.wave.peu_id,
            .waves = {},
        });
        it->second = slots.size() - 1;
      }
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

uint64_t WaveTag(const WaveState& wave) {
  return (static_cast<uint64_t>(wave.block_id) << 32) | wave.wave_id;
}

enum class MemoryWaitDomain {
  None,
  Global,
  Shared,
  Private,
  ScalarBuffer,
};

MemoryWaitDomain MemoryDomainForOpcode(Opcode opcode) {
  switch (opcode) {
    case Opcode::MLoadGlobal:
    case Opcode::MStoreGlobal:
    case Opcode::MAtomicAddGlobal:
      return MemoryWaitDomain::Global;
    case Opcode::MLoadShared:
    case Opcode::MStoreShared:
    case Opcode::MAtomicAddShared:
      return MemoryWaitDomain::Shared;
    case Opcode::MLoadPrivate:
    case Opcode::MStorePrivate:
      return MemoryWaitDomain::Private;
    case Opcode::MLoadConst:
      return MemoryWaitDomain::ScalarBuffer;
    default:
      return MemoryWaitDomain::None;
  }
}

struct WaitCntThresholds {
  uint32_t global = UINT32_MAX;
  uint32_t shared = UINT32_MAX;
  uint32_t private_mem = UINT32_MAX;
  uint32_t scalar_buffer = UINT32_MAX;
};

uint32_t PendingMemoryOpsForDomain(const WaveState& wave, MemoryWaitDomain domain) {
  switch (domain) {
    case MemoryWaitDomain::Global:
      return wave.pending_global_mem_ops;
    case MemoryWaitDomain::Shared:
      return wave.pending_shared_mem_ops;
    case MemoryWaitDomain::Private:
      return wave.pending_private_mem_ops;
    case MemoryWaitDomain::ScalarBuffer:
      return wave.pending_scalar_buffer_mem_ops;
    case MemoryWaitDomain::None:
      return 0;
  }
  return 0;
}

void IncrementPendingMemoryOps(WaveState& wave, MemoryWaitDomain domain) {
  switch (domain) {
    case MemoryWaitDomain::Global:
      ++wave.pending_global_mem_ops;
      return;
    case MemoryWaitDomain::Shared:
      ++wave.pending_shared_mem_ops;
      return;
    case MemoryWaitDomain::Private:
      ++wave.pending_private_mem_ops;
      return;
    case MemoryWaitDomain::ScalarBuffer:
      ++wave.pending_scalar_buffer_mem_ops;
      return;
    case MemoryWaitDomain::None:
      return;
  }
}

void DecrementPendingMemoryOps(WaveState& wave, MemoryWaitDomain domain) {
  switch (domain) {
    case MemoryWaitDomain::Global:
      if (wave.pending_global_mem_ops > 0) {
        --wave.pending_global_mem_ops;
      }
      return;
    case MemoryWaitDomain::Shared:
      if (wave.pending_shared_mem_ops > 0) {
        --wave.pending_shared_mem_ops;
      }
      return;
    case MemoryWaitDomain::Private:
      if (wave.pending_private_mem_ops > 0) {
        --wave.pending_private_mem_ops;
      }
      return;
    case MemoryWaitDomain::ScalarBuffer:
      if (wave.pending_scalar_buffer_mem_ops > 0) {
        --wave.pending_scalar_buffer_mem_ops;
      }
      return;
    case MemoryWaitDomain::None:
      return;
  }
}

WaitCntThresholds WaitCntThresholdsForInstruction(const Instruction& instruction) {
  WaitCntThresholds thresholds;
  if (instruction.opcode != Opcode::SWaitCnt) {
    return thresholds;
  }
  thresholds.global = static_cast<uint32_t>(instruction.operands.at(0).immediate);
  thresholds.shared = static_cast<uint32_t>(instruction.operands.at(1).immediate);
  thresholds.private_mem = static_cast<uint32_t>(instruction.operands.at(2).immediate);
  thresholds.scalar_buffer = static_cast<uint32_t>(instruction.operands.at(3).immediate);
  return thresholds;
}

bool WaitCntSatisfied(const WaveState& wave, const Instruction& instruction) {
  if (instruction.opcode != Opcode::SWaitCnt) {
    return true;
  }
  const auto thresholds = WaitCntThresholdsForInstruction(instruction);
  return wave.pending_global_mem_ops <= thresholds.global &&
         wave.pending_shared_mem_ops <= thresholds.shared &&
         wave.pending_private_mem_ops <= thresholds.private_mem &&
         wave.pending_scalar_buffer_mem_ops <= thresholds.scalar_buffer;
}

std::optional<std::string> WaitCntBlockReason(const WaveState& wave,
                                              const Instruction& instruction) {
  if (instruction.opcode != Opcode::SWaitCnt) {
    return std::nullopt;
  }
  const auto thresholds = WaitCntThresholdsForInstruction(instruction);
  if (wave.pending_global_mem_ops > thresholds.global) {
    return "waitcnt_global";
  }
  if (wave.pending_shared_mem_ops > thresholds.shared) {
    return "waitcnt_shared";
  }
  if (wave.pending_private_mem_ops > thresholds.private_mem) {
    return "waitcnt_private";
  }
  if (wave.pending_scalar_buffer_mem_ops > thresholds.scalar_buffer) {
    return "waitcnt_scalar_buffer";
  }
  return std::nullopt;
}

std::optional<std::string> MemoryDomainBlockReason(const WaveState& wave,
                                                   const Instruction& instruction) {
  switch (MemoryDomainForOpcode(instruction.opcode)) {
    case MemoryWaitDomain::Global:
      if (wave.pending_global_mem_ops > 0) {
        return "waitcnt_global";
      }
      break;
    case MemoryWaitDomain::Shared:
      if (wave.pending_shared_mem_ops > 0) {
        return "waitcnt_shared";
      }
      break;
    case MemoryWaitDomain::Private:
      if (wave.pending_private_mem_ops > 0) {
        return "waitcnt_private";
      }
      break;
    case MemoryWaitDomain::ScalarBuffer:
      if (wave.pending_scalar_buffer_mem_ops > 0) {
        return "waitcnt_scalar_buffer";
      }
      break;
    case MemoryWaitDomain::None:
      break;
  }
  return std::nullopt;
}

bool CanIssueInstruction(const ScheduledWave& scheduled_wave, const Instruction& instruction) {
  const auto& wave = scheduled_wave.wave;
  const MemoryWaitDomain memory_domain = MemoryDomainForOpcode(instruction.opcode);
  return scheduled_wave.dispatch_enabled && wave.status == WaveStatus::Active && wave.valid_entry &&
         (memory_domain == MemoryWaitDomain::None ||
          PendingMemoryOpsForDomain(wave, memory_domain) == 0) &&
         WaitCntSatisfied(wave, instruction) &&
         !wave.branch_pending &&
         !wave.waiting_at_barrier;
}

std::optional<std::string> IssueBlockReason(const ScheduledWave& scheduled_wave,
                                            const Instruction& instruction,
                                            uint64_t cycle) {
  const auto& wave = scheduled_wave.wave;
  if (!scheduled_wave.dispatch_enabled || wave.status != WaveStatus::Active) {
    return std::nullopt;
  }
  if (!wave.valid_entry) {
    return std::string("front_end_wait");
  }
  if (wave.waiting_at_barrier) {
    return std::string("barrier_wait");
  }
  if (wave.branch_pending) {
    return std::string("branch_wait");
  }
  if (const auto reason = WaitCntBlockReason(wave, instruction)) {
    return reason;
  }
  if (const auto reason = MemoryDomainBlockReason(wave, instruction)) {
    return reason;
  }
  if (!DependenciesReady(instruction, scheduled_wave.scoreboard, cycle)) {
    return std::string("dependency_wait");
  }
  return std::nullopt;
}

void ScheduleWaveLaunch(ScheduledWave& scheduled_wave,
                        uint64_t cycle,
                        EventQueue& events,
                        TraceSink& trace) {
  scheduled_wave.dispatch_enabled = true;
  scheduled_wave.launch_cycle = cycle;
  if (scheduled_wave.launch_scheduled || scheduled_wave.wave.status == WaveStatus::Active ||
      scheduled_wave.wave.status == WaveStatus::Exited) {
    return;
  }
  scheduled_wave.launch_scheduled = true;
  ScheduledWave* wave_ptr = &scheduled_wave;
  events.Schedule(TimedEvent{
      .cycle = cycle,
      .action =
          [wave_ptr, &trace, block_id = scheduled_wave.wave.block_id, peu_id = scheduled_wave.wave.peu_id]() {
            wave_ptr->launch_scheduled = false;
            if (wave_ptr->wave.status == WaveStatus::Exited) {
              return;
            }
            wave_ptr->wave.status = WaveStatus::Active;
            wave_ptr->wave.valid_entry = true;
            trace.OnEvent(TraceEvent{
                .kind = TraceEventKind::WaveLaunch,
                .cycle = wave_ptr->launch_cycle,
                .dpc_id = wave_ptr->wave.dpc_id,
                .ap_id = wave_ptr->wave.ap_id,
                .peu_id = wave_ptr->wave.peu_id,
                .block_id = block_id,
                .wave_id = wave_ptr->wave.wave_id,
                .pc = wave_ptr->wave.pc,
                .message = "peu=" + std::to_string(peu_id),
            });
          },
  });
}

void ActivateBlock(ExecutableBlock& block,
                   uint64_t cycle,
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
  std::map<uint32_t, uint32_t> peu_launch_order;
  for (auto& scheduled_wave : block.waves) {
    scheduled_wave.wave.status = WaveStatus::Stalled;
    scheduled_wave.dispatch_enabled = false;
    scheduled_wave.launch_scheduled = false;
    const uint32_t launch_order = peu_launch_order[scheduled_wave.wave.peu_id]++;
    if (launch_order < max_issuable_waves) {
      const uint64_t launch_cycle =
          cycle + static_cast<uint64_t>(launch_order) * wave_launch_cycles;
      ScheduleWaveLaunch(scheduled_wave, launch_cycle, events, trace);
    }
  }
}

void FillDispatchWindow(PeuSlot& slot,
                        uint64_t cycle,
                        uint32_t max_issuable_waves,
                        EventQueue& events,
                        TraceSink& trace) {
  uint32_t active_count = 0;
  for (auto* scheduled_wave : slot.waves) {
    if (!scheduled_wave->block->active || !scheduled_wave->dispatch_enabled) {
      continue;
    }
    if ((scheduled_wave->wave.status == WaveStatus::Active && scheduled_wave->wave.valid_entry) ||
        scheduled_wave->launch_scheduled) {
      ++active_count;
    }
  }
  if (active_count >= max_issuable_waves) {
    return;
  }

  for (auto* scheduled_wave : slot.waves) {
    if (active_count >= max_issuable_waves) {
      break;
    }
    if (!scheduled_wave->block->active || scheduled_wave->dispatch_enabled ||
        scheduled_wave->wave.status == WaveStatus::Exited) {
      continue;
    }
    ScheduleWaveLaunch(*scheduled_wave, cycle, events, trace);
    ++active_count;
  }
}

ScheduledWave* PickNextReadyWave(PeuSlot& slot,
                                 const KernelProgram& kernel,
                                 uint64_t cycle) {
  if (slot.waves.empty()) {
    return nullptr;
  }

  const size_t count = slot.waves.size();
  const size_t start =
      slot.last_issue_index == std::numeric_limits<size_t>::max() ? 0 : (slot.last_issue_index + 1) % count;
  for (size_t offset = 0; offset < count; ++offset) {
    const size_t index = (start + offset) % count;
    ScheduledWave* scheduled_wave = slot.waves[index];
    if (scheduled_wave->wave.pc >= kernel.instructions().size()) {
      throw std::out_of_range("wave pc out of range");
    }
    const auto& instruction = kernel.instructions().at(scheduled_wave->wave.pc);
    if (!CanIssueInstruction(*scheduled_wave, instruction)) {
      continue;
    }
    if (!DependenciesReady(instruction, scheduled_wave->scoreboard, cycle)) {
      continue;
    }
    slot.last_issue_index = index;
    return scheduled_wave;
  }

  return nullptr;
}

std::optional<std::pair<ScheduledWave*, std::string>> PickFirstBlockedWave(PeuSlot& slot,
                                                                           const KernelProgram& kernel,
                                                                           uint64_t cycle) {
  if (slot.waves.empty()) {
    return std::nullopt;
  }

  const size_t count = slot.waves.size();
  const size_t start =
      slot.last_issue_index == std::numeric_limits<size_t>::max() ? 0 : (slot.last_issue_index + 1) % count;
  for (size_t offset = 0; offset < count; ++offset) {
    const size_t index = (start + offset) % count;
    ScheduledWave* scheduled_wave = slot.waves[index];
    if (!scheduled_wave->dispatch_enabled || scheduled_wave->wave.status != WaveStatus::Active) {
      continue;
    }
    if (scheduled_wave->wave.pc >= kernel.instructions().size()) {
      continue;
    }
    const auto& instruction = kernel.instructions().at(scheduled_wave->wave.pc);
    if (const auto reason = IssueBlockReason(*scheduled_wave, instruction, cycle)) {
      return std::make_pair(scheduled_wave, *reason);
    }
  }
  return std::nullopt;
}

}  // namespace

uint64_t CycleExecutor::Run(ExecutionContext& context) {
  auto blocks = MaterializeBlocks(context.placement, context.launch_config);
  auto slots = BuildPeuSlots(blocks);
  EventQueue events;
  std::map<L1Key, CacheModel> l1_caches;
  CacheModel l2_cache(timing_config_.cache_model);
  SharedBankModel shared_bank_model(timing_config_.shared_bank_model);
  std::map<uint32_t, std::vector<ExecutableBlock*>> ap_queues;

  for (auto& slot : slots) {
    slot.busy_until = 0;
    l1_caches.emplace(L1Key{.dpc_id = slot.dpc_id, .ap_id = slot.ap_id},
                      CacheModel(timing_config_.cache_model));
  }

  for (auto& block : blocks) {
    auto& queue = ap_queues[block.global_ap_id];
    block.ap_queue_index = static_cast<uint32_t>(queue.size());
    queue.push_back(&block);
  }
  for (auto& [global_ap_id, queue] : ap_queues) {
    (void)global_ap_id;
    if (!queue.empty()) {
      ActivateBlock(*queue.front(), context.cycle, context.spec.max_issuable_waves,
                    timing_config_.launch_timing.wave_launch_cycles, events, context.trace);
    }
  }

  uint64_t cycle = context.cycle;
  while (true) {
    context.cycle = cycle;
    events.RunReady(cycle);
    for (auto& slot : slots) {
      FillDispatchWindow(slot, cycle, context.spec.max_issuable_waves, events, context.trace);
    }
    events.RunReady(cycle);

    if (AllWavesExited(blocks) && events.empty()) {
      return cycle;
    }

    bool issued_any = false;
    for (auto& slot : slots) {
      if (slot.busy_until > cycle) {
        continue;
      }

      ScheduledWave* candidate = PickNextReadyWave(slot, context.kernel, cycle);

      if (candidate == nullptr) {
        if (const auto blocked = PickFirstBlockedWave(slot, context.kernel, cycle)) {
          context.trace.OnEvent(TraceEvent{
              .kind = TraceEventKind::Stall,
              .cycle = cycle,
              .dpc_id = blocked->first->wave.dpc_id,
              .ap_id = blocked->first->wave.ap_id,
              .peu_id = blocked->first->wave.peu_id,
              .block_id = blocked->first->wave.block_id,
              .wave_id = blocked->first->wave.wave_id,
              .pc = blocked->first->wave.pc,
              .message = blocked->second,
          });
        }
        continue;
      }

      WaveState& wave = candidate->wave;
      const Instruction instruction = context.kernel.instructions().at(wave.pc);
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
          .block_id = wave.block_id,
          .wave_id = wave.wave_id,
          .pc = wave.pc,
          .message = FormatWaveStepMessage(instruction, wave),
      });

      const OpPlan plan = semantics_.BuildPlan(instruction, wave, context);
      const uint64_t wave_tag = WaveTag(wave);
      const uint64_t switch_penalty =
          slot.last_wave_tag != std::numeric_limits<uint64_t>::max() &&
                  slot.last_wave_tag != wave_tag
              ? timing_config_.launch_timing.warp_switch_cycles
              : 0;
      if (switch_penalty > 0) {
        context.trace.OnEvent(TraceEvent{
            .kind = TraceEventKind::Stall,
            .cycle = cycle,
            .dpc_id = wave.dpc_id,
            .ap_id = wave.ap_id,
            .peu_id = wave.peu_id,
            .block_id = wave.block_id,
            .wave_id = wave.wave_id,
            .pc = wave.pc,
            .message = "warp_switch",
        });
      }
      const uint64_t commit_cycle = cycle + switch_penalty + plan.issue_cycles;
      slot.busy_until = commit_cycle;
      slot.last_wave_tag = wave_tag;
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
              [&, candidate, instruction, plan, commit_cycle]() {
                context.cycle = commit_cycle;

                for (const auto& write : plan.scalar_writes) {
                  candidate->wave.sgpr.Write(write.reg_index, write.value);
                }
                for (const auto& write : plan.vector_writes) {
                  for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
                    if (write.mask.test(lane)) {
                      candidate->wave.vgpr.Write(write.reg_index, lane, write.values[lane]);
                    }
                  }
                }
                if (plan.cmask_write.has_value()) {
                  candidate->wave.cmask = *plan.cmask_write;
                }
                if (plan.smask_write.has_value()) {
                  candidate->wave.smask = *plan.smask_write;
                }
                if (plan.exec_write.has_value()) {
                  candidate->wave.exec = *plan.exec_write;
                  std::ostringstream mask_text;
                  mask_text << candidate->wave.exec;
                  context.trace.OnEvent(TraceEvent{
                      .kind = TraceEventKind::ExecMaskUpdate,
                      .cycle = commit_cycle,
                      .dpc_id = candidate->wave.dpc_id,
                      .ap_id = candidate->wave.ap_id,
                      .peu_id = candidate->wave.peu_id,
                      .block_id = candidate->wave.block_id,
                      .wave_id = candidate->wave.wave_id,
                      .pc = candidate->wave.pc,
                      .message = mask_text.str(),
                  });
                }

                context.trace.OnEvent(TraceEvent{
                    .kind = TraceEventKind::Commit,
                    .cycle = commit_cycle,
                    .dpc_id = candidate->wave.dpc_id,
                    .ap_id = candidate->wave.ap_id,
                    .peu_id = candidate->wave.peu_id,
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
                            [&, candidate, request, addrs, arrive_cycle]() {
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
                        loaded_value =
                            LoadLaneValue(context.kernel.const_segment().bytes, request.lanes[lane]);
                        break;
                      }
                      candidate->wave.sgpr.Write(request.dst->index, loaded_value);
                      candidate->scoreboard.MarkReady(
                          ScalarRef(request.dst->index), commit_cycle);
                    } else {
                      for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
                        if (request.lanes[lane].active) {
                          const uint64_t value =
                              LoadLaneValue(context.kernel.const_segment().bytes, request.lanes[lane]);
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
                  candidate->wave.waiting_at_barrier = true;
                  candidate->wave.barrier_generation = candidate->block->barrier_generation;
                  ++candidate->block->barrier_arrivals;
                  context.trace.OnEvent(TraceEvent{
                      .kind = TraceEventKind::Barrier,
                      .cycle = commit_cycle,
                      .dpc_id = candidate->wave.dpc_id,
                      .ap_id = candidate->wave.ap_id,
                      .peu_id = candidate->wave.peu_id,
                      .block_id = candidate->wave.block_id,
                      .wave_id = candidate->wave.wave_id,
                      .pc = candidate->wave.pc,
                      .message = "arrive",
                  });

                  if (candidate->block->barrier_arrivals == candidate->block->waves.size()) {
                    for (auto& waiting_wave : candidate->block->waves) {
                      if (waiting_wave.wave.waiting_at_barrier &&
                          waiting_wave.wave.barrier_generation ==
                              candidate->block->barrier_generation) {
                        waiting_wave.wave.waiting_at_barrier = false;
                        waiting_wave.wave.valid_entry = true;
                        waiting_wave.wave.status = WaveStatus::Active;
                        ++waiting_wave.wave.pc;
                      }
                    }
                    candidate->block->barrier_arrivals = 0;
                    ++candidate->block->barrier_generation;
                    context.trace.OnEvent(TraceEvent{
                        .kind = TraceEventKind::Barrier,
                        .cycle = commit_cycle,
                        .dpc_id = candidate->wave.dpc_id,
                        .ap_id = candidate->wave.ap_id,
                        .block_id = candidate->wave.block_id,
                        .pc = candidate->wave.pc,
                        .message = "release",
                    });
                  }
                  return;
                }

                if (plan.exit_wave) {
                  if (context.stats != nullptr) {
                    ++context.stats->wave_exits;
                  }
                  candidate->wave.status = WaveStatus::Exited;
                  context.trace.OnEvent(TraceEvent{
                      .kind = TraceEventKind::WaveExit,
                      .cycle = commit_cycle,
                      .dpc_id = candidate->wave.dpc_id,
                      .ap_id = candidate->wave.ap_id,
                      .peu_id = candidate->wave.peu_id,
                      .block_id = candidate->wave.block_id,
                      .wave_id = candidate->wave.wave_id,
                      .pc = candidate->wave.pc,
                      .message = "exit",
                  });
                  if (candidate->block->active && !candidate->block->completed &&
                      AllWavesExited(*candidate->block)) {
                    candidate->block->active = false;
                    candidate->block->completed = true;
                    const auto queue_it = ap_queues.find(candidate->block->global_ap_id);
                    if (queue_it != ap_queues.end()) {
                      const uint32_t next_index = candidate->block->ap_queue_index + 1;
                      if (next_index < queue_it->second.size()) {
                        ExecutableBlock* next_block = queue_it->second[next_index];
                        events.Schedule(TimedEvent{
                            .cycle = commit_cycle + timing_config_.launch_timing.block_launch_cycles,
                            .action =
                                [next_block, &context, &events, commit_cycle, this]() {
                                  ActivateBlock(
                                      *next_block,
                                      commit_cycle + timing_config_.launch_timing.block_launch_cycles,
                                      context.spec.max_issuable_waves,
                                      timing_config_.launch_timing.wave_launch_cycles,
                                      events,
                                      context.trace);
                                },
                        });
                      }
                    }
                  }
                  return;
                }

                if (plan.branch_target.has_value()) {
                  candidate->wave.pc = *plan.branch_target;
                } else if (plan.advance_pc) {
                  ++candidate->wave.pc;
                }
                candidate->wave.branch_pending = false;
                candidate->wave.valid_entry = true;
                candidate->wave.status = WaveStatus::Active;
              },
      });

      issued_any = true;
    }

    if (!issued_any && events.empty()) {
      throw std::runtime_error("cycle execution stalled without pending events");
    }

    ++cycle;
  }
}

}  // namespace gpu_model
