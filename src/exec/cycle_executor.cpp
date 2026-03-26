#include "gpu_model/exec/cycle_executor.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "gpu_model/debug/trace_event.h"
#include "gpu_model/exec/event_queue.h"
#include "gpu_model/exec/scoreboard.h"
#include "gpu_model/isa/opcode.h"

namespace gpu_model {

namespace {

struct ScheduledWave {
  uint32_t dpc_id = 0;
  struct ExecutableBlock* block = nullptr;
  WaveState wave;
  Scoreboard scoreboard;
};

struct ExecutableBlock {
  uint32_t block_id = 0;
  uint32_t dpc_id = 0;
  uint32_t ap_id = 0;
  uint64_t barrier_generation = 0;
  uint32_t barrier_arrivals = 0;
  std::vector<std::byte> shared_memory;
  std::vector<ScheduledWave> waves;
};

struct PeuSlot {
  uint32_t dpc_id = 0;
  uint32_t ap_id = 0;
  uint32_t peu_id = 0;
  uint64_t busy_until = 0;
  std::vector<ScheduledWave*> waves;
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
    case Opcode::SysBlockIdxX:
    case Opcode::SysBlockDimX:
    case Opcode::SysLaneId:
    case Opcode::BBranch:
    case Opcode::BExit:
      break;
    case Opcode::SMov:
      AddOperandDependency(instruction.operands.at(1), refs);
      break;
    case Opcode::SAdd:
    case Opcode::SMul:
      AddOperandDependency(instruction.operands.at(1), refs);
      AddOperandDependency(instruction.operands.at(2), refs);
      break;
    case Opcode::SCmpLt:
    case Opcode::SCmpEq:
      AddOperandDependency(instruction.operands.at(0), refs);
      AddOperandDependency(instruction.operands.at(1), refs);
      break;
    case Opcode::VMov:
      refs.push_back(ExecRef());
      AddOperandDependency(instruction.operands.at(1), refs);
      break;
    case Opcode::VAdd:
    case Opcode::VMul:
      refs.push_back(ExecRef());
      AddOperandDependency(instruction.operands.at(1), refs);
      AddOperandDependency(instruction.operands.at(2), refs);
      break;
    case Opcode::VCmpLtCmask:
      refs.push_back(ExecRef());
      AddOperandDependency(instruction.operands.at(0), refs);
      AddOperandDependency(instruction.operands.at(1), refs);
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
    case Opcode::MStoreGlobal:
    case Opcode::MStoreShared:
    case Opcode::MStorePrivate:
      refs.push_back(ExecRef());
      AddOperandDependency(instruction.operands.at(0), refs);
      AddOperandDependency(instruction.operands.at(1), refs);
      if (instruction.opcode == Opcode::MStoreGlobal) {
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
      scheduled.wave.wave_id = wave_placement.wave_id;
      scheduled.wave.peu_id = wave_placement.peu_id;
      scheduled.wave.ap_id = block_placement.ap_id;
      scheduled.wave.thread_count = wave_placement.lane_count;
      scheduled.wave.ResetInitialExec();
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

}  // namespace

uint64_t CycleExecutor::Run(ExecutionContext& context) {
  auto blocks = MaterializeBlocks(context.placement, context.launch_config);
  auto slots = BuildPeuSlots(blocks);
  EventQueue events;

  for (auto& slot : slots) {
    slot.busy_until = 0;
  }

  uint64_t cycle = 0;
  while (true) {
    context.cycle = cycle;
    events.RunReady(cycle);

    if (AllWavesExited(blocks) && events.empty()) {
      return cycle;
    }

    bool issued_any = false;
    for (auto& slot : slots) {
      if (slot.busy_until > cycle) {
        continue;
      }

      ScheduledWave* candidate = nullptr;
      for (auto* scheduled_wave : slot.waves) {
        if (scheduled_wave->wave.status != WaveStatus::Active) {
          continue;
        }
        if (scheduled_wave->wave.pc >= context.kernel.instructions().size()) {
          throw std::out_of_range("wave pc out of range");
        }
        const auto& instruction = context.kernel.instructions().at(scheduled_wave->wave.pc);
        if (DependenciesReady(instruction, scheduled_wave->scoreboard, cycle)) {
          candidate = scheduled_wave;
          break;
        }
      }

      if (candidate == nullptr) {
        continue;
      }

      WaveState& wave = candidate->wave;
      const Instruction instruction = context.kernel.instructions().at(wave.pc);
      context.trace.OnEvent(TraceEvent{
          .kind = TraceEventKind::WaveStep,
          .cycle = cycle,
          .block_id = wave.block_id,
          .wave_id = wave.wave_id,
          .pc = wave.pc,
          .message = std::string(ToString(instruction.opcode)),
      });

      const OpPlan plan = semantics_.BuildPlan(instruction, wave, context);
      const uint64_t commit_cycle = cycle + plan.issue_cycles;
      slot.busy_until = commit_cycle;
      wave.status = WaveStatus::Stalled;

      if (plan.memory.has_value()) {
        context.trace.OnEvent(TraceEvent{
            .kind = TraceEventKind::MemoryAccess,
            .cycle = cycle,
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
                      .block_id = candidate->wave.block_id,
                      .wave_id = candidate->wave.wave_id,
                      .pc = candidate->wave.pc,
                      .message = mask_text.str(),
                  });
                }

                if (plan.memory.has_value()) {
                  const MemoryRequest request = *plan.memory;
                  if (request.space == MemorySpace::Global) {
                    const uint64_t arrive_cycle = commit_cycle + fixed_global_latency_;
                    events.Schedule(TimedEvent{
                        .cycle = arrive_cycle,
                        .action =
                            [&, candidate, request, arrive_cycle]() {
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
                              } else if (request.kind == AccessKind::Store) {
                                for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
                                  if (request.lanes[lane].active) {
                                    StoreLaneValue(context.memory, request.lanes[lane]);
                                  }
                                }
                              }

                              context.trace.OnEvent(TraceEvent{
                                  .kind = TraceEventKind::Arrive,
                                  .cycle = arrive_cycle,
                                  .block_id = candidate->wave.block_id,
                                  .wave_id = candidate->wave.wave_id,
                                  .pc = candidate->wave.pc,
                                  .message = request.kind == AccessKind::Load ? "load_arrive"
                                                                              : "store_arrive",
                              });
                            },
                    });
                  } else if (request.space == MemorySpace::Shared) {
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
                          VectorRef(request.dst->index), commit_cycle);
                    } else if (request.kind == AccessKind::Store) {
                      for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
                        if (request.lanes[lane].active) {
                          StoreLaneValue(candidate->block->shared_memory, request.lanes[lane]);
                        }
                      }
                    }
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
                  } else if (request.space == MemorySpace::Constant) {
                    if (!request.dst.has_value()) {
                      throw std::invalid_argument("load request missing destination");
                    }
                    for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
                      if (request.lanes[lane].active) {
                        const uint64_t value =
                            LoadLaneValue(context.kernel.const_segment().bytes, request.lanes[lane]);
                        candidate->wave.vgpr.Write(request.dst->index, lane, value);
                      }
                    }
                    candidate->scoreboard.MarkReady(
                        VectorRef(request.dst->index), commit_cycle);
                  } else {
                    throw std::invalid_argument("unsupported memory space in cycle executor");
                  }
                }

                if (plan.sync_barrier) {
                  candidate->wave.waiting_at_barrier = true;
                  candidate->wave.barrier_generation = candidate->block->barrier_generation;
                  ++candidate->block->barrier_arrivals;
                  context.trace.OnEvent(TraceEvent{
                      .kind = TraceEventKind::Barrier,
                      .cycle = commit_cycle,
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
                        waiting_wave.wave.status = WaveStatus::Active;
                        ++waiting_wave.wave.pc;
                      }
                    }
                    candidate->block->barrier_arrivals = 0;
                    ++candidate->block->barrier_generation;
                    context.trace.OnEvent(TraceEvent{
                        .kind = TraceEventKind::Barrier,
                        .cycle = commit_cycle,
                        .block_id = candidate->wave.block_id,
                        .pc = candidate->wave.pc,
                        .message = "release",
                    });
                  }
                  return;
                }

                if (plan.exit_wave) {
                  candidate->wave.status = WaveStatus::Exited;
                  context.trace.OnEvent(TraceEvent{
                      .kind = TraceEventKind::WaveExit,
                      .cycle = commit_cycle,
                      .block_id = candidate->wave.block_id,
                      .wave_id = candidate->wave.wave_id,
                      .pc = candidate->wave.pc,
                      .message = "exit",
                  });
                  return;
                }

                if (plan.branch_target.has_value()) {
                  candidate->wave.pc = *plan.branch_target;
                } else if (plan.advance_pc) {
                  ++candidate->wave.pc;
                }
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
