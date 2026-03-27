#include "gpu_model/exec/functional_executor.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "gpu_model/debug/instruction_trace.h"
#include "gpu_model/debug/trace_event.h"
#include "gpu_model/isa/opcode.h"

namespace gpu_model {

namespace {

struct ExecutableBlock {
  uint32_t block_id = 0;
  uint32_t dpc_id = 0;
  uint32_t ap_id = 0;
  uint64_t barrier_generation = 0;
  uint32_t barrier_arrivals = 0;
  std::vector<std::byte> shared_memory;
  std::vector<WaveState> waves;
};

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

uint64_t LoadLaneValue(std::array<std::vector<std::byte>, kWaveSize>& memory,
                       uint32_t lane_id,
                       const LaneAccess& lane) {
  auto& lane_memory = memory.at(lane_id);
  const size_t end = static_cast<size_t>(lane.addr) + lane.bytes;
  if (lane_memory.size() < end) {
    lane_memory.resize(end, std::byte{0});
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
      WaveState wave;
      wave.block_id = block_placement.block_id;
      wave.wave_id = wave_placement.wave_id;
      wave.peu_id = wave_placement.peu_id;
      wave.ap_id = block_placement.ap_id;
      wave.thread_count = wave_placement.lane_count;
      wave.ResetInitialExec();
      block.waves.push_back(wave);
    }
    blocks.push_back(std::move(block));
  }

  return blocks;
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

}  // namespace

uint64_t FunctionalExecutor::Run(ExecutionContext& context) {
  auto blocks = MaterializeBlocks(context.placement, context.launch_config);

  auto has_active_wave = [&blocks]() {
    for (const auto& block : blocks) {
      for (const auto& wave : block.waves) {
        if (wave.status == WaveStatus::Active) {
          return true;
        }
      }
    }
    return false;
  };

  while (has_active_wave()) {
    bool made_progress = false;

    for (auto& block : blocks) {
      for (auto& wave : block.waves) {
        if (wave.status != WaveStatus::Active) {
          continue;
        }
        if (wave.pc >= context.kernel.instructions().size()) {
          throw std::out_of_range("wave pc out of range");
        }

        const Instruction& instruction = context.kernel.instructions().at(wave.pc);
        if (context.stats != nullptr) {
          ++context.stats->wave_steps;
          ++context.stats->instructions_issued;
        }
        context.trace.OnEvent(TraceEvent{
            .kind = TraceEventKind::WaveStep,
            .cycle = context.cycle,
            .block_id = wave.block_id,
            .wave_id = wave.wave_id,
            .pc = wave.pc,
            .message = FormatWaveStepMessage(instruction, wave),
        });

        const OpPlan plan = semantics_.BuildPlan(instruction, wave, context);

        for (const auto& write : plan.scalar_writes) {
          wave.sgpr.Write(write.reg_index, write.value);
        }
        for (const auto& write : plan.vector_writes) {
          for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
            if (write.mask.test(lane)) {
              wave.vgpr.Write(write.reg_index, lane, write.values[lane]);
            }
          }
        }
        if (plan.cmask_write.has_value()) {
          wave.cmask = *plan.cmask_write;
        }
        if (plan.smask_write.has_value()) {
          wave.smask = *plan.smask_write;
        }
        if (plan.exec_write.has_value()) {
          wave.exec = *plan.exec_write;
          std::ostringstream mask_text;
          mask_text << wave.exec;
          context.trace.OnEvent(TraceEvent{
              .kind = TraceEventKind::ExecMaskUpdate,
              .cycle = context.cycle,
              .block_id = wave.block_id,
              .wave_id = wave.wave_id,
              .pc = wave.pc,
              .message = mask_text.str(),
          });
        }
        if (plan.memory.has_value()) {
          const auto& request = *plan.memory;
          if (context.stats != nullptr) {
            ++context.stats->memory_ops;
            if (request.space == MemorySpace::Global) {
              if (request.kind == AccessKind::Load) {
                ++context.stats->global_loads;
              } else if (request.kind == AccessKind::Store) {
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
              .cycle = context.cycle,
              .block_id = wave.block_id,
              .wave_id = wave.wave_id,
              .pc = wave.pc,
              .message = request.kind == AccessKind::Load ? "load" : "store",
          });

          if (request.kind == AccessKind::Load) {
            if (!request.dst.has_value()) {
              throw std::invalid_argument("load request missing destination");
            }
            std::array<uint64_t, 64> loaded_values{};
            for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
              if (request.lanes[lane].active) {
                if (request.space == MemorySpace::Global) {
                  loaded_values[lane] = LoadLaneValue(context.memory, request.lanes[lane]);
                } else if (request.space == MemorySpace::Shared) {
                  loaded_values[lane] = LoadLaneValue(block.shared_memory, request.lanes[lane]);
                } else if (request.space == MemorySpace::Private) {
                  loaded_values[lane] =
                      LoadLaneValue(wave.private_memory, lane, request.lanes[lane]);
                } else if (request.space == MemorySpace::Constant) {
                  loaded_values[lane] =
                      LoadLaneValue(context.kernel.const_segment().bytes, request.lanes[lane]);
                } else {
                  throw std::invalid_argument("unsupported load memory space");
                }
              }
            }
            for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
              if (request.exec_snapshot.test(lane)) {
                wave.vgpr.Write(request.dst->index, lane, loaded_values[lane]);
              }
            }
          } else if (request.kind == AccessKind::Store) {
            for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
              if (request.lanes[lane].active) {
                if (request.space == MemorySpace::Global) {
                  StoreLaneValue(context.memory, request.lanes[lane]);
                } else if (request.space == MemorySpace::Shared) {
                  StoreLaneValue(block.shared_memory, request.lanes[lane]);
                } else if (request.space == MemorySpace::Private) {
                  StoreLaneValue(wave.private_memory, lane, request.lanes[lane]);
                } else {
                  throw std::invalid_argument("unsupported store memory space");
                }
              }
            }
          } else if (request.kind == AccessKind::Atomic) {
            if (request.space != MemorySpace::Shared) {
              throw std::invalid_argument("unsupported atomic memory space");
            }
            for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
              if (!request.lanes[lane].active) {
                continue;
              }
              const int32_t prior = static_cast<int32_t>(
                  LoadLaneValue(block.shared_memory, request.lanes[lane]));
              const int32_t updated = prior + static_cast<int32_t>(request.lanes[lane].value);
              LaneAccess writeback = request.lanes[lane];
              writeback.value = static_cast<uint64_t>(static_cast<int64_t>(updated));
              StoreLaneValue(block.shared_memory, writeback);
            }
          } else {
            throw std::invalid_argument("unsupported memory access kind in functional executor");
          }
        }

        if (plan.sync_wave_barrier) {
          if (context.stats != nullptr) {
            ++context.stats->barriers;
          }
          context.trace.OnEvent(TraceEvent{
              .kind = TraceEventKind::Barrier,
              .cycle = context.cycle,
              .block_id = wave.block_id,
              .wave_id = wave.wave_id,
              .pc = wave.pc,
              .message = "wave",
          });
          ++wave.pc;
        } else if (plan.sync_barrier) {
          if (context.stats != nullptr) {
            ++context.stats->barriers;
          }
          wave.status = WaveStatus::Stalled;
          wave.waiting_at_barrier = true;
          wave.barrier_generation = block.barrier_generation;
          ++block.barrier_arrivals;
          context.trace.OnEvent(TraceEvent{
              .kind = TraceEventKind::Barrier,
              .cycle = context.cycle,
              .block_id = wave.block_id,
              .wave_id = wave.wave_id,
              .pc = wave.pc,
              .message = "arrive",
          });

          if (block.barrier_arrivals == block.waves.size()) {
            for (auto& waiting_wave : block.waves) {
              if (waiting_wave.waiting_at_barrier &&
                  waiting_wave.barrier_generation == block.barrier_generation) {
                waiting_wave.waiting_at_barrier = false;
                waiting_wave.status = WaveStatus::Active;
                ++waiting_wave.pc;
              }
            }
            block.barrier_arrivals = 0;
            ++block.barrier_generation;
            context.trace.OnEvent(TraceEvent{
                .kind = TraceEventKind::Barrier,
                .cycle = context.cycle,
                .block_id = block.block_id,
                .pc = wave.pc,
                .message = "release",
            });
          }
        } else if (plan.exit_wave) {
          if (context.stats != nullptr) {
            ++context.stats->wave_exits;
          }
          wave.status = WaveStatus::Exited;
          context.trace.OnEvent(TraceEvent{
              .kind = TraceEventKind::WaveExit,
              .cycle = context.cycle,
              .block_id = wave.block_id,
              .wave_id = wave.wave_id,
              .pc = wave.pc,
              .message = "exit",
          });
        } else if (plan.branch_target.has_value()) {
          wave.pc = *plan.branch_target;
        } else if (plan.advance_pc) {
          ++wave.pc;
        }

        made_progress = true;
      }
    }

    if (!made_progress) {
      throw std::runtime_error("functional execution stalled without progress");
    }
  }

  return 0;
}

}  // namespace gpu_model
