#include "gpu_model/exec/functional_execution_core.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <vector>

#ifdef GPU_MODEL_HAS_MARL
#include "marl/scheduler.h"
#include "marl/waitgroup.h"
#endif

#include "gpu_model/debug/instruction_trace.h"
#include "gpu_model/debug/trace_event.h"
#include "gpu_model/isa/opcode.h"
#include "gpu_model/loader/device_image_loader.h"

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
  std::vector<std::vector<size_t>> wave_indices_per_peu;
  std::vector<size_t> next_wave_rr_per_peu;
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
        .wave_indices_per_peu = {},
        .next_wave_rr_per_peu = {},
    };
    block.waves.reserve(block_placement.waves.size());
    for (const auto& wave_placement : block_placement.waves) {
      WaveState wave;
      wave.block_id = block_placement.block_id;
      wave.block_idx_x = block_placement.block_idx_x;
      wave.block_idx_y = block_placement.block_idx_y;
      wave.dpc_id = block_placement.dpc_id;
      wave.wave_id = wave_placement.wave_id;
      wave.peu_id = wave_placement.peu_id;
      wave.ap_id = block_placement.ap_id;
      wave.thread_count = wave_placement.lane_count;
      wave.ResetInitialExec();
      if (block.wave_indices_per_peu.size() <= wave_placement.peu_id) {
        block.wave_indices_per_peu.resize(static_cast<size_t>(wave_placement.peu_id) + 1);
        block.next_wave_rr_per_peu.resize(static_cast<size_t>(wave_placement.peu_id) + 1, 0);
      }
      block.wave_indices_per_peu[wave_placement.peu_id].push_back(block.waves.size());
      block.waves.push_back(wave);
    }
    blocks.push_back(std::move(block));
  }

  return blocks;
}

uint64_t LoadLaneValue(const MemorySystem& memory, MemoryPoolKind pool, const LaneAccess& lane) {
  switch (lane.bytes) {
    case 4: {
      const int32_t value = memory.LoadValue<int32_t>(pool, lane.addr);
      return static_cast<uint64_t>(static_cast<int64_t>(value));
    }
    case 8:
      return memory.LoadValue<uint64_t>(pool, lane.addr);
    default:
      throw std::invalid_argument("unsupported load width");
  }
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

void MergeStats(ExecutionStats& dst, const ExecutionStats& src) {
  dst.wave_steps += src.wave_steps;
  dst.instructions_issued += src.instructions_issued;
  dst.memory_ops += src.memory_ops;
  dst.global_loads += src.global_loads;
  dst.global_stores += src.global_stores;
  dst.shared_loads += src.shared_loads;
  dst.shared_stores += src.shared_stores;
  dst.private_loads += src.private_loads;
  dst.private_stores += src.private_stores;
  dst.constant_loads += src.constant_loads;
  dst.barriers += src.barriers;
  dst.wave_exits += src.wave_exits;
  dst.l1_hits += src.l1_hits;
  dst.l2_hits += src.l2_hits;
  dst.cache_misses += src.cache_misses;
  dst.shared_bank_conflict_penalty_cycles += src.shared_bank_conflict_penalty_cycles;
}

class FunctionalExecutionCoreImpl {
 public:
  explicit FunctionalExecutionCoreImpl(ExecutionContext& context)
      : context_(context), semantics_(), blocks_(MaterializeBlocks(context.placement, context.launch_config)) {}

  uint64_t RunSequential() {
    for (auto& block : blocks_) {
      ExecutionStats block_stats;
      ExecuteBlock(block, block_stats);
      CommitStats(block_stats);
    }
    return 0;
  }

  uint64_t RunParallelBlocks(uint32_t worker_threads) {
#ifdef GPU_MODEL_HAS_MARL
    marl::Scheduler::Config scheduler_config;
    if (worker_threads == 0) {
      scheduler_config = marl::Scheduler::Config::allCores();
    } else {
      scheduler_config.setWorkerThreadCount(static_cast<int>(worker_threads));
    }

    marl::Scheduler scheduler(scheduler_config);
    scheduler.bind();

    marl::WaitGroup done(static_cast<int>(blocks_.size()));
    std::exception_ptr failure;
    std::mutex failure_mutex;

    for (size_t i = 0; i < blocks_.size(); ++i) {
      marl::schedule([&, i] {
        ExecutionStats block_stats;
        try {
          ExecuteBlock(blocks_[i], block_stats);
          CommitStats(block_stats);
        } catch (...) {
          std::lock_guard<std::mutex> lock(failure_mutex);
          if (failure == nullptr) {
            failure = std::current_exception();
          }
        }
        done.done();
      });
    }

    done.wait();
    marl::Scheduler::unbind();
    if (failure != nullptr) {
      std::rethrow_exception(failure);
    }
    return 0;
#else
    (void)worker_threads;
    return RunSequential();
#endif
  }

 private:
  ExecutionContext& context_;
  Semantics semantics_;
  std::vector<ExecutableBlock> blocks_;
  std::mutex trace_mutex_;
  std::mutex stats_mutex_;
  std::mutex global_memory_mutex_;

  void TraceEventLocked(TraceEvent event) {
    std::lock_guard<std::mutex> lock(trace_mutex_);
    context_.trace.OnEvent(std::move(event));
  }

  void CommitStats(const ExecutionStats& block_stats) {
    if (context_.stats == nullptr) {
      return;
    }
    std::lock_guard<std::mutex> lock(stats_mutex_);
    MergeStats(*context_.stats, block_stats);
  }

  uint64_t LoadGlobalLaneValue(const LaneAccess& lane) {
    std::lock_guard<std::mutex> lock(global_memory_mutex_);
    switch (lane.bytes) {
      case 4: {
        const int32_t value = context_.memory.LoadGlobalValue<int32_t>(lane.addr);
        return static_cast<uint64_t>(static_cast<int64_t>(value));
      }
      case 8:
        return context_.memory.LoadGlobalValue<uint64_t>(lane.addr);
      default:
        throw std::invalid_argument("unsupported load width");
    }
  }

  void StoreGlobalLaneValue(const LaneAccess& lane) {
    std::lock_guard<std::mutex> lock(global_memory_mutex_);
    switch (lane.bytes) {
      case 4:
        context_.memory.StoreGlobalValue<int32_t>(lane.addr, static_cast<int32_t>(lane.value));
        return;
      case 8:
        context_.memory.StoreGlobalValue<uint64_t>(lane.addr, lane.value);
        return;
      default:
        throw std::invalid_argument("unsupported store width");
    }
  }

  uint64_t LoadConstantLaneValue(const LaneAccess& lane) {
    std::lock_guard<std::mutex> lock(global_memory_mutex_);
    const LaneAccess pool_lane{
        .active = lane.active,
        .addr = ConstantPoolBase(context_) + lane.addr,
        .bytes = lane.bytes,
        .value = lane.value,
    };
    if (context_.memory.HasRange(MemoryPoolKind::Constant, pool_lane.addr, lane.bytes)) {
      return LoadLaneValue(context_.memory, MemoryPoolKind::Constant, pool_lane);
    }
    return LoadLaneValue(context_.kernel.const_segment().bytes, lane);
  }

  void ExecuteBlock(ExecutableBlock& block, ExecutionStats& stats) {
    auto has_active_wave = [&block]() {
      for (const auto& wave : block.waves) {
        if (wave.status == WaveStatus::Active) {
          return true;
        }
      }
      return false;
    };

    auto release_block_barrier_if_ready = [&]() {
      if (block.waves.empty()) {
        return false;
      }
      uint32_t active_wave_count = 0;
      uint32_t waiting_wave_count = 0;
      for (const auto& wave : block.waves) {
        if (wave.status == WaveStatus::Active || wave.status == WaveStatus::Stalled) {
          ++active_wave_count;
          if (wave.waiting_at_barrier) {
            ++waiting_wave_count;
          }
        }
      }
      if (active_wave_count == 0 || waiting_wave_count != active_wave_count) {
        return false;
      }
      for (auto& wave : block.waves) {
        if (wave.waiting_at_barrier && wave.barrier_generation == block.barrier_generation) {
          wave.waiting_at_barrier = false;
          wave.status = WaveStatus::Active;
          ++wave.pc;
        }
      }
      block.barrier_arrivals = 0;
      ++block.barrier_generation;
      TraceEventLocked(TraceEvent{
          .kind = TraceEventKind::Barrier,
          .cycle = context_.cycle,
          .block_id = block.block_id,
          .message = "release",
      });
      return true;
    };

    auto select_next_wave_index_for_peu = [&](size_t peu_index) -> std::optional<size_t> {
      if (peu_index >= block.wave_indices_per_peu.size()) {
        return std::nullopt;
      }
      auto& peu_waves = block.wave_indices_per_peu[peu_index];
      if (peu_waves.empty()) {
        return std::nullopt;
      }
      const size_t start = block.next_wave_rr_per_peu[peu_index] % peu_waves.size();
      for (size_t offset = 0; offset < peu_waves.size(); ++offset) {
        const size_t local_index = (start + offset) % peu_waves.size();
        const size_t wave_index = peu_waves[local_index];
        const auto& wave = block.waves[wave_index];
        if (wave.status == WaveStatus::Active && !wave.waiting_at_barrier) {
          block.next_wave_rr_per_peu[peu_index] = (local_index + 1) % peu_waves.size();
          return wave_index;
        }
      }
      return std::nullopt;
    };

    while (has_active_wave()) {
      bool made_progress = false;

      release_block_barrier_if_ready();

      for (size_t peu_index = 0; peu_index < block.wave_indices_per_peu.size(); ++peu_index) {
        const auto wave_index = select_next_wave_index_for_peu(peu_index);
        if (!wave_index.has_value()) {
          continue;
        }
        auto& wave = block.waves[*wave_index];
        if (wave.pc >= context_.kernel.instructions().size()) {
          throw std::out_of_range("wave pc out of range");
        }

        const Instruction& instruction = context_.kernel.instructions().at(wave.pc);
        ++stats.wave_steps;
        ++stats.instructions_issued;
        TraceEventLocked(TraceEvent{
            .kind = TraceEventKind::WaveStep,
            .cycle = context_.cycle,
            .block_id = wave.block_id,
            .wave_id = wave.wave_id,
            .pc = wave.pc,
            .message = FormatWaveStepMessage(instruction, wave),
        });

        ExecutionContext block_context = context_;
        block_context.stats = nullptr;
        const OpPlan plan = semantics_.BuildPlan(instruction, wave, block_context);

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
          TraceEventLocked(TraceEvent{
              .kind = TraceEventKind::ExecMaskUpdate,
              .cycle = context_.cycle,
              .block_id = wave.block_id,
              .wave_id = wave.wave_id,
              .pc = wave.pc,
              .message = mask_text.str(),
          });
        }
        if (plan.memory.has_value()) {
          const auto& request = *plan.memory;
          ++stats.memory_ops;
          if (request.space == MemorySpace::Global) {
            if (request.kind == AccessKind::Load) {
              ++stats.global_loads;
            } else if (request.kind == AccessKind::Store || request.kind == AccessKind::Atomic) {
              ++stats.global_stores;
            }
          } else if (request.space == MemorySpace::Shared) {
            if (request.kind == AccessKind::Load) {
              ++stats.shared_loads;
            } else if (request.kind == AccessKind::Store || request.kind == AccessKind::Atomic) {
              ++stats.shared_stores;
            }
          } else if (request.space == MemorySpace::Private) {
            if (request.kind == AccessKind::Load) {
              ++stats.private_loads;
            } else if (request.kind == AccessKind::Store) {
              ++stats.private_stores;
            }
          } else if (request.space == MemorySpace::Constant && request.kind == AccessKind::Load) {
            ++stats.constant_loads;
          }

          TraceEventLocked(TraceEvent{
              .kind = TraceEventKind::MemoryAccess,
              .cycle = context_.cycle,
              .block_id = wave.block_id,
              .wave_id = wave.wave_id,
              .pc = wave.pc,
              .message = request.kind == AccessKind::Load ? "load" : "store",
          });

          if (request.kind == AccessKind::Load) {
            if (!request.dst.has_value()) {
              throw std::invalid_argument("load request missing destination");
            }
            if (request.dst->file == RegisterFile::Scalar) {
              uint64_t loaded_value = 0;
              for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
                if (!request.lanes[lane].active) {
                  continue;
                }
                if (request.space == MemorySpace::Constant) {
                  loaded_value = LoadConstantLaneValue(request.lanes[lane]);
                } else {
                  throw std::invalid_argument("scalar load supports constant memory only");
                }
                break;
              }
              wave.sgpr.Write(request.dst->index, loaded_value);
            } else {
              std::array<uint64_t, 64> loaded_values{};
              for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
                if (!request.lanes[lane].active) {
                  continue;
                }
                if (request.space == MemorySpace::Global) {
                  loaded_values[lane] = LoadGlobalLaneValue(request.lanes[lane]);
                } else if (request.space == MemorySpace::Shared) {
                  loaded_values[lane] = LoadLaneValue(block.shared_memory, request.lanes[lane]);
                } else if (request.space == MemorySpace::Private) {
                  loaded_values[lane] =
                      LoadLaneValue(wave.private_memory, lane, request.lanes[lane]);
                } else if (request.space == MemorySpace::Constant) {
                  loaded_values[lane] = LoadConstantLaneValue(request.lanes[lane]);
                } else {
                  throw std::invalid_argument("unsupported load memory space");
                }
              }
              for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
                if (request.exec_snapshot.test(lane)) {
                  wave.vgpr.Write(request.dst->index, lane, loaded_values[lane]);
                }
              }
            }
          } else if (request.kind == AccessKind::Store) {
            for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
              if (!request.lanes[lane].active) {
                continue;
              }
              if (request.space == MemorySpace::Global) {
                StoreGlobalLaneValue(request.lanes[lane]);
              } else if (request.space == MemorySpace::Shared) {
                StoreLaneValue(block.shared_memory, request.lanes[lane]);
              } else if (request.space == MemorySpace::Private) {
                StoreLaneValue(wave.private_memory, lane, request.lanes[lane]);
              } else {
                throw std::invalid_argument("unsupported store memory space");
              }
            }
          } else if (request.kind == AccessKind::Atomic) {
            for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
              if (!request.lanes[lane].active) {
                continue;
              }
              if (request.space == MemorySpace::Shared) {
                int32_t prior = static_cast<int32_t>(LoadLaneValue(block.shared_memory, request.lanes[lane]));
                const int32_t updated = prior + static_cast<int32_t>(request.lanes[lane].value);
                LaneAccess writeback = request.lanes[lane];
                writeback.value = static_cast<uint64_t>(static_cast<int64_t>(updated));
                StoreLaneValue(block.shared_memory, writeback);
              } else if (request.space == MemorySpace::Global) {
                std::lock_guard<std::mutex> lock(global_memory_mutex_);
                int32_t prior = context_.memory.LoadGlobalValue<int32_t>(request.lanes[lane].addr);
                const int32_t updated = prior + static_cast<int32_t>(request.lanes[lane].value);
                context_.memory.StoreGlobalValue<int32_t>(request.lanes[lane].addr, updated);
              } else {
                throw std::invalid_argument("unsupported atomic memory space");
              }
            }
          } else {
            throw std::invalid_argument("unsupported memory access kind in functional executor");
          }
        }

        if (plan.sync_wave_barrier) {
          ++stats.barriers;
          TraceEventLocked(TraceEvent{
              .kind = TraceEventKind::Barrier,
              .cycle = context_.cycle,
              .block_id = wave.block_id,
              .wave_id = wave.wave_id,
              .pc = wave.pc,
              .message = "wave",
          });
          ++wave.pc;
        } else if (plan.sync_barrier) {
          ++stats.barriers;
          wave.status = WaveStatus::Stalled;
          wave.waiting_at_barrier = true;
          wave.barrier_generation = block.barrier_generation;
          ++block.barrier_arrivals;
          TraceEventLocked(TraceEvent{
              .kind = TraceEventKind::Barrier,
              .cycle = context_.cycle,
              .block_id = wave.block_id,
              .wave_id = wave.wave_id,
              .pc = wave.pc,
              .message = "arrive",
          });

          release_block_barrier_if_ready();
        } else if (plan.exit_wave) {
          ++stats.wave_exits;
          wave.status = WaveStatus::Exited;
          TraceEventLocked(TraceEvent{
              .kind = TraceEventKind::WaveExit,
              .cycle = context_.cycle,
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

      if (!made_progress) {
        throw std::runtime_error("functional execution stalled without progress");
      }
    }
  }
};

}  // namespace

uint64_t FunctionalExecutionCore::RunSequential() {
  FunctionalExecutionCoreImpl core(context_);
  return core.RunSequential();
}

uint64_t FunctionalExecutionCore::RunParallelBlocks(uint32_t worker_threads) {
  FunctionalExecutionCoreImpl core(context_);
  return core.RunParallelBlocks(worker_threads);
}

}  // namespace gpu_model
