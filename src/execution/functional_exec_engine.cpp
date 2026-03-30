#include "gpu_model/execution/functional_exec_engine.h"

#include <array>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

#ifdef GPU_MODEL_HAS_MARL
#include "marl/scheduler.h"
#include "marl/waitgroup.h"
#endif

#include "gpu_model/debug/instruction_trace.h"
#include "gpu_model/debug/trace_event.h"
#include "gpu_model/debug/wave_launch_trace.h"
#include "gpu_model/execution/memory_ops.h"
#include "gpu_model/execution/plan_apply.h"
#include "gpu_model/execution/sync_ops.h"
#include "gpu_model/execution/wave_context_builder.h"
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
  std::vector<WaveContext> waves;
  std::vector<std::vector<size_t>> wave_indices_per_peu;
  std::vector<size_t> next_wave_rr_per_peu;
  std::vector<bool> wave_busy;
  std::unique_ptr<std::mutex> control_mutex;
  std::unique_ptr<std::condition_variable> control_cv;
  std::unique_ptr<std::mutex> shared_mutex;
};

uint64_t LoadLaneValue(const std::vector<std::byte>& memory, const LaneAccess& lane) {
  return memory_ops::LoadByteLaneValue(memory, lane);
}

void StoreLaneValue(std::vector<std::byte>& memory, const LaneAccess& lane) {
  memory_ops::StoreByteLaneValue(memory, lane);
}

uint64_t LoadLaneValue(std::array<std::vector<std::byte>, kWaveSize>& memory,
                       uint32_t lane_id,
                       const LaneAccess& lane) {
  return memory_ops::LoadPrivateLaneValue(memory, lane_id, lane, true);
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
        .barrier_generation = shared_block.barrier_generation,
        .barrier_arrivals = shared_block.barrier_arrivals,
        .shared_memory = shared_block.shared_memory,
        .waves = shared_block.waves,
        .wave_indices_per_peu = {},
        .next_wave_rr_per_peu = {},
        .wave_busy = {},
        .control_mutex = std::make_unique<std::mutex>(),
        .control_cv = std::make_unique<std::condition_variable>(),
        .shared_mutex = std::make_unique<std::mutex>(),
    };
    for (size_t wave_index = 0; wave_index < block.waves.size(); ++wave_index) {
      const auto peu_id = block.waves[wave_index].peu_id;
      if (block.wave_indices_per_peu.size() <= peu_id) {
        block.wave_indices_per_peu.resize(static_cast<size_t>(peu_id) + 1);
        block.next_wave_rr_per_peu.resize(static_cast<size_t>(peu_id) + 1, 0);
      }
      block.wave_indices_per_peu[peu_id].push_back(wave_index);
      block.wave_busy.push_back(false);
    }
    blocks.push_back(std::move(block));
  }

  return blocks;
}

uint64_t LoadLaneValue(const MemorySystem& memory, MemoryPoolKind pool, const LaneAccess& lane) {
  return memory_ops::LoadPoolLaneValue(memory, pool, lane);
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

void MarkWaveWaitingAtBarrier(WaveContext& wave, uint64_t barrier_generation) {
  wave.status = WaveStatus::Stalled;
  wave.waiting_at_barrier = true;
  wave.barrier_generation = barrier_generation;
  wave.run_state = WaveRunState::Waiting;
  wave.wait_reason = WaveWaitReason::BlockBarrier;
}

void MarkWaveCompleted(WaveContext& wave) {
  wave.status = WaveStatus::Exited;
  wave.waiting_at_barrier = false;
  wave.run_state = WaveRunState::Completed;
  wave.wait_reason = WaveWaitReason::None;
}

class FunctionalExecutionCoreImpl {
 public:
  explicit FunctionalExecutionCoreImpl(ExecutionContext& context)
      : context_(context), semantics_(), blocks_(MaterializeBlocks(context.placement, context.launch_config)) {}

  uint64_t RunSequential() {
    EmitWaveLaunchEvents();
    for (auto& block : blocks_) {
      ExecutionStats block_stats;
      ExecuteBlock(block, block_stats);
      CommitStats(block_stats);
    }
    return 0;
  }

  uint64_t RunParallelBlocks(uint32_t worker_threads) {
    EmitWaveLaunchEvents();
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
      const size_t peu_count = std::max<size_t>(1, blocks_[i].wave_indices_per_peu.size());
      marl::schedule([&, i, peu_count] {
        marl::WaitGroup block_done(static_cast<int>(peu_count));
        for (size_t peu_index = 0; peu_index < peu_count; ++peu_index) {
          marl::schedule([&, i, peu_index] {
            ExecutionStats peu_stats;
            try {
              while (true) {
                std::optional<size_t> wave_index;
                {
                  std::unique_lock<std::mutex> lock(*blocks_[i].control_mutex);
                  while (true) {
                    if (IsBlockComplete(blocks_[i])) {
                      break;
                    }
                    ReleaseBlockBarrierIfReady(blocks_[i]);
                    wave_index = SelectNextWaveIndexForPeu(blocks_[i], peu_index);
                    if (wave_index.has_value()) {
                      blocks_[i].wave_busy[*wave_index] = true;
                      break;
                    }
                    blocks_[i].control_cv->wait(lock);
                  }
                }
                if (!wave_index.has_value()) {
                  break;
                }
                ExecuteWave(blocks_[i], blocks_[i].waves[*wave_index], peu_stats);
                {
                  std::lock_guard<std::mutex> lock(*blocks_[i].control_mutex);
                  blocks_[i].wave_busy[*wave_index] = false;
                  ReleaseBlockBarrierIfReady(blocks_[i]);
                }
                blocks_[i].control_cv->notify_all();
              }
            } catch (...) {
              std::lock_guard<std::mutex> lock(failure_mutex);
              if (failure == nullptr) {
                failure = std::current_exception();
              }
            }
            CommitStats(peu_stats);
            block_done.done();
          });
        }
        block_done.wait();
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

  void EmitWaveLaunchEvents() {
    for (const auto& block : blocks_) {
      for (const auto& wave : block.waves) {
        TraceEventLocked(TraceEvent{
            .kind = TraceEventKind::WaveLaunch,
            .cycle = context_.cycle,
            .dpc_id = wave.dpc_id,
            .ap_id = wave.ap_id,
            .peu_id = wave.peu_id,
            .block_id = wave.block_id,
            .wave_id = wave.wave_id,
            .pc = wave.pc,
            .message = FormatWaveLaunchTraceMessage(wave),
        });
      }
    }
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
    while (HasActiveWave(block)) {
      bool made_progress = false;
      {
        std::lock_guard<std::mutex> lock(*block.control_mutex);
        ReleaseBlockBarrierIfReady(block);
      }

      for (size_t peu_index = 0; peu_index < block.wave_indices_per_peu.size(); ++peu_index) {
        std::optional<size_t> wave_index;
        {
          std::lock_guard<std::mutex> lock(*block.control_mutex);
          wave_index = SelectNextWaveIndexForPeu(block, peu_index);
          if (wave_index.has_value()) {
            block.wave_busy[*wave_index] = true;
          }
        }
        if (!wave_index.has_value()) {
          continue;
        }
        ExecuteWave(block, block.waves[*wave_index], stats);
        {
          std::lock_guard<std::mutex> lock(*block.control_mutex);
          block.wave_busy[*wave_index] = false;
          ReleaseBlockBarrierIfReady(block);
        }
        made_progress = true;
      }

      if (!made_progress) {
        throw std::runtime_error("functional execution stalled without progress");
      }
    }
  }

  bool HasActiveWave(const ExecutableBlock& block) const {
    for (const auto& wave : block.waves) {
      if (wave.run_state != WaveRunState::Completed) {
        return true;
      }
    }
    return false;
  }

  bool IsBlockComplete(const ExecutableBlock& block) const {
    for (size_t i = 0; i < block.waves.size(); ++i) {
      if (block.wave_busy[i]) {
        return false;
      }
      if (block.waves[i].status != WaveStatus::Exited) {
        return false;
      }
    }
    return true;
  }

  std::string FormatBlockState(const ExecutableBlock& block) const {
    std::ostringstream oss;
    oss << "block=" << block.block_id << " barrier_gen=" << block.barrier_generation
        << " barrier_arrivals=" << block.barrier_arrivals;
    for (size_t i = 0; i < block.waves.size(); ++i) {
      const auto& wave = block.waves[i];
      oss << " | wave=" << i << " peu=" << wave.peu_id << " pc=" << wave.pc
          << " busy=" << (block.wave_busy[i] ? 1 : 0)
          << " wait_barrier=" << (wave.waiting_at_barrier ? 1 : 0)
          << " barrier_gen=" << wave.barrier_generation << " status=";
      switch (wave.status) {
        case WaveStatus::Active:
          oss << "active";
          break;
        case WaveStatus::Exited:
          oss << "exited";
          break;
        case WaveStatus::Stalled:
          oss << "stalled";
          break;
      }
    }
    return oss.str();
  }

  bool ReleaseBlockBarrierIfReady(ExecutableBlock& block) {
    if (!TryResumeBlockedWaves(block)) {
      return false;
    }
    TraceEventLocked(TraceEvent{
        .kind = TraceEventKind::Barrier,
        .cycle = context_.cycle,
        .block_id = block.block_id,
        .message = "release",
    });
    return true;
  }

  bool TryResumeBlockedWaves(ExecutableBlock& block) {
    std::vector<WaveContext*> blocked;
    blocked.reserve(block.waves.size());
    for (auto& wave : block.waves) {
      if (wave.run_state == WaveRunState::Waiting &&
          wave.wait_reason == WaveWaitReason::BlockBarrier) {
        blocked.push_back(&wave);
      }
    }
    if (blocked.empty()) {
      return false;
    }
    if (!sync_ops::ReleaseBarrierIfReady(
            block.waves, block.barrier_generation, block.barrier_arrivals, 1, false)) {
      return false;
    }
    for (auto* wave : blocked) {
      wave->waiting_at_barrier = false;
      wave->run_state = WaveRunState::Runnable;
      wave->wait_reason = WaveWaitReason::None;
    }
    return true;
  }

  std::optional<size_t> SelectNextWaveIndexForPeu(ExecutableBlock& block, size_t peu_index) {
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
      if (wave.status == WaveStatus::Active && wave.run_state == WaveRunState::Runnable &&
          !wave.waiting_at_barrier && !block.wave_busy[wave_index]) {
        block.next_wave_rr_per_peu[peu_index] = (local_index + 1) % peu_waves.size();
        return wave_index;
      }
    }
    return std::nullopt;
  }

  void ExecuteWave(ExecutableBlock& block, WaveContext& wave, ExecutionStats& stats) {
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

    ApplyExecutionPlanRegisterWrites(plan, wave);
    if (const auto mask_text = MaybeFormatExecutionMaskUpdate(plan, wave); mask_text.has_value()) {
      TraceEventLocked(TraceEvent{
          .kind = TraceEventKind::ExecMaskUpdate,
          .cycle = context_.cycle,
          .block_id = wave.block_id,
          .wave_id = wave.wave_id,
          .pc = wave.pc,
          .message = *mask_text,
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
              std::lock_guard<std::mutex> lock(*block.shared_mutex);
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
            std::lock_guard<std::mutex> lock(*block.shared_mutex);
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
            std::lock_guard<std::mutex> lock(*block.shared_mutex);
            int32_t prior =
                static_cast<int32_t>(LoadLaneValue(block.shared_memory, request.lanes[lane]));
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
      wave.run_state = WaveRunState::Runnable;
      wave.wait_reason = WaveWaitReason::None;
    } else if (plan.sync_barrier) {
      ++stats.barriers;
      {
        std::lock_guard<std::mutex> lock(*block.control_mutex);
        sync_ops::MarkWaveAtBarrier(
            wave, block.barrier_generation, block.barrier_arrivals, false);
        MarkWaveWaitingAtBarrier(wave, block.barrier_generation);
      }
      TraceEventLocked(TraceEvent{
          .kind = TraceEventKind::Barrier,
          .cycle = context_.cycle,
          .block_id = wave.block_id,
          .wave_id = wave.wave_id,
          .pc = wave.pc,
          .message = "arrive",
      });
    } else if (plan.exit_wave) {
      ++stats.wave_exits;
      {
        std::lock_guard<std::mutex> lock(*block.control_mutex);
        ApplyExecutionPlanControlFlow(plan, wave, false, false);
        MarkWaveCompleted(wave);
      }
      TraceEventLocked(TraceEvent{
          .kind = TraceEventKind::WaveExit,
          .cycle = context_.cycle,
          .block_id = wave.block_id,
          .wave_id = wave.wave_id,
          .pc = wave.pc,
          .message = "exit",
      });
    } else {
      ApplyExecutionPlanControlFlow(plan, wave, false, false);
      wave.run_state = WaveRunState::Runnable;
      wave.wait_reason = WaveWaitReason::None;
    }
  }

};

}  // namespace

uint64_t FunctionalExecEngine::RunSequential() {
  FunctionalExecutionCoreImpl core(context_);
  return core.RunSequential();
}

uint64_t FunctionalExecEngine::RunParallelBlocks(uint32_t worker_threads) {
  FunctionalExecutionCoreImpl core(context_);
  return core.RunParallelBlocks(worker_threads);
}

}  // namespace gpu_model
