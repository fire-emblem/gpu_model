#include "gpu_model/execution/functional_exec_engine.h"

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <exception>
#include <functional>
#include <future>
#include <map>
#include <mutex>
#include <optional>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

#include "gpu_model/debug/instruction_trace.h"
#include "gpu_model/debug/trace_event.h"
#include "gpu_model/debug/trace_event_builder.h"
#include "gpu_model/debug/wave_launch_trace.h"
#include "gpu_model/execution/internal/issue_eligibility.h"
#include "gpu_model/execution/internal/opcode_execution_info.h"
#include "gpu_model/execution/memory_ops.h"
#include "gpu_model/execution/plan_apply.h"
#include "gpu_model/execution/sync_ops.h"
#include "gpu_model/execution/wave_context_builder.h"
#include "gpu_model/isa/opcode.h"
#include "gpu_model/loader/device_image_loader.h"
#include "gpu_model/runtime/program_cycle_tracker.h"

namespace gpu_model {

namespace {

constexpr uint8_t kFunctionalPendingMemoryCompletionTurns = 5;

uint32_t DefaultFunctionalParallelWorkerCount() {
  const uint32_t cpu_count = std::max(1u, std::thread::hardware_concurrency());
  return std::max(1u, (cpu_count * 9u) / 10u);
}

class GlobalFunctionalWorkerPool {
 public:
  static GlobalFunctionalWorkerPool& Instance() {
    static GlobalFunctionalWorkerPool pool(DefaultFunctionalParallelWorkerCount());
    return pool;
  }

  void EnsureWorkerCount(uint32_t desired) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (desired <= workers_.size()) {
      return;
    }
    for (size_t i = workers_.size(); i < desired; ++i) {
      workers_.emplace_back([this] { WorkerLoop(); });
    }
  }

  std::future<void> Submit(std::function<void()> task) {
    std::packaged_task<void()> packaged(std::move(task));
    auto future = packaged.get_future();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      tasks_.push(std::move(packaged));
    }
    cv_.notify_one();
    return future;
  }

  ~GlobalFunctionalWorkerPool() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto& worker : workers_) {
      if (worker.joinable()) {
        worker.join();
      }
    }
  }

 private:
  explicit GlobalFunctionalWorkerPool(uint32_t initial_workers) {
    workers_.reserve(initial_workers);
    for (uint32_t i = 0; i < initial_workers; ++i) {
      workers_.emplace_back([this] { WorkerLoop(); });
    }
  }

  void WorkerLoop() {
    while (true) {
      std::packaged_task<void()> task;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&] { return stop_ || !tasks_.empty(); });
        if (stop_ && tasks_.empty()) {
          return;
        }
        task = std::move(tasks_.front());
        tasks_.pop();
      }
      task();
    }
  }

  std::mutex mutex_;
  std::condition_variable cv_;
  bool stop_ = false;
  std::vector<std::thread> workers_;
  std::queue<std::packaged_task<void()>> tasks_;
};

struct PendingMemoryOp {
  MemoryWaitDomain domain = MemoryWaitDomain::None;
  uint8_t turns_until_complete = kFunctionalPendingMemoryCompletionTurns;
};

enum class ExecutedFlowEventKind {
  BeginWaveWork,
  CompleteWave,
};

struct ExecutedFlowWorkItem {
  ExecutedStepClass step_class = ExecutedStepClass::ScalarAlu;
  uint64_t cost_cycles = 0;
  uint64_t work_weight = 1;
};

struct ExecutedFlowEvent {
  ExecutedFlowEventKind kind = ExecutedFlowEventKind::BeginWaveWork;
  uint64_t stable_wave_id = 0;
  ExecutedFlowWorkItem work_item{};
};

struct FunctionalWaveState {
  std::deque<PendingMemoryOp> pending_memory_ops;
  std::optional<WaitCntThresholds> waiting_waitcnt_thresholds;
};

struct WaveStatsSnapshot {
  uint32_t launch = 0;
  uint32_t init = 0;
  uint32_t active = 0;
  uint32_t runnable = 0;
  uint32_t waiting = 0;
  uint32_t end = 0;
};

std::string FormatWaveStatsMessage(const WaveStatsSnapshot& stats) {
  std::ostringstream oss;
  oss << "launch=" << stats.launch;
  oss << " init=" << stats.init;
  oss << " active=" << stats.active;
  oss << " runnable=" << stats.runnable;
  oss << " waiting=" << stats.waiting;
  oss << " end=" << stats.end;
  return oss.str();
}

std::optional<WaveWaitReason> MapWaitcntStringToWaveWaitReason(std::string_view reason) {
  if (reason == "waitcnt_global") {
    return WaveWaitReason::PendingGlobalMemory;
  }
  if (reason == "waitcnt_shared") {
    return WaveWaitReason::PendingSharedMemory;
  }
  if (reason == "waitcnt_private") {
    return WaveWaitReason::PendingPrivateMemory;
  }
  if (reason == "waitcnt_scalar_buffer") {
    return WaveWaitReason::PendingScalarBufferMemory;
  }
  return std::nullopt;
}

void MarkWaveWaiting(WaveContext& wave, WaveWaitReason reason) {
  if (wave.run_state == WaveRunState::Completed) {
    return;
  }
  wave.run_state = WaveRunState::Waiting;
  wave.wait_reason = reason;
}

void ResumeWaveToRunnable(WaveContext& wave,
                          uint64_t pc_increment = 0,
                          bool clear_barrier_wait = false) {
  if (clear_barrier_wait) {
    wave.waiting_at_barrier = false;
  }
  wave.pc += pc_increment;
  wave.run_state = WaveRunState::Runnable;
  wave.wait_reason = WaveWaitReason::None;
}

std::optional<MemoryWaitDomain> MemoryWaitDomainForReason(WaveWaitReason reason) {
  switch (reason) {
    case WaveWaitReason::PendingGlobalMemory:
      return MemoryWaitDomain::Global;
    case WaveWaitReason::PendingSharedMemory:
      return MemoryWaitDomain::Shared;
    case WaveWaitReason::PendingPrivateMemory:
      return MemoryWaitDomain::Private;
    case WaveWaitReason::PendingScalarBufferMemory:
      return MemoryWaitDomain::ScalarBuffer;
    case WaveWaitReason::None:
    case WaveWaitReason::BlockBarrier:
      return std::nullopt;
  }
  return std::nullopt;
}

bool IsMemoryWaitReason(WaveWaitReason reason) {
  return MemoryWaitDomainForReason(reason).has_value();
}

bool WaitCntSatisfiedForThresholds(const WaveContext& wave, const WaitCntThresholds& thresholds) {
  return wave.pending_global_mem_ops <= thresholds.global &&
         wave.pending_shared_mem_ops <= thresholds.shared &&
         wave.pending_private_mem_ops <= thresholds.private_mem &&
         wave.pending_scalar_buffer_mem_ops <= thresholds.scalar_buffer;
}

bool IsWaveWaitSatisfied(const FunctionalWaveState& state, const WaveContext& wave) {
  if (wave.run_state != WaveRunState::Waiting || !state.waiting_waitcnt_thresholds.has_value()) {
    return false;
  }
  if (!IsMemoryWaitReason(wave.wait_reason)) {
    return false;
  }
  return WaitCntSatisfiedForThresholds(wave, *state.waiting_waitcnt_thresholds);
}

bool EnterWaitStateFromWaitcnt(const Instruction& instruction,
                               FunctionalWaveState& state,
                               WaveContext& wave) {
  const auto wait_reason = WaitCntBlockReason(wave, instruction);
  if (!wait_reason.has_value()) {
    state.waiting_waitcnt_thresholds.reset();
    return false;
  }
  const auto mapped_wait_reason = MapWaitcntStringToWaveWaitReason(*wait_reason);
  if (!mapped_wait_reason.has_value()) {
    state.waiting_waitcnt_thresholds.reset();
    return false;
  }
  state.waiting_waitcnt_thresholds = WaitCntThresholdsForInstruction(instruction);
  MarkWaveWaiting(wave, *mapped_wait_reason);
  return true;
}

bool ResumeWaveIfWaitSatisfied(FunctionalWaveState& state, WaveContext& wave) {
  if (!IsWaveWaitSatisfied(state, wave)) {
    return false;
  }
  state.waiting_waitcnt_thresholds.reset();
  ResumeWaveToRunnable(wave, /*pc_increment=*/1);
  return true;
}

void ClearWaitcntWaitState(FunctionalWaveState& state) {
  state.waiting_waitcnt_thresholds.reset();
}

bool AdvancePendingMemoryOps(FunctionalWaveState& state, WaveContext& wave) {
  bool advanced = false;
  for (auto it = state.pending_memory_ops.begin(); it != state.pending_memory_ops.end();) {
    advanced = true;
    if (it->turns_until_complete > 0) {
      --it->turns_until_complete;
    }
    if (it->turns_until_complete == 0) {
      DecrementPendingMemoryOps(wave, it->domain);
      it = state.pending_memory_ops.erase(it);
      continue;
    }
    ++it;
  }
  return advanced;
}

uint64_t StableWaveKey(const WaveContext& wave) {
  return (static_cast<uint64_t>(wave.block_id) << 32u) | wave.wave_id;
}

TraceMemoryArriveKind TraceMemoryArriveKindForWaitReason(WaveWaitReason reason) {
  switch (reason) {
    case WaveWaitReason::PendingGlobalMemory:
      return TraceMemoryArriveKind::Load;
    case WaveWaitReason::PendingSharedMemory:
      return TraceMemoryArriveKind::Shared;
    case WaveWaitReason::PendingPrivateMemory:
      return TraceMemoryArriveKind::Private;
    case WaveWaitReason::PendingScalarBufferMemory:
      return TraceMemoryArriveKind::ScalarBuffer;
    case WaveWaitReason::None:
    case WaveWaitReason::BlockBarrier:
      break;
  }
  return TraceMemoryArriveKind::Load;
}

TraceStallReason TraceStallReasonForWaitReason(WaveWaitReason reason) {
  switch (reason) {
    case WaveWaitReason::PendingGlobalMemory:
      return TraceStallReason::WaitCntGlobal;
    case WaveWaitReason::PendingSharedMemory:
      return TraceStallReason::WaitCntShared;
    case WaveWaitReason::PendingPrivateMemory:
      return TraceStallReason::WaitCntPrivate;
    case WaveWaitReason::PendingScalarBufferMemory:
      return TraceStallReason::WaitCntScalarBuffer;
    case WaveWaitReason::None:
    case WaveWaitReason::BlockBarrier:
      break;
  }
  return TraceStallReason::None;
}

std::optional<ExecutedStepClass> ClassifyExecutedInstruction(const Instruction& instruction,
                                                             const OpPlan& plan) {
  if (plan.sync_barrier || plan.sync_wave_barrier) {
    return ExecutedStepClass::Barrier;
  }
  if (plan.wait_cnt) {
    return ExecutedStepClass::Wait;
  }
  if (plan.memory.has_value()) {
    switch (plan.memory->space) {
      case MemorySpace::Global:
        return ExecutedStepClass::GlobalMem;
      case MemorySpace::Shared:
        return ExecutedStepClass::SharedMem;
      case MemorySpace::Private:
        return ExecutedStepClass::PrivateMem;
      case MemorySpace::Constant:
        return ExecutedStepClass::ScalarMem;
    }
  }

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
      return ExecutedStepClass::GlobalMem;
    case SemanticFamily::LocalDataShare:
      return ExecutedStepClass::SharedMem;
    case SemanticFamily::Builtin:
    case SemanticFamily::Mask:
    case SemanticFamily::Branch:
    case SemanticFamily::Sync:
    case SemanticFamily::Special:
      return std::nullopt;
  }
  return std::nullopt;
}

uint64_t CostForExecutedStep(const OpPlan& plan,
                             ExecutedStepClass step_class,
                             const ProgramCycleStatsConfig& config) {
  switch (step_class) {
    case ExecutedStepClass::ScalarAlu:
    case ExecutedStepClass::VectorAlu:
      return plan.issue_cycles;
    case ExecutedStepClass::Tensor:
      return config.tensor_cycles;
    case ExecutedStepClass::SharedMem:
      return config.shared_mem_cycles;
    case ExecutedStepClass::ScalarMem:
      return config.scalar_mem_cycles;
    case ExecutedStepClass::GlobalMem:
      return config.global_mem_cycles;
    case ExecutedStepClass::PrivateMem:
      return config.private_mem_cycles;
    case ExecutedStepClass::Barrier:
    case ExecutedStepClass::Wait:
      return plan.issue_cycles == 0 ? config.default_issue_cycles : plan.issue_cycles;
  }
  return config.default_issue_cycles;
}

class ExecutedFlowEventSource final : public ProgramCycleTickSource {
 public:
  explicit ExecutedFlowEventSource(const std::vector<ExecutedFlowEvent>& events) {
    wave_states_.reserve(events.size());
    for (const auto& event : events) {
      auto& wave = GetOrCreateWaveState(event.stable_wave_id);
      if (event.kind == ExecutedFlowEventKind::BeginWaveWork) {
        wave.work_items.push_back(event.work_item);
      } else if (event.kind == ExecutedFlowEventKind::CompleteWave) {
        wave.saw_completion = true;
      }
    }
  }

  bool Done() const override {
    for (const auto& wave : wave_states_) {
      if (!wave.completed) {
        return false;
      }
    }
    return true;
  }

  void AdvanceOneTick(ProgramCycleTracker& agg) override {
    for (auto& wave : wave_states_) {
      if (!wave.active) {
        continue;
      }
      ++wave.ticks_consumed;
      if (wave.ticks_consumed >= wave.current_cost_cycles) {
        wave.active = false;
        wave.ticks_consumed = 0;
        wave.current_cost_cycles = 0;
        agg.MarkWaveRunnable(wave.agg_wave_id);
      }
    }

    for (auto& wave : wave_states_) {
      if (wave.completed || wave.active) {
        continue;
      }
      if (!wave.work_items.empty()) {
        const auto step = wave.work_items.front();
        wave.work_items.pop_front();
        wave.active = true;
        wave.current_cost_cycles = step.cost_cycles;
        wave.ticks_consumed = 0;
        agg.BeginWaveWork(wave.agg_wave_id,
                          step.step_class,
                          step.cost_cycles,
                          step.work_weight);
        continue;
      }
      if (wave.saw_completion) {
        wave.completed = true;
        agg.MarkWaveCompleted(wave.agg_wave_id);
      }
    }
  }

 private:
  struct WaveQueueState {
    uint32_t agg_wave_id = 0;
    std::deque<ExecutedFlowWorkItem> work_items;
    bool active = false;
    bool completed = false;
    bool saw_completion = false;
    uint64_t current_cost_cycles = 0;
    uint64_t ticks_consumed = 0;
  };

  WaveQueueState& GetOrCreateWaveState(uint64_t stable_wave_id) {
    const auto it = wave_state_indices_.find(stable_wave_id);
    if (it != wave_state_indices_.end()) {
      return wave_states_[it->second];
    }
    const uint32_t agg_wave_id = static_cast<uint32_t>(wave_states_.size());
    wave_state_indices_.emplace(stable_wave_id, wave_states_.size());
    wave_states_.push_back(WaveQueueState{
        .agg_wave_id = agg_wave_id,
        .work_items = {},
        .active = false,
        .completed = false,
        .saw_completion = false,
        .current_cost_cycles = 0,
        .ticks_consumed = 0,
    });
    return wave_states_.back();
  }

  std::unordered_map<uint64_t, size_t> wave_state_indices_;
  std::vector<WaveQueueState> wave_states_;
};

ProgramCycleStats CollectProgramCycleStatsFromExecutedFlow(
    const std::vector<ExecutedFlowEvent>& events,
    const ProgramCycleStatsConfig&) {
  ProgramCycleTracker agg;
  ExecutedFlowEventSource source(events);
  while (!source.Done()) {
    source.AdvanceOneTick(agg);
    agg.AdvanceOneTick();
  }
  return agg.Finish();
}

void RecordPendingMemoryOp(FunctionalWaveState& state,
                           WaveContext& wave,
                           MemoryWaitDomain domain) {
  if (domain == MemoryWaitDomain::None) {
    return;
  }
  IncrementPendingMemoryOps(wave, domain);
  state.pending_memory_ops.push_back(PendingMemoryOp{.domain = domain});
}

struct ExecutableBlock {
  uint32_t block_id = 0;
  uint32_t dpc_id = 0;
  uint32_t ap_id = 0;
  uint32_t global_ap_id = 0;
  std::vector<std::byte> shared_memory;
  std::vector<WaveContext> waves;
  std::vector<FunctionalWaveState> wave_states;
  std::vector<std::vector<size_t>> wave_indices_per_peu;
  std::vector<size_t> next_wave_rr_per_peu;
  std::vector<bool> wave_busy;
  std::unique_ptr<std::mutex> control_mutex;
  std::unique_ptr<std::mutex> wave_state_mutex;
  std::unique_ptr<std::mutex> shared_mutex;
  std::shared_ptr<struct BlockBarrierState> barrier_state;
};

struct WaveTaskRef {
  size_t block_index = 0;
  size_t wave_index = 0;
  uint32_t global_ap_id = 0;
};

struct ApSchedulerState {
  std::deque<WaveTaskRef> runnable;
  std::vector<size_t> block_indices;
};

struct BlockBarrierState {
  explicit BlockBarrierState(uint32_t expected_wave_count)
      : expected_wave_count(expected_wave_count) {}

  uint32_t expected_wave_count = 0;
  std::atomic<uint32_t> arrived_wave_count{0};
  std::atomic<uint32_t> completed_wave_count{0};
  std::atomic<uint64_t> generation{0};
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
        .global_ap_id = placement.blocks.at(shared_block.block_id).global_ap_id,
        .shared_memory = shared_block.shared_memory,
        .waves = shared_block.waves,
        .wave_states = std::vector<FunctionalWaveState>(shared_block.waves.size()),
        .wave_indices_per_peu = {},
        .next_wave_rr_per_peu = {},
        .wave_busy = {},
        .control_mutex = std::make_unique<std::mutex>(),
        .wave_state_mutex = std::make_unique<std::mutex>(),
        .shared_mutex = std::make_unique<std::mutex>(),
        .barrier_state = std::make_shared<BlockBarrierState>(
            static_cast<uint32_t>(shared_block.waves.size())),
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
  MarkWaveWaiting(wave, WaveWaitReason::BlockBarrier);
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
      : context_(context), semantics_(), blocks_(MaterializeBlocks(context.placement, context.launch_config)) {
    AssignLogicalSlotIds();
  }

  uint64_t RunSequential() {
    EmitWaveLaunchEvents();
    EmitWaveStatsSnapshot();
    for (auto& block : blocks_) {
      ExecutionStats block_stats;
      ExecuteBlock(block, block_stats);
      CommitStats(block_stats);
    }
    EmitWaveStatsSnapshot();
    program_cycle_stats_ =
        CollectProgramCycleStatsFromExecutedFlow(executed_flow_events_, cycle_stats_config_);
    return program_cycle_stats_->total_cycles;
  }

  uint64_t RunParallelBlocks(uint32_t worker_threads) {
    EmitWaveLaunchEvents();
    EmitWaveStatsSnapshot();
    BuildParallelWaveSchedulerState();
    const uint32_t actual_workers =
        worker_threads == 0 ? DefaultFunctionalParallelWorkerCount() : worker_threads;
    auto& pool = GlobalFunctionalWorkerPool::Instance();
    pool.EnsureWorkerCount(actual_workers);
    std::exception_ptr failure;
    std::mutex failure_mutex;
    std::vector<std::future<void>> futures;
    futures.reserve(actual_workers);

    for (uint32_t worker = 0; worker < actual_workers; ++worker) {
      futures.push_back(pool.Submit([&, worker] {
          (void)worker;
          ExecutionStats block_stats;
          try {
            WorkerRunParallelWaves(block_stats, failure, failure_mutex);
          } catch (...) {
            std::lock_guard<std::mutex> lock(failure_mutex);
            if (failure == nullptr) {
              failure = std::current_exception();
            }
          }
          scheduler_cv_.notify_all();
          CommitStats(block_stats);
        }));
    }

    for (auto& future : futures) {
      future.get();
    }
    if (failure != nullptr) {
      std::rethrow_exception(failure);
    }
    EmitWaveStatsSnapshot();
    program_cycle_stats_ =
        CollectProgramCycleStatsFromExecutedFlow(executed_flow_events_, cycle_stats_config_);
    return program_cycle_stats_->total_cycles;
  }

  std::optional<ProgramCycleStats> TakeProgramCycleStats() const {
    return program_cycle_stats_;
  }

 private:
  ExecutionContext& context_;
  Semantics semantics_;
  std::vector<ExecutableBlock> blocks_;
  std::mutex trace_mutex_;
  std::mutex stats_mutex_;
  std::mutex global_memory_mutex_;
  std::mutex executed_flow_mutex_;
  std::mutex scheduler_mutex_;
  std::mutex peu_schedule_trace_mutex_;
  std::condition_variable scheduler_cv_;
  std::vector<ApSchedulerState> ap_schedulers_;
  size_t next_ap_rr_ = 0;
  size_t total_waves_ = 0;
  size_t completed_waves_ = 0;
  size_t active_wave_tasks_ = 0;
  ProgramCycleStatsConfig cycle_stats_config_{};
  std::vector<ExecutedFlowEvent> executed_flow_events_;
  std::optional<ProgramCycleStats> program_cycle_stats_;
  std::atomic<uint64_t> trace_cycle_{0};
  std::unordered_map<uint64_t, uint32_t> logical_slot_ids_;
  std::unordered_map<uint64_t, uint64_t> last_wave_tag_per_ap_peu_;

  uint64_t NextTraceCycle() { return trace_cycle_.fetch_add(1, std::memory_order_relaxed); }

  void AssignLogicalSlotIds() {
    std::map<std::tuple<uint32_t, uint32_t, uint32_t>, uint32_t> next_slot_per_peu;
    for (const auto& block : blocks_) {
      for (const auto& wave : block.waves) {
        auto& next_slot = next_slot_per_peu[std::make_tuple(wave.dpc_id, wave.ap_id, wave.peu_id)];
        logical_slot_ids_.emplace(StableWaveKey(wave), next_slot++);
      }
    }
  }

  uint32_t TraceSlotId(const WaveContext& wave) const {
    const auto it = logical_slot_ids_.find(StableWaveKey(wave));
    return it == logical_slot_ids_.end() ? 0u : it->second;
  }

  void MaybeEmitWaveSwitchAwayEvent(const ExecutableBlock& block,
                                    const WaveContext& wave,
                                    uint64_t pc) {
    const uint64_t ap_peu_key =
        (static_cast<uint64_t>(block.global_ap_id) << 32u) | static_cast<uint64_t>(wave.peu_id);
    const uint64_t wave_tag =
        (static_cast<uint64_t>(wave.block_id) << 32u) | static_cast<uint64_t>(wave.wave_id);

    bool emit_switch_away = false;
    {
      std::lock_guard<std::mutex> lock(peu_schedule_trace_mutex_);
      const auto it = last_wave_tag_per_ap_peu_.find(ap_peu_key);
      emit_switch_away = it != last_wave_tag_per_ap_peu_.end() && it->second != wave_tag;
      last_wave_tag_per_ap_peu_[ap_peu_key] = wave_tag;
    }

    if (emit_switch_away) {
      TraceEventLocked(MakeTraceWaveSwitchStallEvent(
          TraceWaveView{.dpc_id = wave.dpc_id,
                        .ap_id = wave.ap_id,
                        .peu_id = wave.peu_id,
                        .slot_id = TraceSlotId(wave),
                        .block_id = wave.block_id,
                        .wave_id = wave.wave_id,
                        .pc = wave.pc},
          NextTraceCycle(),
          TraceSlotModelKind::LogicalUnbounded,
          pc));
    }
  }

  TraceEvent MakeWaveTraceEvent(const WaveContext& wave,
                                TraceEventKind kind,
                                uint64_t cycle,
                                std::string message,
                                uint64_t pc = std::numeric_limits<uint64_t>::max()) const {
    return MakeTraceWaveEvent(
        TraceWaveView{.dpc_id = wave.dpc_id,
                      .ap_id = wave.ap_id,
                      .peu_id = wave.peu_id,
                      .slot_id = TraceSlotId(wave),
                      .block_id = wave.block_id,
                      .wave_id = wave.wave_id,
                      .pc = wave.pc},
        kind,
        cycle,
        TraceSlotModelKind::LogicalUnbounded,
        std::move(message),
        pc);
  }

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

  void RecordExecutedWorkEvent(const WaveContext& wave,
                               ExecutedStepClass step_class,
                               uint64_t cost_cycles) {
    if (cost_cycles == 0) {
      return;
    }
    const uint64_t work_weight = wave.exec.count();
    std::lock_guard<std::mutex> lock(executed_flow_mutex_);
    executed_flow_events_.push_back(ExecutedFlowEvent{
        .kind = ExecutedFlowEventKind::BeginWaveWork,
        .stable_wave_id = StableWaveKey(wave),
        .work_item =
            ExecutedFlowWorkItem{
                .step_class = step_class,
                .cost_cycles = cost_cycles,
                .work_weight = work_weight,
            },
    });
  }

  void RecordExecutedCompletionEvent(const WaveContext& wave) {
    std::lock_guard<std::mutex> lock(executed_flow_mutex_);
    executed_flow_events_.push_back(ExecutedFlowEvent{
        .kind = ExecutedFlowEventKind::CompleteWave,
        .stable_wave_id = StableWaveKey(wave),
    });
  }

  void EmitWaveLaunchEvents() {
    for (const auto& block : blocks_) {
      for (const auto& wave : block.waves) {
        TraceEventLocked(MakeTraceWaveLaunchEvent(
            TraceWaveView{.dpc_id = wave.dpc_id,
                          .ap_id = wave.ap_id,
                          .peu_id = wave.peu_id,
                          .slot_id = TraceSlotId(wave),
                          .block_id = wave.block_id,
                          .wave_id = wave.wave_id,
                          .pc = wave.pc},
            NextTraceCycle(),
            FormatWaveLaunchTraceMessage(wave),
            TraceSlotModelKind::LogicalUnbounded));
      }
    }
  }

  WaveStatsSnapshot CaptureWaveStatsSnapshot() const {
    WaveStatsSnapshot stats;
    for (const auto& block : blocks_) {
      std::lock_guard<std::mutex> state_lock(*block.wave_state_mutex);
      for (const auto& wave : block.waves) {
        ++stats.launch;
        // Task 2 scope: "init" currently mirrors the count of materialized waves.
        ++stats.init;
        switch (wave.run_state) {
          case WaveRunState::Runnable:
            ++stats.runnable;
            break;
          case WaveRunState::Waiting:
            ++stats.waiting;
            break;
          case WaveRunState::Completed:
            ++stats.end;
            break;
        }
      }
    }
    stats.active = stats.runnable + stats.waiting;
    return stats;
  }

  void EmitWaveStatsSnapshot() {
    TraceEventLocked(MakeTraceEvent(TraceEventKind::WaveStats,
                                    NextTraceCycle(),
                                    FormatWaveStatsMessage(CaptureWaveStatsSnapshot())));
  }

  void BuildParallelWaveSchedulerState() {
    std::lock_guard<std::mutex> lock(scheduler_mutex_);
    next_ap_rr_ = 0;
    completed_waves_ = 0;
    total_waves_ = 0;
    active_wave_tasks_ = 0;

    uint32_t max_global_ap_id = 0;
    for (const auto& block : blocks_) {
      max_global_ap_id = std::max(max_global_ap_id, block.global_ap_id);
      total_waves_ += block.waves.size();
    }

    ap_schedulers_.clear();
    ap_schedulers_.resize(static_cast<size_t>(max_global_ap_id) + 1);
    for (size_t block_index = 0; block_index < blocks_.size(); ++block_index) {
      auto& block = blocks_[block_index];
      ap_schedulers_[block.global_ap_id].block_indices.push_back(block_index);
      std::lock_guard<std::mutex> state_lock(*block.wave_state_mutex);
      for (size_t wave_index = 0; wave_index < block.waves.size(); ++wave_index) {
        auto& wave = block.waves[wave_index];
        if (wave.run_state == WaveRunState::Completed || wave.status == WaveStatus::Exited) {
          ++completed_waves_;
        }
        block.wave_busy[wave_index] = false;
      }
    }
    for (size_t block_index = 0; block_index < blocks_.size(); ++block_index) {
      (void)RequeueRunnableWavesForBlockLocked(block_index);
    }
  }

  bool PopRunnableWaveLocked(WaveTaskRef& task) {
    if (ap_schedulers_.empty()) {
      return false;
    }
    for (size_t offset = 0; offset < ap_schedulers_.size(); ++offset) {
      const size_t ap_index = (next_ap_rr_ + offset) % ap_schedulers_.size();
      auto& ap = ap_schedulers_[ap_index];
      if (ap.runnable.empty()) {
        continue;
      }
      task = ap.runnable.front();
      ap.runnable.pop_front();
      next_ap_rr_ = (ap_index + 1) % ap_schedulers_.size();
      return true;
    }
    return false;
  }

  void EnqueueWaveLocked(const WaveTaskRef& task) {
    ap_schedulers_[task.global_ap_id].runnable.push_back(task);
  }

  bool RequeueRunnableWavesForBlockLocked(size_t block_index) {
    auto& block = blocks_[block_index];
    std::lock_guard<std::mutex> state_lock(*block.wave_state_mutex);
    bool enqueued = false;
    for (size_t wave_index = 0; wave_index < block.waves.size(); ++wave_index) {
      auto& wave = block.waves[wave_index];
      if (block.wave_busy[wave_index]) {
        continue;
      }
      if (wave.status != WaveStatus::Active ||
          wave.run_state != WaveRunState::Runnable ||
          wave.waiting_at_barrier) {
        continue;
      }
      block.wave_busy[wave_index] = true;
      EnqueueWaveLocked(WaveTaskRef{
          .block_index = block_index,
          .wave_index = wave_index,
          .global_ap_id = block.global_ap_id,
      });
      enqueued = true;
    }
    return enqueued;
  }

  bool AdvanceWaitingWavesForBlockLocked(size_t block_index) {
    bool progressed = false;
    progressed = ProcessWaitingWaves(blocks_[block_index]) || progressed;
    progressed = RequeueRunnableWavesForBlockLocked(block_index) || progressed;
    return progressed;
  }

  bool AdvanceWaitingWavesLocked() {
    bool progressed = false;
    for (auto& ap : ap_schedulers_) {
      for (const size_t block_index : ap.block_indices) {
        progressed = AdvanceWaitingWavesForBlockLocked(block_index) || progressed;
      }
    }
    return progressed;
  }

  bool AllParallelWavesCompletedLocked() const {
    return completed_waves_ >= total_waves_;
  }

  void ReconcileWaveTaskLocked(const WaveTaskRef& task) {
    auto& block = blocks_[task.block_index];
    {
      std::lock_guard<std::mutex> state_lock(*block.wave_state_mutex);
      const auto& wave = block.waves[task.wave_index];
      if (wave.run_state == WaveRunState::Completed || wave.status == WaveStatus::Exited) {
        if (block.wave_busy[task.wave_index]) {
          block.wave_busy[task.wave_index] = false;
          ++completed_waves_;
        }
      } else {
        block.wave_busy[task.wave_index] = false;
      }
    }
    (void)RequeueRunnableWavesForBlockLocked(task.block_index);
  }

  void WorkerRunParallelWaves(ExecutionStats& stats,
                              std::exception_ptr& failure,
                              std::mutex& failure_mutex) {
    while (true) {
      WaveTaskRef task;
      {
        std::unique_lock<std::mutex> lock(scheduler_mutex_);
        for (;;) {
          {
            std::lock_guard<std::mutex> failure_lock(failure_mutex);
            if (failure != nullptr) {
              return;
            }
          }
          if (AllParallelWavesCompletedLocked()) {
            return;
          }
          if (PopRunnableWaveLocked(task)) {
            ++active_wave_tasks_;
            break;
          }
          if (active_wave_tasks_ > 0) {
            scheduler_cv_.wait(lock);
            continue;
          }
          if (AdvanceWaitingWavesLocked()) {
            continue;
          }
          scheduler_cv_.wait(lock);
        }
      }

      ExecuteWave(blocks_[task.block_index], task.wave_index, stats);

      {
        std::lock_guard<std::mutex> lock(scheduler_mutex_);
        if (active_wave_tasks_ > 0) {
          --active_wave_tasks_;
        }
        ReconcileWaveTaskLocked(task);
        (void)AdvanceWaitingWavesForBlockLocked(task.block_index);
      }
      scheduler_cv_.notify_all();
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
    while (HasUncompletedWave(block)) {
      bool made_progress = false;
      {
        std::lock_guard<std::mutex> lock(*block.control_mutex);
        made_progress = ProcessWaitingWaves(block) || made_progress;
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
        ExecuteWave(block, *wave_index, stats);
        {
          std::lock_guard<std::mutex> lock(*block.control_mutex);
          block.wave_busy[*wave_index] = false;
        }
        made_progress = true;
      }

      if (!made_progress) {
        throw std::runtime_error("functional execution stalled without progress");
      }
    }
  }

  bool HasUncompletedWave(const ExecutableBlock& block) const {
    std::lock_guard<std::mutex> state_lock(*block.wave_state_mutex);
    for (const auto& wave : block.waves) {
      if (wave.run_state != WaveRunState::Completed) {
        return true;
      }
    }
    return false;
  }

  bool IsBlockComplete(const ExecutableBlock& block) const {
    std::lock_guard<std::mutex> state_lock(*block.wave_state_mutex);
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
    oss << "block=" << block.block_id
        << " barrier_gen=" << block.barrier_state->generation.load()
        << " barrier_arrivals=" << block.barrier_state->arrived_wave_count.load();
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
    std::lock_guard<std::mutex> state_lock(*block.wave_state_mutex);
    const uint32_t arrived = block.barrier_state->arrived_wave_count.load();
    const uint32_t completed = block.barrier_state->completed_wave_count.load();
    const uint64_t generation = block.barrier_state->generation.load();
    uint32_t waiting_in_generation = 0;
    for (const auto& wave : block.waves) {
      if (wave.waiting_at_barrier && wave.barrier_generation == generation &&
          wave.run_state == WaveRunState::Waiting &&
          wave.wait_reason == WaveWaitReason::BlockBarrier) {
        ++waiting_in_generation;
      }
    }
    if (waiting_in_generation == 0 ||
        arrived + completed != block.barrier_state->expected_wave_count) {
      return false;
    }
    for (auto& wave : block.waves) {
      if (wave.waiting_at_barrier && wave.barrier_generation == generation) {
        ResumeWaveToRunnable(wave, /*pc_increment=*/1, /*clear_barrier_wait=*/true);
        wave.status = WaveStatus::Active;
      }
    }
    block.barrier_state->arrived_wave_count.store(0);
    block.barrier_state->generation.store(generation + 1);
    return true;
  }

  void RecordWaitingWaveTicks(const ExecutableBlock& block) {
    std::lock_guard<std::mutex> state_lock(*block.wave_state_mutex);
    for (const auto& wave : block.waves) {
      if (wave.run_state != WaveRunState::Waiting) {
        continue;
      }
      if (wave.wait_reason == WaveWaitReason::BlockBarrier) {
        RecordExecutedWorkEvent(wave, ExecutedStepClass::Barrier, 1);
      } else if (IsMemoryWaitReason(wave.wait_reason)) {
        RecordExecutedWorkEvent(wave, ExecutedStepClass::Wait, 1);
      }
    }
  }

  bool ProcessWaitingWaves(ExecutableBlock& block) {
    bool progressed = false;
    RecordWaitingWaveTicks(block);
    progressed = AdvancePendingMemoryOps(block) || progressed;
    progressed = ResumeMemoryWaitingWaves(block) || progressed;
    const bool released_barrier = ReleaseBlockBarrierIfReady(block);
    progressed = released_barrier || progressed;
    if (released_barrier) {
      TraceEventLocked(
          MakeTraceBarrierReleaseEvent(block.dpc_id, block.ap_id, block.block_id, NextTraceCycle()));
      EmitWaveStatsSnapshot();
    }
    EmitWaitingWaveStalls(block);
    return progressed;
  }

  bool AdvancePendingMemoryOps(ExecutableBlock& block) {
    std::lock_guard<std::mutex> state_lock(*block.wave_state_mutex);
    bool advanced = false;
    for (size_t i = 0; i < block.waves.size(); ++i) {
      advanced = ::gpu_model::AdvancePendingMemoryOps(block.wave_states[i], block.waves[i]) || advanced;
    }
    return advanced;
  }

  bool ResumeMemoryWaitingWaves(ExecutableBlock& block) {
    bool resumed = false;
    {
      std::lock_guard<std::mutex> state_lock(*block.wave_state_mutex);
      for (size_t i = 0; i < block.waves.size(); ++i) {
        WaveContext& wave = block.waves[i];
        const WaveWaitReason prior_wait_reason = wave.wait_reason;
        const uint64_t prior_pc = wave.pc;
        if (!ResumeWaveIfWaitSatisfied(block.wave_states[i], wave)) {
          continue;
        }
        if (IsMemoryWaitReason(prior_wait_reason)) {
          TraceEventLocked(MakeTraceMemoryArriveEvent(
              TraceWaveView{.dpc_id = wave.dpc_id,
                            .ap_id = wave.ap_id,
                            .peu_id = wave.peu_id,
                            .slot_id = TraceSlotId(wave),
                            .block_id = wave.block_id,
                            .wave_id = wave.wave_id,
                            .pc = wave.pc},
              NextTraceCycle(),
              TraceMemoryArriveKindForWaitReason(prior_wait_reason),
              TraceSlotModelKind::LogicalUnbounded,
              prior_pc));
        }
        resumed = true;
      }
    }
    if (resumed) {
      EmitWaveStatsSnapshot();
    }
    return resumed;
  }

  void EmitWaitingWaveStalls(const ExecutableBlock& block) {
    std::lock_guard<std::mutex> state_lock(*block.wave_state_mutex);
    for (const auto& wave : block.waves) {
      if (wave.run_state != WaveRunState::Waiting || !IsMemoryWaitReason(wave.wait_reason)) {
        continue;
      }
      TraceEventLocked(MakeTraceWaitStallEvent(
          TraceWaveView{.dpc_id = wave.dpc_id,
                        .ap_id = wave.ap_id,
                        .peu_id = wave.peu_id,
                        .slot_id = TraceSlotId(wave),
                        .block_id = wave.block_id,
                        .wave_id = wave.wave_id,
                        .pc = wave.pc},
          NextTraceCycle(),
          TraceStallReasonForWaitReason(wave.wait_reason),
          TraceSlotModelKind::LogicalUnbounded));
    }
  }

  std::optional<size_t> SelectNextWaveIndexForPeu(ExecutableBlock& block, size_t peu_index) {
    if (peu_index >= block.wave_indices_per_peu.size()) {
      return std::nullopt;
    }
    auto& peu_waves = block.wave_indices_per_peu[peu_index];
    if (peu_waves.empty()) {
      return std::nullopt;
    }
    std::lock_guard<std::mutex> state_lock(*block.wave_state_mutex);
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

  void ExecuteWave(ExecutableBlock& block, size_t wave_index, ExecutionStats& stats) {
    WaveContext& wave = block.waves[wave_index];
    FunctionalWaveState& wave_state = block.wave_states[wave_index];
    if (wave.pc >= context_.kernel.instructions().size()) {
      throw std::out_of_range("wave pc out of range");
    }

    const Instruction& instruction = context_.kernel.instructions().at(wave.pc);
    ++stats.wave_steps;
    ++stats.instructions_issued;
    MaybeEmitWaveSwitchAwayEvent(block, wave, wave.pc);
    const uint64_t issue_cycle = NextTraceCycle();
    const uint64_t issue_pc = wave.pc;
    TraceEventLocked(MakeTraceWaveStepEvent(
        TraceWaveView{.dpc_id = wave.dpc_id,
                      .ap_id = wave.ap_id,
                      .peu_id = wave.peu_id,
                      .slot_id = TraceSlotId(wave),
                      .block_id = wave.block_id,
                      .wave_id = wave.wave_id,
                      .pc = wave.pc},
        issue_cycle,
        TraceSlotModelKind::LogicalUnbounded,
        FormatWaveStepMessage(instruction, wave),
        issue_pc));

    ExecutionContext block_context = context_;
    block_context.stats = nullptr;
    const OpPlan plan = semantics_.BuildPlan(instruction, wave, block_context);
    if (const auto step_class = ClassifyExecutedInstruction(instruction, plan); step_class.has_value()) {
      RecordExecutedWorkEvent(wave, *step_class,
                              CostForExecutedStep(plan, *step_class, cycle_stats_config_));
    }

    ApplyExecutionPlanRegisterWrites(plan, wave);
    if (const auto mask_text = MaybeFormatExecutionMaskUpdate(plan, wave); mask_text.has_value()) {
      TraceEventLocked(
          MakeWaveTraceEvent(wave, TraceEventKind::ExecMaskUpdate, NextTraceCycle(), *mask_text, issue_pc));
    }
    if (plan.memory.has_value()) {
      {
        std::lock_guard<std::mutex> state_lock(*block.wave_state_mutex);
        RecordPendingMemoryOp(wave_state, wave, MemoryDomainForOpcode(instruction.opcode));
      }
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

      TraceEventLocked(MakeWaveTraceEvent(wave,
                                          TraceEventKind::MemoryAccess,
                                          NextTraceCycle(),
                                          request.kind == AccessKind::Load ? "load" : "store",
                                          issue_pc));

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
      TraceEventLocked(MakeTraceCommitEvent(
          TraceWaveView{.dpc_id = wave.dpc_id,
                        .ap_id = wave.ap_id,
                        .peu_id = wave.peu_id,
                        .slot_id = TraceSlotId(wave),
                        .block_id = wave.block_id,
                        .wave_id = wave.wave_id,
                        .pc = wave.pc},
          NextTraceCycle(),
          TraceSlotModelKind::LogicalUnbounded,
          issue_pc));
      TraceEventLocked(MakeTraceBarrierWaveEvent(
          TraceWaveView{.dpc_id = wave.dpc_id,
                        .ap_id = wave.ap_id,
                        .peu_id = wave.peu_id,
                        .slot_id = TraceSlotId(wave),
                        .block_id = wave.block_id,
                        .wave_id = wave.wave_id,
                        .pc = wave.pc},
          NextTraceCycle(),
          TraceSlotModelKind::LogicalUnbounded,
          issue_pc));
      {
        std::lock_guard<std::mutex> state_lock(*block.wave_state_mutex);
        ResumeWaveToRunnable(wave, /*pc_increment=*/1);
      }
    } else if (plan.sync_barrier) {
      ++stats.barriers;
      bool released_barrier = false;
      {
        std::lock_guard<std::mutex> lock(*block.control_mutex);
        std::lock_guard<std::mutex> state_lock(*block.wave_state_mutex);
        const uint64_t barrier_generation = block.barrier_state->generation.load();
        MarkWaveWaitingAtBarrier(wave, barrier_generation);
        block.barrier_state->arrived_wave_count.fetch_add(1);
      }
      TraceEventLocked(MakeTraceCommitEvent(
          TraceWaveView{.dpc_id = wave.dpc_id,
                        .ap_id = wave.ap_id,
                        .peu_id = wave.peu_id,
                        .slot_id = TraceSlotId(wave),
                        .block_id = wave.block_id,
                        .wave_id = wave.wave_id,
                        .pc = wave.pc},
          NextTraceCycle(),
          TraceSlotModelKind::LogicalUnbounded,
          issue_pc));
      TraceEventLocked(MakeTraceBarrierArriveEvent(
          TraceWaveView{.dpc_id = wave.dpc_id,
                        .ap_id = wave.ap_id,
                        .peu_id = wave.peu_id,
                        .slot_id = TraceSlotId(wave),
                        .block_id = wave.block_id,
                        .wave_id = wave.wave_id,
                        .pc = wave.pc},
          NextTraceCycle(),
          TraceSlotModelKind::LogicalUnbounded,
          issue_pc));
      EmitWaveStatsSnapshot();
      {
        std::lock_guard<std::mutex> lock(*block.control_mutex);
        released_barrier = ReleaseBlockBarrierIfReady(block);
      }
      if (released_barrier) {
        TraceEventLocked(MakeTraceBarrierReleaseEvent(
            wave.dpc_id, wave.ap_id, wave.block_id, NextTraceCycle()));
        EmitWaveStatsSnapshot();
      }
    } else if (plan.exit_wave) {
      ++stats.wave_exits;
      bool wave_completed = false;
      {
        std::lock_guard<std::mutex> lock(*block.control_mutex);
        std::lock_guard<std::mutex> state_lock(*block.wave_state_mutex);
        ApplyExecutionPlanControlFlow(plan, wave, false, false);
        ClearWaitcntWaitState(wave_state);
        wave_state.pending_memory_ops.clear();
        MarkWaveCompleted(wave);
        block.barrier_state->completed_wave_count.fetch_add(1);
        RecordExecutedCompletionEvent(wave);
        wave_completed = true;
      }
      TraceEventLocked(MakeTraceCommitEvent(
          TraceWaveView{.dpc_id = wave.dpc_id,
                        .ap_id = wave.ap_id,
                        .peu_id = wave.peu_id,
                        .slot_id = TraceSlotId(wave),
                        .block_id = wave.block_id,
                        .wave_id = wave.wave_id,
                        .pc = wave.pc},
          NextTraceCycle(),
          TraceSlotModelKind::LogicalUnbounded,
          issue_pc));
      TraceEventLocked(MakeTraceWaveExitEvent(
          TraceWaveView{.dpc_id = wave.dpc_id,
                        .ap_id = wave.ap_id,
                        .peu_id = wave.peu_id,
                        .slot_id = TraceSlotId(wave),
                        .block_id = wave.block_id,
                        .wave_id = wave.wave_id,
                        .pc = wave.pc},
          NextTraceCycle(),
          TraceSlotModelKind::LogicalUnbounded,
          issue_pc));
      if (wave_completed) {
        EmitWaveStatsSnapshot();
      }
    } else {
      bool emit_waitcnt_wave_stats = false;
      {
        std::lock_guard<std::mutex> state_lock(*block.wave_state_mutex);
        if (EnterWaitStateFromWaitcnt(instruction, wave_state, wave)) {
          TraceEventLocked(MakeTraceCommitEvent(
              TraceWaveView{.dpc_id = wave.dpc_id,
                            .ap_id = wave.ap_id,
                            .peu_id = wave.peu_id,
                            .slot_id = TraceSlotId(wave),
                            .block_id = wave.block_id,
                            .wave_id = wave.wave_id,
                            .pc = wave.pc},
              NextTraceCycle(),
              TraceSlotModelKind::LogicalUnbounded,
              issue_pc));
          TraceEventLocked(MakeTraceWaitStallEvent(
              TraceWaveView{.dpc_id = wave.dpc_id,
                            .ap_id = wave.ap_id,
                            .peu_id = wave.peu_id,
                            .slot_id = TraceSlotId(wave),
                            .block_id = wave.block_id,
                            .wave_id = wave.wave_id,
                            .pc = wave.pc},
              NextTraceCycle(),
              TraceStallReasonForWaitReason(wave.wait_reason),
              TraceSlotModelKind::LogicalUnbounded,
              issue_pc));
          emit_waitcnt_wave_stats = true;
        } else {
          ApplyExecutionPlanControlFlow(plan, wave, false, false);
          if (instruction.opcode != Opcode::SWaitCnt) {
            wave.run_state = WaveRunState::Runnable;
            wave.wait_reason = WaveWaitReason::None;
          }
        }
      }
      if (emit_waitcnt_wave_stats) {
        EmitWaveStatsSnapshot();
        return;
      }
      TraceEventLocked(MakeTraceCommitEvent(
          TraceWaveView{.dpc_id = wave.dpc_id,
                        .ap_id = wave.ap_id,
                        .peu_id = wave.peu_id,
                        .slot_id = TraceSlotId(wave),
                        .block_id = wave.block_id,
                        .wave_id = wave.wave_id,
                        .pc = wave.pc},
          NextTraceCycle(),
          TraceSlotModelKind::LogicalUnbounded,
          issue_pc));
    }
  }

};

}  // namespace

uint64_t FunctionalExecEngine::RunSequential() {
  FunctionalExecutionCoreImpl core(context_);
  const uint64_t total_cycles = core.RunSequential();
  program_cycle_stats_ = core.TakeProgramCycleStats();
  return total_cycles;
}

uint64_t FunctionalExecEngine::RunParallelBlocks(uint32_t worker_threads) {
  FunctionalExecutionCoreImpl core(context_);
  const uint64_t total_cycles = core.RunParallelBlocks(worker_threads);
  program_cycle_stats_ = core.TakeProgramCycleStats();
  return total_cycles;
}

}  // namespace gpu_model
