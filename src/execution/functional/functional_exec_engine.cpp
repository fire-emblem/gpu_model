#include "execution/functional/functional_exec_engine.h"

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <chrono>
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
#include <unordered_set>
#include <vector>

#include "debug/trace/document.h"
#include "debug/trace/event.h"
#include "debug/trace/event_factory.h"
#include "debug/trace/instruction_trace.h"
#include "debug/trace/wave_launch_trace.h"
#include "execution/internal/commit_logic/async_scoreboard.h"
#include "execution/internal/issue_logic/issue_eligibility.h"
#include "gpu_arch/memory/memory_arrive_kind.h"
#include "instruction/isa/opcode_info.h"
#include "state/wave/wave_stats.h"
#include "state/wave/wave_timing.h"
#include "execution/internal/commit_logic/memory_ops.h"
#include "execution/internal/commit_logic/plan_apply.h"
#include "execution/internal/sync_ops/sync_ops.h"
#include "execution/internal/block_schedule/wave_context_builder.h"
#include "instruction/isa/opcode.h"
#include "program/loader/device_image_loader.h"
#include "runtime/model_runtime/program_cycle_tracker.h"

namespace gpu_model {

namespace {

// Shared issue-quantum helpers live in state/wave/wave_timing.h.

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
  MemoryArriveKind arrive_kind = MemoryArriveKind::Load;
  uint64_t ready_cycle = 0;
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
  // Wave-local modeled time: total includes waiting/idle time that belongs to this wave's
  // lifecycle; active only counts effective instruction progress.
  uint64_t wave_cycle_total = 0;
  uint64_t wave_cycle_active = 0;
  uint64_t last_issue_cycle = 0;
  uint64_t next_issue_cycle = 0;
};

// Wave stats snapshots are owned by state/wave/wave_stats.h.

void MarkWaveWaiting(WaveContext& wave, WaveWaitReason reason) {
  if (wave.run_state == WaveRunState::Completed) {
    return;
  }
  wave.run_state = WaveRunState::Waiting;
  wave.wait_reason = reason;
}

void ResumeWaveToRunnable(const ExecutableKernel& kernel,
                          WaveContext& wave,
                          bool advance_pc = false,
                          bool clear_barrier_wait = false) {
  if (clear_barrier_wait) {
    wave.waiting_at_barrier = false;
  }
  if (advance_pc) {
    const auto next_pc = kernel.NextPc(wave.pc);
    if (!next_pc.has_value()) {
      std::ostringstream oss;
      oss << "next instruction pc not found"
          << " block=" << wave.block_id
          << " wave=" << wave.wave_id
          << " pc=" << wave.pc;
      if (kernel.ContainsPc(wave.pc)) {
        oss << " opcode=" << static_cast<int>(kernel.InstructionAtPc(wave.pc).opcode);
      }
      throw std::out_of_range(oss.str());
    }
    wave.pc = *next_pc;
  }
  wave.run_state = WaveRunState::Runnable;
  wave.wait_reason = WaveWaitReason::None;
}

bool EnterWaitStateFromWaitcnt(const Instruction& instruction,
                               FunctionalWaveState& state,
                               WaveContext& wave) {
  const WaitCntThresholds thresholds = WaitCntThresholdsForInstruction(instruction);
  if (!EnterMemoryWaitState(thresholds, wave)) {
    state.waiting_waitcnt_thresholds.reset();
    return false;
  }
  state.waiting_waitcnt_thresholds = thresholds;
  return true;
}

bool ResumeWaveIfWaitSatisfied(const ExecutableKernel& kernel,
                              FunctionalWaveState& state,
                              WaveContext& wave) {
  if (!state.waiting_waitcnt_thresholds.has_value() ||
      !ResumeMemoryWaitStateIfSatisfied(*state.waiting_waitcnt_thresholds, wave)) {
    return false;
  }
  state.waiting_waitcnt_thresholds.reset();
  bool advance_pc = false;
  if (kernel.ContainsPc(wave.pc)) {
    advance_pc = kernel.InstructionAtPc(wave.pc).opcode == Opcode::SWaitCnt;
  }
  ResumeWaveToRunnable(kernel, wave, advance_pc);
  return true;
}

void ClearWaitcntWaitState(FunctionalWaveState& state) {
  state.waiting_waitcnt_thresholds.reset();
}

bool AdvancePendingMemoryOpsUntil(FunctionalWaveState& state,
                                  WaveContext& wave,
                                  uint64_t ready_through_cycle,
                                  std::vector<PendingMemoryOp>* completed_ops = nullptr);

bool AdvancePendingMemoryOps(FunctionalWaveState& state,
                             WaveContext& wave,
                             std::vector<PendingMemoryOp>* completed_ops = nullptr) {
  return AdvancePendingMemoryOpsUntil(state, wave, state.wave_cycle_total, completed_ops);
}

bool AdvancePendingMemoryOpsUntil(FunctionalWaveState& state,
                                  WaveContext& wave,
                                  uint64_t ready_through_cycle,
                                  std::vector<PendingMemoryOp>* completed_ops) {
  bool advanced = false;
  for (auto it = state.pending_memory_ops.begin(); it != state.pending_memory_ops.end();) {
    advanced = true;
    if (it->ready_cycle <= ready_through_cycle) {
      if (completed_ops != nullptr) {
        completed_ops->push_back(*it);
      }
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

std::optional<ExecutedStepClass> ClassifyExecutedInstruction(const Instruction& instruction,
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

uint64_t CostForExecutedStep(const OpPlan& plan,
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
      // Mark wave launched on first processing
      if (!wave.launched) {
        agg.MarkWaveLaunched(wave.agg_wave_id);
        wave.launched = true;
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
    bool launched = false;
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
        .launched = false,
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
                           MemoryWaitDomain domain,
                           MemoryArriveKind arrive_kind,
                           uint64_t ready_cycle) {
  if (domain == MemoryWaitDomain::None) {
    return;
  }
  IncrementPendingMemoryOps(wave, domain);
  state.pending_memory_ops.push_back(PendingMemoryOp{
      .domain = domain,
      .arrive_kind = arrive_kind,
      .ready_cycle = ready_cycle,
  });
}

MemoryArriveKind MemoryArriveKindForMemoryRequest(const MemoryRequest& request) {
  switch (request.space) {
    case MemorySpace::Global:
      return request.kind == AccessKind::Load ? MemoryArriveKind::Load
                                              : MemoryArriveKind::Store;
    case MemorySpace::Shared:
      return MemoryArriveKind::Shared;
    case MemorySpace::Private:
      return MemoryArriveKind::Private;
    case MemorySpace::Constant:
      return MemoryArriveKind::ScalarBuffer;
  }
  return MemoryArriveKind::Load;
}

struct FunctionalBlockBarrierState {
  explicit FunctionalBlockBarrierState(uint32_t expected_wave_count)
      : expected_wave_count(expected_wave_count) {}

  uint32_t expected_wave_count = 0;
  std::atomic<uint32_t> arrived_wave_count{0};
  std::atomic<uint32_t> completed_wave_count{0};
  std::atomic<uint64_t> generation{0};
};

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
  std::vector<bool> wave_enqueued;
  std::vector<std::unique_ptr<std::mutex>> wave_exec_mutexes;
  std::unique_ptr<std::mutex> control_mutex;
  std::unique_ptr<std::mutex> wave_state_mutex;
  std::unique_ptr<std::mutex> shared_mutex;
  std::shared_ptr<FunctionalBlockBarrierState> barrier_state;
};

struct WaveTaskRef {
  size_t block_index = 0;
  size_t wave_index = 0;
  uint32_t global_ap_id = 0;
};

struct ApSchedulerState {
  std::deque<WaveTaskRef> resumed;
  std::deque<WaveTaskRef> runnable;
  std::vector<size_t> block_indices;
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
        .wave_enqueued = {},
        .wave_exec_mutexes = {},
        .control_mutex = std::make_unique<std::mutex>(),
        .wave_state_mutex = std::make_unique<std::mutex>(),
        .shared_mutex = std::make_unique<std::mutex>(),
        .barrier_state = std::make_shared<FunctionalBlockBarrierState>(
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
      block.wave_enqueued.push_back(false);
      block.wave_exec_mutexes.push_back(std::make_unique<std::mutex>());
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
    return RunParallelBlocks(1);
  }

  uint64_t RunParallelBlocks(uint32_t worker_threads) {
    EmitWaveLaunchEvents();
    EmitWaveStatsSnapshot(/*cycle=*/0);
    BuildParallelWaveSchedulerState();
    const uint32_t requested_workers =
        worker_threads == 0 ? DefaultFunctionalParallelWorkerCount() : worker_threads;
    const uint32_t runnable_worker_cap = std::max<uint32_t>(1u, static_cast<uint32_t>(total_waves_));
    const uint32_t actual_workers = std::min(requested_workers, runnable_worker_cap);
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
    EmitWaveStatsSnapshot(CurrentMaxWaveNextIssueCycle());
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
  size_t waiting_ap_rr_ = 0;
  size_t total_waves_ = 0;
  size_t completed_waves_ = 0;
  size_t active_wave_tasks_ = 0;
  ProgramCycleStatsConfig cycle_stats_config_{};
  std::vector<ExecutedFlowEvent> executed_flow_events_;
  std::optional<ProgramCycleStats> program_cycle_stats_;
  std::unordered_map<uint64_t, uint32_t> logical_slot_ids_;
  struct LastScheduledWaveTraceState {
    uint64_t wave_tag = 0;
    TraceWaveView wave{};
    uint64_t pc = 0;
  };
  std::unordered_map<uint64_t, LastScheduledWaveTraceState> last_wave_per_ap_peu_;

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

  TraceWaveView MakeTraceWaveView(const WaveContext& wave) const {
    return TraceWaveView{
        .dpc_id = wave.dpc_id,
        .ap_id = wave.ap_id,
        .peu_id = wave.peu_id,
        .slot_id = TraceSlotId(wave),
        .block_id = wave.block_id,
        .wave_id = wave.wave_id,
        .pc = wave.pc,
    };
  }

  uint64_t ApPeuKey(const ExecutableBlock& block, const WaveContext& wave) const {
    return (static_cast<uint64_t>(block.global_ap_id) << 32u) | static_cast<uint64_t>(wave.peu_id);
  }

  uint64_t WaveTag(const WaveContext& wave) const {
    return (static_cast<uint64_t>(wave.block_id) << 32u) | static_cast<uint64_t>(wave.wave_id);
  }

  void EmitBlockingWaveSwitchAwayEvent(const WaveContext& wave,
                                       uint64_t cycle,
                                       uint64_t pc) {
    TraceEventLocked(MakeTraceWaveSwitchAwayEvent(MakeTraceWaveView(wave),
                                                  cycle,
                                                  TraceSlotModelKind::LogicalUnbounded,
                                                  pc));
    TraceEventLocked(MakeTraceWaveSwitchStallEvent(MakeTraceWaveView(wave),
                                                   cycle,
                                                   TraceSlotModelKind::LogicalUnbounded,
                                                   pc));
  }

  void RememberScheduledWaveForPeu(const ExecutableBlock& block, const WaveContext& wave) {
    const uint64_t ap_peu_key = ApPeuKey(block, wave);
    std::lock_guard<std::mutex> lock(peu_schedule_trace_mutex_);
    last_wave_per_ap_peu_[ap_peu_key] = LastScheduledWaveTraceState{
        .wave_tag = WaveTag(wave),
        .wave = MakeTraceWaveView(wave),
        .pc = wave.pc,
    };
  }

  void ClearLastScheduledWaveIfCompleted(const ExecutableBlock& block, const WaveContext& wave) {
    const uint64_t ap_peu_key =
        ApPeuKey(block, wave);
    const uint64_t wave_tag = WaveTag(wave);
    std::lock_guard<std::mutex> lock(peu_schedule_trace_mutex_);
    const auto it = last_wave_per_ap_peu_.find(ap_peu_key);
    if (it != last_wave_per_ap_peu_.end() && it->second.wave_tag == wave_tag) {
      last_wave_per_ap_peu_.erase(it);
    }
  }

  TraceEvent MakeWaveTraceEvent(const WaveContext& wave,
                                TraceEventKind kind,
                                uint64_t cycle,
                                std::string message,
                                uint64_t pc = std::numeric_limits<uint64_t>::max()) const {
    return MakeTraceWaveEvent(
        MakeTraceWaveView(wave),
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
      for (size_t wave_index = 0; wave_index < block.waves.size(); ++wave_index) {
        const auto& wave = block.waves[wave_index];
        const uint64_t launch_cycle = block.wave_states[wave_index].next_issue_cycle;
        // Record wave init snapshot for structured trace output.
        TraceWaveInitSnapshot snapshot;
        snapshot.stable_wave_id = StableWaveKey(wave);
        snapshot.block_id = wave.block_id;
        snapshot.dpc_id = wave.dpc_id;
        snapshot.ap_id = wave.ap_id;
        snapshot.peu_id = wave.peu_id;
        snapshot.slot_id = wave.wave_id;  // wave_id within block serves as slot
        snapshot.slot_model = "logical_unbounded";
        snapshot.start_pc = wave.pc;
        snapshot.ready_at_global_cycle = launch_cycle;
        snapshot.next_issue_earliest_global_cycle = launch_cycle;
        context_.trace.OnWaveInitSnapshot(snapshot);
        TraceEventLocked(MakeTraceWaveLaunchEvent(
            MakeTraceWaveView(wave),
            launch_cycle,
            FormatWaveLaunchTraceMessage(wave),
            TraceSlotModelKind::LogicalUnbounded));
        TraceEventLocked(MakeTraceActivePromoteEvent(
            MakeTraceWaveView(wave), launch_cycle, TraceSlotModelKind::LogicalUnbounded));
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

  uint64_t CurrentMaxWaveNextIssueCycle() const {
    uint64_t max_cycle = 0;
    for (const auto& block : blocks_) {
      std::lock_guard<std::mutex> state_lock(*block.wave_state_mutex);
      for (size_t i = 0; i < block.wave_states.size(); ++i) {
        max_cycle = std::max(max_cycle, block.wave_states[i].next_issue_cycle);
      }
    }
    return max_cycle;
  }

  void EmitWaveStatsSnapshot(uint64_t cycle) {
    TraceEventLocked(MakeTraceEvent(TraceEventKind::WaveStats,
                                    cycle,
                                    FormatWaveStatsMessage(CaptureWaveStatsSnapshot())));
  }

  void BuildParallelWaveSchedulerState() {
    std::lock_guard<std::mutex> lock(scheduler_mutex_);
    next_ap_rr_ = 0;
    waiting_ap_rr_ = 0;
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
        block.wave_enqueued[wave_index] = false;
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
      if (!ap.resumed.empty()) {
        task = ap.resumed.front();
        ap.resumed.pop_front();
        blocks_[task.block_index].wave_enqueued[task.wave_index] = false;
        next_ap_rr_ = (ap_index + 1) % ap_schedulers_.size();
        return true;
      }
      if (ap.runnable.empty()) {
        continue;
      }
      task = ap.runnable.front();
      ap.runnable.pop_front();
      blocks_[task.block_index].wave_enqueued[task.wave_index] = false;
      next_ap_rr_ = (ap_index + 1) % ap_schedulers_.size();
      return true;
    }
    return false;
  }

  bool HasRunnableWaveLocked() const {
    for (const auto& ap : ap_schedulers_) {
      if (!ap.resumed.empty()) {
        return true;
      }
      if (!ap.runnable.empty()) {
        return true;
      }
    }
    return false;
  }

  void EnqueueWaveLocked(const WaveTaskRef& task) {
    ap_schedulers_[task.global_ap_id].runnable.push_back(task);
  }

  void EnqueueResumedWaveLocked(const WaveTaskRef& task) {
    ap_schedulers_[task.global_ap_id].resumed.push_back(task);
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
      if (block.wave_enqueued[wave_index]) {
        continue;
      }
      if (wave.status != WaveStatus::Active ||
          wave.run_state != WaveRunState::Runnable ||
          wave.waiting_at_barrier) {
        continue;
      }
      block.wave_busy[wave_index] = true;
      block.wave_enqueued[wave_index] = true;
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
    if (ap_schedulers_.empty()) {
      return false;
    }
    for (size_t ap_offset = 0; ap_offset < ap_schedulers_.size(); ++ap_offset) {
      const size_t ap_index = (waiting_ap_rr_ + ap_offset) % ap_schedulers_.size();
      auto& ap = ap_schedulers_[ap_index];
      for (const size_t block_index : ap.block_indices) {
        if (AdvanceWaitingWavesForBlockLocked(block_index)) {
          waiting_ap_rr_ = (ap_index + 1) % ap_schedulers_.size();
          return true;
        }
      }
    }
    waiting_ap_rr_ = (waiting_ap_rr_ + 1) % ap_schedulers_.size();
    return false;
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
          if (active_wave_tasks_ == 0 && AdvanceWaitingWavesLocked()) {
            continue;
          }
          scheduler_cv_.wait(lock, [&] {
            {
              std::lock_guard<std::mutex> failure_lock(failure_mutex);
              if (failure != nullptr) {
                return true;
              }
            }
            return AllParallelWavesCompletedLocked() || HasRunnableWaveLocked() ||
                   active_wave_tasks_ == 0;
          });
        }
      }

      while (true) {
        ExecuteWave(blocks_[task.block_index], task.wave_index, stats);

        bool claimed_next_task = false;
        {
          std::lock_guard<std::mutex> lock(scheduler_mutex_);
          if (active_wave_tasks_ > 0) {
            --active_wave_tasks_;
          }
          ReconcileWaveTaskLocked(task);
          (void)AdvanceWaitingWavesForBlockLocked(task.block_index);
          if (AllParallelWavesCompletedLocked()) {
            scheduler_cv_.notify_all();
            return;
          }
          if (PopRunnableWaveLocked(task)) {
            ++active_wave_tasks_;
            claimed_next_task = true;
          } else {
            const bool has_runnable = HasRunnableWaveLocked();
            if (has_runnable || active_wave_tasks_ == 0) {
              scheduler_cv_.notify_one();
            }
          }
        }
        if (!claimed_next_task) {
          break;
        }
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

  std::optional<uint64_t> ReleaseBlockBarrierIfReady(ExecutableBlock& block) {
    std::lock_guard<std::mutex> state_lock(*block.wave_state_mutex);
    const uint32_t arrived = block.barrier_state->arrived_wave_count.load();
    const uint32_t completed = block.barrier_state->completed_wave_count.load();
    const uint64_t generation = block.barrier_state->generation.load();
    uint32_t waiting_in_generation = 0;
    uint64_t release_cycle = 0;
    for (size_t i = 0; i < block.waves.size(); ++i) {
      const auto& wave = block.waves[i];
      if (wave.waiting_at_barrier && wave.barrier_generation == generation &&
          wave.run_state == WaveRunState::Waiting &&
          wave.wait_reason == WaveWaitReason::BlockBarrier) {
        ++waiting_in_generation;
        release_cycle = std::max(release_cycle, block.wave_states[i].next_issue_cycle);
      }
    }
    if (waiting_in_generation == 0 ||
        arrived + completed != block.barrier_state->expected_wave_count) {
      return std::nullopt;
    }
    for (size_t i = 0; i < block.waves.size(); ++i) {
      auto& wave = block.waves[i];
      if (wave.waiting_at_barrier && wave.barrier_generation == generation) {
        block.wave_states[i].next_issue_cycle =
            std::max(block.wave_states[i].next_issue_cycle, release_cycle);
        ResumeWaveToRunnable(context_.kernel, wave, /*advance_pc=*/true, /*clear_barrier_wait=*/true);
        wave.status = WaveStatus::Active;
        TraceEventLocked(MakeTraceWaveResumeEvent(
            MakeTraceWaveView(wave), release_cycle, TraceSlotModelKind::LogicalUnbounded));
      }
    }
    block.barrier_state->arrived_wave_count.store(0);
    block.barrier_state->generation.store(generation + 1);
    return release_cycle;
  }

  bool RecordWaitingWaveTicks(ExecutableBlock& block) {
    std::lock_guard<std::mutex> state_lock(*block.wave_state_mutex);
    bool any_waiting = false;
    for (size_t i = 0; i < block.waves.size(); ++i) {
      const auto& wave = block.waves[i];
      if (wave.run_state != WaveRunState::Waiting) {
        continue;
      }
      any_waiting = true;
      block.wave_states[i].wave_cycle_total += 1;
      // Both barrier and memory wait are sync operations
      if (wave.wait_reason == WaveWaitReason::BlockBarrier ||
          IsMemoryWaitReason(wave.wait_reason)) {
        RecordExecutedWorkEvent(wave, ExecutedStepClass::Sync, 1);
      }
    }
    return any_waiting;
  }

  bool ProcessWaitingWaves(ExecutableBlock& block) {
    bool progressed = false;
    progressed = RecordWaitingWaveTicks(block) || progressed;
    progressed = AdvancePendingMemoryOps(block) || progressed;
    progressed = ResumeMemoryWaitingWaves(block) || progressed;
    const auto released_barrier_cycle = ReleaseBlockBarrierIfReady(block);
    progressed = released_barrier_cycle.has_value() || progressed;
    if (released_barrier_cycle.has_value()) {
      TraceEventLocked(
          MakeTraceBarrierReleaseEvent(
              block.dpc_id, block.ap_id, block.block_id, *released_barrier_cycle));
      EmitWaveStatsSnapshot(*released_barrier_cycle);
    }
    EmitWaitingWaveStalls(block);
    return progressed;
  }

  bool AdvancePendingMemoryOps(ExecutableBlock& block) {
    std::lock_guard<std::mutex> state_lock(*block.wave_state_mutex);
    bool advanced = false;
    for (size_t i = 0; i < block.waves.size(); ++i) {
      std::vector<PendingMemoryOp> completed_ops;
      advanced = ::gpu_model::AdvancePendingMemoryOps(block.wave_states[i], block.waves[i], &completed_ops) ||
                 advanced;
      for (const auto& op : completed_ops) {
        block.wave_states[i].next_issue_cycle =
            std::max(block.wave_states[i].next_issue_cycle, QuantizeToNextIssueQuantum(op.ready_cycle));
        TraceEvent event = MakeTraceMemoryArriveEvent(
            MakeTraceWaveView(block.waves[i]),
            op.ready_cycle,
            ToTraceMemoryArriveKind(op.arrive_kind),
            TraceSlotModelKind::LogicalUnbounded);
        const AsyncArriveResult arrive_result = MakeAsyncArriveResult(
            block.waves[i], op.domain, block.wave_states[i].waiting_waitcnt_thresholds);
        event.waitcnt_state = arrive_result.waitcnt_state;
        event.arrive_progress = arrive_result.arrive_progress;
        TraceEventLocked(std::move(event));
        TraceEventLocked(MakeTraceWaveArriveEvent(
            MakeTraceWaveView(block.waves[i]),
            op.ready_cycle,
            ToTraceMemoryArriveKind(op.arrive_kind),
            TraceSlotModelKind::LogicalUnbounded,
            arrive_result.arrive_progress,
            std::numeric_limits<uint64_t>::max(),
            arrive_result.waitcnt_state));
      }
    }
    return advanced;
  }

  bool ResumeMemoryWaitingWaves(ExecutableBlock& block) {
    bool resumed = false;
    uint64_t snapshot_cycle = 0;
    {
      std::lock_guard<std::mutex> state_lock(*block.wave_state_mutex);
      for (size_t i = 0; i < block.waves.size(); ++i) {
        WaveContext& wave = block.waves[i];
        if (!ResumeWaveIfWaitSatisfied(context_.kernel, block.wave_states[i], wave)) {
          continue;
        }
        block.wave_busy[i] = true;
        block.wave_enqueued[i] = true;
        const size_t block_index = static_cast<size_t>(&block - blocks_.data());
        EnqueueResumedWaveLocked(WaveTaskRef{
            .block_index = block_index,
            .wave_index = i,
            .global_ap_id = block.global_ap_id,
        });
        TraceEventLocked(MakeTraceWaveResumeEvent(
            MakeTraceWaveView(wave),
            block.wave_states[i].next_issue_cycle,
            TraceSlotModelKind::LogicalUnbounded));
        snapshot_cycle = std::max(snapshot_cycle, block.wave_states[i].next_issue_cycle);
        resumed = true;
      }
    }
    if (resumed) {
      EmitWaveStatsSnapshot(snapshot_cycle);
    }
    return resumed;
  }

  void EmitWaitingWaveStalls(const ExecutableBlock& block) {
    std::lock_guard<std::mutex> state_lock(*block.wave_state_mutex);
    for (size_t i = 0; i < block.waves.size(); ++i) {
      const auto& wave = block.waves[i];
      if (wave.run_state != WaveRunState::Waiting || !IsMemoryWaitReason(wave.wait_reason)) {
        continue;
      }
      TraceEventLocked(MakeTraceWaitStallEvent(
          MakeTraceWaveView(wave),
          block.wave_states[i].next_issue_cycle,
          TraceStallReasonForWaitReason(wave.wait_reason),
          TraceSlotModelKind::LogicalUnbounded,
          std::numeric_limits<uint64_t>::max(),
          MakeOptionalTraceWaitcntState(wave, block.wave_states[i].waiting_waitcnt_thresholds)));
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
        block.next_wave_rr_per_peu[peu_index] = local_index;
        return wave_index;
      }
    }
    return std::nullopt;
  }

  void ExecuteWave(ExecutableBlock& block, size_t wave_index, ExecutionStats& stats) {
    std::lock_guard<std::mutex> exec_lock(*block.wave_exec_mutexes[wave_index]);
    WaveContext& wave = block.waves[wave_index];
    FunctionalWaveState& wave_state = block.wave_states[wave_index];
    if (!context_.kernel.ContainsPc(wave.pc)) {
      throw std::out_of_range("wave pc out of range");
    }

    auto emit_completed_memory_ops = [&](const std::vector<PendingMemoryOp>& completed_ops) {
      for (const auto& op : completed_ops) {
        wave_state.next_issue_cycle =
            std::max(wave_state.next_issue_cycle, QuantizeToNextIssueQuantum(op.ready_cycle));
        TraceEvent event = MakeTraceMemoryArriveEvent(
            MakeTraceWaveView(wave),
            op.ready_cycle,
            ToTraceMemoryArriveKind(op.arrive_kind),
            TraceSlotModelKind::LogicalUnbounded);
        const AsyncArriveResult arrive_result =
            MakeAsyncArriveResult(wave, op.domain, wave_state.waiting_waitcnt_thresholds);
        event.waitcnt_state = arrive_result.waitcnt_state;
        event.arrive_progress = arrive_result.arrive_progress;
        TraceEventLocked(std::move(event));
        TraceEventLocked(MakeTraceWaveArriveEvent(
            MakeTraceWaveView(wave),
            op.ready_cycle,
            ToTraceMemoryArriveKind(op.arrive_kind),
            TraceSlotModelKind::LogicalUnbounded,
            arrive_result.arrive_progress,
            std::numeric_limits<uint64_t>::max(),
            arrive_result.waitcnt_state));
      }
    };
    {
      std::vector<PendingMemoryOp> completed_ops;
      std::lock_guard<std::mutex> state_lock(*block.wave_state_mutex);
      if (AdvancePendingMemoryOpsUntil(wave_state, wave, wave_state.next_issue_cycle, &completed_ops)) {
        emit_completed_memory_ops(completed_ops);
      }
    }

    const Instruction& instruction = context_.kernel.InstructionAtPc(wave.pc);
    ++stats.wave_steps;
    ++stats.instructions_issued;
    ExecutionContext block_context = context_;
    block_context.stats = nullptr;
    const OpPlan plan = semantics_.BuildPlan(instruction, wave, block_context);
    const uint64_t issue_duration = std::max<uint64_t>(1u, plan.issue_cycles);
    const uint64_t issue_cycle = wave_state.next_issue_cycle;
    const uint64_t commit_cycle = issue_cycle + issue_duration;
    wave_state.last_issue_cycle = issue_cycle;
    wave_state.next_issue_cycle = commit_cycle;
    wave_state.wave_cycle_total += issue_duration;
    wave_state.wave_cycle_active += issue_duration;
    const uint64_t issue_pc = wave.pc;
    TraceEventLocked(MakeTraceIssueSelectEvent(
        MakeTraceWaveView(wave), issue_cycle, TraceSlotModelKind::LogicalUnbounded));
    TraceEventLocked(MakeTraceWaveStepEvent(MakeTraceWaveView(wave),
                                           issue_cycle,
                                           TraceSlotModelKind::LogicalUnbounded,
                                           FormatWaveStepMessage(instruction, wave),
                                           BuildWaveStepDetail(instruction, wave),
                                           issue_pc,
                                           QuantizeIssueDuration(issue_duration)));
    RememberScheduledWaveForPeu(block, wave);
    if (const auto step_class = ClassifyExecutedInstruction(instruction, plan); step_class.has_value()) {
      RecordExecutedWorkEvent(wave, *step_class,
                              CostForExecutedStep(plan, *step_class, cycle_stats_config_));
    }

    ApplyExecutionPlanRegisterWrites(plan, wave);
    if (const auto mask_text = MaybeFormatExecutionMaskUpdate(plan, wave); mask_text.has_value()) {
      TraceEventLocked(
          MakeWaveTraceEvent(wave, TraceEventKind::ExecMaskUpdate, issue_cycle, *mask_text, issue_pc));
    }
    if (plan.memory.has_value()) {
      const auto& request = *plan.memory;
      {
        std::lock_guard<std::mutex> state_lock(*block.wave_state_mutex);
        const uint64_t memory_completion_turns =
            request.space == MemorySpace::Global ? context_.global_memory_latency_cycles : 5u;
        RecordPendingMemoryOp(wave_state,
                              wave,
                              MemoryDomainForOpcode(instruction.opcode),
                              MemoryArriveKindForMemoryRequest(*plan.memory),
                              commit_cycle + memory_completion_turns);
      }
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
                                          issue_cycle,
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
            int32_t updated;
            switch (request.atomic_op) {
              case AtomicOp::Add:
                updated = prior + static_cast<int32_t>(request.lanes[lane].value);
                break;
              case AtomicOp::Max:
                updated = std::max(prior, static_cast<int32_t>(request.lanes[lane].value));
                break;
              case AtomicOp::Min:
                updated = std::min(prior, static_cast<int32_t>(request.lanes[lane].value));
                break;
              case AtomicOp::Exch:
                updated = static_cast<int32_t>(request.lanes[lane].value);
                break;
              default:
                throw std::invalid_argument("unsupported atomic op");
            }
            LaneAccess writeback = request.lanes[lane];
            writeback.value = static_cast<uint64_t>(static_cast<int64_t>(updated));
            StoreLaneValue(block.shared_memory, writeback);
          } else if (request.space == MemorySpace::Global) {
            std::lock_guard<std::mutex> lock(global_memory_mutex_);
            int32_t prior = context_.memory.LoadGlobalValue<int32_t>(request.lanes[lane].addr);
            int32_t updated;
            switch (request.atomic_op) {
              case AtomicOp::Add:
                updated = prior + static_cast<int32_t>(request.lanes[lane].value);
                break;
              case AtomicOp::Max:
                updated = std::max(prior, static_cast<int32_t>(request.lanes[lane].value));
                break;
              case AtomicOp::Min:
                updated = std::min(prior, static_cast<int32_t>(request.lanes[lane].value));
                break;
              case AtomicOp::Exch:
                updated = static_cast<int32_t>(request.lanes[lane].value);
                break;
              default:
                throw std::invalid_argument("unsupported atomic op");
            }
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
          MakeTraceWaveView(wave),
          commit_cycle,
          TraceSlotModelKind::LogicalUnbounded,
          issue_pc));
      TraceEventLocked(MakeTraceBarrierWaveEvent(
          MakeTraceWaveView(wave),
          commit_cycle,
          TraceSlotModelKind::LogicalUnbounded,
          issue_pc));
      {
        std::lock_guard<std::mutex> state_lock(*block.wave_state_mutex);
        ResumeWaveToRunnable(context_.kernel, wave, /*advance_pc=*/true);
      }
    } else if (plan.sync_barrier) {
      ++stats.barriers;
      std::optional<uint64_t> released_barrier_cycle;
      {
        std::lock_guard<std::mutex> lock(*block.control_mutex);
        std::lock_guard<std::mutex> state_lock(*block.wave_state_mutex);
        const uint64_t barrier_generation = block.barrier_state->generation.load();
        MarkWaveWaitingAtBarrier(wave, barrier_generation);
        block.barrier_state->arrived_wave_count.fetch_add(1);
      }
      TraceEventLocked(MakeTraceCommitEvent(
          MakeTraceWaveView(wave),
          commit_cycle,
          TraceSlotModelKind::LogicalUnbounded,
          issue_pc));
      TraceEventLocked(MakeTraceBarrierArriveEvent(
          MakeTraceWaveView(wave),
          commit_cycle,
          TraceSlotModelKind::LogicalUnbounded,
          issue_pc));
      TraceEventLocked(MakeTraceWaveWaitEvent(
          MakeTraceWaveView(wave),
          commit_cycle,
          TraceSlotModelKind::LogicalUnbounded,
          TraceStallReason::None,
          issue_pc));
      EmitBlockingWaveSwitchAwayEvent(wave, commit_cycle, issue_pc);
      EmitWaveStatsSnapshot(commit_cycle);
      {
        std::lock_guard<std::mutex> lock(*block.control_mutex);
        released_barrier_cycle = ReleaseBlockBarrierIfReady(block);
      }
      if (released_barrier_cycle.has_value()) {
        TraceEventLocked(MakeTraceBarrierReleaseEvent(
            wave.dpc_id, wave.ap_id, wave.block_id, *released_barrier_cycle));
        EmitWaveStatsSnapshot(*released_barrier_cycle);
      }
    } else if (plan.exit_wave) {
      ++stats.wave_exits;
      bool wave_completed = false;
      {
        std::lock_guard<std::mutex> lock(*block.control_mutex);
        std::lock_guard<std::mutex> state_lock(*block.wave_state_mutex);
        ApplyExecutionPlanControlFlow(context_.kernel, plan, wave, false, false);
        ClearWaitcntWaitState(wave_state);
        wave_state.pending_memory_ops.clear();
        MarkWaveCompleted(wave);
        block.barrier_state->completed_wave_count.fetch_add(1);
        RecordExecutedCompletionEvent(wave);
        wave_completed = true;
      }
      TraceEventLocked(MakeTraceCommitEvent(
          MakeTraceWaveView(wave),
          commit_cycle,
          TraceSlotModelKind::LogicalUnbounded,
          issue_pc));
      TraceEventLocked(MakeTraceWaveExitEvent(
          MakeTraceWaveView(wave),
          commit_cycle,
          TraceSlotModelKind::LogicalUnbounded,
          issue_pc));
      ClearLastScheduledWaveIfCompleted(block, wave);
      if (wave_completed) {
        EmitWaveStatsSnapshot(commit_cycle);
      }
    } else {
      bool emit_waitcnt_wave_stats = false;
      {
        std::lock_guard<std::mutex> state_lock(*block.wave_state_mutex);
        if (EnterWaitStateFromWaitcnt(instruction, wave_state, wave)) {
          const TraceWaitcntState waitcnt_state =
              MakeTraceWaitcntState(wave, *wave_state.waiting_waitcnt_thresholds);
          TraceEventLocked(MakeTraceCommitEvent(
              MakeTraceWaveView(wave),
              commit_cycle,
              TraceSlotModelKind::LogicalUnbounded,
              issue_pc));
          TraceEventLocked(MakeTraceWaveWaitEvent(
              MakeTraceWaveView(wave),
              commit_cycle,
              TraceSlotModelKind::LogicalUnbounded,
              TraceStallReasonForWaitReason(wave.wait_reason),
              issue_pc,
              waitcnt_state));
          TraceEventLocked(MakeTraceWaitStallEvent(
              MakeTraceWaveView(wave),
              commit_cycle,
              TraceStallReasonForWaitReason(wave.wait_reason),
              TraceSlotModelKind::LogicalUnbounded,
              issue_pc,
              waitcnt_state));
          EmitBlockingWaveSwitchAwayEvent(wave, commit_cycle, issue_pc);
          emit_waitcnt_wave_stats = true;
        } else {
          ApplyExecutionPlanControlFlow(context_.kernel, plan, wave, false, false);
          if (instruction.opcode != Opcode::SWaitCnt) {
            wave.run_state = WaveRunState::Runnable;
            wave.wait_reason = WaveWaitReason::None;
          }
        }
      }
      if (emit_waitcnt_wave_stats) {
        EmitWaveStatsSnapshot(commit_cycle);
        return;
      }
      TraceEventLocked(MakeTraceCommitEvent(
          MakeTraceWaveView(wave),
          commit_cycle,
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
