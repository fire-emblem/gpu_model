#pragma once

#include <cstdint>
#include <deque>
#include <limits>
#include <map>
#include <optional>
#include <vector>

#include "gpu_model/debug/trace/event_factory.h"  // TraceWaveView
#include "gpu_model/execution/internal/cycle_issue_policy.h"  // CycleIssuePolicyForSpec
#include "gpu_model/execution/internal/op_plan.h"  // OpPlan
#include "gpu_model/state/wave/wave_runtime_state.h"  // WaveContext
#include "gpu_model/gpu_arch/issue_config/issue_config.h"  // ArchitecturalIssuePolicy, ArchitecturalIssueLimits
#include "gpu_model/instruction/isa/instruction.h"  // Instruction
#include "gpu_model/runtime/program_cycle_stats.h"  // ProgramCycleStats, ProgramCycleStatsConfig
#include "gpu_model/runtime/program_cycle_tracker.h"  // ExecutedStepClass

namespace gpu_model {

struct CycleTimingConfig;  // forward-declared; defined in cycle_exec_engine.h

namespace cycle_internal {

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

struct ExecutableBlock;  // forward declaration

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
  // Selection timing: when PEU can next select a wave for issue
  uint64_t selection_ready_cycle = 0;
  // Commit timing: when current bundle finishes committing
  uint64_t last_bundle_commit_cycle = 0;
  // Legacy busy_until for backward compatibility (max of selection and commit)
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

// ---------------------------------------------------------------------------
// Cost model function declarations
// ---------------------------------------------------------------------------

constexpr uint64_t kIssueTimelineQuantumCycles = 4;

uint64_t QuantizeIssueDuration(uint64_t cycles);

std::optional<ExecutedStepClass> ClassifyCycleInstruction(const Instruction& instruction,
                                                          const OpPlan& plan);

uint64_t CostForCycleStep(const OpPlan& plan,
                          ExecutedStepClass step_class,
                          const ProgramCycleStatsConfig& config);

void AccumulateProgramCycleStep(ProgramCycleStats& stats,
                                ExecutedStepClass step_class,
                                uint64_t cost_cycles,
                                uint64_t work_weight);

bool IssueLimitsUnset(const ArchitecturalIssueLimits& limits);

// CycleTimingConfig is forward-declared above; full definition in cycle_exec_engine.h
ArchitecturalIssuePolicy ResolveIssuePolicy(const CycleTimingConfig& timing_config,
                                            const GpuArchSpec& spec);

uint64_t ModeledAsyncCompletionDelay(uint32_t issue_cycles, uint32_t default_issue_cycles);

}  // namespace cycle_internal
}  // namespace gpu_model
