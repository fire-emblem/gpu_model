#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <optional>
#include <string>

#include "execution/internal/issue_eligibility.h"
#include "gpu_arch/memory/memory_arrive_kind.h"
#include "state/wave/wave_runtime_state.h"

namespace gpu_model {

// ============================================================================
// Wave State Structures - Unified across all execution engines
// ============================================================================

/// Common wave scheduling state shared across execution engines.
/// This replaces the duplicated RawWave/ScheduledWave/WaveTaskRef patterns.
struct WaveSchedulingState {
  size_t block_index = 0;
  size_t wave_index = 0;
  uint64_t generate_cycle = 0;
  uint64_t dispatch_cycle = 0;
  uint64_t launch_cycle = 0;
  bool generate_completed = false;
  bool generate_scheduled = false;
  bool dispatch_completed = false;
  bool dispatch_scheduled = false;
  bool dispatch_enabled = false;
  bool launch_scheduled = false;
  size_t peu_slot_index = std::numeric_limits<size_t>::max();
  size_t resident_slot_id = std::numeric_limits<size_t>::max();
};

/// Pending memory operation tracking for waitcnt semantics.
/// Used by both functional and encoded execution engines.
struct PendingMemoryOp {
  MemoryWaitDomain domain = MemoryWaitDomain::None;
  uint8_t turns_until_complete = 0;
  uint64_t ready_cycle = 0;
  bool uses_ready_cycle = false;
  std::optional<MemoryArriveKind> arrive_kind;
  uint64_t flow_id = 0;
};

/// Wave execution state tracking cycle counts and waitcnt state.
struct WaveExecutionState {
  std::deque<PendingMemoryOp> pending_memory_ops;
  std::optional<WaitCntThresholds> waiting_waitcnt_thresholds;
  uint64_t waiting_resume_pc_increment = 0;
  uint64_t wave_cycle_total = 0;
  uint64_t wave_cycle_active = 0;
  uint64_t last_issue_cycle = 0;
  uint64_t next_issue_cycle = 0;
  uint64_t eligible_since_cycle = 0;
  bool eligible_since_valid = false;
};

/// PEU (Processing Element Unit) slot state.
struct PeuSlotState {
  uint32_t dpc_id = 0;
  uint32_t ap_id = 0;
  uint32_t peu_id = 0;
  uint64_t busy_until = 0;
  uint64_t last_wave_tag = std::numeric_limits<uint64_t>::max();
};

/// AP (Array Processor) resident state for wave scheduling.
struct ApResidentState {
  uint32_t dpc_id = 0;
  uint32_t ap_id = 0;
  uint32_t resident_count = 0;
  uint64_t last_wave_tag = std::numeric_limits<uint64_t>::max();
};

/// Block barrier state for synchronization.
struct BlockBarrierState {
  uint64_t generation = 0;
  uint32_t arrivals = 0;
  bool slot_acquired = false;
};

/// Wave statistics snapshot for tracking execution progress.
struct WaveStatsSnapshot {
  uint32_t launch = 0;
  uint32_t init = 0;
  uint32_t active = 0;
  uint32_t runnable = 0;
  uint32_t waiting = 0;
  uint32_t end = 0;
};

// ============================================================================
// Helper Functions
// ============================================================================

/// Format wave stats for logging.
std::string FormatWaveStatsMessage(const WaveStatsSnapshot& stats);

// ============================================================================
// Issue Quantum Constants (shared across execution engines)
// ============================================================================

/// Minimum issue quantum for execution timing.
constexpr uint64_t kIssueQuantumCycles = 4;

/// Quantize a cycle value to the next issue quantum boundary.
inline uint64_t QuantizeToNextIssueQuantum(uint64_t cycle) {
  const uint64_t remainder = cycle % kIssueQuantumCycles;
  if (remainder == 0) {
    return cycle;
  }
  return cycle + (kIssueQuantumCycles - remainder);
}

/// Quantize an issue duration to valid quantum boundaries.
inline uint64_t QuantizeIssueDuration(uint64_t cycles) {
  return std::max(kIssueQuantumCycles, QuantizeToNextIssueQuantum(cycles));
}

}  // namespace gpu_model
