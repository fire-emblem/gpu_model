#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace gpu_model {

// TraceRunSnapshot captures run-level static context.
// All fields are producer-owned facts; trace does not infer or compute.
struct TraceRunSnapshot {
  std::string invocation;            // Full invocation line: env vars + command
  std::string execution_model;       // "functional" or "cycle"
  std::string functional_mode;       // "st", "mt", or "" for cycle mode
  std::string trace_time_basis;      // "modeled_cycle"
  bool trace_cycle_is_physical_time = false;
};

// TraceModelConfigSnapshot captures model configuration facts.
struct TraceModelConfigSnapshot {
  uint32_t num_dpcs = 0;
  uint32_t num_aps_per_dpc = 0;
  uint32_t num_peus_per_ap = 0;
  uint32_t num_slots_per_peu = 0;
  std::string slot_model;            // "resident_fixed" or "logical_unbounded"
};

// TraceKernelSnapshot captures kernel launch context.
struct TraceKernelSnapshot {
  std::string kernel_name;
  uint64_t kernel_launch_uid = 0;
  uint64_t launch_index = 0;         // Monotonic launch counter
  uint32_t grid_dim_x = 1;
  uint32_t grid_dim_y = 1;
  uint32_t grid_dim_z = 1;
  uint32_t block_dim_x = 1;
  uint32_t block_dim_y = 1;
  uint32_t block_dim_z = 1;
};

// TraceWaveInitSnapshot captures wave initialization facts.
// Producer-owned: execution engine decides all values.
struct TraceWaveInitSnapshot {
  uint64_t stable_wave_id = 0;
  uint32_t block_id = 0;
  uint32_t dpc_id = 0;
  uint32_t ap_id = 0;
  uint32_t peu_id = 0;
  uint32_t slot_id = 0;
  std::string slot_model;            // "resident_fixed" or "logical_unbounded"
  uint64_t start_pc = 0;
  uint64_t ready_at_global_cycle = 0;
  uint64_t next_issue_earliest_global_cycle = 0;
  std::string exec_mask_init;
  std::string waitcnt_init;
  std::string barrier_init;
};

// TraceSummarySnapshot captures run completion summary.
// All stats are producer-owned facts from execution engine.
struct TraceSummarySnapshot {
  std::string kernel_status;         // "PASS" or "FAIL"
  uint64_t launch_index = 0;         // Launch counter for this summary
  uint64_t submit_cycle = 0;         // Cycle when kernel was submitted
  uint64_t begin_cycle = 0;          // Cycle when execution began
  uint64_t end_cycle = 0;            // Cycle when execution ended
  uint64_t gpu_tot_sim_cycle = 0;
  uint64_t gpu_tot_sim_insn = 0;
  double gpu_tot_ipc = 0.0;
  uint64_t gpu_tot_wave_exits = 0;
  // Stall breakdown
  uint64_t stall_waitcnt_global = 0;
  uint64_t stall_waitcnt_shared = 0;
  uint64_t stall_waitcnt_private = 0;
  uint64_t stall_warp_switch = 0;
  uint64_t stall_barrier_slot = 0;
  uint64_t stall_other = 0;
};

// TraceWarningSnapshot captures producer-detected warnings.
struct TraceWarningSnapshot {
  std::string warning_kind;
  std::string message;
  uint64_t cycle = 0;
  uint32_t wave_id = 0;
  uint32_t block_id = 0;
};

}  // namespace gpu_model
