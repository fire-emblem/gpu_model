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

  // === Theoretical Occupancy (static, computed at launch) ===
  uint32_t theoretical_max_waves_per_peu = 0;
  uint32_t theoretical_max_blocks_per_ap = 0;
  float theoretical_occupancy_pct = 0.0f;
  std::string occupancy_wave_limiter;
  std::string occupancy_block_limiter;
  uint32_t kernel_vgpr_count = 0;
  uint32_t kernel_sgpr_count = 0;
  uint32_t kernel_agpr_count = 0;
  uint32_t kernel_shared_memory_bytes = 0;
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

  // === Stall Breakdown ===
  uint64_t stall_waitcnt_global = 0;
  uint64_t stall_waitcnt_shared = 0;
  uint64_t stall_waitcnt_private = 0;
  uint64_t stall_warp_switch = 0;
  uint64_t stall_barrier_slot = 0;
  uint64_t stall_other = 0;

  // === Instruction Mix (hardware execution unit classification) ===
  uint64_t scalar_alu_insts = 0;      // SOP1, SOP2, SOPC, SOPK (non-branch, non-sync)
  uint64_t scalar_mem_insts = 0;      // SMRD, SMEM
  uint64_t vector_alu_insts = 0;      // VOP1, VOP2, VOP3, VOPC, VINTRP
  uint64_t vector_mem_insts = 0;      // FLAT, MUBUF, MTBUF, MIMG, DS
  uint64_t branch_insts = 0;          // s_branch, s_cbranch_*
  uint64_t sync_insts = 0;            // s_barrier, s_waitcnt
  uint64_t tensor_insts = 0;          // v_mfma_*, v_accvgpr_*
  uint64_t other_insts = 0;           // s_endpgm, s_nop, mask instructions

  // === Memory Operations ===
  uint64_t global_loads = 0;
  uint64_t global_stores = 0;
  uint64_t shared_loads = 0;
  uint64_t shared_stores = 0;
  uint64_t private_loads = 0;
  uint64_t private_stores = 0;
  uint64_t scalar_loads = 0;
  uint64_t scalar_stores = 0;

  // === Memory Bytes (for bandwidth analysis) ===
  uint64_t global_load_bytes = 0;
  uint64_t global_store_bytes = 0;
  uint64_t shared_load_bytes = 0;
  uint64_t shared_store_bytes = 0;

  // === FLOPs (for compute analysis) ===
  uint64_t fp32_ops = 0;
  uint64_t fp64_ops = 0;
  uint64_t int32_ops = 0;
  uint64_t tensor_ops = 0;

  // === Wave Statistics ===
  uint32_t waves_launched = 0;
  uint32_t waves_completed = 0;
  uint32_t max_concurrent_waves = 0;

  // === Theoretical Occupancy (static, from launch analysis) ===
  uint32_t theoretical_max_waves_per_peu = 0;
  uint32_t theoretical_max_blocks_per_ap = 0;
  float theoretical_occupancy_pct = 0.0f;
  std::string occupancy_wave_limiter;
  std::string occupancy_block_limiter;

  // === Utilization ===
  double active_utilization_pct = 0.0;  // active_cycles / total_cycles * 100

  // === Performance Optimization Metrics ===
  uint64_t total_flops = 0;
  uint64_t total_bytes = 0;
  double arithmetic_intensity = 0.0;
  std::string bound_classification;  // "memory_bound", "compute_bound", "balanced"
  double bytes_per_cycle = 0.0;
  double flops_per_cycle = 0.0;
  double memory_intensity = 0.0;
  double compute_intensity = 0.0;
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
