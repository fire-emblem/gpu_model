#pragma once

#include <cstdint>

namespace gpu_model {

struct ProgramCycleStatsConfig {
  uint32_t default_issue_cycles = 4;
  uint32_t tensor_cycles = 16;
  uint32_t shared_mem_cycles = 24;     // gem5 DS latency: Cycles(24)
  uint32_t scalar_mem_cycles = 128;
  uint32_t global_mem_cycles = 1024;
  uint32_t private_mem_cycles = 1024;
  uint32_t store_latency_multiplier = 2;  // gem5: store uses 2x bus latency
};

struct ProgramCycleStats {
  // === Timing ===
  uint64_t total_cycles = 0;
  uint64_t active_cycles = 0;        // Cycles with at least one active wave
  uint64_t idle_cycles = 0;          // Cycles with no active waves

  // === Instruction Counts ===
  uint64_t instructions_executed = 0;
  uint64_t scalar_alu_insts = 0;
  uint64_t scalar_mem_insts = 0;
  uint64_t vector_alu_insts = 0;
  uint64_t vector_mem_insts = 0;
  uint64_t branch_insts = 0;
  uint64_t sync_insts = 0;
  uint64_t tensor_insts = 0;
  uint64_t other_insts = 0;

  // === Memory Operations ===
  uint64_t global_loads = 0;
  uint64_t global_stores = 0;
  uint64_t shared_loads = 0;
  uint64_t shared_stores = 0;
  uint64_t scalar_loads = 0;
  uint64_t scalar_stores = 0;
  uint64_t private_loads = 0;
  uint64_t private_stores = 0;

  // === Memory Bytes (for bandwidth analysis) ===
  uint64_t global_load_bytes = 0;
  uint64_t global_store_bytes = 0;
  uint64_t shared_load_bytes = 0;
  uint64_t shared_store_bytes = 0;

  // === FLOPs (for compute analysis) ===
  uint64_t fp32_ops = 0;             // v_add_f32, v_mul_f32, etc.
  uint64_t fp64_ops = 0;             // v_add_f64, v_mul_f64, etc.
  uint64_t int32_ops = 0;            // v_add_u32, v_mul_u32, etc.
  uint64_t tensor_ops = 0;           // v_mfma_* (counts as multiple FLOPs)

  // === Cycle Breakdown ===
  uint64_t total_issued_work_cycles = 0;
  uint64_t scalar_alu_cycles = 0;
  uint64_t vector_alu_cycles = 0;
  uint64_t tensor_cycles = 0;
  uint64_t shared_mem_cycles = 0;
  uint64_t scalar_mem_cycles = 0;
  uint64_t global_mem_cycles = 0;
  uint64_t private_mem_cycles = 0;
  uint64_t barrier_cycles = 0;
  uint64_t wait_cycles = 0;

  // === Stall Breakdown ===
  uint64_t stall_barrier = 0;        // Waiting at barrier
  uint64_t stall_waitcnt = 0;        // Waiting for memory counter
  uint64_t stall_resource = 0;       // Waiting for execution resource
  uint64_t stall_dependency = 0;     // RAW/WAW/WAR dependency
  uint64_t stall_switch_away = 0;    // Switched away for other wave

  // === Wave Statistics ===
  uint32_t waves_launched = 0;
  uint32_t waves_completed = 0;
  uint32_t max_concurrent_waves = 0;

  // === Derived Metrics ===
  double IPC() const {
    return active_cycles > 0 ?
      static_cast<double>(instructions_executed) / active_cycles : 0.0;
  }

  double WaveOccupancy() const {
    return waves_launched > 0 ?
      static_cast<double>(waves_completed) / waves_launched : 0.0;
  }

  double ActiveUtilization() const {
    return total_cycles > 0 ?
      static_cast<double>(active_cycles) / total_cycles : 0.0;
  }

  double MemoryOpFraction() const {
    uint64_t total_ops = global_loads + global_stores +
                         shared_loads + shared_stores;
    uint64_t all_ops = instructions_executed;
    return all_ops > 0 ?
      static_cast<double>(total_ops) / all_ops : 0.0;
  }

  double StallFraction() const {
    uint64_t total_stalls = stall_barrier + stall_waitcnt +
                            stall_resource + stall_dependency +
                            stall_switch_away;
    return total_cycles > 0 ?
      static_cast<double>(total_stalls) / total_cycles : 0.0;
  }

  // === Performance Optimization Metrics ===

  // Total FLOPs (floating point operations)
  uint64_t TotalFLOPs() const {
    return fp32_ops + fp64_ops * 2 + tensor_ops;
  }

  // Total bytes transferred (for bandwidth calculation)
  uint64_t TotalBytes() const {
    return global_load_bytes + global_store_bytes +
           shared_load_bytes + shared_store_bytes;
  }

  // Arithmetic Intensity (FLOPs/Byte) - key metric for Roofline analysis
  double ArithmeticIntensity() const {
    uint64_t bytes = TotalBytes();
    return bytes > 0 ? static_cast<double>(TotalFLOPs()) / bytes : 0.0;
  }

  // Memory-bound vs Compute-bound classification
  // Returns "memory_bound" if AI < 1, "compute_bound" if AI > 10, "balanced" otherwise
  const char* BoundClassification() const {
    double ai = ArithmeticIntensity();
    if (ai < 1.0) return "memory_bound";
    if (ai > 10.0) return "compute_bound";
    return "balanced";
  }

  // Memory bandwidth utilization (bytes per cycle)
  double BytesPerCycle() const {
    return total_cycles > 0 ?
      static_cast<double>(TotalBytes()) / total_cycles : 0.0;
  }

  // Compute throughput (FLOPs per cycle)
  double FLOPsPerCycle() const {
    return total_cycles > 0 ?
      static_cast<double>(TotalFLOPs()) / total_cycles : 0.0;
  }

  // Memory intensity (memory ops / total ops)
  double MemoryIntensity() const {
    uint64_t total_ops = instructions_executed;
    uint64_t mem_ops = global_loads + global_stores + shared_loads + shared_stores;
    return total_ops > 0 ? static_cast<double>(mem_ops) / total_ops : 0.0;
  }

  // Compute intensity (compute ops / total ops)
  double ComputeIntensity() const {
    uint64_t total_ops = instructions_executed;
    uint64_t compute_ops = vector_alu_insts + tensor_insts;
    return total_ops > 0 ? static_cast<double>(compute_ops) / total_ops : 0.0;
  }
};

}  // namespace gpu_model
