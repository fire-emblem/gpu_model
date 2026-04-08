#pragma once

#include <cstdint>

namespace gpu_model {

struct WaveStats {
  // === Identification ===
  uint32_t block_id = 0;
  uint32_t wave_id = 0;
  uint32_t peu_id = 0;
  uint32_t ap_id = 0;

  // === Timing ===
  uint64_t start_cycle = 0;
  uint64_t end_cycle = 0;
  uint64_t cycles_active = 0;
  uint64_t cycles_waiting = 0;

  // === Instructions ===
  uint64_t instructions_executed = 0;
  uint64_t scalar_alu_insts = 0;
  uint64_t vector_alu_insts = 0;
  uint64_t tensor_insts = 0;
  uint64_t branch_insts = 0;
  uint64_t memory_insts = 0;

  // === Memory ===
  uint64_t global_loads = 0;
  uint64_t global_stores = 0;
  uint64_t shared_loads = 0;
  uint64_t shared_stores = 0;

  // === Stalls ===
  uint64_t barrier_stalls = 0;
  uint64_t waitcnt_stalls = 0;
  uint64_t switch_away_stalls = 0;

  // === Derived Metrics ===
  uint64_t TotalCycles() const {
    return end_cycle > start_cycle ? end_cycle - start_cycle : 0;
  }

  double IPC() const {
    return cycles_active > 0 ?
      static_cast<double>(instructions_executed) / cycles_active : 0.0;
  }

  double ActiveFraction() const {
    uint64_t total = TotalCycles();
    return total > 0 ?
      static_cast<double>(cycles_active) / total : 0.0;
  }
};

}  // namespace gpu_model
