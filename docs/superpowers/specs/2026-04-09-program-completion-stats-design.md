# Program Completion Statistics Design

## Overview

Add comprehensive program completion statistics inspired by gem5 GPU stats and Nsight Compute performance counters.

## Reference: gem5 Statistics

### WavefrontStats (per-wave slot)
- `numInstrExecuted` - Total instructions executed
- `schCycles` - Cycles in scheduler stage
- `schStalls` - Stall cycles encountered
- `schRfAccessStalls` - Register file access stalls
- `schResourceStalls` - Execution resource stalls
- `schOpdNrdyStalls` - Operand not ready stalls
- `schLdsArbStalls` - LDS arbitration stalls
- `numTimesBlockedDueWAXDependencies` - WAR/WAW dependency blocks
- `numTimesBlockedDueRAWDependencies` - RAW dependency blocks
- `vecRawDistance` - Distribution of RAW distances
- `readsPerWrite` - Register read/write ratio

### ComputeUnitStats (per-CU)
- `vALUInsts` / `sALUInsts` - Vector/Scalar ALU instructions
- `instCyclesVALU` / `instCyclesSALU` - Cycles in VALU/SALU
- `threadCyclesVALU` - Thread-cycles in VALU
- `vectorMemReads` / `vectorMemWrites` - Vector memory ops
- `scalarMemReads` / `scalarMemWrites` - Scalar memory ops
- `globalReads` / `globalWrites` - Global memory ops
- `groupReads` / `groupWrites` - LDS (shared) memory ops
- `privReads` / `privWrites` - Private memory ops
- `ldsBankAccesses` / `ldsBankConflictDist` - LDS bank conflicts
- `waveLevelParallelism` - Active waves distribution
- `numInstrExecuted` - Total instructions
- `execRateDist` - Instruction execution rate distribution

### ShaderStats (global)
- `allLatencyDist` / `loadLatencyDist` / `storeLatencyDist` - Memory latency
- `shaderActiveTicks` - Active shader cycles

## Reference: Nsight Compute Metrics

### Key Metric Groups
1. **SASS Metrics**: Instructions executed by type
2. **Memory Throughput**: Global/Shared memory bandwidth
3. **Warp Execution**: Active warps, warp occupancy
4. **Stall Reasons**: Per-category stall breakdown
5. **Cache Metrics**: Hit rate, miss rate, bytes transferred
6. **SM Utilization**: Compute throughput, memory throughput

### Key Counters
- `inst_executed` - Total instructions
- `inst_executed_per_warp` - Instructions per warp
- `warp_executed` - Warps executed
- `active_cycles` - Cycles with active warps
- `stall_*` - Various stall reasons
- `shared_load_transactions` / `shared_store_transactions`
- `global_load_transactions` / `global_store_transactions`
- `l2_read_throughput` / `l2_write_throughput`

## Proposed Design

### 1. Extended ProgramCycleStats

```cpp
// src/gpu_model/runtime/program_cycle_stats.h

struct ProgramCycleStats {
  // === Timing ===
  uint64_t total_cycles = 0;
  uint64_t active_cycles = 0;        // Cycles with at least one active wave
  uint64_t idle_cycles = 0;          // Cycles with no active waves

  // === Instruction Counts ===
  uint64_t instructions_executed = 0;
  uint64_t scalar_alu_insts = 0;
  uint64_t vector_alu_insts = 0;
  uint64_t tensor_insts = 0;
  uint64_t branch_insts = 0;
  uint64_t barrier_insts = 0;

  // === Memory Operations ===
  uint64_t global_loads = 0;
  uint64_t global_stores = 0;
  uint64_t shared_loads = 0;
  uint64_t shared_stores = 0;
  uint64_t scalar_loads = 0;
  uint64_t scalar_stores = 0;
  uint64_t private_loads = 0;
  uint64_t private_stores = 0;

  // === Cycle Breakdown ===
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
};
```

### 2. WaveStats (Per-Wave Statistics)

```cpp
// src/gpu_model/execution/wave_stats.h

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
```

### 3. StatsCollector (Collection Infrastructure)

```cpp
// src/gpu_model/execution/stats_collector.h

class StatsCollector {
 public:
  // Record wave lifecycle
  void RecordWaveLaunch(const WaveContext& wave);
  void RecordWaveComplete(const WaveContext& wave);

  // Record instruction execution
  void RecordInstruction(const WaveContext& wave, const Instruction* inst,
                         uint64_t issue_cycles);

  // Record stalls
  void RecordStall(WaitReason reason, uint64_t cycles);

  // Record memory operations
  void RecordMemoryOp(MemoryDomain domain, bool is_load, uint64_t bytes);

  // Get aggregated stats
  ProgramCycleStats GetAggregatedStats() const;
  std::vector<WaveStats> GetWaveStats() const;

 private:
  std::vector<WaveStats> wave_stats_;
  uint64_t current_cycle_ = 0;
  uint32_t active_waves_ = 0;
  uint32_t max_concurrent_waves_ = 0;
};
```

## Implementation Plan

### Phase 1: Extend ProgramCycleStats (Day 1)
1. Add new fields to `ProgramCycleStats`
2. Add derived metric methods
3. Update `CycleExecEngine` to populate new fields

### Phase 2: Add WaveStats (Day 2)
1. Create `wave_stats.h`
2. Add `WaveStats` member to `WaveContext`
3. Update cycle engine to track per-wave stats

### Phase 3: Add StatsCollector (Day 3)
1. Create `stats_collector.h/cpp`
2. Integrate with cycle engine
3. Add JSON output for stats

### Phase 4: Output Integration (Day 4)
1. Add stats to trace output
2. Add stats to Perfetto timeline
3. Add standalone stats JSON file

## Output Format

### JSON Stats Output

```json
{
  "program_stats": {
    "total_cycles": 1000000,
    "active_cycles": 850000,
    "idle_cycles": 150000,
    "instructions_executed": 500000,
    "ipc": 0.588,
    "active_utilization": 0.85,
    "waves_launched": 128,
    "waves_completed": 128,
    "stall_fraction": 0.12
  },
  "instruction_breakdown": {
    "scalar_alu": 100000,
    "vector_alu": 300000,
    "tensor": 0,
    "branch": 5000,
    "barrier": 2000,
    "memory": 93000
  },
  "memory_stats": {
    "global_loads": 50000,
    "global_stores": 20000,
    "shared_loads": 15000,
    "shared_stores": 8000
  },
  "stall_breakdown": {
    "barrier": 5000,
    "waitcnt": 80000,
    "resource": 10000,
    "dependency": 15000,
    "switch_away": 20000
  },
  "wave_stats": [
    {
      "block_id": 0,
      "wave_id": 0,
      "start_cycle": 0,
      "end_cycle": 50000,
      "instructions_executed": 4000,
      "ipc": 0.08
    }
  ]
}
```

## Testing

1. Unit tests for stats aggregation
2. Integration tests with known workloads
3. Validation against gem5 reference outputs
