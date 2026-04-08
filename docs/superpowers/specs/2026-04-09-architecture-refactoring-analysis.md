# GPU Model Architecture Refactoring Analysis

## 1. Current Architecture Analysis

### 1.1 Architecture Strengths

1. **Clear Five-Layer Architecture**: runtime -> program -> instruction -> execution -> wave
2. **Clean Trace/Debug Separation**: `TraceSink` interface doesn't affect business logic
3. **Execution Engine Abstraction**: Pluggable `IExecutionEngine` for Functional/Cycle/ProgramObject
4. **Memory System Modularity**: Self-contained pool-based memory management
5. **ISA vs Instruction Separation**: Clean distinction between definitions and encoding

### 1.2 Architecture Weaknesses

#### Critical: Layering Violations

| Issue | File | Problem |
|-------|------|---------|
| Arch -> Execution Internal | `src/gpu_model/arch/gpu_arch_spec.h` | Includes `execution/internal/issue_model.h` |
| Execution -> Runtime | `src/gpu_model/execution/internal/semantics.h` | Includes `runtime/kernel_arg_pack.h`, `launch_config.h`, etc. |
| Debug -> Execution | `src/gpu_model/debug/trace/*.h` | Some include `wave_context.h` |

#### Large Implementation Files

| File | Lines | Issue |
|------|-------|-------|
| `program_object_exec_engine.cpp` | 2793 | Multiple responsibilities |
| `cycle_exec_engine.cpp` | 1857 | Multiple responsibilities |
| `functional_exec_engine.cpp` | 1753 | Multiple responsibilities |

#### Code Organization Issues

1. **No Public API Separation**: No `include/gpu_model/` directory
2. **Duplicate Module Paths**: Both `src/gpu_model/runtime/` and `src/runtime/` exist
3. **Spec Directory Misplaced**: `src/spec/` contains docs, not code
4. **Empty Directories**: `src/runtime/abi/`, `src/gpu_model/instruction/modeled/`

#### Test Organization Issues

1. **Duplicate Test Files**: `program_object_launch_test.cpp` in both `tests/loader/` and `tests/runtime/`
2. **Empty Test Directories**: `tests/state/`, `tests/arch/` (minimal)
3. **Instruction Tests Scattered**: Both `tests/instruction/` and `tests/execution/internal/encoded/`

---

## 2. Reference Project Patterns

### 2.1 gem5 Key Patterns

1. **Pipeline Stage Architecture**
   - `FetchStage` -> `ScoreboardCheckStage` -> `ScheduleStage` -> `ExecStage`
   - Typed buffers between stages
   - Tick-driven simulation

2. **Static vs Dynamic Instruction**
   - `GPUStaticInst`: Decoded instruction (no runtime state)
   - `GPUDynInst`: Adds execution context, sequence number, addresses
   - Enables single decode, multiple executions

3. **Wavefront State Machine**
   - Explicit states: `S_STOPPED`, `S_RUNNING`, `S_STALLED`, `S_WAITCNT`, `S_BARRIER`
   - Per-domain wait counters: `vmWaitCnt`, `lgkmWaitCnt`, `expWaitCnt`

4. **Table-Driven Decode**
   - Method pointer tables for instruction dispatch
   - Per-instruction classes with `execute(GPUDynInstPtr)`

### 2.2 MIAOW Key Patterns

1. **Encoding Union Types**: Direct bit-field mapping to hardware encoding
2. **Instruction Classification**: `SCALAR_ALU`, `VECTOR_ALU`, `SCALAR_MEM`, `VECTOR_MEM`
3. **Simple Function Pointer Dispatch**

---

## 3. Refactoring Recommendations

### 3.1 High Priority: Fix Layering Violations

#### Fix Arch -> Execution Dependency

**Current:**
```cpp
// src/gpu_model/arch/gpu_arch_spec.h
#include "gpu_model/execution/internal/issue_model.h"  // WRONG DIRECTION
```

**Proposed:**
```cpp
// src/gpu_model/arch/issue_spec.h (NEW FILE)
struct ArchitecturalIssueType { ... };
struct ArchitecturalIssueLimits { ... };
struct ArchitecturalIssuePolicy { ... };

// src/gpu_model/arch/gpu_arch_spec.h
#include "gpu_model/arch/issue_spec.h"  // CORRECT

// src/gpu_model/execution/internal/issue_model.h
#include "gpu_model/arch/issue_spec.h"  // CORRECT
```

#### Fix Execution -> Runtime Dependency

**Current:**
```cpp
// src/gpu_model/execution/internal/semantics.h
#include "runtime/kernel_arg_pack.h"     // WRONG DIRECTION
#include "runtime/launch_config.h"       // WRONG DIRECTION
#include "runtime/launch_request.h"      // WRONG DIRECTION
```

**Proposed:**
```cpp
// src/gpu_model/execution/launch_context.h (NEW FILE)
struct LaunchContext {
  uint32_t grid_size_x, grid_size_y, grid_size_z;
  uint32_t block_size_x, block_size_y, block_size_z;
  // Move essential types here
};

// src/gpu_model/execution/internal/semantics.h
#include "gpu_model/execution/launch_context.h"  // CORRECT
```

### 3.2 Medium Priority: Split Large Files

#### Split `program_object_exec_engine.cpp`

```
src/execution/program_object_exec_engine.cpp (2793 lines)
  ->
src/execution/program_object_exec_engine.cpp      (~800 lines) - main engine
src/execution/encoded_semantic_handler.cpp        (~600 lines) - semantic dispatch
src/execution/encoded_memory_handler.cpp          (~500 lines) - memory ops
src/execution/encoded_control_handler.cpp         (~400 lines) - branch/barrier
src/execution/encoded_wave_scheduler.cpp          (~400 lines) - wave scheduling
```

#### Split `cycle_exec_engine.cpp`

```
src/execution/cycle_exec_engine.cpp (1857 lines)
  ->
src/execution/cycle_exec_engine.cpp          (~600 lines) - main engine
src/execution/cycle_pipeline.cpp             (~400 lines) - pipeline stages
src/execution/cycle_wave_scheduler.cpp       (~400 lines) - wave scheduling
src/execution/cycle_memory_scheduler.cpp     (~300 lines) - memory scheduling
```

### 3.3 Medium Priority: Adopt gem5 Patterns

#### Add Dynamic Instruction Layer

```cpp
// src/gpu_model/execution/dyn_inst.h (NEW FILE)
struct DynInst {
  const Instruction* static_inst;
  uint64_t seq_num;
  WaveContext* wave;
  
  // Memory tracking
  std::vector<Addr> mem_addrs;
  MemoryOpType mem_op;
  
  // Timing
  Cycle issue_cycle;
  Cycle complete_cycle;
  
  // Completion callback
  std::function<void()> on_complete;
};
```

#### Add Pipeline Stage Buffers

```cpp
// src/gpu_model/execution/cycle/pipeline_buffers.h (NEW FILE)
struct FetchToScoreboard {
  std::vector<WaveContext*> waves;
};

struct ScoreboardToSchedule {
  std::vector<DynInst*> ready;
};

struct ScheduleToExecute {
  std::vector<DynInst*> dispatch;
};
```

#### Refine Wait Counter Model

```cpp
// src/gpu_model/execution/wait_counters.h (NEW FILE)
struct WaitCounters {
  int vm_cnt = 0;       // Vector memory
  int lgkm_cnt = 0;     // LDS/GDS/Scalar/Message
  int exp_cnt = 0;      // Export
  
  int vm_wait_target = -1;
  int lgkm_wait_target = -1;
  int exp_wait_target = -1;
  
  bool IsSatisfied() const;
  void Decrement(MemoryDomain domain);
};
```

### 3.4 Low Priority: Code Organization Cleanup

#### Establish Public API

```
include/gpu_model/
├── runtime.h           # Public runtime API
├── memory.h            # Public memory API
├── execution.h         # Public execution API
└── trace.h             # Public trace API
```

#### Consolidate Runtime Directory

```
src/runtime/
├── hip_runtime.cpp     # HIP compatibility
├── model_runtime.cpp   # Core runtime
├── exec_engine.cpp     # Execution controller
├── logging.cpp         # Logging service
└── config.cpp          # Configuration
```

#### Move Documentation

```
src/spec/ -> docs/spec/ or third_party/spec/
```

---

## 4. Test Reorganization Plan

### 4.1 Current Test Structure

```
tests/
├── arch/           # 1 test
├── cycle/          # Cycle-specific tests
├── execution/      # Execution tests
├── functional/     # Functional tests
├── instruction/    # Instruction tests
├── loader/         # Loader tests
├── memory/         # Memory tests
├── runtime/        # Runtime tests
├── state/          # EMPTY
└── test_utils/     # Test utilities
```

### 4.2 Proposed Test Structure

```
tests/
├── unit/                    # Unit tests (fast, isolated)
│   ├── arch/
│   ├── instruction/
│   ├── memory/
│   └── execution/
│       └── internal/
├── integration/             # Integration tests (slower, multi-module)
│   ├── runtime/
│   ├── loader/
│   └── program/
├── functional/              # Functional correctness tests
│   ├── st/
│   ├── mt/
│   └── cycle/
├── regression/              # Regression tests for specific issues
│   ├── waitcnt/
│   ├── barrier/
│   └── memory/
├── perf/                    # Performance benchmarks
└── test_utils/              # Shared test utilities
```

### 4.3 Test Consolidation Actions

1. **Merge Duplicate Tests**
   - `tests/loader/program_object_launch_test.cpp` + `tests/runtime/program_object_launch_test.cpp`
   - Keep the more comprehensive one, rename to `program_object_launch_integration_test.cpp`

2. **Consolidate Instruction Tests**
   - Move `tests/execution/internal/encoded/*_test.cpp` to `tests/unit/instruction/encoded/`

3. **Remove Empty Directories**
   - Remove `tests/state/` or add state machine tests

---

## 5. Implementation Priority

### Phase 1: Critical Fixes (Week 1-2)

1. Fix Arch -> Execution dependency
2. Fix Execution -> Runtime dependency
3. Add `DynInst` layer

### Phase 2: File Splits (Week 3-4)

1. Split `program_object_exec_engine.cpp`
2. Split `cycle_exec_engine.cpp`
3. Split `functional_exec_engine.cpp`

### Phase 3: Pattern Adoption (Week 5-6)

1. Add pipeline stage buffers
2. Refine wait counter model
3. Add per-domain tracking

### Phase 4: Organization (Week 7-8)

1. Establish public API headers
2. Consolidate runtime directory
3. Reorganize tests
4. Move documentation

---

## 6. Statistics Enhancement

### 6.1 Add Program Completion Stats

Current `ProgramCycleStats` should be extended:

```cpp
// src/gpu_model/runtime/program_cycle_stats.h
struct ProgramCycleStats {
  // Existing
  uint64_t total_cycles = 0;
  uint64_t active_cycles = 0;
  uint64_t stall_cycles = 0;
  
  // Proposed additions
  uint64_t issue_cycles = 0;           // Cycles spent issuing
  uint64_t memory_wait_cycles = 0;     // Cycles waiting for memory
  uint64_t barrier_wait_cycles = 0;    // Cycles at barriers
  uint64_t dispatch_cycles = 0;        // Cycles in dispatch
  
  uint64_t instructions_issued = 0;    // Total instructions
  uint64_t waves_completed = 0;        // Waves finished
  uint64_t barrier_count = 0;          // Barrier encounters
  uint64_t waitcnt_count = 0;          // Waitcnt encounters
  
  // Per-domain memory stats
  uint64_t global_loads = 0;
  uint64_t global_stores = 0;
  uint64_t shared_loads = 0;
  uint64_t shared_stores = 0;
  
  // Utilization
  double IssueUtilization() const {
    return total_cycles > 0 ? 
      static_cast<double>(issue_cycles) / total_cycles : 0.0;
  }
  double MemoryWaitFraction() const {
    return total_cycles > 0 ? 
      static_cast<double>(memory_wait_cycles) / total_cycles : 0.0;
  }
};
```

### 6.2 Add Wave-Level Stats

```cpp
// src/gpu_model/execution/wave_stats.h (NEW FILE)
struct WaveStats {
  uint64_t instructions_executed = 0;
  uint64_t cycles_active = 0;
  uint64_t cycles_waiting = 0;
  uint64_t memory_ops = 0;
  uint64_t branch_ops = 0;
  
  Cycle start_cycle;
  Cycle end_cycle;
  
  double IPC() const {
    return cycles_active > 0 ? 
      static_cast<double>(instructions_executed) / cycles_active : 0.0;
  }
};
```

---

## 7. Summary

| Category | Issues Found | Priority |
|----------|-------------|----------|
| Layering Violations | 3 | High |
| Large Files | 3 | Medium |
| Code Organization | 4 | Low |
| Test Organization | 3 | Medium |
| Missing Patterns | 4 | Medium |

**Estimated Effort**: 8 weeks for complete refactoring

**Risk Level**: Medium - Changes touch core execution paths

**Recommendation**: Implement incrementally, starting with layering fixes and `DynInst` addition.
