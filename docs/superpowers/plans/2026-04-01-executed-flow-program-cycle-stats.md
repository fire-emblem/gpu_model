# Executed Flow Program Cycle Stats Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Status (2026-04-01):** The early API/tracker slices from this plan are already present in the tree, including `program_cycle_stats.h`, `program_cycle_tracker.h/.cpp`, `tests/runtime/executed_flow_program_cycle_stats_test.cpp`, and the runtime API test mentioned below. The checklist was not backfilled, so reconcile this document with the live tree before reusing it as a TODO list.

**Goal:** Add an internal executed-flow program cycle stats path that derives global program cycle from `SingleThreaded / MarlParallel` functional wave execution flow, while exposing only a unified `ProgramCycleStats` result to callers.

**Architecture:** Keep executed-flow sampling internal to `FunctionalExecEngine`, but do not let the cycle-stats path logic depend on functional-only internals. Instead, introduce a shared `ProgramCycleTracker` and event-source abstraction so the current executed-flow cycle stats and future true `CycleExecEngine` can both project into the same `ProgramCycleStats` result shape while detailed executed-flow events remain internal and hidden from `LaunchResult`.

**Tech Stack:** C++20, gtest, existing `RuntimeEngine`, `FunctionalExecEngine`, `LaunchResult`, cycle tests, execution stats

---

## File Map

- Create: `include/gpu_model/runtime/program_cycle_stats.h`
  - Own public result/config types exposed through runtime APIs.
- Modify: `include/gpu_model/runtime/launch_request.h`
  - Add optional `ProgramCycleStats` to `LaunchResult`.
- Modify: `src/runtime/runtime_engine.cpp`
  - Populate `LaunchResult` with program cycle stats for functional `st/mt`.
- Create: `include/gpu_model/runtime/program_cycle_tracker.h`
  - Own the shared program-level cycle aggregation interface.
- Create: `src/runtime/program_cycle_tracker.cpp`
  - Own tick-based wave/program aggregation implementation.
- Modify: `src/execution/functional_exec_engine.cpp`
  - Own internal structured executed-flow sampling and the executed-flow event-source implementation.
- Modify: `include/gpu_model/execution/functional_exec_engine.h`
  - Expose only the minimal internal hook/result path needed by `RuntimeEngine`.
- Modify: `tests/runtime/execution_stats_test.cpp`
  - Lock public-facing stats/result behavior.
- Create: `tests/runtime/executed_flow_program_cycle_stats_test.cpp`
  - Own representative program cycle stats correctness and calibration tests.
- Modify: `docs/module-development-status.md`
  - Sync status after cycle-stats path lands and tests pass.

## Task 1: Add public program-cycle result/config API

**Files:**
- Create: `include/gpu_model/runtime/program_cycle_stats.h`
- Modify: `include/gpu_model/runtime/launch_request.h`
- Test: `tests/runtime/execution_stats_test.cpp`

- [ ] **Step 1: Write the failing API-level test**

Add a failing test in `tests/runtime/execution_stats_test.cpp` that proves functional launches expose program cycle stats:

```cpp
TEST(ExecutionStatsTest, FunctionalLaunchReportsProgramCycleStats) {
  RuntimeEngine runtime;
  runtime.SetFunctionalExecutionMode(FunctionalExecutionMode::SingleThreaded);

  const auto kernel = BuildStatsFunctionalKernel();
  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  EXPECT_TRUE(result.program_cycle_stats.has_value());
  EXPECT_GT(result.program_cycle_stats->total_cycles, 0u);
  EXPECT_GE(result.program_cycle_stats->total_issued_work_cycles,
            result.program_cycle_stats->total_cycles);
}
```

- [ ] **Step 2: Run the focused runtime stats test and confirm it fails**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='ExecutionStatsTest.FunctionalLaunchReportsProgramCycleStats'
```

Expected:

- FAIL because `LaunchResult` does not yet expose `program_cycle_stats`

- [ ] **Step 3: Add public result/config types**

Create `include/gpu_model/runtime/program_cycle_stats.h`:

```cpp
#pragma once

#include <cstdint>

namespace gpu_model {

struct ProgramCycleStatsConfig {
  uint32_t default_issue_cycles = 4;
  uint32_t tensor_cycles = 16;
  uint32_t shared_mem_cycles = 32;
  uint32_t scalar_mem_cycles = 128;
  uint32_t global_mem_cycles = 1024;
  uint32_t private_mem_cycles = 1024;
};

struct ProgramCycleStats {
  uint64_t total_cycles = 0;
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
};

}  // namespace gpu_model
```

Then include it from `include/gpu_model/runtime/launch_request.h` and extend `LaunchResult`:

```cpp
#include <optional>
#include "gpu_model/runtime/program_cycle_stats.h"

struct LaunchResult {
  bool ok = true;
  std::string error_message;
  uint64_t begin_cycle = 0;
  uint64_t end_cycle = 0;
  uint64_t total_cycles = 0;
  ExecutionStats stats;
  std::optional<ProgramCycleStats> program_cycle_stats;
};
```

- [ ] **Step 4: Re-run the focused API test and confirm it still fails for missing population**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='ExecutionStatsTest.FunctionalLaunchReportsProgramCycleStats'
```

Expected:

- FAIL because the field exists but functional launches do not yet populate it

- [ ] **Step 5: Commit the public API slice**

```bash
git add include/gpu_model/runtime/program_cycle_stats.h include/gpu_model/runtime/launch_request.h tests/runtime/execution_stats_test.cpp
git commit -m "feat: add program cycle stats runtime API"
```

## Task 2: Add shared program-cycle aggregation abstractions

**Files:**
- Create: `include/gpu_model/runtime/program_cycle_tracker.h`
- Create: `src/runtime/program_cycle_tracker.cpp`
- Create: `tests/runtime/executed_flow_program_cycle_stats_test.cpp`

- [ ] **Step 1: Write the failing tracker contract test**

Create `tests/runtime/executed_flow_program_cycle_stats_test.cpp` with a focused contract test:

```cpp
TEST(ExecutedFlowProgramCycleStatsTest, ProgramCycleTrackerAccumulatesWaveWorkByTick) {
  ProgramCycleTracker agg(ProgramCycleStatsConfig{});
  agg.BeginWaveWork(/*wave_id=*/0, ExecutedStepClass::VectorAlu, /*cost_cycles=*/4);
  agg.BeginWaveWork(/*wave_id=*/1, ExecutedStepClass::VectorAlu, /*cost_cycles=*/4);

  for (int i = 0; i < 4; ++i) {
    agg.AdvanceOneTick();
  }

  const auto stats = agg.Finish();
  EXPECT_EQ(stats.total_cycles, 4u);
  EXPECT_EQ(stats.total_issued_work_cycles, 8u);
  EXPECT_EQ(stats.vector_alu_cycles, 8u);
}
```

- [ ] **Step 2: Run the new tracker test and confirm it fails**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='ExecutedFlowProgramCycleStatsTest.ProgramCycleTrackerAccumulatesWaveWorkByTick'
```

Expected:

- FAIL because `ProgramCycleTracker` does not yet exist

- [ ] **Step 3: Add shared tracker and event-source interfaces**

Create `include/gpu_model/runtime/program_cycle_tracker.h`:

```cpp
#pragma once

#include <cstdint>
#include "gpu_model/runtime/program_cycle_stats.h"

namespace gpu_model {

enum class ExecutedStepClass {
  ScalarAlu,
  VectorAlu,
  Tensor,
  SharedMem,
  ScalarMem,
  GlobalMem,
  PrivateMem,
  Barrier,
  Wait,
};

class ProgramCycleTracker {
 public:
  explicit ProgramCycleTracker(ProgramCycleStatsConfig config);

  void BeginWaveWork(uint32_t wave_id, ExecutedStepClass step_class, uint64_t cost_cycles);
  void MarkWaveWaiting(uint32_t wave_id, ExecutedStepClass wait_class, uint64_t cost_cycles);
  void MarkWaveRunnable(uint32_t wave_id);
  void MarkWaveCompleted(uint32_t wave_id);
  void AdvanceOneTick();

  bool Done() const;
  ProgramCycleStats Finish() const;
};

struct ProgramCycleTickSource {
  virtual bool Done() const = 0;
  virtual void AdvanceOneTick(ProgramCycleTracker& agg) = 0;
  virtual ~ProgramCycleTickSource() = default;
};

}  // namespace gpu_model
```

Implement the minimal tick-based logic in `src/runtime/program_cycle_tracker.cpp`.

- [ ] **Step 4: Re-run the tracker-focused test and make it pass**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='ExecutedFlowProgramCycleStatsTest.ProgramCycleTrackerAccumulatesWaveWorkByTick'
```

Expected:

- PASS

- [ ] **Step 5: Commit the shared abstraction slice**

```bash
git add include/gpu_model/runtime/program_cycle_tracker.h src/runtime/program_cycle_tracker.cpp tests/runtime/executed_flow_program_cycle_stats_test.cpp
git commit -m "feat: add program cycle tracker abstractions"
```

## Task 3: Add internal structured executed-flow sampling to `FunctionalExecEngine`

**Files:**
- Modify: `src/execution/functional_exec_engine.cpp`
- Modify: `include/gpu_model/execution/functional_exec_engine.h`
- Test: `tests/runtime/executed_flow_program_cycle_stats_test.cpp`

- [ ] **Step 1: Write the failing program-cycle-stats test for a pure ALU kernel**

Extend `tests/runtime/executed_flow_program_cycle_stats_test.cpp` with an initial failing runtime-driven test:

```cpp
TEST(ExecutedFlowProgramCycleStatsTest, PureAluKernelProducesExpectedProgramCycleStats) {
  RuntimeEngine runtime;
  runtime.SetFunctionalExecutionMode(FunctionalExecutionMode::SingleThreaded);

  InstructionBuilder builder;
  builder.VMov("v1", 1);
  builder.VAdd("v2", "v1", "v1");
  builder.BExit();
  const auto kernel = builder.Build("cycle_stats_pure_alu");

  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  ASSERT_TRUE(result.program_cycle_stats.has_value());
  EXPECT_EQ(result.program_cycle_stats->vector_alu_cycles, 8u);
  EXPECT_EQ(result.program_cycle_stats->total_cycles, 8u);
}
```

- [ ] **Step 2: Run the new program-cycle-stats test and confirm it fails**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='ExecutedFlowProgramCycleStatsTest.PureAluKernelProducesExpectedProgramCycleStats'
```

Expected:

- FAIL because no internal executed-flow event source exists yet

- [ ] **Step 3: Add internal executed-flow event/state types in `functional_exec_engine.cpp`**

Add internal-only executed-flow source types near the top of `src/execution/functional_exec_engine.cpp`:

```cpp
struct ExecutedWaveStep {
  uint32_t block_id = 0;
  uint32_t wave_id = 0;
  ExecutedStepClass step_class = ExecutedStepClass::ScalarAlu;
  uint64_t cost_cycles = 0;
};

class ExecutedFlowEventSource final : public ProgramCycleTickSource {
 public:
  bool Done() const override;
  void AdvanceOneTick(ProgramCycleTracker& agg) override;
};
```

Keep them internal to `FunctionalExecEngine`; do not expose them in `LaunchResult`.

- [ ] **Step 4: Add internal sampling hooks at executed-wave decision points**

In `src/execution/functional_exec_engine.cpp`, add local helpers such as:

```cpp
ExecutedStepClass ClassifyExecutedInstruction(const Instruction& instruction);
uint64_t CostForExecutedStep(ExecutedStepClass step_class,
                             const ProgramCycleStatsConfig& config);
```

and append `ExecutedWaveStep` records when:

- an instruction actually executes
- a barrier wait/release contributes waiting work
- waitcnt waiting contributes waiting work

Do not serialize these records into trace strings. Keep them internal and feed them only into `ExecutedFlowEventSource`.

- [ ] **Step 5: Commit the internal sampling slice**

```bash
git add src/execution/functional_exec_engine.cpp include/gpu_model/execution/functional_exec_engine.h tests/runtime/executed_flow_program_cycle_stats_test.cpp
git commit -m "feat: add internal executed flow event source"
```

## Task 4: Implement the first executed-flow cycle stats on top of the shared tracker

**Files:**
- Modify: `src/execution/functional_exec_engine.cpp`
- Test: `tests/runtime/executed_flow_program_cycle_stats_test.cpp`

- [ ] **Step 1: Add failing multi-wave/barrier program-cycle-stats tests**

Extend `tests/runtime/executed_flow_program_cycle_stats_test.cpp` with focused cases:

```cpp
TEST(ExecutedFlowProgramCycleStatsTest, BarrierKernelCountsProgramCyclesByWaveAndTick) {
  auto result = LaunchBarrierKernelAndEstimate(FunctionalExecutionMode::SingleThreaded);
  ASSERT_TRUE(result.program_cycle_stats.has_value());
  EXPECT_GT(result.program_cycle_stats->barrier_cycles, 0u);
  EXPECT_GT(result.program_cycle_stats->total_cycles, 0u);
}

TEST(ExecutedFlowProgramCycleStatsTest, SingleThreadedAndMarlParallelRemainCloseOnBarrierKernel) {
  const auto st = LaunchBarrierKernelAndEstimate(FunctionalExecutionMode::SingleThreaded);
  const auto mt = LaunchBarrierKernelAndEstimate(FunctionalExecutionMode::MarlParallel);
  ASSERT_TRUE(st.program_cycle_stats.has_value());
  ASSERT_TRUE(mt.program_cycle_stats.has_value());
  EXPECT_NEAR(static_cast<double>(st.program_cycle_stats->total_cycles),
              static_cast<double>(mt.program_cycle_stats->total_cycles),
              16.0);
}
```

- [ ] **Step 2: Run the program-cycle-stats-focused suite and confirm failures**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='ExecutedFlowProgramCycleStatsTest.*'
```

Expected:

- FAIL because the internal event source is not yet driving the shared `ProgramCycleTracker`

- [ ] **Step 3: Implement the tick-based tracker**

Add an internal program-cycle-stats helper in `src/execution/functional_exec_engine.cpp` that uses the shared tracker:

```cpp
ProgramCycleStats EstimateProgramCyclesFromExecutedFlow(
    ExecutedFlowEventSource& source,
    const ProgramCycleStatsConfig& config) {
  ProgramCycleTracker agg(config);
  while (!source.Done()) {
    source.AdvanceOneTick(agg);
    agg.AdvanceOneTick();
  }
  return agg.Finish();
}
```

Keep the first version simple:

- no PEU capacity limit
- wave-level work can overlap
- waiting work blocks only the affected wave

- [ ] **Step 4: Re-run the program-cycle-stats-focused suite and calibrate only if needed**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='ExecutedFlowProgramCycleStatsTest.*'
```

Expected:

- PASS, or a small number of predictable mismatches that can be fixed by adjusting only the program-cycle-stats cost mapping or test tolerance

- [ ] **Step 5: Commit the tracker slice**

```bash
git add src/execution/functional_exec_engine.cpp tests/runtime/executed_flow_program_cycle_stats_test.cpp
git commit -m "feat: add executed flow program cycle stats"
```

## Task 5: Wire program cycle stats output into `RuntimeEngine` and compare with cycle-mode shape

**Files:**
- Modify: `src/runtime/runtime_engine.cpp`
- Modify: `tests/runtime/execution_stats_test.cpp`
- Modify: `tests/runtime/executed_flow_program_cycle_stats_test.cpp`

- [ ] **Step 1: Add a failing runtime integration test for `st/mt` result exposure**

Extend `tests/runtime/executed_flow_program_cycle_stats_test.cpp`:

```cpp
TEST(ExecutedFlowProgramCycleStatsTest, FunctionalModesExposeComparableProgramCycleStats) {
  const auto st = LaunchProgramCycleStatsKernel(FunctionalExecutionMode::SingleThreaded);
  const auto mt = LaunchProgramCycleStatsKernel(FunctionalExecutionMode::MarlParallel);

  ASSERT_TRUE(st.program_cycle_stats.has_value());
  ASSERT_TRUE(mt.program_cycle_stats.has_value());
  EXPECT_GT(st.program_cycle_stats->total_cycles, 0u);
  EXPECT_GT(mt.program_cycle_stats->total_cycles, 0u);
}
```

- [ ] **Step 2: Run the runtime program-cycle-stats suite and confirm failure before wiring**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='ExecutedFlowProgramCycleStatsTest.FunctionalModesExposeComparableProgramCycleStats'
```

Expected:

- FAIL because `RuntimeEngine` does not yet populate the program cycle stats result

- [ ] **Step 3: Populate `LaunchResult::program_cycle_stats` for functional launches**

In `src/runtime/runtime_engine.cpp`, after functional execution returns:

```cpp
if (request.mode == ExecutionMode::Functional) {
  result.program_cycle_stats = executor.TakeProgramCycleStats();
}
```

Keep cycle-mode behavior unchanged for this task unless a trivial projection already exists.

- [ ] **Step 4: Re-run focused runtime tests**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='ExecutionStatsTest.FunctionalLaunchReportsProgramCycleStats:ExecutedFlowProgramCycleStatsTest.*'
```

Expected:

- PASS
- Functional launches now expose populated program cycle stats

- [ ] **Step 5: Commit the runtime wiring slice**

```bash
git add src/runtime/runtime_engine.cpp tests/runtime/execution_stats_test.cpp tests/runtime/executed_flow_program_cycle_stats_test.cpp
git commit -m "feat: expose executed flow cycle stats through runtime"
```

## Task 6: Sync status board and final verification

**Files:**
- Modify: `docs/module-development-status.md`
- Test: `tests/runtime/executed_flow_program_cycle_stats_test.cpp`
- Test: `tests/runtime/execution_stats_test.cpp`

- [ ] **Step 1: Update the module status board**

In `docs/module-development-status.md`, add program cycle stats progress where it now belongs:

```md
| `M6` | ... | ...；functional `st/mt` 已新增基于 executed-flow 的 program cycle stats，可按 wave / tick 输出程序级 cycle 统计 | ... |
| `M10` | ... | ...；除 trace 外，runtime 结果现已暴露统一 `ProgramCycleStats`，为后续 cycle model 对齐提供统一接口 | ... |
| `M13` | ... | ...；当前 naive cycle model 之外，已新增与其兼容的 executed-flow 程序级 cycle 统计接口，可用于理论值对照和后续校准 | ... |
```

- [ ] **Step 2: Run the affected verification ring**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='ExecutionStatsTest.*:ExecutedFlowProgramCycleStatsTest.*'
```

Expected:

- PASS

- [ ] **Step 3: Run the next-larger cycle/runtime regression ring**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='ExecutionStatsTest.*:ExecutedFlowProgramCycleStatsTest.*:CycleSmokeTest.*:AsyncMemoryCycleTest.*'
```

Expected:

- PASS
- Existing cycle-mode tests are not regressed by the new public result/config types

- [ ] **Step 4: Run full project regression**

Run:

```bash
./build-ninja/tests/gpu_model_tests
```

Expected:

- PASS

- [ ] **Step 5: Commit the status/verification slice**

```bash
git add docs/module-development-status.md
git commit -m "docs: record executed flow program cycle stats progress"
```

## Self-Review

- Spec coverage:
  - Shared tracker and event-source abstraction: Task 2
  - Internal structured executed-flow events only: Task 3
  - Public `ProgramCycleStats` API only: Tasks 1 and 5
  - Tick-based wave-aware global cycle accumulation: Tasks 2 and 4
  - `st/mt` representative validation and calibration hooks: Tasks 4, 5, 6
  - Cycle-model compatibility boundary: Tasks 2, 4, 5, 6
- Placeholder scan:
  - No `TODO` / `TBD` placeholders remain
  - All steps include exact files, code, and commands
- Type consistency:
  - Plan consistently uses `ProgramCycleStats`, `ProgramCycleStatsConfig`, `ProgramCycleTracker`, `ProgramCycleTickSource`, `ExecutedFlowEventSource`, and internal-only executed-flow sampling
