# Functional MT Wave Scheduler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Change `FunctionalExecutionMode::MarlParallel` so the functional model executes runnable work at `wave` granularity with a global worker pool and AP-local runnable queues, while keeping block barrier and wait semantics correct and keeping cycle-model concepts out of the functional executor.

**Architecture:** Keep `SingleThreaded` unchanged. Replace the current block-parallel `MarlParallel` path in `FunctionalExecEngine` with a wave scheduler: all launched waves are materialized immediately, each AP owns runnable/waiting bookkeeping for its waves, and a single shared worker pool pulls wave work across AP-local queues. Block barrier state remains block-local and only requeues that block’s waves on release. Worker budget is global, with default size `max(1, floor(cpu_cores * 0.9))`.

**Tech Stack:** C++20, marl scheduler already vendored in-tree, existing `FunctionalExecEngine`, `WaveContext`, trace sink, gtest

---

### Task 1: Lock The New MT Semantics With Focused Failing Tests

**Files:**
- Modify: `tests/functional/shared_barrier_functional_test.cpp`
- Modify: `tests/functional/shared_sync_functional_test.cpp`
- Modify: `tests/functional/waitcnt_functional_test.cpp`
- Modify: `tests/runtime/parallel_execution_mode_test.cpp`
- Modify: `src/runtime/exec_engine.cpp`

- [x] **Step 1: Add a failing same-AP cross-block progress regression**

Add a focused functional regression showing that when one block has waves waiting at a block barrier, runnable waves from a different block on the same AP can still make progress in `MarlParallel`.

- [x] **Step 2: Add a failing same-block wave concurrency regression**

Add a regression showing that different waves of the same block can both execute in `MarlParallel` before barrier release, instead of the current implicit block-serial behavior.

- [x] **Step 3: Add a failing runtime default worker-count regression**

Extend `tests/runtime/parallel_execution_mode_test.cpp` so the default `MarlParallel` worker budget expectation matches `floor(cpu_cores * 0.9)` with a minimum of `1`.

- [x] **Step 4: Run the focused failing ring**

Run:
```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='ParallelExecutionModeTest.*:WaitcntFunctionalTest.*:SharedSyncFunctionalTest.*:SharedBarrierFunctionalTest.*'
```

Expected: at least one new regression fails because `MarlParallel` is still block-parallel and the default worker heuristic is still `2/3`.

### Task 2: Replace Block-Parallel MT With Wave-Parallel Scheduling

**Files:**
- Modify: `src/execution/functional_exec_engine.cpp`

- [x] **Step 1: Introduce AP-local scheduler state**

Restructure the parallel executor around AP-local ownership:

```cpp
struct ApSchedulerState {
  std::mutex mutex;
  std::deque<WaveTaskRef> runnable;
  uint64_t rr_cursor = 0;
};
```

Use `WaveTaskRef` to identify `{block_index, wave_index}` instead of scheduling whole blocks.

- [x] **Step 2: Materialize all launched waves into AP-local runnable queues**

Build initial runnable state once at launch:

```cpp
for each block:
  for each wave in block:
    if wave is active/runnable:
      ap_states[global_ap_id].runnable.push_back({block_index, wave_index});
```

Do not add resident-capacity logic here.

- [x] **Step 3: Change the worker body from `ExecuteBlock(...)` to `ExecuteOneWaveTask(...)`**

Replace the current per-block scheduled lambda with a wave scheduler loop:

```cpp
while (TryPopRunnableWave(ap_states, task)) {
  ExecuteWaveTask(task, stats);
  RequeueOrRetire(task);
}
```

Each task execution advances one wave until it:
- completes one instruction and stays runnable
- enters waitcnt/memory waiting
- enters block barrier waiting
- exits

- [x] **Step 4: Keep barrier state block-local and requeue on release**

When `SyncBarrier` releases:

```cpp
for each released wave in that block:
  wave.run_state = Runnable;
  ap_states[wave.global_ap_id].runnable.push_back(task);
```

Do not release unrelated blocks’ waves.

- [x] **Step 5: Keep waitcnt/memory wait wave-local and AP-requeue on readiness**

When waiting conditions clear:

```cpp
if (ResumeWaveIfWaitSatisfied(...)) {
  ap_states[wave.global_ap_id].runnable.push_back(task);
}
```

- [x] **Step 6: Preserve trace/stat collection under existing locks**

Do not loosen correctness around:
- `TraceEventLocked(...)`
- `CommitStats(...)`
- executed-flow cycle stats collection

### Task 3: Update Marl Default Worker Budget To 90 Percent Of CPU Cores

**Files:**
- Modify: `src/runtime/exec_engine.cpp`

- [x] **Step 1: Change the default worker heuristic**

Update:

```cpp
uint32_t DefaultMarlWorkerThreadCount() {
  const uint32_t cpu_count = std::max(1u, std::thread::hardware_concurrency());
  return std::max(1u, static_cast<uint32_t>(cpu_count * 9u / 10u));
}
```

- [x] **Step 2: Re-run the focused runtime expectation**

Run:
```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='ParallelExecutionModeTest.*'
```

Expected: PASS

### Task 4: Verify Wave-Parallel MT Behavior And No Regressions

**Files:**
- Verify only

- [x] **Step 1: Run the functional/race-sensitive focused ring**

Run:
```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='ParallelExecutionModeTest.*:WaitcntFunctionalTest.*:SharedSyncFunctionalTest.*:SharedBarrierFunctionalTest.*:ExecutedFlowProgramCycleStatsTest.*'
```

Expected: PASS

- [x] **Step 2: Run representative example scripts**

Run:
```bash
examples/03-shared-reverse/run.sh
examples/04-atomic-reduction/run.sh
examples/07-vecadd-cycle-splitting/run.sh
```

Expected: PASS, with `results/st`, `results/mt`, and `results/cycle` populated as before.

- [x] **Step 3: Commit**

Run:
```bash
git add docs/superpowers/specs/2026-04-02-functional-mt-wave-scheduler-design.md \
        docs/superpowers/plans/2026-04-02-functional-mt-wave-scheduler.md \
        src/execution/functional_exec_engine.cpp \
        src/runtime/exec_engine.cpp \
        tests/functional/shared_barrier_functional_test.cpp \
        tests/functional/shared_sync_functional_test.cpp \
        tests/functional/waitcnt_functional_test.cpp \
        tests/runtime/parallel_execution_mode_test.cpp
git commit -m "refactor: schedule functional mt at wave granularity"
```
