# Cycle AP Resident Blocks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Status (2026-04-01):** The focused `CycleApResidentBlocksTest` cases and their test file already exist in the tree. The checklist was not backfilled, so treat this plan as historical implementation notes unless you first reconcile it with current source/tests.

**Goal:** Add cycle-path AP multi-block residency so one `AP` can hold `2` resident blocks, making `>4 resident waves / PEU` reachable and enabling a real `active-window / standby-window` front-end model.

**Architecture:** Keep `Mapper` unchanged and treat `block -> global_ap_id` / `wave -> peu_id` as static placement only. Implement all new runtime residency in `CycleExecEngine`: first add `ApResidentState` for `pending -> resident -> retired` block lifecycle, then add `PEU`-local `resident_waves / active_window / standby_waves` so `dispatch_enabled` means “currently in the active window”, not merely “belongs to an active block”. `waitcnt/dependency/front_end_wait` blocked waves stay in-window; `block barrier` waiting waves stay resident but must yield the active slot so overflow waves can continue toward the barrier.

**Tech Stack:** C++20, gtest, existing `CycleExecEngine`, trace events, `RuntimeEngine`, cycle smoke tests

---

## File Map

- Modify: `src/execution/cycle_exec_engine.cpp`
  - Add AP resident block state, PEU active/standby wave state, and retire/backfill logic.
- Create: `tests/cycle/cycle_ap_resident_blocks_test.cpp`
  - Add focused cycle regressions for AP multi-block residency and PEU active-window promotion.
- Modify: `tests/CMakeLists.txt`
  - Register the new cycle resident-block test file.
- Modify: `docs/module-development-status.md`
  - Record that cycle now has a reference model for multi-block resident AP scheduling and active-window/standby behavior.

## Task 1: Add failing AP resident admission/backfill regressions

**Files:**
- Create: `tests/cycle/cycle_ap_resident_blocks_test.cpp`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Write the new focused cycle resident-block test file**

Create `tests/cycle/cycle_ap_resident_blocks_test.cpp` with local helpers and the first two failing tests:

```cpp
#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <vector>

#include "gpu_model/arch/arch_registry.h"
#include "gpu_model/debug/trace_sink.h"
#include "gpu_model/isa/instruction_builder.h"
#include "gpu_model/runtime/runtime_engine.h"

namespace gpu_model {
namespace {

uint32_t WrappedBlockId(const GpuArchSpec& spec, uint32_t ordinal) {
  return ordinal * spec.total_ap_count();
}

ExecutableKernel BuildCycleResidentExitKernel() {
  InstructionBuilder builder;
  builder.VMov("v0", 1);
  builder.BExit();
  return builder.Build("cycle_resident_exit_kernel");
}

size_t FirstBlockLaunchIndex(const std::vector<TraceEvent>& events, uint32_t block_id) {
  for (size_t i = 0; i < events.size(); ++i) {
    if (events[i].kind == TraceEventKind::BlockLaunch && events[i].block_id == block_id) {
      return i;
    }
  }
  return std::numeric_limits<size_t>::max();
}

size_t FirstWaveExitIndex(const std::vector<TraceEvent>& events, uint32_t block_id) {
  for (size_t i = 0; i < events.size(); ++i) {
    if (events[i].kind == TraceEventKind::WaveExit && events[i].block_id == block_id) {
      return i;
    }
  }
  return std::numeric_limits<size_t>::max();
}

TEST(CycleApResidentBlocksTest, SingleApAdmitsTwoResidentBlocksBeforeBackfillingThird) {
  CollectingTraceSink trace;
  RuntimeEngine runtime(&trace);
  runtime.SetLaunchTimingProfile(/*kernel_launch_gap_cycles=*/8,
                                 /*kernel_launch_cycles=*/0,
                                 /*block_launch_cycles=*/0,
                                 /*wave_launch_cycles=*/0,
                                 /*warp_switch_cycles=*/1,
                                 /*arg_load_cycles=*/4);

  const auto spec = ArchRegistry::Get("c500");
  ASSERT_NE(spec, nullptr);
  const auto kernel = BuildCycleResidentExitKernel();

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 2 * spec->total_ap_count() + 1;
  request.config.block_dim_x = 64;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  const uint32_t block0 = WrappedBlockId(*spec, 0);
  const uint32_t block1 = WrappedBlockId(*spec, 1);
  const uint32_t block2 = WrappedBlockId(*spec, 2);
  ASSERT_NE(FirstBlockLaunchIndex(trace.events(), block0), std::numeric_limits<size_t>::max());
  ASSERT_NE(FirstBlockLaunchIndex(trace.events(), block1), std::numeric_limits<size_t>::max());
  ASSERT_NE(FirstBlockLaunchIndex(trace.events(), block2), std::numeric_limits<size_t>::max());

  EXPECT_EQ(trace.events()[FirstBlockLaunchIndex(trace.events(), block0)].cycle, 0u);
  EXPECT_EQ(trace.events()[FirstBlockLaunchIndex(trace.events(), block1)].cycle, 0u);
  EXPECT_GT(trace.events()[FirstBlockLaunchIndex(trace.events(), block2)].cycle, 0u);
}

TEST(CycleApResidentBlocksTest, RetiredBlockBackfillsPendingBlockOnSameAp) {
  CollectingTraceSink trace;
  RuntimeEngine runtime(&trace);
  runtime.SetLaunchTimingProfile(/*kernel_launch_gap_cycles=*/8,
                                 /*kernel_launch_cycles=*/0,
                                 /*block_launch_cycles=*/0,
                                 /*wave_launch_cycles=*/0,
                                 /*warp_switch_cycles=*/1,
                                 /*arg_load_cycles=*/4);

  const auto spec = ArchRegistry::Get("c500");
  ASSERT_NE(spec, nullptr);
  const auto kernel = BuildCycleResidentExitKernel();

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = 2 * spec->total_ap_count() + 1;
  request.config.block_dim_x = 64;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  const uint32_t block0 = WrappedBlockId(*spec, 0);
  const uint32_t block2 = WrappedBlockId(*spec, 2);
  const size_t block0_exit = FirstWaveExitIndex(trace.events(), block0);
  const size_t block2_launch = FirstBlockLaunchIndex(trace.events(), block2);
  ASSERT_NE(block0_exit, std::numeric_limits<size_t>::max());
  ASSERT_NE(block2_launch, std::numeric_limits<size_t>::max());
  EXPECT_LT(block0_exit, block2_launch);
}

}  // namespace
}  // namespace gpu_model
```

Also register the file in `tests/CMakeLists.txt`:

```cmake
  cycle/cycle_ap_resident_blocks_test.cpp
```

- [ ] **Step 2: Build and run the new resident-block tests to confirm they fail**

Run:

```bash
cmake --build /data/gpu_model/build-ninja --target gpu_model_tests -j4
./build-ninja/tests/gpu_model_tests --gtest_filter='CycleApResidentBlocksTest.SingleApAdmitsTwoResidentBlocksBeforeBackfillingThird:CycleApResidentBlocksTest.RetiredBlockBackfillsPendingBlockOnSameAp'
```

Expected:

- FAIL because current cycle front-end only activates `ap_queue.front()`
- `block1` for the wrapped `AP` does not launch at cycle `0`

- [ ] **Step 3: Commit the failing admission/backfill test slice**

```bash
git add tests/CMakeLists.txt tests/cycle/cycle_ap_resident_blocks_test.cpp
git commit -m "test: add cycle ap resident block regressions"
```

## Task 2: Implement AP resident block admission and pending backfill

**Files:**
- Modify: `src/execution/cycle_exec_engine.cpp`

- [ ] **Step 1: Add AP resident block state and initialization**

Extend the internal cycle state with a dedicated AP resident container:

```cpp
struct ApResidentState {
  uint32_t global_ap_id = 0;
  std::deque<ExecutableBlock*> pending_blocks;
  std::vector<ExecutableBlock*> resident_blocks;
  uint32_t resident_block_limit = 2;
};
```

Build it during `Run()` instead of the current `ap_queues`-only setup:

```cpp
std::map<uint32_t, ApResidentState> ap_states;
for (auto& block : blocks) {
  auto& state = ap_states[block.global_ap_id];
  state.global_ap_id = block.global_ap_id;
  state.pending_blocks.push_back(&block);
}
```

- [ ] **Step 2: Replace one-block activation with resident admission helpers**

Add local helpers in `src/execution/cycle_exec_engine.cpp`:

```cpp
void AdmitResidentBlocks(ApResidentState& ap_state,
                         uint64_t cycle,
                         uint32_t max_issuable_waves,
                         uint64_t wave_launch_cycles,
                         EventQueue& events,
                         TraceSink& trace) {
  while (ap_state.resident_blocks.size() < ap_state.resident_block_limit &&
         !ap_state.pending_blocks.empty()) {
    ExecutableBlock* block = ap_state.pending_blocks.front();
    ap_state.pending_blocks.pop_front();
    ap_state.resident_blocks.push_back(block);
    ActivateBlock(*block, cycle, max_issuable_waves, wave_launch_cycles, events, trace);
  }
}

bool RetireResidentBlock(ApResidentState& ap_state, ExecutableBlock* block) {
  auto it = std::find(ap_state.resident_blocks.begin(), ap_state.resident_blocks.end(), block);
  if (it == ap_state.resident_blocks.end()) {
    return false;
  }
  ap_state.resident_blocks.erase(it);
  return true;
}
```

Then use `AdmitResidentBlocks()` at startup and after block retirement.

- [ ] **Step 3: Wire block-exit backfill through the new AP state**

Replace the current `ap_queues`-based “next block” scheduling in the `exit_wave` path with resident-block retire + readmit:

```cpp
if (candidate->block->active && !candidate->block->completed &&
    AllWavesExited(*candidate->block)) {
  candidate->block->active = false;
  candidate->block->completed = true;
  auto& ap_state = ap_states.at(candidate->block->global_ap_id);
  const bool removed = RetireResidentBlock(ap_state, candidate->block);
  if (removed) {
    events.Schedule(TimedEvent{
        .cycle = commit_cycle + timing_config_.launch_timing.block_launch_cycles,
        .action = [&, global_ap_id = candidate->block->global_ap_id, commit_cycle]() {
          AdmitResidentBlocks(ap_states.at(global_ap_id),
                              commit_cycle + timing_config_.launch_timing.block_launch_cycles,
                              context.spec.max_issuable_waves,
                              timing_config_.launch_timing.wave_launch_cycles,
                              events,
                              context.trace);
        },
    });
  }
}
```

- [ ] **Step 4: Re-run the admission/backfill tests and make them pass**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='CycleApResidentBlocksTest.SingleApAdmitsTwoResidentBlocksBeforeBackfillingThird:CycleApResidentBlocksTest.RetiredBlockBackfillsPendingBlockOnSameAp'
```

Expected:

- PASS

- [ ] **Step 5: Commit the AP resident block implementation**

```bash
git add src/execution/cycle_exec_engine.cpp
git commit -m "feat: add cycle ap resident block admission"
```

## Task 3: Add failing active-window/standby regressions

**Files:**
- Modify: `tests/cycle/cycle_ap_resident_blocks_test.cpp`

- [ ] **Step 1: Add async resident-overflow kernels and launch-count helpers**

Extend `tests/cycle/cycle_ap_resident_blocks_test.cpp`:

```cpp
ExecutableKernel BuildCycleResidentAsyncLoadKernel(uint64_t base_addr) {
  InstructionBuilder builder;
  builder.SMov("s0", base_addr);
  builder.SMov("s1", 0);
  builder.MLoadGlobal("v0", "s0", "s1", 4);
  builder.BExit();
  return builder.Build("cycle_resident_async_load_kernel");
}

uint32_t CountWaveLaunchesForBlockAtCycle(const std::vector<TraceEvent>& events,
                                         uint32_t block_id,
                                         uint64_t cycle) {
  uint32_t count = 0;
  for (const auto& event : events) {
    if (event.kind == TraceEventKind::WaveLaunch &&
        event.block_id == block_id &&
        event.cycle == cycle) {
      ++count;
    }
  }
  return count;
}

size_t FirstWaveLaunchIndexForBlock(const std::vector<TraceEvent>& events, uint32_t block_id) {
  for (size_t i = 0; i < events.size(); ++i) {
    if (events[i].kind == TraceEventKind::WaveLaunch && events[i].block_id == block_id) {
      return i;
    }
  }
  return std::numeric_limits<size_t>::max();
}
```

- [ ] **Step 2: Write the failing active-window/standby tests**

Add these tests:

```cpp
TEST(CycleApResidentBlocksTest, ResidentStandbyBlockDoesNotLaunchWavesUntilActiveSlotOpens) {
  CollectingTraceSink trace;
  RuntimeEngine runtime(&trace);
  runtime.SetFixedGlobalMemoryLatency(40);
  runtime.SetLaunchTimingProfile(/*kernel_launch_gap_cycles=*/8,
                                 /*kernel_launch_cycles=*/0,
                                 /*block_launch_cycles=*/0,
                                 /*wave_launch_cycles=*/0,
                                 /*warp_switch_cycles=*/1,
                                 /*arg_load_cycles=*/4);

  const auto spec = ArchRegistry::Get("c500");
  ASSERT_NE(spec, nullptr);
  const uint64_t base_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(base_addr, 7);
  const auto kernel = BuildCycleResidentAsyncLoadKernel(base_addr);

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = spec->total_ap_count() + 1;
  request.config.block_dim_x = 1024;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  const uint32_t block0 = WrappedBlockId(*spec, 0);
  const uint32_t block1 = WrappedBlockId(*spec, 1);
  EXPECT_EQ(trace.events()[FirstBlockLaunchIndex(trace.events(), block0)].cycle, 0u);
  EXPECT_EQ(trace.events()[FirstBlockLaunchIndex(trace.events(), block1)].cycle, 0u);
  EXPECT_EQ(CountWaveLaunchesForBlockAtCycle(trace.events(), block0, 0u), 16u);
  EXPECT_EQ(CountWaveLaunchesForBlockAtCycle(trace.events(), block1, 0u), 0u);
}

TEST(CycleApResidentBlocksTest, StandbyWavePromotesAfterActiveWaveExits) {
  CollectingTraceSink trace;
  RuntimeEngine runtime(&trace);
  runtime.SetFixedGlobalMemoryLatency(40);
  runtime.SetLaunchTimingProfile(/*kernel_launch_gap_cycles=*/8,
                                 /*kernel_launch_cycles=*/0,
                                 /*block_launch_cycles=*/0,
                                 /*wave_launch_cycles=*/0,
                                 /*warp_switch_cycles=*/1,
                                 /*arg_load_cycles=*/4);

  const auto spec = ArchRegistry::Get("c500");
  ASSERT_NE(spec, nullptr);
  const uint64_t base_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(base_addr, 7);
  const auto kernel = BuildCycleResidentAsyncLoadKernel(base_addr);

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = spec->total_ap_count() + 1;
  request.config.block_dim_x = 1024;

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  const uint32_t block0 = WrappedBlockId(*spec, 0);
  const uint32_t block1 = WrappedBlockId(*spec, 1);
  const size_t block0_exit = FirstWaveExitIndex(trace.events(), block0);
  const size_t block1_launch = FirstWaveLaunchIndexForBlock(trace.events(), block1);
  ASSERT_NE(block0_exit, std::numeric_limits<size_t>::max());
  ASSERT_NE(block1_launch, std::numeric_limits<size_t>::max());
  EXPECT_LT(block0_exit, block1_launch);
}
```

- [ ] **Step 3: Build and run the active-window tests to confirm they fail**

Run:

```bash
cmake --build /data/gpu_model/build-ninja --target gpu_model_tests -j4
./build-ninja/tests/gpu_model_tests --gtest_filter='CycleApResidentBlocksTest.ResidentStandbyBlockDoesNotLaunchWavesUntilActiveSlotOpens:CycleApResidentBlocksTest.StandbyWavePromotesAfterActiveWaveExits'
```

Expected:

- FAIL because current cycle front-end still launches up to `max_issuable_waves` per resident block rather than per `PEU` active window
- or FAIL because the second block does not become resident at cycle `0`

- [ ] **Step 4: Commit the failing active-window regression slice**

```bash
git add tests/cycle/cycle_ap_resident_blocks_test.cpp
git commit -m "test: add cycle active window regressions"
```

## Task 4: Implement PEU resident wave pools, active windows, barrier slot-yield, and standby promotion

**Files:**
- Modify: `src/execution/cycle_exec_engine.cpp`

- [ ] **Step 1: Add PEU resident-state tracking**

Extend the cycle front-end internals:

```cpp
struct PeuSlot {
  uint32_t dpc_id = 0;
  uint32_t ap_id = 0;
  uint32_t peu_id = 0;
  uint64_t busy_until = 0;
  uint64_t last_wave_tag = std::numeric_limits<uint64_t>::max();
  size_t last_issue_index = std::numeric_limits<size_t>::max();
  std::vector<ScheduledWave*> resident_waves;
  std::vector<ScheduledWave*> active_window;
  std::deque<ScheduledWave*> standby_waves;
};
```

During block admission, register the block’s waves into `slot.resident_waves`.

- [ ] **Step 2: Make `dispatch_enabled` mean “currently in active window”**

Add a refill helper:

```cpp
void RefillPeuActiveWindow(PeuSlot& slot,
                           uint64_t cycle,
                           uint32_t active_wave_limit,
                           EventQueue& events,
                           TraceSink& trace) {
  while (slot.active_window.size() < active_wave_limit && !slot.standby_waves.empty()) {
    ScheduledWave* wave = slot.standby_waves.front();
    slot.standby_waves.pop_front();
    wave->dispatch_enabled = true;
    slot.active_window.push_back(wave);
    ScheduleWaveLaunch(*wave, cycle, events, trace);
  }
}
```

And update `ActivateBlock()` so resident waves are inserted into either `active_window` or `standby_waves`, instead of scheduling up to `max_issuable_waves` per block.

- [ ] **Step 3: Keep non-barrier blocked waves resident/in-window, but let barrier waiting waves yield active slots**

Adjust `FillDispatchWindow()`, `PickNextReadyWave()`, and the `exit_wave` path so:

```cpp
// exited or retired waves leave the resident/active sets
if (plan.exit_wave) {
  RemoveWaveFromActiveWindow(slot, candidate);
  RefillPeuActiveWindow(slot, commit_cycle, /*active_wave_limit=*/4, events, context.trace);
}
```

Do not remove a wave from `active_window` merely because `IssueBlockReason()` reports
`waitcnt`, `dependency`, or `front_end_wait`.

For `block barrier`, use a separate path:

```cpp
if (plan.sync_barrier) {
  RemoveWaveFromActiveWindow(slot, candidate);
  candidate->dispatch_enabled = false;
  RefillPeuActiveWindow(slot, commit_cycle, /*active_wave_limit=*/4, events, context.trace);
}
```

The barrier-waiting wave must remain resident, and after barrier release it must be able to re-enter
the active window through the normal refill path.

- [ ] **Step 4: Add and pass the overflow-barrier focused regression**

Before claiming Task 4 complete, add one focused regression to
`tests/cycle/cycle_ap_resident_blocks_test.cpp`:

```cpp
TEST(CycleApResidentBlocksTest, BarrierWaitingResidentWaveYieldsActiveSlotUntilRelease) {
  // Build an overflow block where the first active waves reach SyncBarrier early.
  // Assert that:
  // - overflow resident waves are still able to launch before any active wave exits
  // - barrier release eventually occurs
  // - early barrier-waiting waves can continue after release
}
```

Then run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='CycleApResidentBlocksTest.*'
```

Expected:

- PASS, including the new barrier-overflow regression

- [ ] **Step 5: Re-run the focused cycle resident-block test suite and make it pass**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='CycleApResidentBlocksTest.*'
```

Expected:

- PASS for admission/backfill
- PASS for resident standby behavior
- PASS for active-window promotion after an exit
- PASS for barrier-wait slot yielding and post-release refill

- [ ] **Step 6: Commit the cycle active-window implementation**

```bash
git add src/execution/cycle_exec_engine.cpp
git commit -m "feat: add cycle peu active and standby windows"
```

## Task 5: Update status tracking and run final verification

**Files:**
- Modify: `docs/module-development-status.md`

- [ ] **Step 1: Update the cycle-model status entry**

In `docs/module-development-status.md`, extend the `M13` narrative with the new reference model facts:

```md
- cycle front-end 现已支持同一 `AP` 最多 `2` 个 resident blocks
- 同一 `PEU` 上 `>4 resident waves` 现已在 cycle 路径可达
- `active_window = 4` / `standby` promotion 现已由 focused cycle regression 锁定
- `waitcnt/dependency/front_end_wait` blocked active wave 在 cycle 路径保持 resident，不会被错误逐出窗口
- `barrier` waiting resident wave 会让出 active slot，并在 release 后重新参与 refill
```

- [ ] **Step 2: Run the complete related verification set**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='CycleApResidentBlocksTest.*:CycleSmokeTest.*:SharedBarrierCycleTest.*:SharedSyncCycleTest.*:ParallelExecutionModeTest.*:WaitcntFunctionalTest.*:SharedSyncFunctionalTest.*'
```

Expected:

- PASS

- [ ] **Step 3: Commit the status update**

```bash
git add docs/module-development-status.md
git commit -m "docs: update cycle resident block status"
```

- [ ] **Step 4: Sanity-check the worktree before handoff**

Run:

```bash
git status --short
```

Expected:

- only unrelated pre-existing changes remain, or the tree is clean

- [ ] **Step 5: Prepare the handoff summary**

Report:

```text
- cycle front-end now admits 2 resident blocks per AP
- >4 resident waves per PEU is reachable in cycle mode
- active-window/standby promotion is covered by focused regressions
- barrier-wait resident waves yield active slots and re-enter after release
- functional path is still intentionally deferred for this sub-project
```
