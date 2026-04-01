# Multi-Wave Dispatch Front-End Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align the current reachable multi-wave dispatch behavior so `FunctionalExecEngine` and `CycleExecEngine` share the same dispatch-visible readiness contract for resident waves, while proving that a blocked wave does not stall a ready sibling wave on the same `PEU`.

**Architecture:** Keep the current block-level resident-wave pool shape, but tighten the shared `issue_eligibility` contract so waiting waves, barrier-blocked waves, `valid_entry` stalls, and explicit `waitcnt` stalls are all represented consistently before a `PEU` chooses its next wave. Then make `FunctionalExecEngine::SelectNextWaveIndexForPeu()` instruction-aware by consulting that shared contract instead of only checking `status/run_state/busy`. Do not add active-window/standby-window yet, because with the current `Mapper` and `1024-thread` block limit, one block still reaches at most `4 waves/PEU`.

**Tech Stack:** C++20, gtest, existing `FunctionalExecEngine`, `CycleExecEngine`, `issue_eligibility`, `RuntimeEngine`, trace-based functional regressions

---

## File Map

- Modify: `src/execution/internal/issue_eligibility.cpp`
  - Implement waiting-wave block-reason mapping and tighten `CanIssueInstruction()` / `IssueBlockReason()`.
- Modify: `src/execution/functional_exec_engine.cpp`
  - Make `SelectNextWaveIndexForPeu()` instruction-aware and shared-contract-driven.
- Modify: `tests/execution/internal/issue_eligibility_test.cpp`
  - Lock the waiting-wave contract at unit scope.
- Modify: `tests/functional/waitcnt_functional_test.cpp`
  - Add a same-`PEU` sibling-progress regression for `waitcnt`.
- Modify: `tests/functional/shared_sync_functional_test.cpp`
  - Add a barrier-release regression proving post-release waves re-enter dispatch.
- Modify: `tests/runtime/executed_flow_program_cycle_stats_test.cpp`
  - Add one representative same-`PEU` overlap stats regression so the program-level cycle contract stays intact.
- Modify: `docs/module-development-status.md`
  - Record that current dispatch work closes reachable resident-pool semantics only, not multi-block/AP residency.

## Task 1: Tighten the shared dispatch-readiness contract

**Files:**
- Modify: `src/execution/internal/issue_eligibility.cpp`
- Test: `tests/execution/internal/issue_eligibility_test.cpp`

- [ ] **Step 1: Write the failing unit tests for waiting-wave dispatch eligibility**

Add tests to `tests/execution/internal/issue_eligibility_test.cpp` that prove a wave in explicit waiting state is not dispatchable even when counters are already satisfied:

```cpp
TEST(IssueEligibilityTest, WaitingWaveCannotIssueEvenWhenInstructionDependenciesAreOtherwiseReady) {
  WaveContext wave;
  wave.status = WaveStatus::Active;
  wave.valid_entry = true;
  wave.run_state = WaveRunState::Waiting;
  wave.wait_reason = WaveWaitReason::PendingGlobalMemory;

  Instruction vector_instr;
  vector_instr.opcode = Opcode::VAdd;

  EXPECT_FALSE(CanIssueInstruction(true, wave, vector_instr, true));
}

TEST(IssueEligibilityTest, WaitingWaveReportsExplicitWaitReasonBeforeOtherChecks) {
  WaveContext wave;
  wave.status = WaveStatus::Active;
  wave.valid_entry = true;
  wave.run_state = WaveRunState::Waiting;
  wave.wait_reason = WaveWaitReason::PendingGlobalMemory;

  Instruction vector_instr;
  vector_instr.opcode = Opcode::VAdd;

  const auto reason = IssueBlockReason(true, wave, vector_instr, true);
  ASSERT_TRUE(reason.has_value());
  EXPECT_EQ(*reason, "waitcnt_global");
}

TEST(IssueEligibilityTest, BarrierWaitingWaveReportsBarrierWait) {
  WaveContext wave;
  wave.status = WaveStatus::Active;
  wave.valid_entry = true;
  wave.run_state = WaveRunState::Waiting;
  wave.wait_reason = WaveWaitReason::BlockBarrier;
  wave.waiting_at_barrier = true;

  Instruction vector_instr;
  vector_instr.opcode = Opcode::VAdd;

  const auto reason = IssueBlockReason(true, wave, vector_instr, true);
  ASSERT_TRUE(reason.has_value());
  EXPECT_EQ(*reason, "barrier_wait");
}
```

- [ ] **Step 2: Run the focused issue-eligibility tests and confirm they fail**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='IssueEligibilityTest.WaitingWaveCannotIssueEvenWhenInstructionDependenciesAreOtherwiseReady:IssueEligibilityTest.WaitingWaveReportsExplicitWaitReasonBeforeOtherChecks:IssueEligibilityTest.BarrierWaitingWaveReportsBarrierWait'
```

Expected:

- FAIL because `CanIssueInstruction()` still ignores `run_state`
- FAIL because `IssueBlockReason()` does not yet map explicit waiting state back to a shared reason

- [ ] **Step 3: Implement waiting-state awareness in the shared helpers**

Extend `src/execution/internal/issue_eligibility.cpp` so waiting state is part of the shared contract:

```cpp
namespace {

std::optional<std::string> WaitingStateBlockReason(const WaveContext& wave) {
  if (wave.run_state != WaveRunState::Waiting) {
    return std::nullopt;
  }
  switch (wave.wait_reason) {
    case WaveWaitReason::BlockBarrier:
      return std::string("barrier_wait");
    case WaveWaitReason::PendingGlobalMemory:
      return std::string("waitcnt_global");
    case WaveWaitReason::PendingSharedMemory:
      return std::string("waitcnt_shared");
    case WaveWaitReason::PendingPrivateMemory:
      return std::string("waitcnt_private");
    case WaveWaitReason::PendingScalarBufferMemory:
      return std::string("waitcnt_scalar_buffer");
    case WaveWaitReason::None:
      return std::string("wave_wait");
  }
  return std::nullopt;
}

}  // namespace

bool CanIssueInstruction(bool dispatch_enabled,
                         const WaveContext& wave,
                         const Instruction& instruction,
                         bool dependencies_ready) {
  return dispatch_enabled && wave.status == WaveStatus::Active &&
         wave.run_state == WaveRunState::Runnable &&
         wave.valid_entry &&
         !wave.waiting_at_barrier &&
         !wave.branch_pending &&
         WaitCntSatisfied(wave, instruction) &&
         dependencies_ready &&
         (MemoryDomainForOpcode(instruction.opcode) == MemoryWaitDomain::None ||
          PendingMemoryOpsForDomain(wave, MemoryDomainForOpcode(instruction.opcode)) == 0);
}

std::optional<std::string> IssueBlockReason(bool dispatch_enabled,
                                            const WaveContext& wave,
                                            const Instruction& instruction,
                                            bool dependencies_ready) {
  if (!dispatch_enabled || wave.status != WaveStatus::Active) {
    return std::nullopt;
  }
  if (const auto waiting_reason = WaitingStateBlockReason(wave)) {
    return waiting_reason;
  }
  if (!wave.valid_entry) {
    return std::string("front_end_wait");
  }
  if (wave.waiting_at_barrier) {
    return std::string("barrier_wait");
  }
  if (wave.branch_pending) {
    return std::string("branch_wait");
  }
  if (const auto reason = WaitCntBlockReason(wave, instruction)) {
    return reason;
  }
  if (const auto reason = MemoryDomainBlockReason(wave, instruction)) {
    return reason;
  }
  if (!dependencies_ready) {
    return std::string("dependency_wait");
  }
  return std::nullopt;
}
```

- [ ] **Step 4: Re-run the focused issue-eligibility tests and make them pass**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='IssueEligibilityTest.*'
```

Expected:

- PASS for the new waiting-state tests
- PASS for the existing waitcnt/dependency tests

- [ ] **Step 5: Commit the shared eligibility slice**

```bash
git add src/execution/internal/issue_eligibility.cpp tests/execution/internal/issue_eligibility_test.cpp
git commit -m "feat: align dispatch eligibility with wave wait state"
```

## Task 2: Add focused same-PEU multi-wave regressions

**Files:**
- Modify: `tests/functional/waitcnt_functional_test.cpp`
- Modify: `tests/functional/shared_sync_functional_test.cpp`
- Modify: `tests/runtime/executed_flow_program_cycle_stats_test.cpp`

- [ ] **Step 1: Write the failing same-PEU waitcnt sibling-progress test**

Add a kernel to `tests/functional/waitcnt_functional_test.cpp` that places `wave 0` and `wave 4` on the same `PEU` by launching one `320-thread` block:

```cpp
ExecutableKernel BuildSamePeuSiblingWaitcntKernel() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SLoadArg("s1", 1);
  builder.SysGlobalIdX("v0");

  builder.SMov("s2", 64);
  builder.VCmpLtCmask("v0", "s2");
  builder.MaskSaveExec("s10");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("after_wait_wave");
  builder.MLoadGlobal("v1", "s0", "v0", 4);
  builder.SWaitCnt(/*global_count=*/0, /*shared_count=*/UINT32_MAX,
                   /*private_count=*/UINT32_MAX, /*scalar_buffer_count=*/UINT32_MAX);
  builder.VMov("v2", 11);
  builder.MStoreGlobal("s1", "v0", "v2", 4);
  builder.Label("after_wait_wave");
  builder.MaskRestoreExec("s10");

  builder.SMov("s3", 256);
  builder.VCmpGeCmask("v0", "s3");
  builder.MaskSaveExec("s11");
  builder.MaskAndExecCmask();
  builder.SMov("s4", 320);
  builder.VCmpLtCmask("v0", "s4");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("exit");
  builder.VMov("v3", 22);
  builder.MStoreGlobal("s1", "v0", "v3", 4);
  builder.Label("exit");
  builder.MaskRestoreExec("s11");
  builder.BExit();
  return builder.Build("same_peu_sibling_waitcnt");
}
```

Then add a regression that checks `wave 4` advances while `wave 0` is stalled on `s_waitcnt`:

```cpp
TEST(WaitcntFunctionalTest, WaitingWaveDoesNotBlockReadySiblingOnSamePeu) {
  CollectingTraceSink trace;
  RuntimeEngine runtime(&trace);
  runtime.SetFunctionalExecutionMode(FunctionalExecutionMode::SingleThreaded);

  constexpr uint32_t kBlockDim = 320;
  constexpr uint32_t kElementCount = kBlockDim;
  const auto kernel = BuildSamePeuSiblingWaitcntKernel();
  const uint64_t waitcnt_pc = NthInstructionPcWithOpcode(kernel, Opcode::SWaitCnt, 0);
  const uint64_t wait_wave_store_pc = NthInstructionPcWithOpcode(kernel, Opcode::MStoreGlobal, 0);
  const uint64_t sibling_store_pc = NthInstructionPcWithOpcode(kernel, Opcode::MStoreGlobal, 1);
  ASSERT_NE(waitcnt_pc, std::numeric_limits<uint64_t>::max());
  ASSERT_NE(wait_wave_store_pc, std::numeric_limits<uint64_t>::max());
  ASSERT_NE(sibling_store_pc, std::numeric_limits<uint64_t>::max());

  const uint64_t in_addr = runtime.memory().AllocateGlobal(kElementCount * sizeof(int32_t));
  const uint64_t out_addr = runtime.memory().AllocateGlobal(kElementCount * sizeof(int32_t));
  for (uint32_t i = 0; i < kElementCount; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(in_addr + i * sizeof(int32_t),
                                               static_cast<int32_t>(100 + i));
    runtime.memory().StoreGlobalValue<int32_t>(out_addr + i * sizeof(int32_t), -1);
  }

  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = kBlockDim;
  request.args.PushU64(in_addr);
  request.args.PushU64(out_addr);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  for (uint32_t i = 0; i < 64; ++i) {
    EXPECT_EQ(runtime.memory().LoadGlobalValue<int32_t>(out_addr + i * sizeof(int32_t)), 11);
  }
  for (uint32_t i = 256; i < 320; ++i) {
    EXPECT_EQ(runtime.memory().LoadGlobalValue<int32_t>(out_addr + i * sizeof(int32_t)), 22);
  }

  EXPECT_LT(FirstEventIndexForBlockWave(trace.events(), 0, 0, TraceEventKind::Stall,
                                        waitcnt_pc, "waitcnt_global"),
            FirstEventIndexForBlockWave(trace.events(), 0, 4, TraceEventKind::WaveStep,
                                        sibling_store_pc));
  EXPECT_LT(FirstEventIndexForBlockWave(trace.events(), 0, 4, TraceEventKind::WaveStep,
                                        sibling_store_pc),
            FirstEventIndexForBlockWave(trace.events(), 0, 0, TraceEventKind::WaveStep,
                                        wait_wave_store_pc));
}
```

- [ ] **Step 2: Write the failing barrier-release regression**

Add a `320-thread` one-block kernel to `tests/functional/shared_sync_functional_test.cpp` where `wave 0` reaches a block barrier early and `wave 4` does extra pre-barrier work before releasing it:

```cpp
ExecutableKernel BuildSamePeuBarrierResumeKernel() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SysGlobalIdX("v0");

  builder.SMov("s1", 64);
  builder.VCmpLtCmask("v0", "s1");
  builder.MaskSaveExec("s10");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("after_early_wave");
  builder.SyncBarrier();
  builder.VMov("v1", 31);
  builder.MStoreGlobal("s0", "v0", "v1", 4);
  builder.Label("after_early_wave");
  builder.MaskRestoreExec("s10");

  builder.SMov("s2", 256);
  builder.VCmpGeCmask("v0", "s2");
  builder.MaskSaveExec("s11");
  builder.MaskAndExecCmask();
  builder.SMov("s3", 320);
  builder.VCmpLtCmask("v0", "s3");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("done");
  builder.VMov("v2", 7);
  builder.VAdd("v3", "v2", "v2");
  builder.SyncBarrier();
  builder.VMov("v4", 47);
  builder.MStoreGlobal("s0", "v0", "v4", 4);
  builder.Label("done");
  builder.MaskRestoreExec("s11");
  builder.BExit();
  return builder.Build("same_peu_barrier_resume");
}
```

Add a regression:

```cpp
uint64_t NthInstructionPcWithOpcode(const ExecutableKernel& kernel, Opcode opcode, size_t ordinal) {
  size_t seen = 0;
  for (uint64_t pc = 0; pc < kernel.instructions().size(); ++pc) {
    if (kernel.instructions()[pc].opcode != opcode) {
      continue;
    }
    if (seen == ordinal) {
      return pc;
    }
    ++seen;
  }
  return std::numeric_limits<uint64_t>::max();
}

size_t FirstEventIndexForBlockWave(const std::vector<TraceEvent>& events,
                                   uint32_t block_id,
                                   uint32_t wave_id,
                                   TraceEventKind kind,
                                   uint64_t pc) {
  for (size_t i = 0; i < events.size(); ++i) {
    if (events[i].block_id == block_id && events[i].wave_id == wave_id &&
        events[i].kind == kind && events[i].pc == pc) {
      return i;
    }
  }
  return std::numeric_limits<size_t>::max();
}

size_t FirstBarrierReleaseIndex(const std::vector<TraceEvent>& events) {
  for (size_t i = 0; i < events.size(); ++i) {
    if (events[i].kind == TraceEventKind::Barrier && events[i].message == "release") {
      return i;
    }
  }
  return std::numeric_limits<size_t>::max();
}

TEST(SharedSyncFunctionalTest, BarrierReleaseReturnsEarlyWaveToDispatch) {
  CollectingTraceSink trace;
  RuntimeEngine runtime(&trace);
  runtime.SetFunctionalExecutionMode(FunctionalExecutionMode::MarlParallel);

  constexpr uint32_t kBlockDim = 320;
  constexpr uint32_t kElementCount = kBlockDim;
  const auto kernel = BuildSamePeuBarrierResumeKernel();
  const uint64_t late_pre_barrier_pc = NthInstructionPcWithOpcode(kernel, Opcode::VAdd, 0);
  const uint64_t early_post_barrier_store_pc = NthInstructionPcWithOpcode(kernel, Opcode::MStoreGlobal, 0);
  ASSERT_NE(late_pre_barrier_pc, std::numeric_limits<uint64_t>::max());
  ASSERT_NE(early_post_barrier_store_pc, std::numeric_limits<uint64_t>::max());

  const uint64_t out_addr = runtime.memory().AllocateGlobal(kElementCount * sizeof(int32_t));
  for (uint32_t i = 0; i < kElementCount; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(out_addr + i * sizeof(int32_t), -1);
  }

  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = kBlockDim;
  request.args.PushU64(out_addr);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  for (uint32_t i = 0; i < 64; ++i) {
    EXPECT_EQ(runtime.memory().LoadGlobalValue<int32_t>(out_addr + i * sizeof(int32_t)), 31);
  }
  for (uint32_t i = 256; i < 320; ++i) {
    EXPECT_EQ(runtime.memory().LoadGlobalValue<int32_t>(out_addr + i * sizeof(int32_t)), 47);
  }

  const size_t release_index = FirstBarrierReleaseIndex(trace.events());
  ASSERT_NE(release_index, std::numeric_limits<size_t>::max());
  EXPECT_LT(FirstEventIndexForBlockWave(trace.events(), 0, 4, TraceEventKind::WaveStep,
                                        late_pre_barrier_pc),
            release_index);
  EXPECT_LT(release_index,
            FirstEventIndexForBlockWave(trace.events(), 0, 0, TraceEventKind::WaveStep,
                                        early_post_barrier_store_pc));
}
```

- [ ] **Step 3: Add the representative program-cycle-stats regression**

Extend `tests/runtime/executed_flow_program_cycle_stats_test.cpp` with a local representative same-`PEU` regression:

```cpp
ExecutableKernel BuildSamePeuSiblingWaitcntCycleStatsKernel() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SLoadArg("s1", 1);
  builder.SysGlobalIdX("v0");

  builder.SMov("s2", 64);
  builder.VCmpLtCmask("v0", "s2");
  builder.MaskSaveExec("s10");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("after_wait_wave");
  builder.MLoadGlobal("v1", "s0", "v0", 4);
  builder.SWaitCnt(/*global_count=*/0, /*shared_count=*/UINT32_MAX,
                   /*private_count=*/UINT32_MAX, /*scalar_buffer_count=*/UINT32_MAX);
  builder.VMov("v2", 11);
  builder.MStoreGlobal("s1", "v0", "v2", 4);
  builder.Label("after_wait_wave");
  builder.MaskRestoreExec("s10");

  builder.SMov("s3", 256);
  builder.VCmpGeCmask("v0", "s3");
  builder.MaskSaveExec("s11");
  builder.MaskAndExecCmask();
  builder.SMov("s4", 320);
  builder.VCmpLtCmask("v0", "s4");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("exit");
  builder.VMov("v3", 22);
  builder.MStoreGlobal("s1", "v0", "v3", 4);
  builder.Label("exit");
  builder.MaskRestoreExec("s11");
  builder.BExit();
  return builder.Build("same_peu_sibling_waitcnt_cycle_stats");
}

TEST(ExecutedFlowProgramCycleStatsTest,
     SamePeuWaitcntSiblingProgressMaintainsModeAgreementAndOverlap) {
  const auto kernel = BuildSamePeuSiblingWaitcntCycleStatsKernel();
  const auto st = LaunchProgramCycleStatsKernel(kernel, FunctionalExecutionMode::SingleThreaded,
                                                /*block_dim_x=*/320);
  const auto mt = LaunchProgramCycleStatsKernel(kernel, FunctionalExecutionMode::MarlParallel,
                                                /*block_dim_x=*/320);

  ASSERT_TRUE(st.ok) << st.error_message;
  ASSERT_TRUE(mt.ok) << mt.error_message;
  ASSERT_TRUE(st.program_cycle_stats.has_value());
  ASSERT_TRUE(mt.program_cycle_stats.has_value());
  EXPECT_EQ(st.program_cycle_stats->total_cycles, mt.program_cycle_stats->total_cycles);
  EXPECT_LT(st.program_cycle_stats->total_cycles,
            st.program_cycle_stats->total_issued_work_cycles);
}
```

- [ ] **Step 4: Run the focused regressions and confirm they fail**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='WaitcntFunctionalTest.WaitingWaveDoesNotBlockReadySiblingOnSamePeu:SharedSyncFunctionalTest.BarrierReleaseReturnsEarlyWaveToDispatch:ExecutedFlowProgramCycleStatsTest.SamePeuWaitcntSiblingProgressMaintainsModeAgreementAndOverlap'
```

Expected:

- FAIL because the current functional `PEU` selector still picks waves only by `status/run_state/busy`
- FAIL because the current regressions do not yet prove same-`PEU` sibling progress and barrier re-entry

- [ ] **Step 5: Commit the failing regression slice**

```bash
git add tests/functional/waitcnt_functional_test.cpp tests/functional/shared_sync_functional_test.cpp tests/runtime/executed_flow_program_cycle_stats_test.cpp
git commit -m "test: add multi-wave dispatch front-end regressions"
```

## Task 3: Make functional PEU selection instruction-aware

**Files:**
- Modify: `src/execution/functional_exec_engine.cpp`
- Modify: `tests/functional/waitcnt_functional_test.cpp`
- Modify: `tests/functional/shared_sync_functional_test.cpp`

- [ ] **Step 1: Implement instruction-aware selection inside `SelectNextWaveIndexForPeu()`**

Replace the current readiness test in `src/execution/functional_exec_engine.cpp` with shared-contract selection:

```cpp
std::optional<size_t> SelectNextWaveIndexForPeu(ExecutableBlock& block, size_t peu_index) {
  if (peu_index >= block.wave_indices_per_peu.size()) {
    return std::nullopt;
  }
  auto& peu_waves = block.wave_indices_per_peu[peu_index];
  if (peu_waves.empty()) {
    return std::nullopt;
  }

  std::lock_guard<std::mutex> state_lock(*block.wave_state_mutex);
  const size_t start = block.next_wave_rr_per_peu[peu_index] % peu_waves.size();
  for (size_t offset = 0; offset < peu_waves.size(); ++offset) {
    const size_t local_index = (start + offset) % peu_waves.size();
    const size_t wave_index = peu_waves[local_index];
    const auto& wave = block.waves[wave_index];
    if (block.wave_busy[wave_index]) {
      continue;
    }
    if (wave.pc >= context_.kernel.instructions().size()) {
      throw std::out_of_range("wave pc out of range");
    }
    const auto& instruction = context_.kernel.instructions().at(wave.pc);
    if (!CanIssueInstruction(/*dispatch_enabled=*/true, wave, instruction,
                             /*dependencies_ready=*/true)) {
      continue;
    }
    block.next_wave_rr_per_peu[peu_index] = (local_index + 1) % peu_waves.size();
    return wave_index;
  }
  return std::nullopt;
}
```

- [ ] **Step 2: Keep waiting/resume behavior on the same shared contract**

While editing `src/execution/functional_exec_engine.cpp`, make sure the existing resume path remains the only way back to runnable:

```cpp
bool ResumeMemoryWaitingWaves(ExecutableBlock& block) {
  bool resumed = false;
  {
    std::lock_guard<std::mutex> state_lock(*block.wave_state_mutex);
    for (size_t i = 0; i < block.waves.size(); ++i) {
      resumed = ResumeWaveIfWaitSatisfied(block.wave_states[i], block.waves[i]) || resumed;
    }
  }
  if (resumed) {
    EmitWaveStatsSnapshot();
  }
  return resumed;
}
```

The implementation check here is behavioral: do not add a second ad hoc readiness path that bypasses `run_state/wait_reason`.

- [ ] **Step 3: Re-run the focused regressions and make them pass**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='IssueEligibilityTest.*:WaitcntFunctionalTest.WaitingWaveDoesNotBlockReadySiblingOnSamePeu:SharedSyncFunctionalTest.BarrierReleaseReturnsEarlyWaveToDispatch:ExecutedFlowProgramCycleStatsTest.SamePeuWaitcntSiblingProgressMaintainsModeAgreementAndOverlap'
```

Expected:

- PASS for the new unit tests
- PASS for the new same-`PEU` waitcnt/barrier regressions
- PASS for the representative stats regression

- [ ] **Step 4: Run the broader dispatch-adjacent suites**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='ParallelExecutionModeTest.*:WaitcntFunctionalTest.*:SharedSyncFunctionalTest.*:ExecutedFlowProgramCycleStatsTest.*'
```

Expected:

- PASS with no regressions in existing `st/mt`, waitcnt, barrier, and program-cycle-stats coverage

- [ ] **Step 5: Commit the functional dispatch implementation**

```bash
git add src/execution/functional_exec_engine.cpp
git commit -m "feat: make functional peu selection dispatch-aware"
```

## Task 4: Update status tracking and finish the batch

**Files:**
- Modify: `docs/module-development-status.md`

- [ ] **Step 1: Update the execution status entry**

In `docs/module-development-status.md`, update the `M6` / `M13` narrative so it explicitly says this batch closes reachable resident-pool semantics only:

```md
- functional resident-wave dispatch now uses instruction-aware readiness on top of the shared
  `issue_eligibility` contract
- same-`PEU` regressions now lock that a blocked wave does not stall a ready sibling wave
- barrier release returns waves to runnable dispatch in both `st` and `mt`
- `>4 resident waves / PEU` remains deferred until one `AP` can hold multiple resident blocks
```

- [ ] **Step 2: Run the final focused verification**

Run:

```bash
./build-ninja/tests/gpu_model_tests --gtest_filter='IssueEligibilityTest.*:ParallelExecutionModeTest.*:WaitcntFunctionalTest.*:SharedSyncFunctionalTest.*:ExecutedFlowProgramCycleStatsTest.*'
```

Expected:

- PASS

- [ ] **Step 3: Commit the status update**

```bash
git add docs/module-development-status.md
git commit -m "docs: update multi-wave dispatch status"
```

- [ ] **Step 4: Sanity-check the worktree before handoff**

Run:

```bash
git status --short
```

Expected:

- Only unrelated pre-existing changes remain, or the tree is clean

- [ ] **Step 5: Prepare the handoff summary**

Report:

```text
- shared dispatch-readiness contract now respects explicit wait state
- functional same-PEU round-robin selection is instruction-aware
- same-PEU waitcnt and barrier regressions pass in both st/mt
- active-window/standby-window work is still intentionally deferred
```
