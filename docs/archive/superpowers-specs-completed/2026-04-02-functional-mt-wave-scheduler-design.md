# Functional MT Wave Scheduler Design

**Goal:** Redefine `FunctionalExecutionMode::MarlParallel` so the functional model executes at `wave` granularity with CPU-side parallelism that more closely matches GPU concurrency semantics, while keeping `cycle` model concerns completely separate.

## Scope

This design only changes the `functional` execution backend:

- `SingleThreaded`
- `MarlParallel`

It does **not** change:

- `ExecutionMode::Cycle`
- resident block / active window / dispatch capacity modeling
- AP / PEU hardware capacity rules

Those remain cycle-model concerns.

## Core Model

### Functional Launch Semantics

In the functional model, kernel launch logically materializes all blocks and all waves immediately.

- every block is assigned to an AP
- every wave is assigned to an AP / PEU according to the existing placement map
- all waves exist from launch time onward as software execution entities

This is logical placement, not hardware occupancy modeling.

### ST vs MT

- `SingleThreaded`
  - one CPU thread advances all runnable waves
  - existing semantics remain unchanged

- `MarlParallel`
  - execution unit is `wave`, not `block`
  - a global worker pool executes wave tasks
  - default worker count is `max(1, floor(cpu_cores * 0.9))`
  - AP ownership is represented by AP-local queues and state, not by separate OS thread pools

## Scheduling Model

### Global Worker Pool

Use one shared worker pool for the whole functional launch.

Reasons:

- AP in the functional model is a logical ownership boundary, not a host resource boundary
- one pool avoids creating `AP_count * worker_count` OS threads
- workers can still respect AP-local scheduling by pulling wave work from AP-local queues

### AP-Local Scheduling State

Each AP owns:

- a runnable wave queue
- bookkeeping for blocked / waiting waves that belong to that AP
- wakeup reinsertion of waves into that AP’s runnable queue

Workers consume runnable waves from AP-local queues using a fair global policy, such as round-robin across AP queues.

### Wave Execution Granularity

One wave task advances one runnable wave until one of these outcomes:

- completes one instruction and remains runnable
- enters a wait state
- arrives at a block barrier
- exits

The scheduler then decides the next runnable wave independently.

This means:

- different blocks’ waves can run concurrently
- the same block’s waves can run concurrently
- a blocked wave must not prevent sibling or unrelated waves on the same AP from making progress

## Synchronization Model

### Block Barrier

Barrier state remains block-local:

- `barrier_generation`
- `barrier_arrivals`
- block-local waiting set

When a wave executes `SyncBarrier` in functional MT:

1. mark that wave as barrier-waiting
2. remove it from runnable scheduling
3. update the block barrier arrival count
4. if the block barrier is not ready, do nothing further
5. if the block barrier becomes ready, release only that block’s waiting waves
6. push the released waves back into their AP-local runnable queues

No other block participates in this release decision.

### Waitcnt / Memory Wait

Wait states remain wave-local:

- a wave waiting on memory or waitcnt is not runnable
- once the wait condition is satisfied, it is requeued onto its AP-local runnable queue
- this waiting wave does not block other runnable waves on the same AP

## Ownership And Locking

### AP-Local

AP-level structures should be independently synchronized:

- runnable queue
- scheduling cursor / fairness state

### Block-Local

Block-level synchronization should remain separate:

- shared memory mutex
- control mutex for barrier bookkeeping
- wave-state mutex protecting wave status / wait state transitions

The important rule is:

- AP queue ownership handles *which wave may run next*
- block-local synchronization handles *whether a specific wave is allowed to resume*

## Fairness

The scheduler should avoid starving APs or waves.

Recommended policy:

- global round-robin over APs with runnable work
- within each AP, round-robin over runnable waves

This keeps the model simple while preserving the idea that all launched waves are eligible software work once they are runnable.

## Non-Goals

- no AP resident block limit
- no wave dispatch slot limit
- no PEU issue bandwidth modeling
- no block backfill / active window logic

Those belong only to the cycle model.

## Expected Code Changes

Likely primary files:

- `src/execution/functional_exec_engine.cpp`
- `tests/functional/shared_barrier_functional_test.cpp`
- `tests/functional/shared_sync_functional_test.cpp`
- `tests/functional/waitcnt_functional_test.cpp`
- possibly `tests/runtime/parallel_execution_mode_test.cpp`

## Acceptance Criteria

- `MarlParallel` no longer runs blocks as the unit of parallel work
- wave is the unit of runnable scheduling
- barrier release affects only the waiting waves of that block
- different blocks’ runnable waves on the same AP can keep progressing while another block is waiting at barrier
- existing functional `st/mt` correctness regressions continue to pass after being adapted to the new scheduler structure
