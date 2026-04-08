# Marl Parallel Wave Execution Roadmap

> [!NOTE]
> 历史计划文档。用于保留当时的拆解和决策上下文，不作为当前代码结构的权威描述。当前主线以 `docs/my_design.md` 和 `docs/module-development-status.md` 为准。


## Goal

Introduce a switchable functional execution backend:

- `SingleThreaded`
- `MarlParallel`

The short-term goal is not cycle-accurate parallel hardware replay.
The short-term goal is to land the execution-mode switch, `marl` dependency wiring,
and the executor insertion point that later wave-parallel scheduling will use.

## Phase 1

Land infrastructure only:

- vendor `third_party/marl`
- add CMake integration guarded by `GPU_MODEL_ENABLE_MARL`
- add `FunctionalExecutionMode`
- add `HostRuntime::SetFunctionalExecutionMode()`
- add `ParallelWaveExecutor`
- keep correctness aligned with the existing functional executor

This phase is intentionally conservative.
`MarlParallel` uses marl's scheduler, but still delegates the actual instruction execution
to the existing functional path while the correctness model is preserved.

Status:

- complete
- `HostRuntime` can switch between `SingleThreaded` and `MarlParallel`
- both modes now share the same functional execution core
- block execution already uses a `PEU`-local round-robin wave selection structure
- parallel mode currently parallelizes blocks, not waves

## Phase 2

Move to wave-granular parallel dispatch:

- one marl task per runnable wave
- block-local shared state held behind explicit synchronization
- barrier release driven by block-scoped counters / condition variables
- block-local atomic serialization for shared/global atomics
- explicit wave wait / resume points for:
  - barrier
  - waitcnt
  - pending memory

Status:

- partially complete
- block execution now enters `PEU`-worker mode in `MarlParallel`
- each `PEU` owns a resident wave pool and selects ready waves in round-robin order
- shared-memory, shared-atomic, and block-barrier kernels now run on the same `PEU`-parallel path
- block-local wait now uses condition-variable wakeup instead of pure yield spinning
- remaining gap is to generalize the current block-local wakeup path into explicit wait / resume primitives for:
  - barrier
  - waitcnt
  - pending memory completion

## Phase 3

Tighten semantics toward the target model:

- wave-internal lane loop per instruction with `exec` mask
- wave-to-wave parallelism inside a block
- AP / PEU aware worker assignment
- scheduler-visible wave waiting / switching hooks
- prepare the same structure for naive cycle simulation

## Immediate Constraints

- keep single-thread mode as the default
- mode switching must be runtime configurable
- keep existing gtests green in both modes
- avoid introducing nondeterministic result drift in current functional tests
