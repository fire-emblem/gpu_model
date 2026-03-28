# Marl Parallel Wave Execution Roadmap

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
