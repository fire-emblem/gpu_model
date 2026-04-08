# Trace Canonical Event Model Design

## Goal

Unify all trace consumers around a single canonical typed event model so that:

- producers emit structured trace semantics once
- text trace, JSON trace, timeline rendering, and Perfetto export become serializers over the same canonical interpretation
- `TraceEvent.message` is no longer a primary contract
- `cycle`, `st`, `mt`, and encoded/runtime traces share one mode-agnostic semantic schema

This design extends the existing trace unification work that centralized producer-side event construction in `trace_event_builder.h`.

## Non-Goals

- Do not redesign execution semantics or scheduling behavior
- Do not change the visible slot hierarchy or perfetto hierarchy model
- Do not remove `TraceEvent.message` immediately from the struct in this phase
- Do not split trace schema by execution mode

## Problem Statement

The current system has producer-side semantic construction mostly unified, but consumer-side interpretation is still partially fragmented:

- `FileTraceSink` and `JsonTraceSink` format text/JSON directly from `TraceEvent`
- timeline / perfetto code performs its own canonical-name resolution and event interpretation
- some semantics still rely on fallback interpretation from `message`
- `message` remains both a compatibility field and an implicit semantic carrier

That leaves the system with one producer contract but multiple consumer interpretations.

## Design Summary

Move to a single canonical event model with two layers:

1. `TraceEvent`
   - remains the stored/emitted record type
   - becomes explicitly typed enough that semantic interpretation does not require `message`
2. `TraceEventView`
   - a normalized, read-only interpretation layer built from `TraceEvent`
   - used by all serializers and renderers

`message` is retained only as an optional compatibility/display field during migration, but all primary semantics must be represented by typed fields.

## Execution-First Constraint

Behavioral truth belongs to the execution model, not to trace projection.

- `arrive` completion timing, `waitcnt(N)` threshold accounting, barrier release, and wave switch /
  resume decisions must be computed by execution state machines and must remain correct even when
  trace capture is completely disabled.
- trace-oriented layers (`TraceEvent`, recorder, text/json serializers, timeline, Perfetto) may
  only project already-decided execution outcomes into typed fields and visual names.
- projection code must not decide whether an `arrive` means "still blocked" or "resume" by
  re-deriving execution semantics from partial trace state. If the UI needs that distinction, the
  execution path must emit it explicitly as typed event state first.

This constraint applies in particular to:

- `arrive` events with different completion times
- `waitcnt(1)` vs `waitcnt(0)` threshold-consumption behavior
- "first arrive still blocked, second arrive resumes" visualization
- barrier and wave-switch causal markers

## Canonical Schema Direction

Keep the top-level `TraceEventKind`, and add typed subkind fields rather than flattening everything into one giant event enum.

### Existing top-level kind retained

- `Launch`
- `BlockPlaced`
- `BlockLaunch`
- `WaveLaunch`
- `WaveStats`
- `WaveStep`
- `Commit`
- `ExecMaskUpdate`
- `MemoryAccess`
- `Barrier`
- `WaveExit`
- `Stall`
- `Arrive`

### New typed semantic fields

Add canonical subkind fields to `TraceEvent`.

- `TraceBarrierKind barrier_kind`
  - `None`
  - `Wave`
  - `Arrive`
  - `Release`
- `TraceArriveKind arrive_kind`
  - `None`
  - `Load`
  - `Store`
  - `Shared`
  - `Private`
  - `ScalarBuffer`
- `TraceLifecycleStage lifecycle_stage`
  - `None`
  - `Launch`
  - `Exit`
- `std::string display_name`
  - canonical renderer/display label for instruction-like events
  - examples: `buffer_load_dword`, `s_waitcnt`, `v_add_i32`, `wave_launch`, `wave_exit`, `stall_waitcnt_global`
- keep existing typed fields
  - `slot_model_kind`
  - `stall_reason`

### Meaning of `message`

`message` becomes compatibility-only.

Rules:

- producers should not rely on `message` to carry primary semantics
- serializers should not derive semantics from `message` unless handling legacy records
- tests should prefer typed field assertions over `message`
- compatibility tests remain to ensure old-style textual output is preserved only where still required

## TraceEventView

Introduce a normalization helper in debug layer, for example in a new pair:

- `include/gpu_model/debug/trace_event_view.h`
- `src/debug/trace_event_view.cpp`

This layer derives a canonical read-only view from `TraceEvent`.

### Proposed contents

`TraceEventView` should expose:

- identity/time/location fields copied from `TraceEvent`
- effective typed semantics
  - `kind`
  - `slot_model_kind`
  - `stall_reason`
  - `barrier_kind`
  - `arrive_kind`
  - `lifecycle_stage`
- canonical labels
  - `canonical_name`
  - `display_name`
  - `category`
- compatibility flags
  - whether semantics came from typed fields or legacy fallback

### Canonical naming rules

`TraceEventView` owns canonical event naming, for example:

- `WaveLaunch` + `lifecycle_stage=Launch` -> `wave_launch`
- `WaveExit` + `lifecycle_stage=Exit` -> `wave_exit`
- `Barrier` + `barrier_kind=Arrive` -> `barrier_arrive`
- `Barrier` + `barrier_kind=Release` -> `barrier_release`
- `Barrier` + `barrier_kind=Wave` -> `barrier_wave`
- `Stall` + `stall_reason=WaitCntGlobal` -> `stall_waitcnt_global`
- `Arrive` + `arrive_kind=Load` -> `load_arrive`
- `Arrive` + `arrive_kind=Shared` -> `shared_arrive`

Instruction events:

- `WaveStep`, `Commit`, `MemoryAccess`, `ExecMaskUpdate`
  - prefer explicit `display_name`
  - use legacy parsing only as fallback for pre-migration events

## Serializer Unification

All trace sinks and renderers should first normalize through `TraceEventView`.

### Text trace

`FileTraceSink` should:

- format output from `TraceEventView`
- emit stable field names from one place
- stop inferring semantic meaning from `message`

### JSON trace

`JsonTraceSink` should:

- serialize the same canonical fields as text trace
- include typed subkind fields directly
- preserve `message` only as optional compatibility/display output

### Timeline / Perfetto

`CycleTimelineRenderer` and Perfetto export should:

- use `TraceEventView::canonical_name`
- use typed subkind fields instead of local semantic reconstruction
- keep existing hierarchy and slot semantics unchanged

## Producer Changes

Extend `trace_event_builder.h` so semantic factories populate the new typed fields directly.

Examples:

- `MakeTraceWaveLaunchEvent(...)`
  - sets `kind=WaveLaunch`
  - sets `lifecycle_stage=Launch`
  - sets canonical `display_name=wave_launch` if needed
- `MakeTraceWaveExitEvent(...)`
  - sets `lifecycle_stage=Exit`
- `MakeTraceBarrierArriveEvent(...)`
  - sets `barrier_kind=Arrive`
- `MakeTraceBarrierReleaseEvent(...)`
  - sets `barrier_kind=Release`
- `MakeTraceMemoryArriveEvent(..., Load, ...)`
  - sets `arrive_kind=Load`
- `MakeTraceWaitStallEvent(..., WaitCntGlobal, ...)`
  - sets `stall_reason=WaitCntGlobal`

Instruction-bearing helpers should additionally set `display_name` where the renderer currently needs canonical names.

## Migration Strategy

### Phase 1: Schema and normalization

- add new typed subkind fields to `TraceEvent`
- add `TraceEventView`
- make `TraceEventView` capable of typed-first interpretation plus legacy fallback

### Phase 2: Consumer migration

- migrate `trace_sink.cpp` to `TraceEventView`
- migrate `cycle_timeline.cpp` and perfetto export to `TraceEventView`
- remove duplicated canonical-name logic from consumer code

### Phase 3: Producer completion

- ensure all semantic factories populate typed subkind fields
- ensure remaining direct event construction sites are migrated or wrapped

### Phase 4: Test hardening

- switch representative tests to typed/subkind assertions
- leave a small compatibility suite for legacy `message`

## Compatibility Policy

Short term:

- keep `message` field present
- keep legacy fallback inside `TraceEventView`
- preserve existing text/json expectations where required by tests

Medium term:

- `message` becomes optional display payload only
- canonical meaning comes exclusively from typed fields

Long term:

- remove consumer dependence on `message`
- optionally deprecate producer-side free-form `message` construction except for `display_name`/details payloads

## Risks and Mitigations

### Risk: overloading `display_name`

Mitigation:

- define it narrowly as canonical display/serializer label
- do not use it as another free-form semantic dump

### Risk: breaking existing text/json tests

Mitigation:

- keep compatibility suite
- migrate sinks through `TraceEventView` with typed-first but fallback-enabled logic

### Risk: mode-specific divergence returns

Mitigation:

- forbid mode-specific trace schema branches
- keep mode differences limited to observed slot semantics and timing only

### Risk: renderer behavior drift

Mitigation:

- pin current perfetto/timeline behavior with targeted tests before removing old logic

## Files Expected To Change

Core schema and normalization:

- `include/gpu_model/debug/trace_event.h`
- `include/gpu_model/debug/trace_event_builder.h`
- `include/gpu_model/debug/trace_event_view.h` (new)
- `src/debug/trace_event_view.cpp` (new)

Consumers:

- `include/gpu_model/debug/trace_sink.h`
- `src/debug/trace_sink.cpp`
- `include/gpu_model/debug/cycle_timeline.h`
- `src/debug/cycle_timeline.cpp`
- `src/debug/trace_artifact_recorder.cpp`

Producers:

- `src/execution/cycle_exec_engine.cpp`
- `src/execution/functional_exec_engine.cpp`
- `src/execution/encoded_exec_engine.cpp`
- `src/runtime/exec_engine.cpp`

Tests:

- `tests/runtime/trace_test.cpp`
- `tests/runtime/cycle_timeline_test.cpp`
- representative cycle/functional/runtime tests that still assert raw `message`

## Acceptance Criteria

- text trace and json trace are produced from the same normalized event view
- timeline/perfetto canonical naming comes from the same normalized event view
- primary semantics no longer require reading `TraceEvent.message`
- `message` is compatibility-only, not a required semantic contract
- cycle/st/mt/encoded/runtime continue sharing one trace schema
- existing high-signal trace and perfetto tests continue to pass
- new tests pin typed subkind behavior and compatibility fallback behavior
