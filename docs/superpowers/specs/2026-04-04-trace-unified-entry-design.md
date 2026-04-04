# Trace Unified Entry Design

Date: 2026-04-04

## Goal

Eliminate producer-side and test-side raw trace text construction and replace it with a single,
shared trace construction entry surface.

The trace layer must remain mode-agnostic:

- `cycle`, `st`, and `mt` may observe different slot semantics
- trace schema shape and naming vocabulary must not split by execution mode
- typed fields remain the primary contract
- legacy `message` text remains available for compatibility, but is generated from one place only

## Non-Goals

- Do not redesign Perfetto export formats in this change
- Do not remove the `message` field from `TraceEvent`
- Do not introduce a large tagged-union payload schema in this change
- Do not change the existing mode-specific slot semantics:
  - `cycle` keeps accurate resident-slot modeling
  - `st/mt` keep logical unbounded slot modeling

## Problem Statement

The current trace stack is already much more unified than before:

- typed `slot_model_kind` and `stall_reason` exist
- sinks and renderers are typed-first with string fallback
- native Perfetto and chrome-json exports share the same slot-centric observation model

However, trace construction is still partially fragmented:

- producers still directly call generic builders with free-form message text
- some producer sites still encode event semantics via raw string literals
- tests still construct some `TraceEvent` objects directly
- the canonical trace vocabulary is distributed across producers, tests, and renderer expectations

This makes it too easy for:

- event naming to drift across execution modes
- tests to validate accidental strings instead of intended trace semantics
- new trace producers to bypass typed semantics and shared naming conventions

## Design Summary

Introduce a single unified trace construction surface in the existing
`include/gpu_model/debug/trace_event_builder.h` layer.

This surface has two responsibilities:

1. Define the canonical trace vocabulary
2. Define semantic event factories that own typed-field population and compatibility message output

The generic low-level builders may remain as implementation primitives, but producer code and tests
should migrate to semantic factories instead of directly assembling free-form text.

## Design Principles

- One semantic source of truth for trace names
- One construction path for each trace event family
- Typed schema is primary; legacy text is derived
- Producers should describe semantics, not format strings
- Tests should construct events through the same public trace entry surface used by producers
- Execution mode should influence observations, not trace schema shape

## Unified Entry Model

### Layer 1: Canonical Trace Vocabulary

The builder layer will own all canonical trace names and formatting helpers for stable event
families.

This vocabulary includes:

- lifecycle:
  - `wave_start`
  - `wave_end`
- barrier:
  - `arrive`
  - `release`
  - `wave`
- memory-arrive:
  - `load_arrive`
  - `store_arrive`
  - `shared_arrive`
  - `private_arrive`
  - `scalar_buffer_arrive`
- generic operational labels:
  - `commit`
  - `exit`
- stall reasons:
  - already represented by `TraceStallReason`

This layer also owns all formatting helpers for composite messages such as wave launch detail or
instruction-step formatting.

### Layer 2: Semantic Event Factories

The builder layer will expose semantic factories. These factories are the new unified entry.

Representative factories:

- wave lifecycle:
  - `MakeTraceWaveLaunchEvent(...)`
  - `MakeTraceWaveExitEvent(...)`
- wave execution:
  - `MakeTraceWaveStepEvent(...)`
  - `MakeTraceCommitEvent(...)`
- synchronization:
  - `MakeTraceBarrierWaveEvent(...)`
  - `MakeTraceBarrierArriveEvent(...)`
  - `MakeTraceBarrierReleaseEvent(...)`
- memory completion:
  - `MakeTraceMemoryArriveEvent(...)`
- stalls:
  - `MakeTraceWaitStallEvent(...)`
  - `MakeTraceWaveSwitchStallEvent(...)`
  - `MakeTraceBarrierSlotUnavailableStallEvent(...)`
- runtime/block-level:
  - `MakeTraceRuntimeLaunchEvent(...)`
  - `MakeTraceBlockPlacedEvent(...)`
  - `MakeTraceBlockLaunchEvent(...)`

Each factory owns:

- `TraceEventKind`
- typed field population
- canonical `message`
- compatibility text generation
- default slot model naming behavior
- pc selection rules where relevant

## API Direction

The implementation should preserve the existing low-level builders temporarily:

- `MakeTraceWaveEvent(...)`
- `MakeTraceBlockEvent(...)`
- `MakeTraceEvent(...)`

But those will become internal-style primitives rather than the preferred producer entry.

Producer code and tests should migrate away from generic raw-message construction toward semantic
factory calls.

## Message Policy

After this change:

- no producer should write raw semantic text such as `"arrive"`, `"release"`, `"wave_start"`,
  `"wave_end"`, `"commit"`, `"exit"`, `"load_arrive"`, `"reason=waitcnt_global"`, or
  `"reason=warp_switch"` directly
- no trace-related test should manually spell semantic text when constructing events unless the
  test is explicitly validating the legacy text compatibility contract
- legacy text output remains stable, but is derived from factories

This is the key shift:

- before: callers own semantic strings
- after: callers choose a semantic factory, and the factory owns the strings

## Scope of Migration

### In Scope

- `src/execution/functional_exec_engine.cpp`
- `src/execution/cycle_exec_engine.cpp`
- `src/execution/encoded_exec_engine.cpp`
- `src/runtime/runtime_engine.cpp`
- trace-related runtime/timeline tests
- helper tests that construct trace events directly

### Out of Scope

- non-trace application strings
- renderer internals that only consume already-built events
- full schema redesign beyond the existing typed fields

## Producer Migration Plan

### Step 1: Vocabulary Completion

Complete the shared vocabulary in `trace_event_builder.h` so every currently repeated trace token
has one canonical definition.

### Step 2: Factory Introduction

Introduce semantic factories alongside the existing generic builders.

Factories should cover at minimum:

- wave launch
- wave step
- commit
- wave exit
- barrier arrive/release/wave
- memory arrive
- wait stall
- wave switch stall
- runtime launch
- block placed

### Step 3: Producer Conversion

Convert producer files to semantic factories.

No new producer call sites should directly pass raw semantic strings into generic trace builders.

### Step 4: Test Conversion

Convert trace-related tests to construct events through the same semantic factories used by
producers.

Direct `TraceEvent{...}` construction should remain only when a test is explicitly about the raw
schema object itself.

### Step 5: Guardrails

Add targeted tests that fail if:

- a typed stall event factory does not produce the expected canonical marker name
- a memory-arrive factory does not produce the expected canonical arrive label
- a lifecycle factory drifts from the shared vocabulary
- legacy message output disappears unexpectedly

## Testing Strategy

This change should follow TDD in batches:

1. Add builder/factory tests for the new semantic entry points
2. Verify those tests fail before implementation
3. Implement minimal semantic factories
4. Convert one producer family at a time
5. Re-run targeted runtime/cycle/functional/encoded trace tests after each batch

Required coverage:

- builder-level tests:
  - factory populates typed fields correctly
  - factory emits canonical compatibility message
- renderer-level tests:
  - chrome-json and native Perfetto names remain stable
- producer integration tests:
  - functional timeline gap
  - cycle resident slots
  - wave switch-away
  - barrier arrive/release

## Backward Compatibility

This design preserves:

- existing `TraceEvent` schema
- existing chrome-json export structure
- existing native Perfetto proto track/event structure
- existing legacy `message` output where tests or downstream tools still inspect it

Compatibility is preserved by making factories generate the same stable text, rather than by
allowing producers to freehand those strings.

## Risks

### Risk: Partial migration leaves two entry surfaces alive

Mitigation:

- semantic factories are introduced first
- producer migration follows immediately
- tests are updated to use factories so the preferred path becomes the dominant path

### Risk: Name drift in compatibility output

Mitigation:

- add builder-level golden checks for canonical names
- keep existing exporter tests for Perfetto-visible marker names

### Risk: Over-expanding the abstraction

Mitigation:

- keep `TraceEvent` schema as-is for this change
- unify vocabulary and construction first
- defer larger typed payload redesign

## Implementation Constraints

Implementation should follow these constraints:

- generic builders remain available during migration, but all newly touched producer code and tests
  must prefer semantic factories
- frequently reused trace detail formatters that define canonical trace text should move into the
  trace builder layer when they are needed by more than one producer or by both producer and test
- semantic factories remain in the existing trace builder surface for this change; do not split
  them into a second public helper header during this migration

## Recommendation

Proceed with the semantic factory approach.

This is the smallest change that fully satisfies the requirement to remove raw trace text from
producer and test construction while preserving the current unified typed schema trajectory.
