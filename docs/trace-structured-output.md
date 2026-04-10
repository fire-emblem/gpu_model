# Trace Structured Output

This document captures the immutable trace contract that feeds the structured renderer described in the newer `trace.txt` adaptation. It builds on the runtime constraints laid out in `AGENTS.md` and `docs/my_design.md` so the trace remains a consumer of producer-defined facts and a faithful reflection of modeled timing.

## Hard Constraints

- trace consumes producer-owned facts only
- trace must not infer wait/arrive/resume/business state
- trace `cycle` is modeled time, not physical hardware time
- `WaveResume` means ready/eligible, not issued
- `WaveStep` is the authoritative execution fact
- `GPU_MODEL_DISABLE_TRACE=1` must disable artifact generation without changing execution results

## Phase-1 Output Scope

- sectioned `trace.txt`
- enriched `trace.jsonl`
- unchanged `timeline.perfetto.json` semantics
- run/kernel/model/wave-init/summary snapshots
- structured `WaveStep` detail

## Document Sections

Recorder facts remain the shared protocol for all trace artifacts; text, JSON, and Perfetto outputs are consumer views built from those facts. The phase-1 renderer therefore emits a sectioned `trace.txt` so humans can navigate the statement of facts just like the reference template from `src/debug/ref/`, and no section attempts to reinterpret modeled-cycle counts as wall-clock time. Phase 1 intentionally omits extra config/display/resource sections and does not expect a populated `[WARNINGS]` section until producers supply dedicated `TraceWarningSnapshot` records in a later phase.

### Sectioned `trace.txt`

The structured text renderer will print ordered headers, snapshot contexts, run/kernel/model summaries, a `[WAVE_INIT]` roster, `[EVENTS]` for typed events, and `[SUMMARY]`/`[WARNINGS]` tails. Every line is produced from recorder-held facts; the renderer never guesses ready/wait/issue transitions. This format keeps `cycle` as modeled time, consistent with the AGENTS/my_design rules.

### Enriched `trace.jsonl`

`trace.jsonl` serves as a typed serialization of the shared recorder facts: each entry reflects snapshots, event metadata, or `WaveStep` detail. Structured fields replace message parsing, and the JSON always references modeled-cycle counters (e.g., the `cycle` field continues to match the engine's internal time counter).

### `timeline.perfetto.json`

Perfetto exports continue to rely on the existing modeled-time semantics. The renderer will emit the same `timeline.perfetto.json` once the structured recorder flushes events, so Perfetto consumers do not perceive any new physical-time interpretation.

### Snapshots and WaveStep Detail

Run/kernel/model/wave-init/summary snapshots must live in the recorder before rendering, which allows `trace.txt` and `trace.jsonl` to write unambiguous sections and facts. Every `WaveStep` will embed structured detail (asm, operands, timing, state deltas) so the renderer can describe execution without re-deriving anything from natural-language `message` text. This keeps `WaveStep` as the authoritative execution fact and preserves the `WaveResume` semantics declared above. Warnings are produced by distinct producer-owned `TraceWarningSnapshot` records and remain deferred beyond phase 1, so the renderer does not invent `[WARNINGS]` content for the initial rollout.
