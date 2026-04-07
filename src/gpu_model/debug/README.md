# Debug Module Boundary

This directory exposes the stable debug-facing API surface for the GPU model.

Preferred public entry points live under grouped module directories:

- `debug/trace/...`
- `debug/recorder/...`
- `debug/timeline/...`
- `debug/replay/...`
- `debug/info/...`

Rules:

- Prefer including grouped headers such as `trace/api.h`, `trace/event.h`,
  `trace/event_factory.h`, `trace/sink.h`, `recorder/recorder.h`,
  `recorder/export.h`, `timeline/cycle_timeline.h`, and `info/debug_info.h`.
- Top-level `debug/*.h` compatibility wrappers have been removed. New code must include grouped
  module headers only.
- Headers under `debug/internal/` are implementation details. They may change without notice and
  should not be treated as stable external contracts.
- New render/export/helper logic should stay in `debug/internal/` or `src/debug/`, unless it is
  intentionally being added to the stable public API.

Design intent:

- program execution producer is a single path: `ProgramObject -> LaunchProgramObject -> ExecEngine`.
- debug/trace modules must not branch behavior based on historical `canonical` vs `encoded` source names.
- execution semantics such as `arrive` completion, `waitcnt` threshold satisfaction,
  barrier release, and wave switch scheduling must be decided in the execution model itself.
- trace / recorder / timeline / Perfetto are projection layers only: they consume typed execution
  outcomes and serialize them, but they must not invent execution-state transitions that would not
  exist when trace capture is disabled.
- timeline cycle data must come from execution-produced modeled cycle facts recorded by the recorder.
  Serializers and renderers must not infer or repair timing gaps on their own.
- `trace/event.h` defines the semantic event schema.
- `trace/event_factory.h` defines the public factory helpers used by runtime/execution/tests to
  construct semantic trace events.
- `trace/event_view.h` and `trace/event_export.h` define the stable semantic projection/export
  protocol used by recorder, renderers, and tests.
- `recorder/recorder.h` defines the execution recorder boundary between runtime-produced events and
  serializer / renderer consumers.
- `recorder/export.h` defines text/json serialization entry points that consume the recorder.
- `replay/replayer.h` reserves the future execution replay / state-restore boundary.
- `trace/sink.h` and `trace/artifact_recorder.h` define trace capture/output entry points.
- `timeline/cycle_timeline.h` defines trace rendering entry points.
- Canonical projection, export-field derivation, builder helpers, JSON helpers, and render internals
  are intentionally hidden behind the public surface.
