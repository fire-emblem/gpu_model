# Trace Flow Serialization Design

## Context

- `TraceEventExportFields` already carries the flow metadata from recorder events, but the\n+  text and JSON traces still omit those fields when writing `trace.txt` and `trace.jsonl`.
- The JSON serializer shares the `AppendTraceExportJsonFields` helper with the timeline\n+  exporter, so adding flow metadata there would leak into Perfetto traces (out of scope for\n+  Task 7).

## Desired Outcome

- `trace.txt` and `trace.jsonl` should include `has_flow`, `flow_id`, and `flow_phase` only\n+  when the exporter reports `fields.has_flow`.\n+- Timeline exports must remain unchanged: the shared JSON helper keeps ignoring flow\n+  metadata, and only the trace-specific serializer adds the fields locally.

## Approach

1. Exercise both `FileTraceSink` and `JsonTraceSink` with positive+negative tests to prove\n+   the metadata appears when present and is omitted when absent.\n+2. Update `FormatTextTraceLineFromFields` to append `has_flow=1`, `flow_id=...`, and\n+   `flow_phase=...` whenever `fields.has_flow` is true.\n+3. Keep `AppendTraceExportJsonFields` flow-free so timeline exporters continue to reuse it\n+   unchanged; let `FormatJsonTraceLineFromFields` directly serialize the flow metadata before\n+   writing the rest of the JSON line.

## Testing

- `tests/runtime/trace_test.cpp` now contains targeted assertions for text/json flow serialization.\n+- Verified via `./build-ninja/tests/gpu_model_tests --gtest_filter='TraceTest.FileTraceSinkSerializesFlowMetadata:TraceTest.JsonTraceSinkSerializesFlowMetadata:TraceTest.JsonTraceSinkSkipsFlowMetadataWhenNoFlow'`.
