# Round 1 Summary

## Work Completed
- Extended the execution-owned state-edge schema for functional and cycle execution:
  - added `ActivePromote`, `WaveWait`, `WaveArrive`, `WaveResume`, and `WaveSwitchAway` to [event.h](/data/gpu_model/src/gpu_model/debug/trace/event.h) and [event_factory.h](/data/gpu_model/src/gpu_model/debug/trace/event_factory.h)
  - mapped the new typed edges through [recorder.h](/data/gpu_model/src/gpu_model/debug/recorder/recorder.h) and [recorder.cpp](/data/gpu_model/src/debug/recorder/recorder.cpp)
  - emitted the new edges from real execution transitions in [functional_exec_engine.cpp](/data/gpu_model/src/execution/functional_exec_engine.cpp) and [cycle_exec_engine.cpp](/data/gpu_model/src/execution/cycle_exec_engine.cpp)
- Wired the new schema through current consumers so markers stay stable while task6 is still pending:
  - [trace_event_view.cpp](/data/gpu_model/src/debug/trace/trace_event_view.cpp)
  - [trace_format.cpp](/data/gpu_model/src/debug/trace/trace_format.cpp)
  - [cycle_timeline.cpp](/data/gpu_model/src/debug/timeline/cycle_timeline.cpp)
  - [cycle_timeline_google_trace.cpp](/data/gpu_model/src/debug/timeline/cycle_timeline_google_trace.cpp)
- Added focused regressions for wave-state-edge order and trace-disable parity:
  - functional waitcnt emits `wave_wait -> wave_arrive -> wave_resume -> next WaveStep`
  - cycle waitcnt(0) with two outstanding loads emits `Arrive(still_blocked) -> Arrive(resume) -> WaveResume -> next WaveStep`
  - runtime-level `GPU_MODEL_DISABLE_TRACE=1` parity for both functional and cycle waitcnt flows
- Added the round-0 review-mandated task1 addendum here in round artifacts:
  - residual consumer-side semantic derivation still exists in [trace_event_view.cpp](/data/gpu_model/src/debug/trace/trace_event_view.cpp) and [cycle_timeline_google_trace.cpp](/data/gpu_model/src/debug/timeline/cycle_timeline_google_trace.cpp)
  - those derivations remain explicit task6 removal targets; they are not treated as execution semantics

## Task4 Audit

### Recorder Production Paths
- `functional st` and `functional mt` both emit `TraceEvent`s from [functional_exec_engine.cpp](/data/gpu_model/src/execution/functional_exec_engine.cpp) with `TraceSlotModelKind::LogicalUnbounded`, then land in [trace_artifact_recorder.cpp](/data/gpu_model/src/debug/trace/trace_artifact_recorder.cpp) through `RecorderTraceSink` into the unified [recorder.cpp](/data/gpu_model/src/debug/recorder/recorder.cpp) path.
- `cycle` emits `TraceEvent`s from [cycle_exec_engine.cpp](/data/gpu_model/src/execution/cycle_exec_engine.cpp) with `TraceSlotModelKind::ResidentFixed`, then lands in the same [trace_artifact_recorder.cpp](/data/gpu_model/src/debug/trace/trace_artifact_recorder.cpp) -> [recorder.cpp](/data/gpu_model/src/debug/recorder/recorder.cpp) path.
- `encoded functional/cycle` already chooses logical-unbounded vs resident-fixed in [encoded_exec_engine.cpp](/data/gpu_model/src/execution/encoded_exec_engine.cpp#L1104), and also feeds the same recorder sink path, but it still emits generic `Stall` / `Arrive` semantics and has not been upgraded to the new wave-state-edge schema added this round.

### Slot Export Rule
- The slot-model rule is currently enforced at execution emission sites, not in recorder:
  - functional uses `TraceSlotModelKind::LogicalUnbounded` directly in [functional_exec_engine.cpp](/data/gpu_model/src/execution/functional_exec_engine.cpp)
  - cycle uses `TraceSlotModelKind::ResidentFixed` directly in [cycle_exec_engine.cpp](/data/gpu_model/src/execution/cycle_exec_engine.cpp)
  - encoded chooses via `TraceSlotModel()` in [encoded_exec_engine.cpp](/data/gpu_model/src/execution/encoded_exec_engine.cpp#L1104)
- Recorder currently preserves `slot_model_kind`; hierarchy rendering then derives `DPC/AP/PEU/WAVE_SLOT` from slot keys in [cycle_timeline_perfetto.cpp](/data/gpu_model/src/debug/timeline/cycle_timeline_perfetto.cpp) and [cycle_timeline_layout.cpp](/data/gpu_model/src/debug/timeline/cycle_timeline_layout.cpp).
- This means the slot export contract is shared in practice, but not yet centralized behind one recorder-facing abstraction. That remains a task5 item.

### Remaining Non-Recorder Semantic Paths
- High: [recorder.cpp](/data/gpu_model/src/debug/recorder/recorder.cpp#L11) still synthesizes instruction ranges with `NormalizeInstructionRangeCycles(0)` instead of consuming execution-owned intervals. This is the main AC-3 blocker.
- High: [trace_event_view.cpp](/data/gpu_model/src/debug/trace/trace_event_view.cpp) still derives semantic names/categories from raw fields:
  - waitcnt-specific stall names via `CanonicalNameFromStall(const TraceWaitcntState&, ...)`
  - `wave_switch_away` from generic `Stall + WarpSwitch`
  - arrive categories from `canonical_name.ends_with(...)`
  - legacy message fallbacks for barrier/arrive/lifecycle parsing
- High: [trace_format.cpp](/data/gpu_model/src/debug/trace/trace_format.cpp) still serializes text/json from raw `TraceEvent` via `MakeCanonicalTraceEvent(event)`, not from `RecorderEntry` / `RecorderProgramEvent`.
- Medium: [cycle_timeline_google_trace.cpp](/data/gpu_model/src/debug/timeline/cycle_timeline_google_trace.cpp) still contains fallback naming/category logic for markers when recorder-provided semantic fields are incomplete.
- Medium: [trace_event_export.cpp](/data/gpu_model/src/debug/trace/trace_event_export.cpp) is still an event-view canonicalization layer, not a recorder-export layer; text/json consumers depend on it directly.

### Module-Boundary / Dependency Strain
- High: Recorder currently depends on trace canonicalization internals by calling `MakeCanonicalTraceEvent(event)` inside [recorder.cpp](/data/gpu_model/src/debug/recorder/recorder.cpp). That keeps recorder downstream of trace-view shaping instead of owning the canonical debug protocol itself.
- Medium: text/json are still implemented under `src/debug/trace/*` and centered on `TraceEvent`, while recorder serializers live separately under `src/debug/recorder/*`. The write path is unified only inside [trace_artifact_recorder.cpp](/data/gpu_model/src/debug/trace/trace_artifact_recorder.cpp), not at the serializer interface boundary.
- Medium: encoded execution is on the unified sink path, but semantically lags behind functional/cycle on new typed state edges. That creates model skew at the producer boundary even before task6 consumer cleanup.

## Files Changed
- [event.h](/data/gpu_model/src/gpu_model/debug/trace/event.h)
- [event_factory.h](/data/gpu_model/src/gpu_model/debug/trace/event_factory.h)
- [recorder.h](/data/gpu_model/src/gpu_model/debug/recorder/recorder.h)
- [recorder.cpp](/data/gpu_model/src/debug/recorder/recorder.cpp)
- [trace_event_view.cpp](/data/gpu_model/src/debug/trace/trace_event_view.cpp)
- [trace_format.cpp](/data/gpu_model/src/debug/trace/trace_format.cpp)
- [cycle_timeline.cpp](/data/gpu_model/src/debug/timeline/cycle_timeline.cpp)
- [cycle_timeline_google_trace.cpp](/data/gpu_model/src/debug/timeline/cycle_timeline_google_trace.cpp)
- [functional_exec_engine.cpp](/data/gpu_model/src/execution/functional_exec_engine.cpp)
- [cycle_exec_engine.cpp](/data/gpu_model/src/execution/cycle_exec_engine.cpp)
- [functional_exec_engine_waitcnt_test.cpp](/data/gpu_model/tests/execution/functional_exec_engine_waitcnt_test.cpp)
- [shared_barrier_cycle_test.cpp](/data/gpu_model/tests/cycle/shared_barrier_cycle_test.cpp)
- [async_memory_cycle_test.cpp](/data/gpu_model/tests/cycle/async_memory_cycle_test.cpp)
- [execution_stats_test.cpp](/data/gpu_model/tests/runtime/execution_stats_test.cpp)
- [trace_test.cpp](/data/gpu_model/tests/runtime/trace_test.cpp)
- [cycle_timeline_test.cpp](/data/gpu_model/tests/runtime/cycle_timeline_test.cpp)

## Validation
- `cmake --build build-ninja --target gpu_model_tests -j8`
  - passed
- `./build-ninja/tests/gpu_model_tests --gtest_filter='AsyncMemoryCycleTest.WaitCntZeroKeepsWaveBlockedUntilFinalArriveThenResumesIssue:ExecutionStatsTest.FunctionalWaitcntKeepsCyclesAndResultsWhenTraceIsDisabled:ExecutionStatsTest.CycleWaitcntKeepsCyclesWhenTraceIsDisabled:FunctionalExecEngineWaitcntTest.GlobalWaitcntEmitsWaveWaitArriveAndResumeMarkers:SharedBarrierCycleTest.BarrierLifecycleEmitsWaveWaitAndWaveResumeMarkers:TraceTest.TraceEventViewProvidesStableCanonicalNamesForWaveStateEdgeMarkers:CycleTimelineTest.GoogleTraceRendersWaveStateEdgeMarkersWithStableTypedNames'`
  - 7 tests passed
- `./build-ninja/tests/gpu_model_tests --gtest_filter='AsyncMemoryCycleTest.WaitCntCanWaitForGlobalMemoryOnly:AsyncMemoryCycleTest.WaitCntIgnoresGlobalWhenWaitingSharedOnly:AsyncMemoryCycleTest.WaitCntCanWaitForScalarBufferScalarLoadOnly:FunctionalExecEngineWaitcntTest.ResumesWhenStoredThresholdBecomesSatisfied:FunctionalExecEngineWaitcntTest.WaitcntResumeRequiresAllStoredThresholdDomains:ExecutionStatsTest.GlobalDisableTraceEnvForcesNullTraceSinkWithoutBreakingCycles'`
  - 6 tests passed

## Remaining Items
- `task5` is still open: recorder does not own instruction intervals yet; [recorder.cpp](/data/gpu_model/src/debug/recorder/recorder.cpp) still normalizes from zero.
- `task6` is still open: text/json/trace-view/google-trace still derive semantics from raw `TraceEvent` fields instead of consuming recorder-only facts.
- `task7` and `task8` are still open: no new representative example calibration or visual note landed in this round.
- `task9` is still open: docs were not updated in this round.
- Encoded execution still lacks parity with the new wave-state-edge schema and should be treated as an open producer-side skew until addressed.

## Goal Tracker Update Request

### Requested Changes:
- Mark `task2-calibrate-execution-state-edges` as completed with evidence from functional/cycle producer changes in [functional_exec_engine.cpp](/data/gpu_model/src/execution/functional_exec_engine.cpp), [cycle_exec_engine.cpp](/data/gpu_model/src/execution/cycle_exec_engine.cpp), [event.h](/data/gpu_model/src/gpu_model/debug/trace/event.h), and [recorder.cpp](/data/gpu_model/src/debug/recorder/recorder.cpp).
- Mark `task3-add-focused-regressions` as partial progress, not completed.
  - Evidence: [functional_exec_engine_waitcnt_test.cpp](/data/gpu_model/tests/execution/functional_exec_engine_waitcnt_test.cpp), [shared_barrier_cycle_test.cpp](/data/gpu_model/tests/cycle/shared_barrier_cycle_test.cpp), [async_memory_cycle_test.cpp](/data/gpu_model/tests/cycle/async_memory_cycle_test.cpp), [execution_stats_test.cpp](/data/gpu_model/tests/runtime/execution_stats_test.cpp), [trace_test.cpp](/data/gpu_model/tests/runtime/trace_test.cpp), [cycle_timeline_test.cpp](/data/gpu_model/tests/runtime/cycle_timeline_test.cpp).
- Mark `task4-audit-recorder-unification` as completed with the audit findings recorded in this summary.
- Add to Open Issues: encoded execution still uses the unified recorder sink path but has not adopted the new `ActivePromote/WaveWait/WaveArrive/WaveResume/WaveSwitchAway` producer schema, so model parity is incomplete at the producer boundary.

### Justification:
- Round 1 finished the requested execution-first work for functional/cycle, added validation for wait/resume ordering and trace-disable parity, and completed the requested recorder/consumer audit.
- The remaining blockers are now explicitly narrowed to recorder interval ownership, consumer de-derivation, encoded-producer parity, examples, and docs.

## BitLesson Delta
- Action: none
- Lesson ID(s): NONE
- Notes: `bitlesson-selector` is not available in this environment, and no reusable new project lesson was extracted in this round.
