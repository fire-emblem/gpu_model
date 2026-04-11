#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "gpu_model/debug/recorder/export.h"
#include "gpu_model/debug/recorder/recorder.h"
#include "gpu_model/debug/trace/event_export.h"
#include "gpu_model/debug/trace/event_factory.h"
#include "gpu_model/debug/trace/event_view.h"
#include "tests/test_utils/trace_test_support.h"

namespace gpu_model {
namespace {

using test::MakeRecorder;

// =============================================================================
// Recorder Entry Building
// =============================================================================

TEST(TraceRecorderTest, BuildsPerWaveEntriesAndInstructionCycleRanges) {
  Recorder recorder;
  const TraceWaveView wave0{
      .dpc_id = 0,
      .ap_id = 1,
      .peu_id = 2,
      .slot_id = 3,
      .block_id = 4,
      .wave_id = 5,
      .pc = 0x40,
  };
  const TraceWaveView wave1{
      .dpc_id = 0,
      .ap_id = 1,
      .peu_id = 2,
      .slot_id = 3,
      .block_id = 4,
      .wave_id = 6,
      .pc = 0x80,
  };

  recorder.Record(MakeTraceRuntimeLaunchEvent(0, "kernel=recorder_test arch=mac500"));
  recorder.Record(
      MakeTraceWaveStepEvent(wave0, 8, TraceSlotModelKind::ResidentFixed, "pc=0x40 op=v_add_i32"));
  recorder.Record(MakeTraceCommitEvent(wave0, 12, TraceSlotModelKind::ResidentFixed));
  recorder.Record(
      MakeTraceWaveStepEvent(wave1, 16, TraceSlotModelKind::ResidentFixed, "pc=0x80 op=s_mov_b32"));

  ASSERT_EQ(recorder.events().size(), 4u);
  ASSERT_EQ(recorder.program_events().size(), 1u);
  EXPECT_EQ(recorder.program_events().front().kind, RecorderProgramEventKind::Launch);
  ASSERT_EQ(recorder.waves().size(), 2u);

  const RecorderWave& first_wave = recorder.waves().at(0);
  EXPECT_EQ(first_wave.dpc_id, 0u);
  EXPECT_EQ(first_wave.ap_id, 1u);
  EXPECT_EQ(first_wave.peu_id, 2u);
  EXPECT_EQ(first_wave.slot_id, 3u);
  EXPECT_EQ(first_wave.block_id, 4u);
  EXPECT_EQ(first_wave.wave_id, 5u);
  ASSERT_EQ(first_wave.entries.size(), 2u);
  EXPECT_EQ(first_wave.entries.at(0).kind, RecorderEntryKind::InstructionIssue);
  EXPECT_TRUE(first_wave.entries.at(0).has_cycle_range);
  EXPECT_EQ(first_wave.entries.at(0).begin_cycle, 8u);
  EXPECT_EQ(first_wave.entries.at(0).end_cycle, 12u);
  EXPECT_EQ(first_wave.entries.at(1).kind, RecorderEntryKind::Commit);
  EXPECT_FALSE(first_wave.entries.at(1).has_cycle_range);

  const RecorderWave& second_wave = recorder.waves().at(1);
  ASSERT_EQ(second_wave.entries.size(), 1u);
  EXPECT_EQ(second_wave.wave_id, 6u);
  EXPECT_EQ(second_wave.entries.at(0).kind, RecorderEntryKind::InstructionIssue);
  EXPECT_FALSE(second_wave.entries.at(0).has_cycle_range);
  EXPECT_EQ(second_wave.entries.at(0).begin_cycle, 16u);
  EXPECT_EQ(second_wave.entries.at(0).end_cycle, 16u);
}

TEST(TraceRecorderTest, PreservesSourceWaveIssueRangeWithoutCommitBackfill) {
  Recorder recorder;
  const TraceWaveView wave{
      .dpc_id = 0,
      .ap_id = 1,
      .peu_id = 2,
      .slot_id = 3,
      .block_id = 4,
      .wave_id = 5,
      .pc = 0x40,
  };

  recorder.Record(MakeTraceWaveStepEvent(wave,
                                         8,
                                         TraceSlotModelKind::ResidentFixed,
                                         "pc=0x40 op=v_add_i32",
                                         std::numeric_limits<uint64_t>::max(),
                                         /*issue_duration_cycles=*/12));

  ASSERT_EQ(recorder.waves().size(), 1u);
  const RecorderWave& recorded_wave = recorder.waves().front();
  ASSERT_EQ(recorded_wave.entries.size(), 1u);
  const RecorderEntry& step = recorded_wave.entries.front();
  EXPECT_EQ(step.kind, RecorderEntryKind::InstructionIssue);
  EXPECT_TRUE(step.has_cycle_range);
  EXPECT_EQ(step.begin_cycle, 8u);
  EXPECT_EQ(step.end_cycle, 20u);
}

TEST(TraceRecorderTest, DoesNotOverwriteSourceWaveIssueRangeWhenCommitArrives) {
  Recorder recorder;
  const TraceWaveView wave{
      .dpc_id = 0,
      .ap_id = 1,
      .peu_id = 2,
      .slot_id = 3,
      .block_id = 4,
      .wave_id = 5,
      .pc = 0x40,
  };

  recorder.Record(MakeTraceWaveStepEvent(wave,
                                         8,
                                         TraceSlotModelKind::ResidentFixed,
                                         "pc=0x40 op=v_add_i32",
                                         std::numeric_limits<uint64_t>::max(),
                                         /*issue_duration_cycles=*/12));
  recorder.Record(MakeTraceCommitEvent(wave, 9, TraceSlotModelKind::ResidentFixed));

  ASSERT_EQ(recorder.waves().size(), 1u);
  const RecorderWave& recorded_wave = recorder.waves().front();
  ASSERT_EQ(recorded_wave.entries.size(), 2u);
  const RecorderEntry& step = recorded_wave.entries.front();
  EXPECT_TRUE(step.has_cycle_range);
  EXPECT_EQ(step.begin_cycle, 8u);
  EXPECT_EQ(step.end_cycle, 20u);
}

TEST(TraceRecorderTest, CapturesTypedSemanticSnapshotForReplayFacingUses) {
  Recorder recorder;
  const TraceWaveView wave{
      .dpc_id = 0,
      .ap_id = 0,
      .peu_id = 0,
      .slot_id = 1,
      .block_id = 2,
      .wave_id = 3,
      .pc = 0x44,
  };
  const TraceWaitcntState waitcnt_state{
      .valid = true,
      .threshold_global = 0,
      .threshold_shared = UINT32_MAX,
      .threshold_private = UINT32_MAX,
      .threshold_scalar_buffer = UINT32_MAX,
      .pending_global = 2,
      .pending_shared = 0,
      .pending_private = 0,
      .pending_scalar_buffer = 0,
      .blocked_global = true,
      .blocked_shared = false,
      .blocked_private = false,
      .blocked_scalar_buffer = false,
  };

  recorder.Record(MakeTraceWaveLaunchEvent(
      wave, 1, "lanes=0x40 exec=0xffffffffffffffff", TraceSlotModelKind::ResidentFixed));
  recorder.Record(MakeTraceWaitStallEvent(wave,
                                          8,
                                          TraceStallReason::WaitCntGlobal,
                                          TraceSlotModelKind::ResidentFixed,
                                          std::numeric_limits<uint64_t>::max(),
                                          waitcnt_state));
  recorder.Record(MakeTraceMemoryArriveEvent(
      wave, 12, TraceMemoryArriveKind::Load, TraceSlotModelKind::ResidentFixed));
  recorder.Record(
      MakeTraceBarrierArriveEvent(wave, 16, TraceSlotModelKind::ResidentFixed));

  ASSERT_EQ(recorder.waves().size(), 1u);
  const RecorderWave& recorded_wave = recorder.waves().front();
  ASSERT_EQ(recorded_wave.entries.size(), 4u);

  EXPECT_EQ(recorded_wave.entries.at(0).kind, RecorderEntryKind::WaveLaunch);
  EXPECT_EQ(recorded_wave.entries.at(0).lifecycle_stage, TraceLifecycleStage::Launch);
  EXPECT_EQ(recorded_wave.entries.at(0).canonical_name, "wave_launch");

  EXPECT_EQ(recorded_wave.entries.at(1).kind, RecorderEntryKind::Stall);
  EXPECT_EQ(recorded_wave.entries.at(1).stall_reason, TraceStallReason::WaitCntGlobal);
  EXPECT_TRUE(recorded_wave.entries.at(1).waitcnt_state.valid);
  EXPECT_TRUE(recorded_wave.entries.at(1).waitcnt_state.blocked_global);
  EXPECT_EQ(recorded_wave.entries.at(1).category, "stall/waitcnt_global");

  EXPECT_EQ(recorded_wave.entries.at(2).kind, RecorderEntryKind::Arrive);
  EXPECT_EQ(recorded_wave.entries.at(2).arrive_kind, TraceArriveKind::Load);
  EXPECT_EQ(recorded_wave.entries.at(2).canonical_name, "load_arrive");

  EXPECT_EQ(recorded_wave.entries.at(3).kind, RecorderEntryKind::Barrier);
  EXPECT_EQ(recorded_wave.entries.at(3).barrier_kind, TraceBarrierKind::Arrive);
  EXPECT_EQ(recorded_wave.entries.at(3).canonical_name, "barrier_arrive");
}

// =============================================================================
// Recorder Export
// =============================================================================

TEST(TraceRecorderTest, ExportsTextAndJsonInRecordedOrder) {
  Recorder recorder;
  const TraceWaveView wave{
      .dpc_id = 0,
      .ap_id = 0,
      .peu_id = 0,
      .slot_id = 2,
      .block_id = 0,
      .wave_id = 1,
      .pc = 0x24,
  };

  recorder.Record(MakeTraceRuntimeLaunchEvent(0, "kernel=recorder_export arch=mac500"));
  recorder.Record(MakeTraceWaveLaunchEvent(
      wave, 1, "lanes=0x40 exec=0xffffffffffffffff", TraceSlotModelKind::ResidentFixed));
  recorder.Record(
      MakeTraceWaveStepEvent(wave, 2, TraceSlotModelKind::ResidentFixed, "pc=0x24 op=v_add_i32"));
  recorder.Record(MakeTraceCommitEvent(wave, 6, TraceSlotModelKind::ResidentFixed));
  recorder.Record(MakeTraceWaveExitEvent(wave, 7, TraceSlotModelKind::ResidentFixed));

  const std::string text = RenderRecorderTextTrace(recorder);
  const std::string json = RenderRecorderJsonTrace(recorder);

  // Text trace now only includes WaveStep and WaveExit for cleaner output.
  // Other events are still available in JSON trace.
  // WaveStep uses the instruction mnemonic as canonical_name (e.g., "v_add_i32")
  // WaveExit uses "wave_exit" as canonical_name
  EXPECT_NE(text.find("v_add_i32"), std::string::npos);
  EXPECT_NE(text.find("wave_exit"), std::string::npos);
  // Verify ordering
  EXPECT_LT(text.find("v_add_i32"), text.find("wave_exit"));

  // JSON trace still contains all events
  EXPECT_NE(json.find("\"kind\":\"Launch\""), std::string::npos);
  EXPECT_NE(json.find("\"kind\":\"WaveLaunch\""), std::string::npos);
  EXPECT_NE(json.find("\"kind\":\"WaveStep\""), std::string::npos);
  EXPECT_NE(json.find("\"kind\":\"Commit\""), std::string::npos);
  EXPECT_NE(json.find("\"kind\":\"WaveExit\""), std::string::npos);
  EXPECT_NE(json.find("\"has_cycle_range\":true"), std::string::npos);
  EXPECT_NE(json.find("\"begin_cycle\":\"0x2\""), std::string::npos);
  EXPECT_NE(json.find("\"end_cycle\":\"0x6\""), std::string::npos);
  EXPECT_LT(json.find("\"kind\":\"Launch\""), json.find("\"kind\":\"WaveLaunch\""));
  EXPECT_LT(json.find("\"kind\":\"WaveLaunch\""), json.find("\"kind\":\"WaveStep\""));
  EXPECT_LT(json.find("\"kind\":\"WaveStep\""), json.find("\"kind\":\"Commit\""));
  EXPECT_LT(json.find("\"kind\":\"Commit\""), json.find("\"kind\":\"WaveExit\""));
}

// =============================================================================
// Flow Export Gating
// =============================================================================

TEST(TraceRecorderTest, EntryTraceEventExportRespectsFlowGating) {
  TraceEvent issue;
  issue.kind = TraceEventKind::Commit;
  issue.flow_phase = TraceFlowPhase::Start;
  issue.flow_id = 0;

  RecorderEntry entry;
  entry.kind = RecorderEntryKind::Commit;
  entry.event = issue;

  const TraceEventExportFields fields = MakeTraceEventExportFields(entry);
  EXPECT_FALSE(fields.has_flow);
  EXPECT_TRUE(fields.flow_id.empty());
  EXPECT_TRUE(fields.flow_phase.empty());
}

TEST(TraceRecorderTest, ProgramEventTraceEventExportRespectsFlowGating) {
  TraceEvent issue;
  issue.kind = TraceEventKind::BlockLaunch;
  issue.flow_phase = TraceFlowPhase::Finish;
  issue.flow_id = 0;

  RecorderProgramEvent program_event;
  program_event.kind = RecorderProgramEventKind::BlockLaunch;
  program_event.event = issue;

  const TraceEventExportFields fields = MakeTraceEventExportFields(program_event);
  EXPECT_FALSE(fields.has_flow);
  EXPECT_TRUE(fields.flow_id.empty());
  EXPECT_TRUE(fields.flow_phase.empty());
}

}  // namespace
}  // namespace gpu_model
