#include "debug/timeline/actual_timeline_snapshot.h"

#include <string_view>

namespace gpu_model {

namespace {

TimelineLaneKey MakeLaneKey(const RecorderWave& wave) {
  return TimelineLaneKey{
      .dpc_id = wave.dpc_id,
      .ap_id = wave.ap_id,
      .peu_id = wave.peu_id,
      .slot_id = wave.slot_id,
      .wave_id = wave.wave_id,
  };
}

std::string ExtractMnemonic(const std::string& display_name) {
  // If display_name contains assembly text like "buffer_load_dword v1, s0, ...",
  // extract just the mnemonic (first word)
  const auto space_pos = display_name.find(' ');
  if (space_pos != std::string::npos) {
    return display_name.substr(0, space_pos);
  }
  return display_name;
}

std::string EventName(const RecorderEntry& entry) {
  if (!entry.presentation_name.empty()) {
    return entry.presentation_name;
  }
  if (!entry.canonical_name.empty()) {
    return entry.canonical_name;
  }
  if (!entry.display_name.empty()) {
    return ExtractMnemonic(entry.display_name);
  }
  return entry.compatibility_message;
}

bool IsTimelineMarkerEntryKind(RecorderEntryKind kind) {
  switch (kind) {
    case RecorderEntryKind::WaveLaunch:
    case RecorderEntryKind::WaveGenerate:
    case RecorderEntryKind::WaveDispatch:
    case RecorderEntryKind::SlotBind:
    case RecorderEntryKind::ActivePromote:
    case RecorderEntryKind::IssueSelect:
    case RecorderEntryKind::WaveWait:
    case RecorderEntryKind::WaveArrive:
    case RecorderEntryKind::WaveResume:
    case RecorderEntryKind::WaveSwitchAway:
    case RecorderEntryKind::Barrier:
    case RecorderEntryKind::WaveExit:
    case RecorderEntryKind::Stall:
    case RecorderEntryKind::Arrive:
      return true;
    case RecorderEntryKind::WaveStats:
    case RecorderEntryKind::InstructionIssue:
    case RecorderEntryKind::Commit:
    case RecorderEntryKind::ExecMaskUpdate:
    case RecorderEntryKind::MemoryAccess:
      return false;
  }
  return false;
}

}  // namespace

ActualTimelineSnapshot BuildActualTimelineSnapshot(const Recorder& recorder) {
  ActualTimelineSnapshot snapshot;
  for (const auto& wave : recorder.waves()) {
    const TimelineLaneKey lane = MakeLaneKey(wave);
    for (const auto& entry : wave.entries) {
      if (entry.kind == RecorderEntryKind::InstructionIssue && entry.has_cycle_range) {
        snapshot.slices.push_back(ActualSlice{
            .key =
                TimelineEventKey{
                    .lane = lane,
                    .pc = entry.event.pc,
                    .name = EventName(entry),
                },
            .begin_cycle = entry.begin_cycle,
            .end_cycle = entry.end_cycle,
            .sequence = entry.sequence,
        });
        continue;
      }
      if (!IsTimelineMarkerEntryKind(entry.kind)) {
        continue;
      }
      snapshot.markers.push_back(ActualMarker{
          .key =
              TimelineEventKey{
                  .lane = lane,
                  .pc = entry.event.pc,
                  .name = EventName(entry),
              },
          .cycle = entry.event.cycle,
          .sequence = entry.sequence,
          .stall_reason = entry.stall_reason,
          .arrive_progress = entry.event.arrive_progress,
      });
    }
  }
  return snapshot;
}

}  // namespace gpu_model
