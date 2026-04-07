#include "gpu_model/debug/recorder/recorder.h"

#include <algorithm>

#include "gpu_model/debug/trace/event_export.h"

namespace gpu_model {

namespace {

constexpr uint64_t kRecorderIssueQuantumCycles = 4;

uint64_t NormalizeInstructionRangeCycles(uint64_t cycles) {
  const uint64_t clamped = std::max<uint64_t>(kRecorderIssueQuantumCycles, cycles);
  const uint64_t remainder = clamped % kRecorderIssueQuantumCycles;
  if (remainder == 0) {
    return clamped;
  }
  return clamped + (kRecorderIssueQuantumCycles - remainder);
}

RecorderWave* FindWave(std::vector<RecorderWave>& waves, const TraceEvent& event) {
  for (auto& wave : waves) {
    if (wave.dpc_id == event.dpc_id && wave.ap_id == event.ap_id && wave.peu_id == event.peu_id &&
        wave.slot_id == event.slot_id && wave.block_id == event.block_id &&
        wave.wave_id == event.wave_id) {
      return &wave;
    }
  }
  return nullptr;
}

RecorderWave& GetOrCreateWave(std::vector<RecorderWave>& waves, const TraceEvent& event) {
  if (RecorderWave* wave = FindWave(waves, event)) {
    return *wave;
  }
  waves.push_back(RecorderWave{
      .dpc_id = event.dpc_id,
      .ap_id = event.ap_id,
      .peu_id = event.peu_id,
      .slot_id = event.slot_id,
      .block_id = event.block_id,
      .wave_id = event.wave_id,
      .entries = {},
  });
  return waves.back();
}

bool EventBelongsToWave(const TraceEvent& event) {
  switch (event.kind) {
    case TraceEventKind::WaveLaunch:
    case TraceEventKind::WaveGenerate:
    case TraceEventKind::WaveDispatch:
    case TraceEventKind::SlotBind:
    case TraceEventKind::IssueSelect:
    case TraceEventKind::WaveStats:
    case TraceEventKind::WaveStep:
    case TraceEventKind::Commit:
    case TraceEventKind::ExecMaskUpdate:
    case TraceEventKind::MemoryAccess:
    case TraceEventKind::Barrier:
    case TraceEventKind::WaveExit:
    case TraceEventKind::Stall:
    case TraceEventKind::Arrive:
      return true;
    case TraceEventKind::Launch:
    case TraceEventKind::BlockPlaced:
    case TraceEventKind::BlockAdmit:
    case TraceEventKind::BlockLaunch:
    case TraceEventKind::BlockActivate:
    case TraceEventKind::BlockRetire:
      return false;
  }
  return false;
}

RecorderProgramEventKind ProgramEventKindFromTraceEvent(const TraceEvent& event) {
  switch (event.kind) {
    case TraceEventKind::Launch:
      return RecorderProgramEventKind::Launch;
    case TraceEventKind::BlockPlaced:
      return RecorderProgramEventKind::BlockPlaced;
    case TraceEventKind::BlockAdmit:
      return RecorderProgramEventKind::BlockAdmit;
    case TraceEventKind::BlockLaunch:
      return RecorderProgramEventKind::BlockLaunch;
    case TraceEventKind::BlockActivate:
      return RecorderProgramEventKind::BlockActivate;
    case TraceEventKind::BlockRetire:
      return RecorderProgramEventKind::BlockRetire;
    case TraceEventKind::WaveGenerate:
    case TraceEventKind::WaveDispatch:
    case TraceEventKind::SlotBind:
    case TraceEventKind::IssueSelect:
    case TraceEventKind::WaveLaunch:
    case TraceEventKind::WaveStats:
    case TraceEventKind::WaveStep:
    case TraceEventKind::Commit:
    case TraceEventKind::ExecMaskUpdate:
    case TraceEventKind::MemoryAccess:
    case TraceEventKind::Barrier:
    case TraceEventKind::WaveExit:
    case TraceEventKind::Stall:
    case TraceEventKind::Arrive:
      break;
  }
  return RecorderProgramEventKind::Launch;
}

RecorderEntryKind EntryKindFromTraceEvent(const TraceEvent& event) {
  switch (event.kind) {
    case TraceEventKind::WaveLaunch:
      return RecorderEntryKind::WaveLaunch;
    case TraceEventKind::WaveGenerate:
      return RecorderEntryKind::WaveGenerate;
    case TraceEventKind::WaveDispatch:
      return RecorderEntryKind::WaveDispatch;
    case TraceEventKind::SlotBind:
      return RecorderEntryKind::SlotBind;
    case TraceEventKind::IssueSelect:
      return RecorderEntryKind::IssueSelect;
    case TraceEventKind::BlockAdmit:
    case TraceEventKind::BlockActivate:
    case TraceEventKind::BlockRetire:
      break;
    case TraceEventKind::WaveStats:
      return RecorderEntryKind::WaveStats;
    case TraceEventKind::WaveStep:
      return RecorderEntryKind::InstructionIssue;
    case TraceEventKind::Commit:
      return RecorderEntryKind::Commit;
    case TraceEventKind::ExecMaskUpdate:
      return RecorderEntryKind::ExecMaskUpdate;
    case TraceEventKind::MemoryAccess:
      return RecorderEntryKind::MemoryAccess;
    case TraceEventKind::Barrier:
      return RecorderEntryKind::Barrier;
    case TraceEventKind::WaveExit:
      return RecorderEntryKind::WaveExit;
    case TraceEventKind::Stall:
      return RecorderEntryKind::Stall;
    case TraceEventKind::Arrive:
      return RecorderEntryKind::Arrive;
    case TraceEventKind::Launch:
    case TraceEventKind::BlockPlaced:
    case TraceEventKind::BlockLaunch:
      break;
  }
  return RecorderEntryKind::InstructionIssue;
}

RecorderProgramEvent MakeRecorderProgramEvent(const TraceEvent& event, uint64_t sequence) {
  const CanonicalTraceEvent canonical = MakeCanonicalTraceEvent(event);
  const auto& view = canonical.view;
  return RecorderProgramEvent{
      .sequence = sequence,
      .kind = ProgramEventKindFromTraceEvent(event),
      .event = event,
      .slot_model_kind = view.slot_model_kind,
      .stall_reason = view.stall_reason,
      .barrier_kind = view.barrier_kind,
      .arrive_kind = view.arrive_kind,
      .lifecycle_stage = view.lifecycle_stage,
      .waitcnt_state = view.waitcnt_state,
      .canonical_name = view.canonical_name,
      .presentation_name = view.presentation_name,
      .display_name = view.display_name,
      .category = view.category,
      .compatibility_message = view.compatibility_message,
  };
}

RecorderEntry MakeRecorderEntry(const TraceEvent& event, uint64_t sequence) {
  const CanonicalTraceEvent canonical = MakeCanonicalTraceEvent(event);
  const auto& view = canonical.view;
  RecorderEntry entry{
      .sequence = sequence,
      .kind = EntryKindFromTraceEvent(event),
      .event = event,
      .slot_model_kind = view.slot_model_kind,
      .stall_reason = view.stall_reason,
      .barrier_kind = view.barrier_kind,
      .arrive_kind = view.arrive_kind,
      .lifecycle_stage = view.lifecycle_stage,
      .waitcnt_state = view.waitcnt_state,
      .canonical_name = view.canonical_name,
      .presentation_name = view.presentation_name,
      .display_name = view.display_name,
      .category = view.category,
      .compatibility_message = view.compatibility_message,
      .begin_cycle = event.cycle,
      .end_cycle = event.cycle,
      .has_cycle_range = false,
  };
  if (entry.kind == RecorderEntryKind::InstructionIssue) {
    entry.end_cycle = event.cycle + NormalizeInstructionRangeCycles(0);
    entry.has_cycle_range = true;
  }
  return entry;
}

}  // namespace

void Recorder::Record(const TraceEvent& event) {
  const uint64_t sequence = next_sequence_++;
  events_.push_back(event);
  if (!EventBelongsToWave(event)) {
    program_events_.push_back(MakeRecorderProgramEvent(event, sequence));
    return;
  }
  RecorderWave& wave = GetOrCreateWave(waves_, event);
  wave.entries.push_back(MakeRecorderEntry(event, sequence));
}

}  // namespace gpu_model
