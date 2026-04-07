#include "gpu_model/debug/timeline/cycle_timeline.h"
#include "gpu_model/debug/trace/event_export.h"
#include "cycle_timeline_internal.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

#include "gpu_model/execution/internal/tensor_op_utils.h"

namespace gpu_model {

namespace {

constexpr uint64_t kTimelineInstructionRenderCycles = 4;

uint64_t QuantizeRenderDurationCycles(uint64_t cycles) {
  const uint64_t clamped = std::max<uint64_t>(kTimelineInstructionRenderCycles, cycles);
  const uint64_t remainder = clamped % kTimelineInstructionRenderCycles;
  if (remainder == 0) {
    return clamped;
  }
  return clamped + (kTimelineInstructionRenderCycles - remainder);
}

std::string ExtractOpName(const std::string& message) {
  const auto pos = message.find("op=");
  if (pos == std::string::npos) {
    return message;
  }
  const auto start = pos + 3;
  const auto end = message.find_first_of(" \n", start);
  return message.substr(start, end == std::string::npos ? std::string::npos : end - start);
}

uint64_t ComputeEndCycle(const Recorder& recorder) {
  uint64_t end = 0;
  for (const auto& event : recorder.events()) {
    end = std::max(end, event.cycle);
  }
  for (const auto& wave : recorder.waves()) {
    for (const auto& entry : wave.entries) {
      end = std::max(end, entry.event.cycle);
      if (entry.has_cycle_range) {
        end = std::max(end, entry.end_cycle);
      }
    }
  }
  return end;
}

char AssignSymbol(const std::string& op, std::unordered_map<std::string, char>& symbols) {
  const auto it = symbols.find(op);
  if (it != symbols.end()) {
    return it->second;
  }

  if (IsTensorMnemonic(op)) {
    symbols.emplace(op, 'T');
    return 'T';
  }

  static const std::string palette =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  const size_t index = symbols.size();
  const char symbol = index < palette.size() ? palette[index] : '?';
  symbols.emplace(op, symbol);
  return symbol;
}

TimelineSemanticEvent MakeTimelineSemanticEvent(const RecorderProgramEvent& event) {
  return TimelineSemanticEvent{
      .kind = event.event.kind,
      .cycle = event.event.cycle,
      .dpc_id = event.event.dpc_id,
      .ap_id = event.event.ap_id,
      .peu_id = event.event.peu_id,
      .slot_id = event.event.slot_id,
      .block_id = event.event.block_id,
      .wave_id = event.event.wave_id,
      .pc = event.event.pc,
      .slot_model_kind = event.slot_model_kind,
      .stall_reason = event.stall_reason,
      .barrier_kind = event.barrier_kind,
      .arrive_kind = event.arrive_kind,
      .arrive_progress = event.event.arrive_progress,
      .lifecycle_stage = event.lifecycle_stage,
      .waitcnt_state = event.waitcnt_state,
      .fields = MakeTraceEventExportFields(event),
  };
}

TimelineSemanticEvent MakeTimelineSemanticEvent(const RecorderEntry& event) {
  return TimelineSemanticEvent{
      .kind = event.event.kind,
      .cycle = event.event.cycle,
      .dpc_id = event.event.dpc_id,
      .ap_id = event.event.ap_id,
      .peu_id = event.event.peu_id,
      .slot_id = event.event.slot_id,
      .block_id = event.event.block_id,
      .wave_id = event.event.wave_id,
      .pc = event.event.pc,
      .slot_model_kind = event.slot_model_kind,
      .stall_reason = event.stall_reason,
      .barrier_kind = event.barrier_kind,
      .arrive_kind = event.arrive_kind,
      .arrive_progress = event.event.arrive_progress,
      .lifecycle_stage = event.lifecycle_stage,
      .waitcnt_state = event.waitcnt_state,
      .fields = MakeTraceEventExportFields(event),
  };
}

TimelineData BuildTimelineData(const Recorder& recorder) {
  TimelineData data;

  for (const auto& program_event : recorder.program_events()) {
    const TimelineSemanticEvent semantic = MakeTimelineSemanticEvent(program_event);
    const std::string_view slot_model = TraceSlotModelName(program_event.slot_model_kind);
    if (!slot_model.empty()) {
      data.slot_models.insert(std::string(slot_model));
    }

    if (program_event.kind == RecorderProgramEventKind::Launch ||
        program_event.kind == RecorderProgramEventKind::BlockPlaced ||
        program_event.kind == RecorderProgramEventKind::BlockAdmit ||
        program_event.kind == RecorderProgramEventKind::BlockLaunch ||
        program_event.kind == RecorderProgramEventKind::BlockActivate ||
        program_event.kind == RecorderProgramEventKind::BlockRetire) {
      data.runtime_events.push_back(semantic);
      continue;
    }

    continue;
  }

  for (const auto& wave : recorder.waves()) {
    std::queue<const RecorderEntry*> open_issue;
    for (const auto& entry : wave.entries) {
      const TraceEvent& event = entry.event;
      const TimelineSemanticEvent semantic = MakeTimelineSemanticEvent(entry);
      const std::string_view slot_model = TraceSlotModelName(entry.slot_model_kind);
      if (!slot_model.empty()) {
        data.slot_models.insert(std::string(slot_model));
      }

      const SlotKey slot_key{
          .dpc_id = wave.dpc_id,
          .ap_id = wave.ap_id,
          .peu_id = wave.peu_id,
          .slot_id = wave.slot_id,
      };

      if (entry.kind == RecorderEntryKind::InstructionIssue) {
        open_issue.push(&entry);
        const std::string op =
            entry.display_name.empty() ? ExtractOpName(entry.compatibility_message)
                                       : entry.display_name;
        AssignSymbol(op, data.symbols);
        continue;
      }

      if (entry.kind == RecorderEntryKind::Commit) {
        if (open_issue.empty()) {
          continue;
        }
        const RecorderEntry& issue = *open_issue.front();
        open_issue.pop();
        const std::string op = issue.display_name.empty()
                                   ? ExtractOpName(issue.compatibility_message)
                                   : issue.display_name;
        if (op == "s_waitcnt") {
          continue;
        }
        data.segments[slot_key].push_back(Segment{
            .issue_cycle = issue.begin_cycle,
            .commit_cycle = event.cycle,
            .render_duration_cycles =
                issue.has_cycle_range ? issue.end_cycle - issue.begin_cycle
                                      : QuantizeRenderDurationCycles(0),
            .op = op,
            .slot_model = std::string(slot_model),
            .block_id = wave.block_id,
            .wave_id = wave.wave_id,
            .pc = issue.event.pc,
        });
        continue;
      }

      if (entry.kind == RecorderEntryKind::Arrive || entry.kind == RecorderEntryKind::Barrier ||
          entry.kind == RecorderEntryKind::WaveExit || entry.kind == RecorderEntryKind::Stall ||
          entry.kind == RecorderEntryKind::WaveLaunch ||
          entry.kind == RecorderEntryKind::WaveGenerate ||
          entry.kind == RecorderEntryKind::WaveDispatch ||
          entry.kind == RecorderEntryKind::SlotBind ||
          entry.kind == RecorderEntryKind::ActivePromote ||
          entry.kind == RecorderEntryKind::IssueSelect ||
          entry.kind == RecorderEntryKind::WaveWait ||
          entry.kind == RecorderEntryKind::WaveArrive ||
          entry.kind == RecorderEntryKind::WaveResume ||
          entry.kind == RecorderEntryKind::WaveSwitchAway) {
        char symbol = '.';
        if (entry.kind == RecorderEntryKind::Arrive) {
          symbol = 'R';
        } else if (entry.kind == RecorderEntryKind::Barrier) {
          symbol = semantic.barrier_kind == TraceBarrierKind::Release ? '|' : 'B';
        } else if (entry.kind == RecorderEntryKind::WaveExit) {
          symbol = 'X';
        } else if (entry.kind == RecorderEntryKind::Stall) {
          symbol = 'S';
        } else if (entry.kind == RecorderEntryKind::WaveLaunch) {
          symbol = 'L';
        } else if (entry.kind == RecorderEntryKind::WaveGenerate) {
          symbol = 'G';
        } else if (entry.kind == RecorderEntryKind::WaveDispatch) {
          symbol = 'D';
        } else if (entry.kind == RecorderEntryKind::SlotBind) {
          symbol = 'P';
        } else if (entry.kind == RecorderEntryKind::ActivePromote) {
          symbol = 'A';
        } else if (entry.kind == RecorderEntryKind::IssueSelect) {
          symbol = 'I';
        } else if (entry.kind == RecorderEntryKind::WaveWait) {
          symbol = 'W';
        } else if (entry.kind == RecorderEntryKind::WaveArrive) {
          symbol = 'Y';
        } else if (entry.kind == RecorderEntryKind::WaveResume) {
          symbol = 'U';
        } else if (entry.kind == RecorderEntryKind::WaveSwitchAway) {
          symbol = 'Z';
        }
        data.markers[slot_key].push_back(Marker{
            .symbol = symbol,
            .semantic = semantic,
        });
      }
    }
  }

  return data;
}

}  // namespace

std::string CycleTimelineRenderer::RenderGoogleTrace(const Recorder& recorder,
                                                     CycleTimelineOptions options) {
  if (recorder.events().empty()) {
    return "{\"traceEvents\":[],\"metadata\":{\"time_unit\":\"cycle\",\"slot_models\":[]}}\n";
  }

  const TimelineData data = BuildTimelineData(recorder);
  const uint64_t begin = options.cycle_begin.value_or(0);
  const uint64_t end = options.cycle_end.value_or(ComputeEndCycle(recorder));
  return RenderGoogleTraceExport(data, begin, end, options.group_by);
}

std::string CycleTimelineRenderer::RenderPerfettoTraceProto(const Recorder& recorder,
                                                            CycleTimelineOptions options) {
  if (recorder.events().empty()) {
    return {};
  }

  const TimelineData data = BuildTimelineData(recorder);
  const uint64_t begin = options.cycle_begin.value_or(0);
  const uint64_t end = options.cycle_end.value_or(ComputeEndCycle(recorder));
  return RenderPerfettoTraceExport(data, begin, end);
}

}  // namespace gpu_model
