#include "gpu_model/debug/cycle_timeline.h"
#include "gpu_model/debug/trace_event_export.h"
#include "gpu_model/debug/trace_event_builder.h"
#include "cycle_timeline_internal.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <optional>
#include <queue>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "gpu_model/execution/internal/tensor_op_utils.h"

namespace gpu_model {

namespace {

struct WaveKey {
  uint32_t dpc_id = 0;
  uint32_t ap_id = 0;
  uint32_t peu_id = 0;
  uint32_t block_id = 0;
  uint32_t wave_id = 0;

  bool operator<(const WaveKey& other) const {
    return std::tie(dpc_id, ap_id, peu_id, block_id, wave_id) <
           std::tie(other.dpc_id, other.ap_id, other.peu_id, other.block_id, other.wave_id);
  }
};

std::string ExtractOpName(const std::string& message) {
  const auto pos = message.find("op=");
  if (pos == std::string::npos) {
    return message;
  }
  const auto start = pos + 3;
  const auto end = message.find_first_of(" \n", start);
  return message.substr(start, end == std::string::npos ? std::string::npos : end - start);
}

uint64_t ComputeEndCycle(const std::vector<TraceEvent>& events) {
  uint64_t end = 0;
  for (const auto& event : events) {
    end = std::max(end, event.cycle);
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

TimelineData BuildTimelineData(const std::vector<TraceEvent>& events) {
  TimelineData data;
  struct OpenIssue {
    uint64_t issue_cycle = 0;
    std::string op;
    SlotKey slot;
    uint32_t block_id = 0;
    uint32_t wave_id = 0;
    uint64_t pc = 0;
  };
  std::map<WaveKey, std::queue<OpenIssue>> open_issue;

  for (const auto& event : events) {
    const TraceEventView view = MakeTraceEventView(event);
    const TraceEventExportFields fields = MakeTraceEventExportFields(view);
    const std::string_view slot_model = TraceSlotModelName(view.slot_model_kind);
    if (!slot_model.empty()) {
      data.slot_models.insert(std::string(slot_model));
    }
    const WaveKey wave_key{.dpc_id = event.dpc_id,
                           .ap_id = event.ap_id,
                           .peu_id = event.peu_id,
                           .block_id = event.block_id,
                           .wave_id = event.wave_id};
    const SlotKey slot_key{.dpc_id = event.dpc_id,
                           .ap_id = event.ap_id,
                           .peu_id = event.peu_id,
                           .slot_id = event.slot_id};
    if (event.kind == TraceEventKind::WaveStep) {
      const std::string op =
          view.display_name.empty() ? ExtractOpName(event.message) : view.display_name;
      open_issue[wave_key].push(OpenIssue{.issue_cycle = event.cycle,
                                          .op = op,
                                          .slot = slot_key,
                                          .block_id = event.block_id,
                                          .wave_id = event.wave_id,
                                          .pc = event.pc});
      AssignSymbol(op, data.symbols);
    } else if (event.kind == TraceEventKind::Commit) {
      auto& queue = open_issue[wave_key];
      if (!queue.empty()) {
        const OpenIssue issue = queue.front();
        queue.pop();
        const bool suppress_segment = issue.op == "s_waitcnt";
        if (!suppress_segment) {
          data.segments[issue.slot].push_back(Segment{.issue_cycle = issue.issue_cycle,
                                                      .commit_cycle = event.cycle,
                                                      .op = issue.op,
                                                      .slot_model = std::string(slot_model),
                                                      .block_id = issue.block_id,
                                                      .wave_id = issue.wave_id,
                                                      .pc = issue.pc});
        }
      }
    } else if (event.kind == TraceEventKind::Arrive || event.kind == TraceEventKind::Barrier ||
               event.kind == TraceEventKind::WaveExit || event.kind == TraceEventKind::Stall ||
               event.kind == TraceEventKind::WaveLaunch || event.kind == TraceEventKind::BlockLaunch) {
      char symbol = '.';
      if (event.kind == TraceEventKind::Arrive) {
        symbol = 'R';
      } else if (event.kind == TraceEventKind::Barrier) {
        symbol = view.barrier_kind == TraceBarrierKind::Release ? '|' : 'B';
      } else if (event.kind == TraceEventKind::WaveExit) {
        symbol = 'X';
      } else if (event.kind == TraceEventKind::Stall) {
        symbol = 'S';
      } else if (event.kind == TraceEventKind::WaveLaunch) {
        symbol = 'L';
      } else if (event.kind == TraceEventKind::BlockLaunch) {
        symbol = '#';
      }
      data.markers[slot_key].push_back(Marker{.cycle = event.cycle,
                                              .symbol = symbol,
                                              .kind = event.kind,
                                              .stall_reason = view.stall_reason,
                                              .barrier_kind = view.barrier_kind,
                                              .arrive_kind = view.arrive_kind,
                                              .lifecycle_stage = view.lifecycle_stage,
                                              .canonical_name = view.canonical_name,
                                              .presentation_name = view.presentation_name,
                                              .display_name = view.display_name,
                                              .category = view.category,
                                              .message = fields.compatibility_message,
                                              .slot_model = fields.slot_model,
                                              .stall_reason_name = fields.stall_reason,
                                              .barrier_kind_name = fields.barrier_kind,
                                              .arrive_kind_name = fields.arrive_kind,
                                              .lifecycle_stage_name = fields.lifecycle_stage,
                                              .block_id = event.block_id,
                                              .wave_id = event.wave_id});
    } else if (event.kind == TraceEventKind::Launch || event.kind == TraceEventKind::BlockPlaced) {
      data.runtime_events.push_back(event);
    }
  }

  return data;
}

}  // namespace

std::string CycleTimelineRenderer::RenderGoogleTrace(const std::vector<TraceEvent>& events,
                                                     CycleTimelineOptions options) {
  if (events.empty()) {
    return "{\"traceEvents\":[],\"metadata\":{\"time_unit\":\"cycle\",\"slot_models\":[]}}\n";
  }

  const TimelineData data = BuildTimelineData(events);
  const uint64_t begin = options.cycle_begin.value_or(0);
  const uint64_t end = options.cycle_end.value_or(ComputeEndCycle(events));
  return RenderGoogleTraceExport(data, begin, end, options.group_by);
}

std::string CycleTimelineRenderer::RenderPerfettoTraceProto(const std::vector<TraceEvent>& events,
                                                            CycleTimelineOptions options) {
  if (events.empty()) {
    return {};
  }

  const TimelineData data = BuildTimelineData(events);
  const uint64_t begin = options.cycle_begin.value_or(0);
  const uint64_t end = options.cycle_end.value_or(ComputeEndCycle(events));
  return RenderPerfettoTraceExport(data, begin, end);
}

}  // namespace gpu_model
