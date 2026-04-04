#include "gpu_model/debug/cycle_timeline.h"
#include "gpu_model/debug/trace_event_export.h"
#include "gpu_model/debug/trace_event_builder.h"
#include "gpu_model/debug/trace_json_fields.h"
#include "gpu_model/debug/trace_event_view.h"
#include "cycle_timeline_internal.h"
#include "trace_perfetto_proto.h"

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <map>
#include <optional>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
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

std::string HexU64(uint64_t value) {
  std::ostringstream out;
  out << "0x" << std::hex << std::nouppercase << value;
  return out.str();
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

uint64_t Fnv1a64(std::string_view text) {
  uint64_t hash = 1469598103934665603ull;
  for (const unsigned char ch : text) {
    hash ^= static_cast<uint64_t>(ch);
    hash *= 1099511628211ull;
  }
  return hash;
}

uint64_t MakeTrackUuid(std::string_view kind, std::string_view label) {
  const uint64_t hash = Fnv1a64(std::string(kind) + ":" + std::string(label));
  return hash == 0 ? 1 : hash;
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

std::vector<TrackDescriptorNode> BuildPerfettoTrackTree(const TimelineData& data) {
  std::set<TrackDescriptorNode> nodes;

  const uint64_t device_uuid = MakeTrackUuid("track", "Device");
  nodes.insert(TrackDescriptorNode{.uuid = device_uuid,
                                   .parent_uuid = std::nullopt,
                                   .name = "Device",
                                   .sibling_order_rank = std::nullopt,
                                   .child_ordering = 3u});

  const uint64_t runtime_uuid = MakeTrackUuid("track", "Runtime");
  nodes.insert(TrackDescriptorNode{.uuid = runtime_uuid,
                                   .parent_uuid = std::nullopt,
                                   .name = "Runtime",
                                   .sibling_order_rank = std::nullopt,
                                   .child_ordering = std::nullopt});

  auto append_slot_path = [&](const SlotKey& key) {
    const std::string dpc_name = DpcLabel(key);
    const uint64_t dpc_uuid = MakeTrackUuid("track", dpc_name);
    nodes.insert(TrackDescriptorNode{.uuid = dpc_uuid,
                                     .parent_uuid = device_uuid,
                                     .name = dpc_name,
                                     .sibling_order_rank = static_cast<int32_t>(key.dpc_id),
                                     .child_ordering = 3u});

    const std::string ap_path = dpc_name + "/" + ApLabel(key);
    const uint64_t ap_uuid = MakeTrackUuid("track", ap_path);
    nodes.insert(TrackDescriptorNode{.uuid = ap_uuid,
                                     .parent_uuid = dpc_uuid,
                                     .name = ApLabel(key),
                                     .sibling_order_rank = static_cast<int32_t>(key.ap_id),
                                     .child_ordering = 3u});

    const std::string peu_path = ap_path + "/" + PeuLabel(key);
    const uint64_t peu_uuid = MakeTrackUuid("track", peu_path);
    nodes.insert(TrackDescriptorNode{.uuid = peu_uuid,
                                     .parent_uuid = ap_uuid,
                                     .name = PeuLabel(key),
                                     .sibling_order_rank = static_cast<int32_t>(key.peu_id),
                                     .child_ordering = 3u});

    const std::string slot_path = peu_path + "/" + SlotLabel(key);
    const uint64_t slot_uuid = MakeTrackUuid("track", slot_path);
    nodes.insert(TrackDescriptorNode{.uuid = slot_uuid,
                                     .parent_uuid = peu_uuid,
                                     .name = SlotLabel(key),
                                     .sibling_order_rank = static_cast<int32_t>(key.slot_id),
                                     .child_ordering = std::nullopt});
  };

  for (const auto& [key, row_segments] : data.segments) {
    if (!row_segments.empty()) {
      append_slot_path(key);
    }
  }
  for (const auto& [key, row_markers] : data.markers) {
    if (!row_markers.empty()) {
      append_slot_path(key);
    }
  }
  return {nodes.begin(), nodes.end()};
}

uint64_t SlotTrackUuid(const SlotKey& key) {
  const std::string path = DpcLabel(key) + "/" + ApLabel(key) + "/" + PeuLabel(key) + "/" +
                           SlotLabel(key);
  return MakeTrackUuid("track", path);
}

}  // namespace

std::string CycleTimelineRenderer::RenderAscii(const std::vector<TraceEvent>& events,
                                               CycleTimelineOptions options) {
  if (events.empty()) {
    return "cycle_timeline: <no events>\n";
  }

  const uint64_t begin = options.cycle_begin.value_or(0);
  const uint64_t end = options.cycle_end.value_or(ComputeEndCycle(events));
  if (end < begin) {
    return "cycle_timeline: <invalid range>\n";
  }

  const uint64_t total_cycles = end - begin + 1;
  const uint32_t max_columns = std::max<uint32_t>(1, options.max_columns);
  const uint64_t cycles_per_column =
      std::max<uint64_t>(1, (total_cycles + max_columns - 1) / max_columns);
  const uint32_t width =
      static_cast<uint32_t>((total_cycles + cycles_per_column - 1) / cycles_per_column);
  const TimelineData data = BuildTimelineData(events);

  std::ostringstream out;
  out << "cycle_timeline scale=" << cycles_per_column << " cycle(s)/col range=["
      << HexU64(begin) << ", " << HexU64(end) << "]\n";
  out << "legend:";
  if (std::any_of(data.symbols.begin(), data.symbols.end(),
                  [](const auto& entry) { return entry.second == 'T'; })) {
    out << " T=tensor-op";
  }
  for (const auto& [op, symbol] : data.symbols) {
    out << ' ' << symbol << '=' << op;
  }
  out << " R=arrive B=barrier-arrive |=barrier-release X=exit S=stall L=wave-launch #=block-launch\n";
  out << "cycles ";
  for (uint32_t col = 0; col < width; ++col) {
    const uint64_t cycle = begin + static_cast<uint64_t>(col) * cycles_per_column;
    if (col % 8 == 0) {
      out << std::setw(8) << std::setfill(' ') << HexU64(cycle);
    }
  }
  out << '\n';

  struct AsciiRow {
    std::vector<Segment> segments;
    std::vector<Marker> markers;
  };
  std::map<RowDescriptor, AsciiRow> grouped_rows;
  for (const auto& [key, row_segments] : data.segments) {
    for (const auto& segment : row_segments) {
      grouped_rows[DescribeRow(key, options.group_by, segment.block_id)].segments.push_back(segment);
    }
  }
  for (const auto& [key, row_markers] : data.markers) {
    for (const auto& marker : row_markers) {
      grouped_rows[DescribeRow(key, options.group_by, marker.block_id)].markers.push_back(marker);
    }
  }

  for (const auto& [row_info, contents] : grouped_rows) {
    std::string line(width, '.');
    for (const auto& segment : contents.segments) {
      const char symbol = data.symbols.at(segment.op);
      for (uint32_t col = 0; col < width; ++col) {
        const uint64_t col_begin = begin + static_cast<uint64_t>(col) * cycles_per_column;
        const uint64_t col_end = col_begin + cycles_per_column;
        if (segment.issue_cycle < col_end && segment.commit_cycle > col_begin) {
          line[col] = symbol;
        }
      }
    }
    for (const auto& marker : contents.markers) {
      if (marker.cycle < begin || marker.cycle > end) {
        continue;
      }
      const uint32_t col = static_cast<uint32_t>((marker.cycle - begin) / cycles_per_column);
      if (col < line.size()) {
        line[col] = marker.symbol;
      }
    }
    out << std::left << std::setw(8) << row_info.thread_name << ' ' << line << '\n';
  }

  return out.str();
}

std::string CycleTimelineRenderer::RenderGoogleTrace(const std::vector<TraceEvent>& events,
                                                     CycleTimelineOptions options) {
  if (events.empty()) {
    return "{\"traceEvents\":[],\"metadata\":{\"time_unit\":\"cycle\",\"slot_models\":[]}}\n";
  }

  const TimelineData data = BuildTimelineData(events);
  const uint64_t begin = options.cycle_begin.value_or(0);
  const uint64_t end = options.cycle_end.value_or(ComputeEndCycle(events));
  if (end < begin) {
    return RenderGoogleTraceExport(data, begin, end, options.group_by);
  }
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
  if (end < begin) {
    return {};
  }

  std::string trace;
  const std::vector<TrackDescriptorNode> tracks = BuildPerfettoTrackTree(data);
  for (const auto& track : tracks) {
    AppendTracePacket(EncodeTrackDescriptorPacket(track), trace);
  }

  const uint64_t runtime_track_uuid = MakeTrackUuid("track", "Runtime");
  for (const auto& runtime_event : data.runtime_events) {
    if (runtime_event.cycle < begin || runtime_event.cycle > end) {
      continue;
    }
    const TraceEventView view = MakeTraceEventView(runtime_event);
    AppendTracePacket(
        EncodeTrackEventPacket(runtime_event.cycle, runtime_track_uuid, 3u, view.presentation_name),
        trace);
  }

  for (const auto& [key, row_segments] : data.segments) {
    const uint64_t track_uuid = SlotTrackUuid(key);
    for (const auto& segment : row_segments) {
      if (segment.commit_cycle < begin || segment.issue_cycle > end) {
        continue;
      }
      const uint64_t clipped_begin = std::max(begin, segment.issue_cycle);
      const uint64_t clipped_end = std::min(end, segment.commit_cycle);
      AppendTracePacket(
          EncodeTrackEventPacket(clipped_begin, track_uuid, 1u, segment.op), trace);
      AppendTracePacket(EncodeTrackEventPacket(
                            clipped_end > clipped_begin ? clipped_end : clipped_begin + 1,
                            track_uuid,
                            2u,
                            std::nullopt),
                        trace);
    }
  }

  for (const auto& [key, row_markers] : data.markers) {
    const uint64_t track_uuid = SlotTrackUuid(key);
    for (const auto& marker : row_markers) {
      if (marker.cycle < begin || marker.cycle > end) {
        continue;
      }
      AppendTracePacket(
          EncodeTrackEventPacket(marker.cycle, track_uuid, 3u, MarkerEventName(marker)), trace);
    }
  }

  return trace;
}

}  // namespace gpu_model
