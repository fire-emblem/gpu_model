#include "gpu_model/debug/cycle_timeline.h"

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

struct SlotKey {
  uint32_t dpc_id = 0;
  uint32_t ap_id = 0;
  uint32_t peu_id = 0;
  uint32_t slot_id = 0;

  bool operator<(const SlotKey& other) const {
    return std::tie(dpc_id, ap_id, peu_id, slot_id) <
           std::tie(other.dpc_id, other.ap_id, other.peu_id, other.slot_id);
  }
};

struct RowDescriptor {
  uint32_t pid = 0;
  uint32_t tid = 0;
  int32_t process_sort_index = 0;
  int32_t thread_sort_index = 0;
  std::string process_name;
  std::string thread_name;

  bool operator<(const RowDescriptor& other) const {
    return std::tie(process_sort_index, pid, thread_sort_index, tid, process_name, thread_name) <
           std::tie(other.process_sort_index, other.pid, other.thread_sort_index, other.tid,
                    other.process_name, other.thread_name);
  }
};

struct Segment {
  uint64_t issue_cycle = 0;
  uint64_t commit_cycle = 0;
  std::string op;
  uint32_t block_id = 0;
  uint32_t wave_id = 0;
  uint64_t pc = 0;
};

struct Marker {
  uint64_t cycle = 0;
  char symbol = '.';
  TraceEventKind kind = TraceEventKind::Launch;
  std::string message;
  uint32_t block_id = 0;
  uint32_t wave_id = 0;
};

struct TimelineData {
  std::map<SlotKey, std::vector<Segment>> segments;
  std::map<SlotKey, std::vector<Marker>> markers;
  std::unordered_map<std::string, char> symbols;
  std::vector<TraceEvent> runtime_events;
};

std::string HexU64(uint64_t value) {
  std::ostringstream out;
  out << "0x" << std::hex << std::nouppercase << value;
  return out.str();
}

std::string_view StallReasonFromMessage(std::string_view message) {
  constexpr std::string_view kPrefix = "reason=";
  if (!message.starts_with(kPrefix)) {
    return {};
  }
  std::string_view rest = message.substr(kPrefix.size());
  const size_t start = rest.find_first_not_of(" \t\n");
  if (start == std::string_view::npos) {
    return {};
  }
  rest.remove_prefix(start);
  const size_t end = rest.find_first_of(" \t\n");
  if (end == std::string_view::npos) {
    return rest;
  }
  return rest.substr(0, end);
}

std::string StallLabel(std::string_view message) {
  const std::string_view reason = StallReasonFromMessage(message);
  if (!reason.empty()) {
    return std::string(reason);
  }
  return std::string(message);
}

std::string SlotLabel(const SlotKey& key) {
  return "S" + std::to_string(key.slot_id);
}

std::string PeuLabel(const SlotKey& key) {
  return "P" + std::to_string(key.peu_id);
}

std::string ApLabel(const SlotKey& key) {
  return "A" + std::to_string(key.ap_id);
}

std::string DpcLabel(const SlotKey& key) {
  return "D" + std::to_string(key.dpc_id);
}

std::string ProcessName(const SlotKey& key, CycleTimelineGroupBy group_by) {
  switch (group_by) {
    case CycleTimelineGroupBy::Wave:
      return "Device/" + DpcLabel(key) + "/" + ApLabel(key) + "/" + PeuLabel(key);
    case CycleTimelineGroupBy::Block:
      return "Device/Blocks";
    case CycleTimelineGroupBy::Peu:
      return "Device/" + DpcLabel(key) + "/" + ApLabel(key);
    case CycleTimelineGroupBy::Ap:
      return "Device/" + DpcLabel(key);
    case CycleTimelineGroupBy::Dpc:
      return "Device";
  }
  return "Device";
}

std::string BlockLabel(uint32_t block_id) {
  return "B" + std::to_string(block_id);
}

std::string RuntimeLabel() {
  return "Runtime";
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

std::string EscapeJson(std::string_view text) {
  std::string escaped;
  escaped.reserve(text.size());
  for (const char ch : text) {
    switch (ch) {
      case '\\':
        escaped += "\\\\";
        break;
      case '"':
        escaped += "\\\"";
        break;
      case '\n':
        escaped += "\\n";
        break;
      default:
        escaped.push_back(ch);
        break;
    }
  }
  return escaped;
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
      const std::string op = ExtractOpName(event.message);
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
        data.segments[issue.slot].push_back(Segment{.issue_cycle = issue.issue_cycle,
                                                    .commit_cycle = event.cycle,
                                                    .op = issue.op,
                                                    .block_id = issue.block_id,
                                                    .wave_id = issue.wave_id,
                                                    .pc = issue.pc});
      }
    } else if (event.kind == TraceEventKind::Arrive || event.kind == TraceEventKind::Barrier ||
               event.kind == TraceEventKind::WaveExit || event.kind == TraceEventKind::Stall ||
               event.kind == TraceEventKind::WaveLaunch || event.kind == TraceEventKind::BlockLaunch) {
      char symbol = '.';
      if (event.kind == TraceEventKind::Arrive) {
        symbol = 'R';
      } else if (event.kind == TraceEventKind::Barrier) {
        symbol = event.message.find("release") != std::string::npos ? '|' : 'B';
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
                                              .message = event.message,
                                              .block_id = event.block_id,
                                              .wave_id = event.wave_id});
    } else if (event.kind == TraceEventKind::Launch || event.kind == TraceEventKind::BlockPlaced) {
      data.runtime_events.push_back(event);
    }
  }

  return data;
}

std::string ThreadLabel(const SlotKey& key,
                        CycleTimelineGroupBy group_by,
                        std::optional<uint32_t> block_id = std::nullopt) {
  switch (group_by) {
    case CycleTimelineGroupBy::Wave:
      return SlotLabel(key);
    case CycleTimelineGroupBy::Block:
      return BlockLabel(block_id.value_or(0));
    case CycleTimelineGroupBy::Peu:
      return PeuLabel(key);
    case CycleTimelineGroupBy::Ap:
      return ApLabel(key);
    case CycleTimelineGroupBy::Dpc:
      return DpcLabel(key);
  }
  return SlotLabel(key);
}

uint32_t TracePid(const SlotKey& key, CycleTimelineGroupBy group_by) {
  switch (group_by) {
    case CycleTimelineGroupBy::Wave:
      return 1u + (key.dpc_id << 12) + (key.ap_id << 4) + key.peu_id;
    case CycleTimelineGroupBy::Block:
      return 1u;
    case CycleTimelineGroupBy::Peu:
      return 1u + (key.dpc_id << 8) + key.ap_id;
    case CycleTimelineGroupBy::Ap:
      return 1u + key.dpc_id;
    case CycleTimelineGroupBy::Dpc:
      return 1u;
  }
  return 1u;
}

uint32_t TraceTid(const SlotKey& key,
                  CycleTimelineGroupBy group_by,
                  std::optional<uint32_t> block_id = std::nullopt) {
  switch (group_by) {
    case CycleTimelineGroupBy::Wave:
      return key.slot_id;
    case CycleTimelineGroupBy::Block:
      return block_id.value_or(0);
    case CycleTimelineGroupBy::Peu:
      return key.peu_id;
    case CycleTimelineGroupBy::Ap:
      return key.ap_id;
    case CycleTimelineGroupBy::Dpc:
      return key.dpc_id;
  }
  return key.slot_id;
}

int32_t ProcessSortIndex(const SlotKey& key, CycleTimelineGroupBy group_by) {
  switch (group_by) {
    case CycleTimelineGroupBy::Wave:
      return static_cast<int32_t>((key.dpc_id << 8) + (key.ap_id << 4) + key.peu_id);
    case CycleTimelineGroupBy::Block:
      return 0;
    case CycleTimelineGroupBy::Peu:
      return static_cast<int32_t>((key.dpc_id << 8) + key.ap_id);
    case CycleTimelineGroupBy::Ap:
      return static_cast<int32_t>(key.dpc_id);
    case CycleTimelineGroupBy::Dpc:
      return 0;
  }
  return 0;
}

int32_t ThreadSortIndex(const SlotKey& key,
                        CycleTimelineGroupBy group_by,
                        std::optional<uint32_t> block_id = std::nullopt) {
  switch (group_by) {
    case CycleTimelineGroupBy::Wave:
      return static_cast<int32_t>(key.slot_id);
    case CycleTimelineGroupBy::Block:
      return static_cast<int32_t>(block_id.value_or(0));
    case CycleTimelineGroupBy::Peu:
      return static_cast<int32_t>(key.peu_id);
    case CycleTimelineGroupBy::Ap:
      return static_cast<int32_t>(key.ap_id);
    case CycleTimelineGroupBy::Dpc:
      return static_cast<int32_t>(key.dpc_id);
  }
  return 0;
}

RowDescriptor DescribeRow(const SlotKey& key,
                          CycleTimelineGroupBy group_by,
                          std::optional<uint32_t> block_id = std::nullopt) {
  return RowDescriptor{.pid = TracePid(key, group_by),
                       .tid = TraceTid(key, group_by, block_id),
                       .process_sort_index = ProcessSortIndex(key, group_by),
                       .thread_sort_index = ThreadSortIndex(key, group_by, block_id),
                       .process_name = ProcessName(key, group_by),
                       .thread_name = ThreadLabel(key, group_by, block_id)};
}

std::string MarkerName(const Marker& marker) {
  switch (marker.kind) {
    case TraceEventKind::Arrive:
      return marker.message.empty() ? "arrive" : marker.message;
    case TraceEventKind::Barrier:
      return marker.message.empty() ? "barrier" : "barrier_" + marker.message;
    case TraceEventKind::WaveExit:
      return "wave_exit";
    case TraceEventKind::Stall:
      if (marker.message.empty()) {
        return "stall";
      }
      return "stall_" + StallLabel(marker.message);
    case TraceEventKind::WaveLaunch:
      return "wave_launch";
    case TraceEventKind::BlockLaunch:
      return "block_launch";
    default:
      return marker.message.empty() ? "event" : marker.message;
  }
}

std::string MarkerCategory(const Marker& marker) {
  switch (marker.kind) {
    case TraceEventKind::Arrive:
      return marker.message.find("load") != std::string::npos ? "memory/arrive_load"
                                                              : "memory/arrive_store";
    case TraceEventKind::Barrier:
      return "sync/barrier";
    case TraceEventKind::WaveExit:
      return "control/exit";
    case TraceEventKind::Stall:
      if (marker.message.empty()) {
        return "stall";
      }
      return "stall/" + StallLabel(marker.message);
    case TraceEventKind::WaveLaunch:
      return "launch/wave";
    case TraceEventKind::BlockLaunch:
      return "launch/block";
    default:
      return "marker";
  }
}

std::string SegmentArgs(const SlotKey& key, const Segment& segment) {
  std::ostringstream out;
  out << "\"dpc\":" << key.dpc_id << ",\"ap\":" << key.ap_id << ",\"peu\":" << key.peu_id
      << ",\"slot\":" << key.slot_id << ",\"block\":" << segment.block_id << ",\"wave\":"
      << segment.wave_id;
  if (segment.pc != 0) {
    out << ",\"pc\":\"" << EscapeJson(HexU64(segment.pc)) << "\"";
  }
  out << ",\"issue_cycle\":" << segment.issue_cycle << ",\"commit_cycle\":"
      << segment.commit_cycle;
  return out.str();
}

std::string MarkerArgs(const SlotKey& key, const Marker& marker) {
  std::ostringstream out;
  out << "\"dpc\":" << key.dpc_id << ",\"ap\":" << key.ap_id << ",\"peu\":" << key.peu_id
      << ",\"slot\":" << key.slot_id << ",\"block\":" << marker.block_id << ",\"wave\":"
      << marker.wave_id << ",\"cycle\":" << marker.cycle << ",\"message\":\""
      << EscapeJson(marker.message) << "\"";
  return out.str();
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
    return "{\"traceEvents\":[],\"metadata\":{\"time_unit\":\"cycle\"}}\n";
  }

  const uint64_t begin = options.cycle_begin.value_or(0);
  const uint64_t end = options.cycle_end.value_or(ComputeEndCycle(events));
  if (end < begin) {
    return "{\"traceEvents\":[],\"metadata\":{\"time_unit\":\"cycle\",\"error\":\"invalid_range\"}}\n";
  }

  const TimelineData data = BuildTimelineData(events);
  std::ostringstream out;
  out << "{\"traceEvents\":[";

  bool first = true;
  auto append = [&](const std::string& text) {
    if (!first) {
      out << ',';
    }
    first = false;
    out << text;
  };

  append("{\"name\":\"process_name\",\"ph\":\"M\",\"pid\":0,\"tid\":0,\"args\":{\"name\":\"" +
         EscapeJson(RuntimeLabel()) + "\"}}");
  for (const auto& runtime_event : data.runtime_events) {
    if (runtime_event.cycle < begin || runtime_event.cycle > end) {
      continue;
    }
    const std::string name =
        runtime_event.kind == TraceEventKind::Launch ? "launch" : "block_placed";
    append("{\"name\":\"" + EscapeJson(name) +
           "\",\"cat\":\"runtime\",\"ph\":\"i\",\"s\":\"g\",\"pid\":0,\"tid\":0,\"ts\":" +
           std::to_string(runtime_event.cycle) + ",\"args\":{\"message\":\"" +
           EscapeJson(runtime_event.message) + "\"}}");
  }

  std::set<RowDescriptor> declared_rows;
  for (const auto& [key, row_segments] : data.segments) {
    for (const auto& segment : row_segments) {
      declared_rows.insert(DescribeRow(key, options.group_by, segment.block_id));
    }
  }
  for (const auto& [key, row_markers] : data.markers) {
    for (const auto& marker : row_markers) {
      declared_rows.insert(DescribeRow(key, options.group_by, marker.block_id));
    }
  }

  std::set<uint32_t> declared_processes;
  for (const auto& row : declared_rows) {
    if (declared_processes.insert(row.pid).second) {
      append("{\"name\":\"process_name\",\"ph\":\"M\",\"pid\":" + std::to_string(row.pid) +
             ",\"tid\":0,\"args\":{\"name\":\"" + EscapeJson(row.process_name) + "\"}}");
      append("{\"name\":\"process_sort_index\",\"ph\":\"M\",\"pid\":" + std::to_string(row.pid) +
             ",\"tid\":0,\"args\":{\"sort_index\":" +
             std::to_string(row.process_sort_index) + "}}");
    }
    append("{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":" + std::to_string(row.pid) +
           ",\"tid\":" + std::to_string(row.tid) + ",\"args\":{\"name\":\"" +
           EscapeJson(row.thread_name) + "\"}}");
    append("{\"name\":\"thread_sort_index\",\"ph\":\"M\",\"pid\":" + std::to_string(row.pid) +
           ",\"tid\":" + std::to_string(row.tid) + ",\"args\":{\"sort_index\":" +
           std::to_string(row.thread_sort_index) + "}}");
  }

  for (const auto& [key, row_segments] : data.segments) {
    for (const auto& segment : row_segments) {
      const RowDescriptor row = DescribeRow(key, options.group_by, segment.block_id);
      if (segment.commit_cycle < begin || segment.issue_cycle > end) {
        continue;
      }
      const uint64_t clipped_begin = std::max(begin, segment.issue_cycle);
      const uint64_t clipped_end = std::min(end, segment.commit_cycle);
      const uint64_t duration = clipped_end > clipped_begin ? clipped_end - clipped_begin : 1;
      const std::string category = IsTensorMnemonic(segment.op) ? "tensor" : "instruction";
      append("{\"name\":\"" + EscapeJson(segment.op) +
             "\",\"cat\":\"" + category + "\",\"ph\":\"X\",\"pid\":" +
             std::to_string(row.pid) + ",\"tid\":" + std::to_string(row.tid) + ",\"ts\":" +
             std::to_string(clipped_begin) + ",\"dur\":" + std::to_string(duration) +
             ",\"args\":{" + SegmentArgs(key, segment) + "}}");
    }
  }

  for (const auto& [key, row_markers] : data.markers) {
    for (const auto& marker : row_markers) {
      const RowDescriptor row = DescribeRow(key, options.group_by, marker.block_id);
      if (marker.cycle < begin || marker.cycle > end) {
        continue;
      }
      append("{\"name\":\"" + EscapeJson(MarkerName(marker)) +
             "\",\"cat\":\"" + EscapeJson(MarkerCategory(marker)) +
             "\",\"ph\":\"i\",\"s\":\"t\",\"pid\":" + std::to_string(row.pid) +
             ",\"tid\":" + std::to_string(row.tid) + ",\"ts\":" +
             std::to_string(marker.cycle) + ",\"args\":{" + MarkerArgs(key, marker) + "}}");
    }
  }

  out << "],\"metadata\":{\"time_unit\":\"cycle\"}}\n";
  return out.str();
}

}  // namespace gpu_model
