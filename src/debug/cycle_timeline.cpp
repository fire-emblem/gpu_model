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

namespace gpu_model {

namespace {

struct WaveKey {
  uint32_t block_id = 0;
  uint32_t wave_id = 0;

  bool operator<(const WaveKey& other) const {
    return std::tie(block_id, wave_id) < std::tie(other.block_id, other.wave_id);
  }
};

struct Segment {
  uint64_t issue_cycle = 0;
  uint64_t commit_cycle = 0;
  std::string op;
};

struct Marker {
  uint64_t cycle = 0;
  char symbol = '.';
  TraceEventKind kind = TraceEventKind::Launch;
  std::string message;
};

struct TimelineData {
  std::map<WaveKey, std::vector<Segment>> segments;
  std::map<WaveKey, std::vector<Marker>> markers;
  std::unordered_map<std::string, char> symbols;
  std::set<WaveKey> seen_waves;
  std::vector<TraceEvent> runtime_events;
};

std::string HexU64(uint64_t value) {
  std::ostringstream out;
  out << "0x" << std::hex << std::nouppercase << value;
  return out.str();
}

std::string WaveLabel(const WaveKey& key) {
  std::ostringstream out;
  out << "B" << key.block_id << "W" << key.wave_id;
  return out.str();
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
  std::map<WaveKey, std::queue<std::pair<uint64_t, std::string>>> open_issue;

  for (const auto& event : events) {
    const WaveKey key{.block_id = event.block_id, .wave_id = event.wave_id};
    if (event.kind == TraceEventKind::WaveStep) {
      const std::string op = ExtractOpName(event.message);
      open_issue[key].push({event.cycle, op});
      data.seen_waves.insert(key);
      AssignSymbol(op, data.symbols);
    } else if (event.kind == TraceEventKind::Commit) {
      auto& queue = open_issue[key];
      if (!queue.empty()) {
        const auto [issue_cycle, op] = queue.front();
        queue.pop();
        data.segments[key].push_back(
            Segment{.issue_cycle = issue_cycle, .commit_cycle = event.cycle, .op = op});
        data.seen_waves.insert(key);
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
      data.markers[key].push_back(
          Marker{.cycle = event.cycle, .symbol = symbol, .kind = event.kind, .message = event.message});
      data.seen_waves.insert(key);
    } else if (event.kind == TraceEventKind::Launch || event.kind == TraceEventKind::BlockPlaced) {
      data.runtime_events.push_back(event);
    }
  }

  return data;
}

std::string ThreadLabel(const WaveKey& key, CycleTimelineGroupBy group_by) {
  return group_by == CycleTimelineGroupBy::Block ? BlockLabel(key.block_id) : WaveLabel(key);
}

uint32_t TracePid(const WaveKey& key, CycleTimelineGroupBy group_by) {
  return group_by == CycleTimelineGroupBy::Block ? 1u : key.block_id + 1;
}

uint32_t TraceTid(const WaveKey& key, CycleTimelineGroupBy group_by) {
  return group_by == CycleTimelineGroupBy::Block ? key.block_id : key.wave_id;
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
      return marker.message.empty() ? "stall" : "stall_" + marker.message;
    case TraceEventKind::WaveLaunch:
      return "wave_launch";
    case TraceEventKind::BlockLaunch:
      return "block_launch";
    default:
      return marker.message.empty() ? "event" : marker.message;
  }
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

  std::map<std::string, std::vector<Segment>> grouped_segments;
  std::map<std::string, std::vector<Marker>> grouped_markers;
  for (const auto& key : data.seen_waves) {
    const std::string label = ThreadLabel(key, options.group_by);
    const auto segment_it = data.segments.find(key);
    if (segment_it != data.segments.end()) {
      grouped_segments[label].insert(grouped_segments[label].end(), segment_it->second.begin(),
                                     segment_it->second.end());
    }
    const auto marker_it = data.markers.find(key);
    if (marker_it != data.markers.end()) {
      grouped_markers[label].insert(grouped_markers[label].end(), marker_it->second.begin(),
                                    marker_it->second.end());
    }
  }

  for (const auto& [label, row_segments] : grouped_segments) {
    std::string row(width, '.');
    for (const auto& segment : row_segments) {
      const char symbol = data.symbols.at(segment.op);
      for (uint32_t col = 0; col < width; ++col) {
        const uint64_t col_begin = begin + static_cast<uint64_t>(col) * cycles_per_column;
        const uint64_t col_end = col_begin + cycles_per_column;
        if (segment.issue_cycle < col_end && segment.commit_cycle > col_begin) {
          row[col] = symbol;
        }
      }
    }
    for (const auto& marker : grouped_markers[label]) {
      if (marker.cycle < begin || marker.cycle > end) {
        continue;
      }
      const uint32_t col = static_cast<uint32_t>((marker.cycle - begin) / cycles_per_column);
      if (col < row.size()) {
        row[col] = marker.symbol;
      }
    }
    out << std::left << std::setw(8) << label << ' ' << row << '\n';
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

  std::set<std::pair<uint32_t, uint32_t>> declared_rows;
  for (const auto& key : data.seen_waves) {
    const uint32_t pid = TracePid(key, options.group_by);
    const uint32_t tid = TraceTid(key, options.group_by);
    if (!declared_rows.insert({pid, tid}).second) {
      continue;
    }
    const std::string process_name =
        options.group_by == CycleTimelineGroupBy::Block ? "Blocks" : BlockLabel(key.block_id);
    append("{\"name\":\"process_name\",\"ph\":\"M\",\"pid\":" + std::to_string(pid) +
           ",\"tid\":0,\"args\":{\"name\":\"" + EscapeJson(process_name) + "\"}}");
    append("{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":" + std::to_string(pid) + ",\"tid\":" +
           std::to_string(tid) + ",\"args\":{\"name\":\"" +
           EscapeJson(ThreadLabel(key, options.group_by)) + "\"}}");
  }

  for (const auto& [key, row_segments] : data.segments) {
    const uint32_t pid = TracePid(key, options.group_by);
    const uint32_t tid = TraceTid(key, options.group_by);
    for (const auto& segment : row_segments) {
      if (segment.commit_cycle < begin || segment.issue_cycle > end) {
        continue;
      }
      const uint64_t clipped_begin = std::max(begin, segment.issue_cycle);
      const uint64_t clipped_end = std::min(end, segment.commit_cycle);
      const uint64_t duration = clipped_end > clipped_begin ? clipped_end - clipped_begin : 1;
      append("{\"name\":\"" + EscapeJson(segment.op) +
             "\",\"cat\":\"instruction\",\"ph\":\"X\",\"pid\":" + std::to_string(pid) +
             ",\"tid\":" + std::to_string(tid) + ",\"ts\":" + std::to_string(clipped_begin) +
             ",\"dur\":" + std::to_string(duration) + ",\"args\":{\"block_id\":\"" +
             HexU64(key.block_id) + "\",\"wave_id\":\"" + HexU64(key.wave_id) +
             "\",\"issue_cycle\":\"" + HexU64(segment.issue_cycle) + "\",\"commit_cycle\":\"" +
             HexU64(segment.commit_cycle) + "\"}}");
    }
  }

  for (const auto& [key, row_markers] : data.markers) {
    const uint32_t pid = TracePid(key, options.group_by);
    const uint32_t tid = TraceTid(key, options.group_by);
    for (const auto& marker : row_markers) {
      if (marker.cycle < begin || marker.cycle > end) {
        continue;
      }
      append("{\"name\":\"" + EscapeJson(MarkerName(marker)) +
             "\",\"cat\":\"marker\",\"ph\":\"i\",\"s\":\"t\",\"pid\":" + std::to_string(pid) +
             ",\"tid\":" + std::to_string(tid) + ",\"ts\":" + std::to_string(marker.cycle) +
             ",\"args\":{\"message\":\"" + EscapeJson(marker.message) + "\"}}");
    }
  }

  out << "],\"metadata\":{\"time_unit\":\"cycle\"}}\n";
  return out.str();
}

}  // namespace gpu_model
