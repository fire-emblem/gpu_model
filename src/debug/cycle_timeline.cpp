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

  std::map<WaveKey, std::queue<std::pair<uint64_t, std::string>>> open_issue;
  std::map<WaveKey, std::vector<Segment>> segments;
  std::map<WaveKey, std::vector<Marker>> markers;
  std::unordered_map<std::string, char> symbols;
  std::set<WaveKey> seen_waves;

  for (const auto& event : events) {
    const WaveKey key{.block_id = event.block_id, .wave_id = event.wave_id};
    if (event.kind == TraceEventKind::WaveStep) {
      const std::string op = ExtractOpName(event.message);
      open_issue[key].push({event.cycle, op});
      seen_waves.insert(key);
      AssignSymbol(op, symbols);
    } else if (event.kind == TraceEventKind::Commit) {
      auto& queue = open_issue[key];
      if (!queue.empty()) {
        const auto [issue_cycle, op] = queue.front();
        queue.pop();
        segments[key].push_back(
            Segment{.issue_cycle = issue_cycle, .commit_cycle = event.cycle, .op = op});
        seen_waves.insert(key);
      }
    } else if (event.kind == TraceEventKind::Arrive) {
      markers[key].push_back(Marker{.cycle = event.cycle, .symbol = 'R'});
      seen_waves.insert(key);
    } else if (event.kind == TraceEventKind::Barrier) {
      const char symbol = event.message.find("release") != std::string::npos ? '|' : 'B';
      markers[key].push_back(Marker{.cycle = event.cycle, .symbol = symbol});
      seen_waves.insert(key);
    } else if (event.kind == TraceEventKind::WaveExit) {
      markers[key].push_back(Marker{.cycle = event.cycle, .symbol = 'X'});
      seen_waves.insert(key);
    }
  }

  std::ostringstream out;
  out << "cycle_timeline scale=" << cycles_per_column << " cycle(s)/col range=["
      << HexU64(begin) << ", " << HexU64(end) << "]\n";
  out << "legend:";
  for (const auto& [op, symbol] : symbols) {
    out << ' ' << symbol << '=' << op;
  }
  out << " R=arrive B=barrier-arrive |=barrier-release X=exit\n";
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
  for (const auto& key : seen_waves) {
    const std::string label =
        options.group_by == CycleTimelineGroupBy::Block ? BlockLabel(key.block_id) : WaveLabel(key);
    grouped_segments[label].insert(grouped_segments[label].end(), segments[key].begin(), segments[key].end());
    grouped_markers[label].insert(grouped_markers[label].end(), markers[key].begin(), markers[key].end());
  }

  for (const auto& [label, row_segments] : grouped_segments) {
    std::string row(width, '.');
    for (const auto& segment : row_segments) {
      const char symbol = symbols.at(segment.op);
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

}  // namespace gpu_model
