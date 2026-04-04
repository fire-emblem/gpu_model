#include "cycle_timeline_internal.h"

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace gpu_model {

namespace {

std::string HexU64(uint64_t value) {
  std::ostringstream out;
  out << "0x" << std::hex << std::nouppercase << value;
  return out.str();
}

struct AsciiRow {
  std::vector<Segment> segments;
  std::vector<Marker> markers;
};

std::map<RowDescriptor, AsciiRow> GroupAsciiRows(const TimelineData& data,
                                                 CycleTimelineGroupBy group_by) {
  std::map<RowDescriptor, AsciiRow> grouped_rows;
  for (const auto& [key, row_segments] : data.segments) {
    for (const auto& segment : row_segments) {
      grouped_rows[DescribeRow(key, group_by, segment.block_id)].segments.push_back(segment);
    }
  }
  for (const auto& [key, row_markers] : data.markers) {
    for (const auto& marker : row_markers) {
      grouped_rows[DescribeRow(key, group_by, marker.block_id)].markers.push_back(marker);
    }
  }
  return grouped_rows;
}

}  // namespace

std::string RenderAsciiTimelineExport(const TimelineData& data,
                                      uint64_t begin,
                                      uint64_t end,
                                      uint32_t max_columns,
                                      CycleTimelineGroupBy group_by) {
  if (end < begin) {
    return "cycle_timeline: <invalid range>\n";
  }

  const uint64_t total_cycles = end - begin + 1;
  const uint32_t clamped_max_columns = std::max<uint32_t>(1, max_columns);
  const uint64_t cycles_per_column =
      std::max<uint64_t>(1, (total_cycles + clamped_max_columns - 1) / clamped_max_columns);
  const uint32_t width =
      static_cast<uint32_t>((total_cycles + cycles_per_column - 1) / cycles_per_column);

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

  const std::map<RowDescriptor, AsciiRow> grouped_rows = GroupAsciiRows(data, group_by);
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

}  // namespace gpu_model
