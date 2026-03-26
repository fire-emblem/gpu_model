#include "gpu_model/memory/cache_model.h"

#include <algorithm>
#include <unordered_set>

namespace gpu_model {

namespace {

void RemoveIfPresent(std::vector<uint64_t>& lines, uint64_t line) {
  lines.erase(std::remove(lines.begin(), lines.end(), line), lines.end());
}

std::vector<uint64_t> UniqueLines(const std::vector<uint64_t>& addresses, uint32_t line_bytes) {
  std::unordered_set<uint64_t> seen;
  std::vector<uint64_t> lines;
  for (const uint64_t addr : addresses) {
    const uint64_t line = line_bytes == 0 ? addr : addr / line_bytes;
    if (seen.insert(line).second) {
      lines.push_back(line);
    }
  }
  return lines;
}

}  // namespace

uint64_t CacheModel::Probe(const std::vector<uint64_t>& addresses) const {
  if (addresses.empty()) {
    return 0;
  }
  if (!spec_.enabled) {
    return spec_.dram_latency;
  }

  uint64_t worst_latency = 0;
  for (const uint64_t line : UniqueLines(addresses, spec_.line_bytes)) {
    if (ContainsL1(line)) {
      worst_latency = std::max(worst_latency, spec_.l1_hit_latency);
    } else if (ContainsL2(line)) {
      worst_latency = std::max(worst_latency, spec_.l2_hit_latency);
    } else {
      worst_latency = std::max(worst_latency, spec_.dram_latency);
    }
  }
  return worst_latency;
}

void CacheModel::Promote(const std::vector<uint64_t>& addresses) {
  if (!spec_.enabled) {
    return;
  }
  for (const uint64_t line : UniqueLines(addresses, spec_.line_bytes)) {
    TouchL2(line);
    TouchL1(line);
  }
}

bool CacheModel::ContainsL1(uint64_t line) const {
  return std::find(l1_lines_.begin(), l1_lines_.end(), line) != l1_lines_.end();
}

bool CacheModel::ContainsL2(uint64_t line) const {
  return std::find(l2_lines_.begin(), l2_lines_.end(), line) != l2_lines_.end();
}

void CacheModel::TouchL1(uint64_t line) {
  RemoveIfPresent(l1_lines_, line);
  l1_lines_.push_back(line);
  if (spec_.l1_line_capacity > 0 && l1_lines_.size() > spec_.l1_line_capacity) {
    l1_lines_.erase(l1_lines_.begin());
  }
}

void CacheModel::TouchL2(uint64_t line) {
  RemoveIfPresent(l2_lines_, line);
  l2_lines_.push_back(line);
  if (spec_.l2_line_capacity > 0 && l2_lines_.size() > spec_.l2_line_capacity) {
    l2_lines_.erase(l2_lines_.begin());
  }
}

}  // namespace gpu_model
