#pragma once

#include <algorithm>
#include <cstdint>

namespace gpu_model {

constexpr uint64_t kIssueQuantumCycles = 4;

inline uint64_t QuantizeToNextIssueQuantum(uint64_t cycle) {
  const uint64_t remainder = cycle % kIssueQuantumCycles;
  if (remainder == 0) {
    return cycle;
  }
  return cycle + (kIssueQuantumCycles - remainder);
}

inline uint64_t QuantizeIssueDuration(uint64_t cycles) {
  return std::max(kIssueQuantumCycles, QuantizeToNextIssueQuantum(cycles));
}

}  // namespace gpu_model
