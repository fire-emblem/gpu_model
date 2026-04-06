#pragma once

#include <cstdint>
#include <mutex>
#include <vector>

#include "gpu_model/arch/gpu_arch_spec.h"

namespace gpu_model {

struct CacheProbeResult {
  uint64_t latency = 0;
  uint64_t l1_hits = 0;
  uint64_t l2_hits = 0;
  uint64_t misses = 0;
};

class CacheModel {
 public:
  explicit CacheModel(CacheModelSpec spec = {}) : spec_(spec) {}
  CacheModel(const CacheModel& other);
  CacheModel& operator=(const CacheModel& other);
  CacheModel(CacheModel&& other) noexcept;
  CacheModel& operator=(CacheModel&& other) noexcept;

  CacheProbeResult Probe(const std::vector<uint64_t>& addresses) const;
  void Promote(const std::vector<uint64_t>& addresses);

 private:
  bool ContainsL1(uint64_t line) const;
  bool ContainsL2(uint64_t line) const;
  void TouchL1(uint64_t line);
  void TouchL2(uint64_t line);

  CacheModelSpec spec_;
  mutable std::mutex mutex_;
  std::vector<uint64_t> l1_lines_;
  std::vector<uint64_t> l2_lines_;
};

}  // namespace gpu_model
