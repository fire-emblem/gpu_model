#pragma once

#include <cstdint>
#include <vector>

#include "gpu_model/arch/gpu_arch_spec.h"

namespace gpu_model {

class CacheModel {
 public:
  explicit CacheModel(CacheModelSpec spec = {}) : spec_(spec) {}

  uint64_t Probe(const std::vector<uint64_t>& addresses) const;
  void Promote(const std::vector<uint64_t>& addresses);

 private:
  bool ContainsL1(uint64_t line) const;
  bool ContainsL2(uint64_t line) const;
  void TouchL1(uint64_t line);
  void TouchL2(uint64_t line);

  CacheModelSpec spec_;
  std::vector<uint64_t> l1_lines_;
  std::vector<uint64_t> l2_lines_;
};

}  // namespace gpu_model
