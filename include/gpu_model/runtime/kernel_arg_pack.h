#pragma once

#include <bit>
#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace gpu_model {

class KernelArgPack {
 public:
  void PushU64(uint64_t value) { values_.push_back(value); }
  void PushI32(int32_t value) { values_.push_back(static_cast<uint32_t>(value)); }
  void PushU32(uint32_t value) { values_.push_back(value); }
  void PushF32(float value) { values_.push_back(std::bit_cast<uint32_t>(value)); }

  uint64_t GetU64(size_t index) const {
    if (index >= values_.size()) {
      throw std::out_of_range("kernel arg index out of range");
    }
    return values_[index];
  }

  size_t size() const { return values_.size(); }
  const std::vector<uint64_t>& values() const { return values_; }

 private:
  std::vector<uint64_t> values_;
};

}  // namespace gpu_model
