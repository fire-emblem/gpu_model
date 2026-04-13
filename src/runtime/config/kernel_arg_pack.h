#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <vector>

namespace gpu_model {

class KernelArgPack {
 public:
  void PushU64(uint64_t value) { PushScalar(value); }
  void PushI32(int32_t value) { PushScalar(static_cast<uint32_t>(value)); }
  void PushU32(uint32_t value) { PushScalar(value); }
  void PushF32(float value) { PushScalar(std::bit_cast<uint32_t>(value)); }
  void PushBytes(std::span<const std::byte> bytes) {
    values_.emplace_back(bytes.begin(), bytes.end());
  }
  void PushBytes(const void* data, size_t size) {
    const auto* first = static_cast<const std::byte*>(data);
    PushBytes(std::span<const std::byte>(first, size));
  }

  uint64_t GetU64(size_t index) const {
    if (index >= values_.size()) {
      throw std::out_of_range("kernel arg index out of range");
    }
    uint64_t value = 0;
    const auto& bytes = values_[index];
    std::memcpy(&value, bytes.data(), std::min(bytes.size(), sizeof(value)));
    return value;
  }

  size_t size() const { return values_.size(); }
  const std::vector<std::byte>& bytes(size_t index) const {
    if (index >= values_.size()) {
      throw std::out_of_range("kernel arg index out of range");
    }
    return values_[index];
  }

 private:
  template <typename T>
  void PushScalar(T value) {
    std::vector<std::byte> bytes(sizeof(T));
    std::memcpy(bytes.data(), &value, sizeof(T));
    values_.push_back(std::move(bytes));
  }

  std::vector<std::vector<std::byte>> values_;
};

}  // namespace gpu_model
