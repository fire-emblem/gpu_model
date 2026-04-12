#pragma once

#include <bitset>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

namespace gpu_model {

inline std::bitset<64> MaskFromU64(uint64_t value) {
  return std::bitset<64>(value);
}

inline uint32_t LoadU32(std::span<const std::byte> bytes, size_t offset) {
  uint32_t value = 0;
  std::memcpy(&value, bytes.data() + offset, sizeof(value));
  return value;
}

inline uint32_t LoadU32(const std::vector<std::byte>& bytes, size_t offset) {
  return LoadU32(std::span<const std::byte>(bytes.data(), bytes.size()), offset);
}

inline void StoreU32(std::span<std::byte> bytes, size_t offset, uint32_t value) {
  std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

inline void StoreU32(std::vector<std::byte>& bytes, size_t offset, uint32_t value) {
  StoreU32(std::span<std::byte>(bytes.data(), bytes.size()), offset, value);
}

}  // namespace gpu_model
