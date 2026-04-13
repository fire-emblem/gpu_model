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

/// Calculate branch target address from PC and signed offset.
/// This is a pure utility function for branch address calculation.
inline uint64_t BranchTarget(uint64_t pc, int32_t simm16) {
  const int64_t target = static_cast<int64_t>(pc) + 4 + static_cast<int64_t>(simm16) * 4;
  return static_cast<uint64_t>(target);
}

}  // namespace gpu_model
