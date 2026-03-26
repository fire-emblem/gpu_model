#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

namespace gpu_model {

class MemorySystem {
 public:
  uint64_t AllocateGlobal(size_t bytes);
  void EnsureGlobalSize(size_t bytes);
  void WriteGlobal(uint64_t addr, std::span<const std::byte> data);
  void ReadGlobal(uint64_t addr, std::span<std::byte> data) const;

  size_t global_memory_size() const { return global_memory_.size(); }

  template <typename T>
  void StoreGlobalValue(uint64_t addr, const T& value) {
    std::array<std::byte, sizeof(T)> bytes{};
    std::memcpy(bytes.data(), &value, sizeof(T));
    WriteGlobal(addr, std::span<const std::byte>(bytes.data(), bytes.size()));
  }

  template <typename T>
  T LoadGlobalValue(uint64_t addr) const {
    std::array<std::byte, sizeof(T)> bytes{};
    ReadGlobal(addr, std::span<std::byte>(bytes.data(), bytes.size()));
    T value{};
    std::memcpy(&value, bytes.data(), sizeof(T));
    return value;
  }

 private:
  std::vector<std::byte> global_memory_;
};

}  // namespace gpu_model
