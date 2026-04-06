#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

#include "gpu_model/memory/memory_pool.h"

namespace gpu_model {

class MemorySystem {
 public:
  uint64_t Allocate(MemoryPoolKind pool, size_t bytes);
  void EnsureSize(MemoryPoolKind pool, size_t bytes);
  void Write(MemoryPoolKind pool, uint64_t addr, std::span<const std::byte> data);
  void Read(MemoryPoolKind pool, uint64_t addr, std::span<std::byte> data) const;
  size_t pool_memory_size(MemoryPoolKind pool) const;
  bool HasRange(MemoryPoolKind pool, uint64_t addr, size_t bytes) const;

  uint64_t AllocateGlobal(size_t bytes);
  void EnsureGlobalSize(size_t bytes);
  void WriteGlobal(uint64_t addr, std::span<const std::byte> data);
  void ReadGlobal(uint64_t addr, std::span<std::byte> data) const;
  bool HasGlobalRange(uint64_t addr, size_t bytes) const;

  size_t global_memory_size() const { return pool_memory_size(MemoryPoolKind::Global); }

  template <typename T>
  void StoreValue(MemoryPoolKind pool, uint64_t addr, const T& value) {
    std::array<std::byte, sizeof(T)> bytes{};
    std::memcpy(bytes.data(), &value, sizeof(T));
    Write(pool, addr, std::span<const std::byte>(bytes.data(), bytes.size()));
  }

  template <typename T>
  T LoadValue(MemoryPoolKind pool, uint64_t addr) const {
    std::array<std::byte, sizeof(T)> bytes{};
    Read(pool, addr, std::span<std::byte>(bytes.data(), bytes.size()));
    T value{};
    std::memcpy(&value, bytes.data(), sizeof(T));
    return value;
  }

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
  static constexpr size_t kPoolCount = 8;
  std::array<std::vector<std::byte>, kPoolCount> pool_memory_{};
};

}  // namespace gpu_model
