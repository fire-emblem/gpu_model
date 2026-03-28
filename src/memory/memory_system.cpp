#include "gpu_model/memory/memory_system.h"

#include <algorithm>
#include <stdexcept>

namespace gpu_model {

namespace {

size_t PoolIndex(MemoryPoolKind pool) {
  return static_cast<size_t>(pool);
}

const char* PoolName(MemoryPoolKind pool) {
  switch (pool) {
    case MemoryPoolKind::Global:
      return "global";
    case MemoryPoolKind::Constant:
      return "constant";
    case MemoryPoolKind::Shared:
      return "shared";
    case MemoryPoolKind::Private:
      return "private";
    case MemoryPoolKind::Managed:
      return "managed";
    case MemoryPoolKind::Kernarg:
      return "kernarg";
    case MemoryPoolKind::Code:
      return "code";
    case MemoryPoolKind::RawData:
      return "raw_data";
  }
  return "unknown";
}

}  // namespace

uint64_t MemorySystem::Allocate(MemoryPoolKind pool, size_t bytes) {
  auto& storage = pool_memory_[PoolIndex(pool)];
  const uint64_t addr = storage.size();
  storage.resize(storage.size() + bytes, std::byte{0});
  return addr;
}

void MemorySystem::EnsureSize(MemoryPoolKind pool, size_t bytes) {
  auto& storage = pool_memory_[PoolIndex(pool)];
  if (storage.size() < bytes) {
    storage.resize(bytes, std::byte{0});
  }
}

void MemorySystem::Write(MemoryPoolKind pool, uint64_t addr, std::span<const std::byte> data) {
  auto& storage = pool_memory_[PoolIndex(pool)];
  const size_t end = static_cast<size_t>(addr) + data.size();
  EnsureSize(pool, end);
  std::copy(data.begin(), data.end(), storage.begin() + static_cast<size_t>(addr));
}

void MemorySystem::Read(MemoryPoolKind pool, uint64_t addr, std::span<std::byte> data) const {
  const auto& storage = pool_memory_[PoolIndex(pool)];
  const size_t end = static_cast<size_t>(addr) + data.size();
  if (end > storage.size()) {
    throw std::out_of_range(std::string(PoolName(pool)) + " memory read out of range");
  }
  std::copy(storage.begin() + static_cast<size_t>(addr),
            storage.begin() + end, data.begin());
}

size_t MemorySystem::pool_memory_size(MemoryPoolKind pool) const {
  return pool_memory_[PoolIndex(pool)].size();
}

bool MemorySystem::HasRange(MemoryPoolKind pool, uint64_t addr, size_t bytes) const {
  const auto& storage = pool_memory_[PoolIndex(pool)];
  const size_t end = static_cast<size_t>(addr) + bytes;
  return end <= storage.size();
}

uint64_t MemorySystem::AllocateGlobal(size_t bytes) {
  return Allocate(MemoryPoolKind::Global, bytes);
}

void MemorySystem::EnsureGlobalSize(size_t bytes) {
  EnsureSize(MemoryPoolKind::Global, bytes);
}

void MemorySystem::WriteGlobal(uint64_t addr, std::span<const std::byte> data) {
  Write(MemoryPoolKind::Global, addr, data);
}

void MemorySystem::ReadGlobal(uint64_t addr, std::span<std::byte> data) const {
  Read(MemoryPoolKind::Global, addr, data);
}

}  // namespace gpu_model
