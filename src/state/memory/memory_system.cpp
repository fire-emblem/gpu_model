#include "state/memory/memory_system.h"

#include <algorithm>
#include <stdexcept>

namespace gpu_model {

namespace {

constexpr uint64_t kPoolTagMask = 0xF000000000000000ull;
constexpr uint64_t kPoolOffsetMask = ~kPoolTagMask;

size_t PoolIndex(MemoryPoolKind pool) {
  return static_cast<size_t>(pool);
}

uint64_t PoolBase(MemoryPoolKind pool) {
  return MemoryPoolBase(pool);
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

uint64_t PoolOffset(MemoryPoolKind pool, uint64_t addr) {
  if ((addr & kPoolTagMask) != PoolBase(pool)) {
    throw std::out_of_range(std::string(PoolName(pool)) + " memory address below pool base");
  }
  return addr & kPoolOffsetMask;
}

bool IsPoolAddress(MemoryPoolKind pool, uint64_t addr) {
  return (addr & kPoolTagMask) == PoolBase(pool);
}

}  // namespace

uint64_t MemorySystem::Allocate(MemoryPoolKind pool, size_t bytes) {
  auto& storage = pool_memory_[PoolIndex(pool)];
  const uint64_t addr = PoolBase(pool) + storage.size();
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
  const uint64_t offset = PoolOffset(pool, addr);
  const size_t end = static_cast<size_t>(offset) + data.size();
  EnsureSize(pool, end);
  std::copy(data.begin(), data.end(), storage.begin() + static_cast<size_t>(offset));
}

void MemorySystem::Read(MemoryPoolKind pool, uint64_t addr, std::span<std::byte> data) const {
  const auto& storage = pool_memory_[PoolIndex(pool)];
  const uint64_t offset = PoolOffset(pool, addr);
  const size_t end = static_cast<size_t>(offset) + data.size();
  if (end > storage.size()) {
    throw std::out_of_range(std::string(PoolName(pool)) + " memory read out of range");
  }
  std::copy(storage.begin() + static_cast<size_t>(offset),
            storage.begin() + end, data.begin());
}

size_t MemorySystem::pool_memory_size(MemoryPoolKind pool) const {
  return pool_memory_[PoolIndex(pool)].size();
}

bool MemorySystem::HasRange(MemoryPoolKind pool, uint64_t addr, size_t bytes) const {
  const auto& storage = pool_memory_[PoolIndex(pool)];
  if (!IsPoolAddress(pool, addr)) {
    return false;
  }
  const uint64_t offset = PoolOffset(pool, addr);
  const size_t end = static_cast<size_t>(offset) + bytes;
  return end <= storage.size();
}

uint64_t MemorySystem::AllocateGlobal(size_t bytes) {
  return Allocate(MemoryPoolKind::Global, bytes);
}

void MemorySystem::EnsureGlobalSize(size_t bytes) {
  EnsureSize(MemoryPoolKind::Global, bytes);
}

void MemorySystem::WriteGlobal(uint64_t addr, std::span<const std::byte> data) {
  for (const auto pool : {MemoryPoolKind::Constant, MemoryPoolKind::Shared, MemoryPoolKind::Private,
                          MemoryPoolKind::Managed, MemoryPoolKind::Kernarg, MemoryPoolKind::Code,
                          MemoryPoolKind::RawData}) {
    if (IsPoolAddress(pool, addr)) {
      Write(pool, addr, data);
      return;
    }
  }
  Write(MemoryPoolKind::Global, addr, data);
}

void MemorySystem::ReadGlobal(uint64_t addr, std::span<std::byte> data) const {
  for (const auto pool : {MemoryPoolKind::Constant, MemoryPoolKind::Shared, MemoryPoolKind::Private,
                          MemoryPoolKind::Managed, MemoryPoolKind::Kernarg, MemoryPoolKind::Code,
                          MemoryPoolKind::RawData}) {
    if (IsPoolAddress(pool, addr)) {
      Read(pool, addr, data);
      return;
    }
  }
  Read(MemoryPoolKind::Global, addr, data);
}

bool MemorySystem::HasGlobalRange(uint64_t addr, size_t bytes) const {
  for (const auto pool : {MemoryPoolKind::Constant, MemoryPoolKind::Shared, MemoryPoolKind::Private,
                          MemoryPoolKind::Managed, MemoryPoolKind::Kernarg, MemoryPoolKind::Code,
                          MemoryPoolKind::RawData}) {
    if (HasRange(pool, addr, bytes)) {
      return true;
    }
  }
  return HasRange(MemoryPoolKind::Global, addr, bytes);
}

}  // namespace gpu_model
