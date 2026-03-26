#include "gpu_model/memory/memory_system.h"

#include <algorithm>
#include <stdexcept>

namespace gpu_model {

uint64_t MemorySystem::AllocateGlobal(size_t bytes) {
  const uint64_t addr = global_memory_.size();
  global_memory_.resize(global_memory_.size() + bytes, std::byte{0});
  return addr;
}

void MemorySystem::EnsureGlobalSize(size_t bytes) {
  if (global_memory_.size() < bytes) {
    global_memory_.resize(bytes, std::byte{0});
  }
}

void MemorySystem::WriteGlobal(uint64_t addr, std::span<const std::byte> data) {
  const size_t end = static_cast<size_t>(addr) + data.size();
  EnsureGlobalSize(end);
  std::copy(data.begin(), data.end(), global_memory_.begin() + static_cast<size_t>(addr));
}

void MemorySystem::ReadGlobal(uint64_t addr, std::span<std::byte> data) const {
  const size_t end = static_cast<size_t>(addr) + data.size();
  if (end > global_memory_.size()) {
    throw std::out_of_range("global memory read out of range");
  }
  std::copy(global_memory_.begin() + static_cast<size_t>(addr),
            global_memory_.begin() + end, data.begin());
}

}  // namespace gpu_model
