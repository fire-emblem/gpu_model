#include "gpu_model/runtime/device_memory_manager.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include <sys/mman.h>
#include <unistd.h>

namespace gpu_model {

DeviceMemoryManager::DeviceMemoryManager(MemorySystem* memory) : memory_(memory) {}

DeviceMemoryManager::~DeviceMemoryManager() {
  Reset();
}

void DeviceMemoryManager::BindMemory(MemorySystem* memory) {
  memory_ = memory;
}

void DeviceMemoryManager::Reset() {
  for (auto& [key, allocation] : allocations_) {
    (void)key;
    UnmapCompatibilitySpan(allocation.mapped_addr, allocation.mapped_bytes);
  }
  allocations_.clear();
}

void* DeviceMemoryManager::AllocateGlobal(size_t bytes, uint64_t model_addr) {
  auto* mapped_addr = MapCompatibilitySpan(bytes, PROT_NONE);
  CompatibilityAllocation allocation;
  allocation.model_addr = model_addr;
  allocation.bytes = bytes;
  allocation.pool = MemoryPoolKind::Global;
  allocation.mapped_addr = mapped_addr;
  allocation.mapped_bytes = PageAlignedBytes(std::max<size_t>(bytes, 1u));
  PutAllocation(reinterpret_cast<uintptr_t>(mapped_addr), std::move(allocation));
  return reinterpret_cast<void*>(mapped_addr);
}

void* DeviceMemoryManager::AllocateManaged(size_t bytes, uint64_t model_addr) {
  auto* mapped_addr = MapCompatibilitySpan(bytes, PROT_READ | PROT_WRITE);
  std::memset(mapped_addr, 0, bytes);
  CompatibilityAllocation allocation{
      .model_addr = model_addr,
      .bytes = bytes,
      .pool = MemoryPoolKind::Managed,
      .mapped_addr = mapped_addr,
      .mapped_bytes = PageAlignedBytes(std::max<size_t>(bytes, 1u))};
  PutAllocation(reinterpret_cast<uintptr_t>(mapped_addr), std::move(allocation));
  return reinterpret_cast<void*>(mapped_addr);
}

bool DeviceMemoryManager::Free(void* device_ptr) {
  const auto* allocation = FindAllocation(device_ptr);
  if (allocation == nullptr) {
    return false;
  }
  UnmapCompatibilitySpan(allocation->mapped_addr, allocation->mapped_bytes);
  EraseAllocation(device_ptr);
  return true;
}

bool DeviceMemoryManager::HasAllocation(const void* ptr) const {
  return allocations_.find(reinterpret_cast<uintptr_t>(ptr)) != allocations_.end();
}

bool DeviceMemoryManager::IsDevicePointer(const void* ptr) const {
  return HasAllocation(ptr);
}

DeviceMemoryManager::CompatibilityAllocation* DeviceMemoryManager::FindAllocation(const void* ptr) {
  const auto it = allocations_.find(reinterpret_cast<uintptr_t>(ptr));
  if (it == allocations_.end()) {
    return nullptr;
  }
  return &it->second;
}

const DeviceMemoryManager::CompatibilityAllocation* DeviceMemoryManager::FindAllocation(
    const void* ptr) const {
  const auto it = allocations_.find(reinterpret_cast<uintptr_t>(ptr));
  if (it == allocations_.end()) {
    return nullptr;
  }
  return &it->second;
}

uint64_t DeviceMemoryManager::ResolveDeviceAddress(const void* ptr) const {
  const auto* allocation = FindAllocation(ptr);
  if (allocation == nullptr) {
    throw std::invalid_argument("unknown compatibility device pointer");
  }
  return allocation->model_addr;
}

void DeviceMemoryManager::SyncManagedHostToDevice() {
  if (memory_ == nullptr) {
    return;
  }
  for (auto& [key, allocation] : allocations_) {
    (void)key;
    if (allocation.pool != MemoryPoolKind::Managed || allocation.mapped_addr == nullptr) {
      continue;
    }
    memory_->WriteGlobal(allocation.model_addr,
                         std::span<const std::byte>(allocation.mapped_addr, allocation.bytes));
  }
}

void DeviceMemoryManager::SyncManagedDeviceToHost() {
  if (memory_ == nullptr) {
    return;
  }
  for (auto& [key, allocation] : allocations_) {
    (void)key;
    if (allocation.pool != MemoryPoolKind::Managed || allocation.mapped_addr == nullptr) {
      continue;
    }
    memory_->ReadGlobal(allocation.model_addr,
                        std::span<std::byte>(allocation.mapped_addr, allocation.bytes));
  }
}

size_t DeviceMemoryManager::PageAlignedBytes(size_t bytes) {
  const long page_size = ::sysconf(_SC_PAGESIZE);
  const size_t alignment = page_size > 0 ? static_cast<size_t>(page_size) : 4096u;
  return ((bytes + alignment - 1) / alignment) * alignment;
}

std::byte* DeviceMemoryManager::MapCompatibilitySpan(size_t bytes, int protection) {
  const size_t mapped_bytes = PageAlignedBytes(std::max<size_t>(bytes, 1u));
  void* addr = ::mmap(nullptr, mapped_bytes, protection, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (addr == MAP_FAILED) {
    throw std::runtime_error("mmap failed for compatibility allocation");
  }
  return reinterpret_cast<std::byte*>(addr);
}

void DeviceMemoryManager::UnmapCompatibilitySpan(std::byte* addr, size_t mapped_bytes) {
  if (addr == nullptr || mapped_bytes == 0) {
    return;
  }
  ::munmap(addr, mapped_bytes);
}

DeviceMemoryManager::CompatibilityAllocation& DeviceMemoryManager::PutAllocation(
    uintptr_t key,
    CompatibilityAllocation allocation) {
  allocations_[key] = std::move(allocation);
  return allocations_.at(key);
}

void DeviceMemoryManager::EraseAllocation(const void* ptr) {
  allocations_.erase(reinterpret_cast<uintptr_t>(ptr));
}

}  // namespace gpu_model
