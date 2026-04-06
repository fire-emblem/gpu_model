#include "gpu_model/runtime/device_memory_manager.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>

#include <sys/mman.h>
#include <unistd.h>

namespace gpu_model {

namespace {

constexpr uintptr_t kGlobalCompatWindowBase = 0x0000600000000000ull;
constexpr uintptr_t kManagedCompatWindowBase = 0x0000610000000000ull;

}  // namespace

DeviceMemoryManager::DeviceMemoryManager(MemorySystem* memory)
    : memory_(memory), windows_(BuildDefaultWindows()) {
  for (const auto& window : windows_) {
    ReserveCompatibilityWindow(window.base, window.size);
  }
}

DeviceMemoryManager::~DeviceMemoryManager() {
  Reset();
}

void DeviceMemoryManager::BindMemory(MemorySystem* memory) {
  memory_ = memory;
}

void DeviceMemoryManager::Reset() {
  for (auto& [key, allocation] : allocations_) {
    (void)key;
    CommitCompatibilitySpan(reinterpret_cast<uintptr_t>(allocation.mapped_addr),
                            allocation.mapped_bytes,
                            PROT_NONE);
  }
  allocations_.clear();
  for (auto& window : windows_) {
    window.next_offset = 0;
  }
}

void* DeviceMemoryManager::AllocateGlobal(size_t bytes, uint64_t model_addr) {
  auto* window = MutableWindow(MemoryPoolKind::Global);
  if (window == nullptr) {
    throw std::runtime_error("missing global compatibility window");
  }
  const size_t mapped_bytes = PageAlignedBytes(std::max<size_t>(bytes, 1u));
  if (window->next_offset + mapped_bytes > window->size) {
    throw std::runtime_error("global compatibility window exhausted");
  }
  const uintptr_t addr = window->base + window->next_offset;
  auto* mapped_addr = CommitCompatibilitySpan(addr, mapped_bytes, PROT_NONE);
  window->next_offset += mapped_bytes;
  CompatibilityAllocation allocation;
  allocation.model_addr = model_addr;
  allocation.bytes = bytes;
  allocation.pool = MemoryPoolKind::Global;
  allocation.mapped_addr = mapped_addr;
  allocation.mapped_bytes = mapped_bytes;
  PutAllocation(reinterpret_cast<uintptr_t>(mapped_addr), std::move(allocation));
  return reinterpret_cast<void*>(mapped_addr);
}

void* DeviceMemoryManager::AllocateManaged(size_t bytes, uint64_t model_addr) {
  auto* window = MutableWindow(MemoryPoolKind::Managed);
  if (window == nullptr) {
    throw std::runtime_error("missing managed compatibility window");
  }
  const size_t mapped_bytes = PageAlignedBytes(std::max<size_t>(bytes, 1u));
  if (window->next_offset + mapped_bytes > window->size) {
    throw std::runtime_error("managed compatibility window exhausted");
  }
  const uintptr_t addr = window->base + window->next_offset;
  auto* mapped_addr = CommitCompatibilitySpan(addr, mapped_bytes, PROT_READ | PROT_WRITE);
  window->next_offset += mapped_bytes;
  std::memset(mapped_addr, 0, bytes);
  CompatibilityAllocation allocation{
      .model_addr = model_addr,
      .bytes = bytes,
      .pool = MemoryPoolKind::Managed,
      .mapped_addr = mapped_addr,
      .mapped_bytes = mapped_bytes};
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

bool DeviceMemoryManager::IsPointerInCompatibilityWindow(const void* ptr) const {
  return FindWindowForPointer(ptr) != nullptr;
}

std::optional<MemoryPoolKind> DeviceMemoryManager::ClassifyCompatibilityPointer(
    const void* ptr) const {
  const auto* window = FindWindowForPointer(ptr);
  if (window == nullptr) {
    return std::nullopt;
  }
  return window->pool;
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

std::array<DeviceMemoryManager::CompatibilityWindow,
           DeviceMemoryManager::kCompatibilityWindowCount>
DeviceMemoryManager::BuildDefaultWindows() {
  return {{{.pool = MemoryPoolKind::Global,
            .base = kGlobalCompatWindowBase,
            .size = kCompatibilityWindowSize,
            .next_offset = 0},
           {.pool = MemoryPoolKind::Managed,
            .base = kManagedCompatWindowBase,
            .size = kCompatibilityWindowSize,
            .next_offset = 0}}};
}

std::byte* DeviceMemoryManager::ReserveCompatibilityWindow(uintptr_t base, size_t bytes) {
  void* addr = ::mmap(reinterpret_cast<void*>(base),
                      bytes,
                      PROT_NONE,
                      MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED_NOREPLACE,
                      -1,
                      0);
  if (addr == MAP_FAILED) {
    throw std::runtime_error("failed to reserve compatibility window");
  }
  return reinterpret_cast<std::byte*>(addr);
}

std::byte* DeviceMemoryManager::CommitCompatibilitySpan(uintptr_t base,
                                                        size_t bytes,
                                                        int protection) {
  void* addr = ::mmap(reinterpret_cast<void*>(base),
                      bytes,
                      protection,
                      MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED,
                      -1,
                      0);
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

DeviceMemoryManager::CompatibilityWindow* DeviceMemoryManager::MutableWindow(MemoryPoolKind pool) {
  for (auto& window : windows_) {
    if (window.pool == pool) {
      return &window;
    }
  }
  return nullptr;
}

const DeviceMemoryManager::CompatibilityWindow* DeviceMemoryManager::FindWindowForPointer(
    const void* ptr) const {
  const auto addr = reinterpret_cast<uintptr_t>(ptr);
  for (const auto& window : windows_) {
    if (addr >= window.base && addr < window.base + window.size) {
      return &window;
    }
  }
  return nullptr;
}

}  // namespace gpu_model
