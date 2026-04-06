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
  for (const auto& window : windows_) {
    ReleaseCompatibilityWindow(window.base, window.size);
  }
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
    window.free_ranges.clear();
  }
}

void* DeviceMemoryManager::AllocateGlobal(size_t bytes, uint64_t model_addr) {
  auto* window = MutableWindow(MemoryPoolKind::Global);
  if (window == nullptr) {
    throw std::runtime_error("missing global compatibility window");
  }
  const size_t mapped_bytes = PageAlignedBytes(std::max<size_t>(bytes, 1u));
  const auto reused_offset = TryReuseFreeRange(*window, mapped_bytes);
  size_t offset = 0;
  if (reused_offset.has_value()) {
    offset = *reused_offset;
  } else {
    if (window->next_offset + mapped_bytes > window->size) {
      throw std::runtime_error("global compatibility window exhausted");
    }
    offset = window->next_offset;
    window->next_offset += mapped_bytes;
  }
  const uintptr_t addr = window->base + offset;
  auto* mapped_addr = CommitCompatibilitySpan(addr, mapped_bytes, PROT_NONE);
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
  const auto reused_offset = TryReuseFreeRange(*window, mapped_bytes);
  size_t offset = 0;
  if (reused_offset.has_value()) {
    offset = *reused_offset;
  } else {
    if (window->next_offset + mapped_bytes > window->size) {
      throw std::runtime_error("managed compatibility window exhausted");
    }
    offset = window->next_offset;
    window->next_offset += mapped_bytes;
  }
  const uintptr_t addr = window->base + offset;
  auto* mapped_addr = CommitCompatibilitySpan(addr, mapped_bytes, PROT_READ | PROT_WRITE);
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
  auto* window = MutableWindow(allocation->pool);
  if (window == nullptr) {
    throw std::runtime_error("missing compatibility window for allocation");
  }
  const size_t offset = reinterpret_cast<uintptr_t>(allocation->mapped_addr) - window->base;
  UnmapCompatibilitySpan(allocation->mapped_addr, allocation->mapped_bytes);
  ReleaseRange(*window, offset, allocation->mapped_bytes);
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

const DeviceMemoryManager::CompatibilityWindow* DeviceMemoryManager::GetCompatibilityWindow(
    MemoryPoolKind pool) const {
  for (const auto& window : windows_) {
    if (window.pool == pool) {
      return &window;
    }
  }
  return nullptr;
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
            .next_offset = 0,
            .free_ranges = {}},
           {.pool = MemoryPoolKind::Managed,
            .base = kManagedCompatWindowBase,
            .size = kCompatibilityWindowSize,
            .next_offset = 0,
            .free_ranges = {}}}};
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

void DeviceMemoryManager::ReleaseCompatibilityWindow(uintptr_t base, size_t bytes) {
  ::munmap(reinterpret_cast<void*>(base), bytes);
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

std::optional<size_t> DeviceMemoryManager::TryReuseFreeRange(CompatibilityWindow& window,
                                                             size_t bytes) {
  for (size_t i = 0; i < window.free_ranges.size(); ++i) {
    auto& range = window.free_ranges[i];
    if (range.size < bytes) {
      continue;
    }
    const size_t offset = range.offset;
    if (range.size == bytes) {
      window.free_ranges.erase(window.free_ranges.begin() + static_cast<std::ptrdiff_t>(i));
    } else {
      range.offset += bytes;
      range.size -= bytes;
    }
    return offset;
  }
  return std::nullopt;
}

void DeviceMemoryManager::ReleaseRange(CompatibilityWindow& window, size_t offset, size_t size) {
  window.free_ranges.push_back({.offset = offset, .size = size});
  std::sort(window.free_ranges.begin(), window.free_ranges.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.offset < rhs.offset;
  });

  std::vector<CompatibilityWindow::FreeRange> merged;
  merged.reserve(window.free_ranges.size());
  for (const auto& range : window.free_ranges) {
    if (!merged.empty() && merged.back().offset + merged.back().size == range.offset) {
      merged.back().size += range.size;
    } else {
      merged.push_back(range);
    }
  }
  window.free_ranges = std::move(merged);
}

}  // namespace gpu_model
