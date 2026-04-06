#pragma once

#include <cstddef>
#include <cstdint>
#include <array>
#include <optional>
#include <vector>
#include <unordered_map>

#include "gpu_model/memory/memory_pool.h"
#include "gpu_model/memory/memory_system.h"

namespace gpu_model {

class DeviceMemoryManager {
 public:
  struct CompatibilityWindow {
    struct FreeRange {
      size_t offset = 0;
      size_t size = 0;
    };

    MemoryPoolKind pool = MemoryPoolKind::Global;
    uintptr_t base = 0;
    size_t size = 0;
    size_t next_offset = 0;
    std::vector<FreeRange> free_ranges;
  };

  struct CompatibilityAllocation {
    uint64_t model_addr = 0;
    size_t bytes = 0;
    MemoryPoolKind pool = MemoryPoolKind::Global;
    std::byte* mapped_addr = nullptr;
    size_t mapped_bytes = 0;
  };

  explicit DeviceMemoryManager(MemorySystem* memory = nullptr);
  ~DeviceMemoryManager();

  DeviceMemoryManager(const DeviceMemoryManager&) = delete;
  DeviceMemoryManager& operator=(const DeviceMemoryManager&) = delete;

  void BindMemory(MemorySystem* memory);
  void Reset();

  void* AllocateGlobal(size_t bytes, uint64_t model_addr);
  void* AllocateManaged(size_t bytes, uint64_t model_addr);
  bool Free(void* device_ptr);

  bool HasAllocation(const void* ptr) const;
  bool IsDevicePointer(const void* ptr) const;
  bool IsPointerInCompatibilityWindow(const void* ptr) const;
  std::optional<MemoryPoolKind> ClassifyCompatibilityPointer(const void* ptr) const;
  const CompatibilityWindow* GetCompatibilityWindow(MemoryPoolKind pool) const;
  CompatibilityAllocation* FindAllocation(const void* ptr);
  const CompatibilityAllocation* FindAllocation(const void* ptr) const;
  uint64_t ResolveDeviceAddress(const void* ptr) const;

  void SyncManagedHostToDevice();
  void SyncManagedDeviceToHost();

 private:
  static constexpr size_t kCompatibilityWindowCount = 2;
  static constexpr size_t kCompatibilityWindowSize = 1ull << 30;
  static std::array<CompatibilityWindow, kCompatibilityWindowCount> BuildDefaultWindows();
  static size_t PageAlignedBytes(size_t bytes);
  static std::byte* ReserveCompatibilityWindow(uintptr_t base, size_t bytes);
  static void ReleaseCompatibilityWindow(uintptr_t base, size_t bytes);
  static std::byte* CommitCompatibilitySpan(uintptr_t base, size_t bytes, int protection);
  static void UnmapCompatibilitySpan(std::byte* addr, size_t mapped_bytes);

  CompatibilityWindow* MutableWindow(MemoryPoolKind pool);
  const CompatibilityWindow* FindWindowForPointer(const void* ptr) const;
  static std::optional<size_t> TryReuseFreeRange(CompatibilityWindow& window, size_t bytes);
  static void ReleaseRange(CompatibilityWindow& window, size_t offset, size_t size);
  CompatibilityAllocation& PutAllocation(uintptr_t key, CompatibilityAllocation allocation);
  void EraseAllocation(const void* ptr);

  MemorySystem* memory_ = nullptr;
  std::array<CompatibilityWindow, kCompatibilityWindowCount> windows_{};
  std::unordered_map<uintptr_t, CompatibilityAllocation> allocations_;
};

}  // namespace gpu_model
