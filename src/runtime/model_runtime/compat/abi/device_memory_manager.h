#pragma once

#include <cstddef>
#include <cstdint>
#include <array>
#include <optional>
#include <vector>
#include <unordered_map>

#include "gpu_arch/memory/memory_pool.h"
#include "state/memory/memory_system.h"

namespace gpu_model {

class DeviceMemoryManager {
 public:
  struct AbiWindow {
    struct FreeRange {
      size_t offset = 0;
      size_t size = 0;
    };

    MemoryPoolKind pool = MemoryPoolKind::Global;
    uintptr_t base = 0;
    size_t size = 0;
    size_t next_offset = 0;
    size_t committed_bytes = 0;
    std::vector<FreeRange> free_ranges;
  };

  struct AbiAllocation {
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
  bool IsPointerInAbiWindow(const void* ptr) const;
  std::optional<MemoryPoolKind> ClassifyAbiPointer(const void* ptr) const;
  const AbiWindow* GetAbiWindow(MemoryPoolKind pool) const;
  AbiAllocation* FindAllocation(const void* ptr);
  const AbiAllocation* FindAllocation(const void* ptr) const;
  uint64_t ResolveDeviceAddress(const void* ptr) const;

  void SyncManagedHostToDevice();
  void SyncManagedDeviceToHost();

 private:
  static constexpr size_t kAbiWindowCount = 2;
  static constexpr size_t kAbiWindowSize = 1ull << 30;
  static std::array<AbiWindow, kAbiWindowCount> BuildDefaultAbiWindows();
  static size_t PageAlignedBytes(size_t bytes);
  static std::byte* TryReserveAbiWindow(uintptr_t base, size_t bytes);
  static std::byte* ReserveAnyAbiWindow(size_t bytes);
  static void ReleaseAbiWindow(uintptr_t base, size_t bytes);
  static std::byte* CommitAbiSpan(uintptr_t base, size_t bytes, int protection);
  static void UnmapAbiSpan(std::byte* addr, size_t mapped_bytes);

  AbiWindow* MutableWindow(MemoryPoolKind pool);
  const AbiWindow* FindWindowForPointer(const void* ptr) const;
  static std::optional<size_t> TryReuseFreeRange(AbiWindow& window, size_t bytes);
  static void ReleaseRange(AbiWindow& window, size_t offset, size_t size);
  AbiAllocation& PutAllocation(uintptr_t key, AbiAllocation allocation);
  void EraseAllocation(const void* ptr);

  MemorySystem* memory_ = nullptr;
  std::array<AbiWindow, kAbiWindowCount> windows_{};
  std::unordered_map<uintptr_t, AbiAllocation> allocations_;
};

}  // namespace gpu_model
