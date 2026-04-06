#include <gtest/gtest.h>

#include "gpu_model/memory/memory_system.h"
#include "gpu_model/runtime/device_memory_manager.h"

namespace gpu_model {
namespace {

TEST(DeviceMemoryManagerTest, ClassifiesPointersByCompatibilityWindow) {
  MemorySystem memory;
  DeviceMemoryManager manager(&memory);

  void* global_ptr = manager.AllocateGlobal(64, memory.AllocateGlobal(64));
  void* managed_ptr =
      manager.AllocateManaged(64, memory.Allocate(MemoryPoolKind::Managed, 64));

  ASSERT_TRUE(manager.IsPointerInCompatibilityWindow(global_ptr));
  ASSERT_TRUE(manager.IsPointerInCompatibilityWindow(managed_ptr));

  const auto global_kind = manager.ClassifyCompatibilityPointer(global_ptr);
  const auto managed_kind = manager.ClassifyCompatibilityPointer(managed_ptr);
  ASSERT_TRUE(global_kind.has_value());
  ASSERT_TRUE(managed_kind.has_value());
  EXPECT_EQ(*global_kind, MemoryPoolKind::Global);
  EXPECT_EQ(*managed_kind, MemoryPoolKind::Managed);
}

TEST(DeviceMemoryManagerTest, ReusesFreedAddressWithinSamePool) {
  MemorySystem memory;
  DeviceMemoryManager manager(&memory);

  void* first = manager.AllocateGlobal(64, memory.AllocateGlobal(64));
  ASSERT_NE(first, nullptr);
  ASSERT_TRUE(manager.Free(first));

  void* second = manager.AllocateGlobal(64, memory.AllocateGlobal(64));
  ASSERT_NE(second, nullptr);
  EXPECT_EQ(second, first);
}

TEST(DeviceMemoryManagerTest, DoesNotMisclassifyNonWindowPointer) {
  MemorySystem memory;
  DeviceMemoryManager manager(&memory);

  int host_value = 7;
  void* host_ptr = &host_value;

  EXPECT_FALSE(manager.IsPointerInCompatibilityWindow(host_ptr));
  EXPECT_FALSE(manager.IsDevicePointer(host_ptr));
  EXPECT_FALSE(manager.ClassifyCompatibilityPointer(host_ptr).has_value());
  EXPECT_EQ(manager.FindAllocation(host_ptr), nullptr);
}

TEST(DeviceMemoryManagerTest, ExposesStableSeparatedWindowsForGlobalAndManagedPools) {
  MemorySystem memory;
  DeviceMemoryManager manager(&memory);

  const auto* global_window = manager.GetCompatibilityWindow(MemoryPoolKind::Global);
  const auto* managed_window = manager.GetCompatibilityWindow(MemoryPoolKind::Managed);
  ASSERT_NE(global_window, nullptr);
  ASSERT_NE(managed_window, nullptr);

  EXPECT_EQ(global_window->pool, MemoryPoolKind::Global);
  EXPECT_EQ(managed_window->pool, MemoryPoolKind::Managed);
  EXPECT_GT(global_window->size, 0u);
  EXPECT_GT(managed_window->size, 0u);
  EXPECT_NE(global_window->base, managed_window->base);
  EXPECT_LT(global_window->base, managed_window->base);
  EXPECT_LE(global_window->base + global_window->size, managed_window->base);

  void* global_ptr = manager.AllocateGlobal(64, memory.AllocateGlobal(64));
  void* managed_ptr = manager.AllocateManaged(64, memory.Allocate(MemoryPoolKind::Managed, 64));
  const auto global_addr = reinterpret_cast<uintptr_t>(global_ptr);
  const auto managed_addr = reinterpret_cast<uintptr_t>(managed_ptr);

  EXPECT_GE(global_addr, global_window->base);
  EXPECT_LT(global_addr, global_window->base + global_window->size);
  EXPECT_GE(managed_addr, managed_window->base);
  EXPECT_LT(managed_addr, managed_window->base + managed_window->size);
}

TEST(DeviceMemoryManagerTest, TracksCommittedBytesPerCompatibilityWindowOnDemand) {
  MemorySystem memory;
  DeviceMemoryManager manager(&memory);

  const auto* global_window_before = manager.GetCompatibilityWindow(MemoryPoolKind::Global);
  const auto* managed_window_before = manager.GetCompatibilityWindow(MemoryPoolKind::Managed);
  ASSERT_NE(global_window_before, nullptr);
  ASSERT_NE(managed_window_before, nullptr);
  EXPECT_GT(global_window_before->size, 0u);
  EXPECT_GT(managed_window_before->size, 0u);
  EXPECT_EQ(global_window_before->committed_bytes, 0u);
  EXPECT_EQ(managed_window_before->committed_bytes, 0u);

  void* global_ptr = manager.AllocateGlobal(64, memory.AllocateGlobal(64));
  const auto* global_window_after_alloc = manager.GetCompatibilityWindow(MemoryPoolKind::Global);
  ASSERT_NE(global_window_after_alloc, nullptr);
  EXPECT_GE(global_window_after_alloc->committed_bytes, 64u);

  void* managed_ptr = manager.AllocateManaged(128, memory.Allocate(MemoryPoolKind::Managed, 128));
  const auto* managed_window_after_alloc = manager.GetCompatibilityWindow(MemoryPoolKind::Managed);
  ASSERT_NE(managed_window_after_alloc, nullptr);
  EXPECT_GE(managed_window_after_alloc->committed_bytes, 128u);

  ASSERT_TRUE(manager.Free(global_ptr));
  const auto* global_window_after_free = manager.GetCompatibilityWindow(MemoryPoolKind::Global);
  ASSERT_NE(global_window_after_free, nullptr);
  EXPECT_EQ(global_window_after_free->committed_bytes, 0u);

  manager.Reset();
  const auto* managed_window_after_reset = manager.GetCompatibilityWindow(MemoryPoolKind::Managed);
  ASSERT_NE(managed_window_after_reset, nullptr);
  EXPECT_EQ(managed_window_after_reset->committed_bytes, 0u);
  (void)managed_ptr;
}

TEST(DeviceMemoryManagerTest, ResetPreservesReservedWindowButClearsCommittedBytes) {
  MemorySystem memory;
  DeviceMemoryManager manager(&memory);

  const auto* global_window_before = manager.GetCompatibilityWindow(MemoryPoolKind::Global);
  ASSERT_NE(global_window_before, nullptr);
  const uintptr_t base_before = global_window_before->base;
  const size_t size_before = global_window_before->size;

  void* global_ptr = manager.AllocateGlobal(64, memory.AllocateGlobal(64));
  ASSERT_NE(global_ptr, nullptr);
  const auto* global_window_after_alloc = manager.GetCompatibilityWindow(MemoryPoolKind::Global);
  ASSERT_NE(global_window_after_alloc, nullptr);
  EXPECT_GT(global_window_after_alloc->committed_bytes, 0u);

  manager.Reset();

  const auto* global_window_after_reset = manager.GetCompatibilityWindow(MemoryPoolKind::Global);
  ASSERT_NE(global_window_after_reset, nullptr);
  EXPECT_EQ(global_window_after_reset->base, base_before);
  EXPECT_EQ(global_window_after_reset->size, size_before);
  EXPECT_EQ(global_window_after_reset->committed_bytes, 0u);
}

}  // namespace
}  // namespace gpu_model
