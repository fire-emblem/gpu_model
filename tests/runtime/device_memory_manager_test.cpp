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

}  // namespace
}  // namespace gpu_model
