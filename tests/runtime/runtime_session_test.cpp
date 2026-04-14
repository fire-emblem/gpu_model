#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#include "debug/trace/artifact_recorder.h"
#include "runtime/model_runtime/runtime_session.h"

namespace gpu_model {
namespace {

TEST(RuntimeSessionTest, SupportsSynchronousMemoryApiMatrixWithoutKernelLaunch) {
  RuntimeSession session;

  constexpr size_t count = 8;
  std::array<uint32_t, count> input{};
  for (size_t i = 0; i < count; ++i) {
    input[i] = static_cast<uint32_t>(100 + i * 7);
  }
  std::array<uint32_t, count> copied{};
  std::array<uint32_t, count> filled32{};
  std::array<uint16_t, count> filled16{};
  std::array<uint8_t, count * sizeof(uint32_t)> filled8{};

  void* global_src = session.AllocateDevice(count * sizeof(uint32_t));
  void* global_dst = session.AllocateDevice(count * sizeof(uint32_t));
  void* managed_dst = session.AllocateManaged(count * sizeof(uint32_t));
  void* fill32_ptr = session.AllocateDevice(count * sizeof(uint32_t));
  void* fill16_ptr = session.AllocateManaged(count * sizeof(uint16_t));
  void* fill8_ptr = session.AllocateDevice(count * sizeof(uint32_t));

  session.MemcpyHostToDevice(global_src, input.data(), count * sizeof(uint32_t));
  session.MemsetDevice(global_dst, 0, count * sizeof(uint32_t));
  session.MemcpyDeviceToDevice(global_dst, global_src, count * sizeof(uint32_t));
  session.MemcpyDeviceToHost(copied.data(), global_dst, count * sizeof(uint32_t));
  EXPECT_EQ(copied, input);

  session.MemcpyDeviceToDevice(managed_dst, global_src, count * sizeof(uint32_t));
  const auto* managed_allocation = session.FindAbiAllocation(managed_dst);
  ASSERT_NE(managed_allocation, nullptr);
  ASSERT_NE(managed_allocation->mapped_addr, nullptr);
  auto* managed_words = reinterpret_cast<const uint32_t*>(managed_allocation->mapped_addr);
  for (size_t i = 0; i < count; ++i) {
    EXPECT_EQ(managed_words[i], input[i]);
  }

  session.MemsetDeviceD32(fill32_ptr, 0xdeadbeefu, count);
  session.MemcpyDeviceToHost(filled32.data(), fill32_ptr, count * sizeof(uint32_t));
  for (uint32_t value : filled32) {
    EXPECT_EQ(value, 0xdeadbeefu);
  }

  session.MemsetDeviceD16(fill16_ptr, 0xbeefu, count);
  session.MemcpyDeviceToHost(filled16.data(), fill16_ptr, count * sizeof(uint16_t));
  for (uint16_t value : filled16) {
    EXPECT_EQ(value, 0xbeefu);
  }

  session.MemsetDevice(fill8_ptr, 0x5a, count * sizeof(uint32_t));
  session.MemcpyDeviceToHost(filled8.data(), fill8_ptr, filled8.size());
  for (uint8_t value : filled8) {
    EXPECT_EQ(value, 0x5a);
  }

  auto* global_src_offset = reinterpret_cast<std::byte*>(global_src) + sizeof(uint32_t);
  uint32_t offset_word = 0;
  session.MemcpyDeviceToHost(&offset_word, global_src_offset, sizeof(offset_word));
  EXPECT_EQ(offset_word, input[1]);

  auto* managed_dst_offset = reinterpret_cast<std::byte*>(managed_dst) + 2 * sizeof(uint32_t);
  const uint32_t patch_value = 0xc001d00du;
  session.MemcpyHostToDevice(managed_dst_offset, &patch_value, sizeof(patch_value));
  uint32_t patched_word = 0;
  session.MemcpyDeviceToHost(&patched_word, managed_dst_offset, sizeof(patched_word));
  EXPECT_EQ(patched_word, patch_value);
  managed_allocation = session.FindAbiAllocation(managed_dst);
  ASSERT_NE(managed_allocation, nullptr);
  managed_words = reinterpret_cast<const uint32_t*>(managed_allocation->mapped_addr);
  EXPECT_EQ(managed_words[0], input[0]);
  EXPECT_EQ(managed_words[1], input[1]);
  EXPECT_EQ(managed_words[2], patch_value);
  EXPECT_EQ(managed_words[3], input[3]);

  auto* managed_fill_offset =
      reinterpret_cast<std::byte*>(managed_dst) + 3 * sizeof(uint32_t);
  session.MemsetDeviceD32(managed_fill_offset, 0xa5a5a5a5u, 2);
  std::array<uint32_t, count> managed_after_fill{};
  session.MemcpyDeviceToHost(managed_after_fill.data(), managed_dst,
                             managed_after_fill.size() * sizeof(uint32_t));
  EXPECT_EQ(managed_after_fill[0], input[0]);
  EXPECT_EQ(managed_after_fill[1], input[1]);
  EXPECT_EQ(managed_after_fill[2], patch_value);
  EXPECT_EQ(managed_after_fill[3], 0xa5a5a5a5u);
  EXPECT_EQ(managed_after_fill[4], 0xa5a5a5a5u);
}

TEST(RuntimeSessionTest, RejectsUnknownPointersAcrossMemcpyAndMemsetApis) {
  RuntimeSession session;

  std::array<uint32_t, 4> host_words{1u, 2u, 3u, 4u};
  void* valid_ptr = session.AllocateDevice(host_words.size() * sizeof(uint32_t));

  EXPECT_THROW(session.ResolveDeviceAddress(host_words.data()), std::invalid_argument);
  EXPECT_THROW(session.MemcpyHostToDevice(host_words.data(), host_words.data(),
                                          host_words.size() * sizeof(uint32_t)),
               std::invalid_argument);
  EXPECT_THROW(session.MemcpyDeviceToHost(host_words.data(), host_words.data(),
                                          host_words.size() * sizeof(uint32_t)),
               std::invalid_argument);
  EXPECT_THROW(session.MemcpyDeviceToDevice(valid_ptr, host_words.data(),
                                            host_words.size() * sizeof(uint32_t)),
               std::invalid_argument);
  EXPECT_THROW(session.MemcpyDeviceToDevice(host_words.data(), valid_ptr,
                                            host_words.size() * sizeof(uint32_t)),
               std::invalid_argument);
  EXPECT_THROW(session.MemsetDevice(host_words.data(), 0, host_words.size() * sizeof(uint32_t)),
               std::invalid_argument);
  EXPECT_THROW(session.MemsetDeviceD16(host_words.data(), 0xbeefu, host_words.size()),
               std::invalid_argument);
  EXPECT_THROW(session.MemsetDeviceD32(host_words.data(), 0xdeadbeefu, host_words.size()),
               std::invalid_argument);
}

TEST(RuntimeSessionTest, ReleasedPointersLoseAbiAllocationMapping) {
  RuntimeSession session;

  void* ptr = session.AllocateDevice(64);
  ASSERT_NE(ptr, nullptr);
  ASSERT_TRUE(session.HasAbiAllocation(ptr));
  ASSERT_TRUE(session.IsDevicePointer(ptr));
  ASSERT_TRUE(session.FreeDevice(ptr));

  EXPECT_FALSE(session.HasAbiAllocation(ptr));
  EXPECT_FALSE(session.IsDevicePointer(ptr));
  EXPECT_EQ(session.FindAbiAllocation(ptr), nullptr);
  EXPECT_FALSE(session.FreeDevice(ptr));
  EXPECT_THROW(session.ResolveDeviceAddress(ptr), std::invalid_argument);
}

TEST(RuntimeSessionTest, RejectsInteriorFreeWithoutInvalidatingBaseAllocation) {
  RuntimeSession session;

  void* ptr = session.AllocateDevice(64);
  auto* interior = reinterpret_cast<std::byte*>(ptr) + 4;
  ASSERT_TRUE(session.IsDevicePointer(ptr));
  ASSERT_TRUE(session.IsDevicePointer(interior));

  EXPECT_FALSE(session.FreeDevice(interior));
  EXPECT_TRUE(session.IsDevicePointer(ptr));
  EXPECT_EQ(session.FindAbiAllocation(ptr), session.FindAbiAllocation(interior));
  EXPECT_NO_THROW(static_cast<void>(session.ResolveDeviceAddress(ptr)));
  EXPECT_TRUE(session.FreeDevice(ptr));
  EXPECT_FALSE(session.IsDevicePointer(ptr));
}

TEST(DeviceMemoryManagerTest, ReleasedPointersLoseAllocationAndResolvedAddress) {
  MemorySystem memory;
  DeviceMemoryManager manager(&memory);

  constexpr uint64_t model_addr = 0x12345000ull;
  void* ptr = manager.AllocateGlobal(64, model_addr);
  ASSERT_NE(ptr, nullptr);
  ASSERT_TRUE(manager.HasAllocation(ptr));
  EXPECT_EQ(manager.ResolveDeviceAddress(ptr), model_addr);

  ASSERT_TRUE(manager.Free(ptr));
  EXPECT_FALSE(manager.HasAllocation(ptr));
  EXPECT_FALSE(manager.IsDevicePointer(ptr));
  EXPECT_EQ(manager.FindAllocation(ptr), nullptr);
  EXPECT_FALSE(manager.Free(ptr));
  EXPECT_THROW(manager.ResolveDeviceAddress(ptr), std::invalid_argument);
}

}  // namespace
}  // namespace gpu_model
