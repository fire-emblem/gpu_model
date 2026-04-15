#include <gtest/gtest.h>

#include <array>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "runtime/model_runtime/runtime_trace_state.h"
#include "runtime/model_runtime/runtime_session.h"

namespace gpu_model {
namespace {

class ScopedEnvUnset {
 public:
  explicit ScopedEnvUnset(const char* name) : name_(name) {
    if (const char* current = std::getenv(name_); current != nullptr) {
      had_value_ = true;
      value_ = current;
      ::unsetenv(name_);
    }
  }

  ~ScopedEnvUnset() {
    if (had_value_) {
      ::setenv(name_, value_.c_str(), 1);
    }
  }

 private:
  const char* name_;
  bool had_value_ = false;
  std::string value_;
};

class ScopedEnvSet {
 public:
  ScopedEnvSet(const char* name, std::string value) : name_(name) {
    if (const char* current = std::getenv(name_); current != nullptr) {
      had_value_ = true;
      value_before_ = current;
    }
    ::setenv(name_, value.c_str(), 1);
  }

  ~ScopedEnvSet() {
    if (had_value_) {
      ::setenv(name_, value_before_.c_str(), 1);
    } else {
      ::unsetenv(name_);
    }
  }

 private:
  const char* name_;
  bool had_value_ = false;
  std::string value_before_;
};

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

TEST(RuntimeTraceStateTest, TracksLaunchIndexAndTraceRecorderStateIndependently) {
  ScopedEnvUnset unset_disable_trace("GPU_MODEL_DISABLE_TRACE");
  ScopedEnvUnset unset_trace_dir("GPU_MODEL_TRACE_DIR");
  RuntimeTraceState state;

  EXPECT_EQ(state.NextLaunchIndex(), 0u);
  EXPECT_EQ(state.NextLaunchIndex(), 1u);
  EXPECT_EQ(state.ResolveTraceArtifactRecorderFromEnv(), nullptr);
  EXPECT_EQ(state.NextLaunchIndex(), 2u);

  state.Reset();
  EXPECT_EQ(state.NextLaunchIndex(), 0u);
}

TEST(RuntimeTraceStateTest, EnablesTraceRecorderOnlyWhenEnvExplicitlyRequestsIt) {
  ScopedEnvSet enable_trace("GPU_MODEL_DISABLE_TRACE", "0");
  const auto trace_dir =
      std::filesystem::temp_directory_path() / "gpu_model_runtime_session_trace_state_test";
  ScopedEnvSet set_trace_dir("GPU_MODEL_TRACE_DIR", trace_dir.string());
  RuntimeTraceState state;

  auto* trace = state.ResolveTraceArtifactRecorderFromEnv();
  ASSERT_NE(trace, nullptr);
  EXPECT_EQ(trace, state.ResolveTraceArtifactRecorderFromEnv());

  ScopedEnvSet disable_trace("GPU_MODEL_DISABLE_TRACE", "1");
  EXPECT_EQ(state.ResolveTraceArtifactRecorderFromEnv(), nullptr);
}

TEST(RuntimeSessionTest, PushesAndPopsPendingLaunchConfigThroughDedicatedState) {
  RuntimeSession session;
  const LaunchConfig first{
      .grid_dim_x = 2,
      .grid_dim_y = 3,
      .block_dim_x = 64,
      .shared_memory_bytes = 128,
  };
  const LaunchConfig second{
      .grid_dim_x = 7,
      .block_dim_x = 256,
      .shared_memory_bytes = 64,
  };

  EXPECT_FALSE(session.PopLaunchConfig().has_value());

  session.PushLaunchConfig(first);
  auto popped = session.PopLaunchConfig();
  ASSERT_TRUE(popped.has_value());
  EXPECT_EQ(popped->grid_dim_x, first.grid_dim_x);
  EXPECT_EQ(popped->grid_dim_y, first.grid_dim_y);
  EXPECT_EQ(popped->block_dim_x, first.block_dim_x);
  EXPECT_EQ(popped->shared_memory_bytes, first.shared_memory_bytes);
  EXPECT_FALSE(session.PopLaunchConfig().has_value());

  session.PushLaunchConfig(first);
  session.PushLaunchConfig(second);
  popped = session.PopLaunchConfig();
  ASSERT_TRUE(popped.has_value());
  EXPECT_EQ(popped->grid_dim_x, second.grid_dim_x);
  EXPECT_EQ(popped->block_dim_x, second.block_dim_x);
  EXPECT_EQ(popped->shared_memory_bytes, second.shared_memory_bytes);
}

TEST(RuntimeSessionTest, ResetClearsPendingLaunchConfigWithoutTouchingOtherHelpers) {
  ScopedEnvUnset unset_disable_trace("GPU_MODEL_DISABLE_TRACE");
  ScopedEnvUnset unset_trace_dir("GPU_MODEL_TRACE_DIR");
  RuntimeSession session;

  session.PushLaunchConfig(LaunchConfig{.grid_dim_x = 4, .block_dim_x = 128});

  session.ResetAbiState();

  EXPECT_FALSE(session.PopLaunchConfig().has_value());
}

TEST(RuntimeSessionTest, RegistersAndResolvesKernelSymbolsThroughDedicatedState) {
  RuntimeSession session;
  int host_symbol_a = 1;
  int host_symbol_b = 2;

  EXPECT_FALSE(session.ResolveKernelSymbol(&host_symbol_a).has_value());

  session.RegisterKernelSymbol(&host_symbol_a, "kernel_a");
  session.RegisterKernelSymbol(&host_symbol_b, "kernel_b");

  auto kernel_a = session.ResolveKernelSymbol(&host_symbol_a);
  auto kernel_b = session.ResolveKernelSymbol(&host_symbol_b);
  ASSERT_TRUE(kernel_a.has_value());
  ASSERT_TRUE(kernel_b.has_value());
  EXPECT_EQ(*kernel_a, "kernel_a");
  EXPECT_EQ(*kernel_b, "kernel_b");

  session.RegisterKernelSymbol(&host_symbol_a, "kernel_a_v2");
  kernel_a = session.ResolveKernelSymbol(&host_symbol_a);
  ASSERT_TRUE(kernel_a.has_value());
  EXPECT_EQ(*kernel_a, "kernel_a_v2");
}

TEST(RuntimeSessionTest, ResetClearsRegisteredKernelSymbols) {
  RuntimeSession session;
  int host_symbol = 7;

  session.RegisterKernelSymbol(&host_symbol, "kernel_before_reset");
  ASSERT_TRUE(session.ResolveKernelSymbol(&host_symbol).has_value());

  session.ResetAbiState();

  EXPECT_FALSE(session.ResolveKernelSymbol(&host_symbol).has_value());
}

TEST(RuntimeSessionTest, LastErrorStateIsPeekableAndConsumable) {
  RuntimeSession session;

  EXPECT_EQ(session.PeekLastError(), 0);
  EXPECT_EQ(session.ConsumeLastError(), 0);

  session.SetLastError(17);
  EXPECT_EQ(session.PeekLastError(), 17);
  EXPECT_EQ(session.PeekLastError(), 17);
  EXPECT_EQ(session.ConsumeLastError(), 17);
  EXPECT_EQ(session.PeekLastError(), 0);
  EXPECT_EQ(session.ConsumeLastError(), 0);
}

TEST(RuntimeSessionTest, ResetAbiStateDoesNotClearLastErrorChannel) {
  RuntimeSession session;

  session.SetLastError(29);
  session.ResetAbiState();

  EXPECT_EQ(session.PeekLastError(), 29);
  EXPECT_EQ(session.ConsumeLastError(), 29);
  EXPECT_EQ(session.PeekLastError(), 0);
}

TEST(RuntimeSessionTest, ParsesAndPacksAbiArgsThroughDedicatedPacker) {
  RuntimeSession session;
  MetadataBlob metadata{.values = {
                            {"arg_layout", "global_buffer:8,by_value:4,by_value:8,by_value:16:12"},
                        }};

  const auto layout = session.ParseAbiArgLayout(metadata);
  ASSERT_EQ(layout.size(), 4u);
  EXPECT_EQ(layout[0].kind, HipRuntimeAbiArgKind::GlobalBuffer);
  EXPECT_EQ(layout[0].size, 8u);
  EXPECT_EQ(layout[1].kind, HipRuntimeAbiArgKind::ByValue);
  EXPECT_EQ(layout[1].size, 4u);
  EXPECT_EQ(layout[2].size, 8u);
  EXPECT_EQ(layout[3].size, 12u);

  void* device_ptr = session.AllocateDevice(64);
  uint32_t scalar32 = 0x11223344u;
  uint64_t scalar64 = 0x5566778899aabbccull;
  struct AggregateArg {
    uint32_t x;
    uint32_t y;
    uint32_t z;
  } aggregate{7u, 8u, 9u};

  void* arg_ptrs[] = {&device_ptr, &scalar32, &scalar64, &aggregate};
  const auto packed = session.PackAbiArgs(metadata, arg_ptrs);
  ASSERT_EQ(packed.size(), 4u);

  uint64_t packed_device_addr = 0;
  std::memcpy(&packed_device_addr, packed.bytes(0).data(), sizeof(packed_device_addr));
  EXPECT_EQ(packed_device_addr, session.ResolveDeviceAddress(device_ptr));

  uint32_t packed_u32 = 0;
  std::memcpy(&packed_u32, packed.bytes(1).data(), sizeof(packed_u32));
  EXPECT_EQ(packed_u32, scalar32);

  uint64_t packed_u64 = 0;
  std::memcpy(&packed_u64, packed.bytes(2).data(), sizeof(packed_u64));
  EXPECT_EQ(packed_u64, scalar64);

  AggregateArg packed_aggregate{};
  ASSERT_EQ(packed.bytes(3).size(), sizeof(AggregateArg));
  std::memcpy(&packed_aggregate, packed.bytes(3).data(), sizeof(AggregateArg));
  EXPECT_EQ(packed_aggregate.x, aggregate.x);
  EXPECT_EQ(packed_aggregate.y, aggregate.y);
  EXPECT_EQ(packed_aggregate.z, aggregate.z);
}

TEST(RuntimeSessionTest, PackAbiArgsRejectsMissingPointers) {
  RuntimeSession session;
  MetadataBlob metadata{.values = {
                            {"arg_layout", "by_value:4"},
                        }};

  EXPECT_THROW(session.PackAbiArgs(metadata, nullptr), std::invalid_argument);
  void* null_args[] = {nullptr};
  EXPECT_THROW(session.PackAbiArgs(metadata, null_args), std::invalid_argument);
}

TEST(RuntimeSessionTest, CurrentExecutablePathResolvesExistingBinary) {
  const auto path = RuntimeSession::CurrentExecutablePath();
  EXPECT_FALSE(path.empty());
  EXPECT_TRUE(std::filesystem::exists(path));
  EXPECT_TRUE(std::filesystem::is_regular_file(path));
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
