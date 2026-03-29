#include <gtest/gtest.h>

#include <cstring>

#include "gpu_model/runtime/kernarg_packer.h"

namespace gpu_model {
namespace {

uint32_t LoadU32(const std::vector<std::byte>& bytes, size_t offset) {
  uint32_t value = 0;
  std::memcpy(&value, bytes.data() + offset, sizeof(value));
  return value;
}

uint64_t LoadU64(const std::vector<std::byte>& bytes, size_t offset) {
  uint64_t value = 0;
  std::memcpy(&value, bytes.data() + offset, sizeof(value));
  return value;
}

TEST(KernargPackerTest, PacksVisibleArgsByTypedLayout) {
  KernelLaunchMetadata metadata;
  metadata.arg_layout = {
      KernelArgLayoutEntry{.kind = KernelArgValueKind::GlobalBuffer, .kind_name = "global_buffer", .size = 8},
      KernelArgLayoutEntry{.kind = KernelArgValueKind::ByValue, .kind_name = "by_value", .size = 4},
      KernelArgLayoutEntry{.kind = KernelArgValueKind::ByValue, .kind_name = "by_value", .size = 2},
  };

  KernelArgPack args;
  args.PushU64(0x1122334455667788ull);
  args.PushU32(0xaabbccdd);
  args.PushU32(0xeeff);

  const auto bytes = BuildKernargImage(
      metadata, args,
      LaunchConfig{.grid_dim_x = 1, .grid_dim_y = 1, .grid_dim_z = 1, .block_dim_x = 64});
  EXPECT_EQ(LoadU64(bytes, 0), 0x1122334455667788ull);
  EXPECT_EQ(LoadU32(bytes, 8), 0xaabbccddu);
  EXPECT_EQ(static_cast<uint16_t>(LoadU32(bytes, 12) & 0xffffu), 0xeeffu);
}

TEST(KernargPackerTest, PacksTypedHiddenArgsIncludingDynamicLds) {
  KernelLaunchMetadata metadata;
  metadata.kernarg_segment_size = 64;
  metadata.arg_layout = {
      KernelArgLayoutEntry{.kind = KernelArgValueKind::GlobalBuffer, .kind_name = "global_buffer", .size = 8},
  };
  metadata.hidden_arg_layout = {
      KernelHiddenArgLayoutEntry{.kind = KernelHiddenArgKind::BlockCountX,
                                 .kind_name = "hidden_block_count_x",
                                 .offset = 16,
                                 .size = 4},
      KernelHiddenArgLayoutEntry{.kind = KernelHiddenArgKind::GroupSizeX,
                                 .kind_name = "hidden_group_size_x",
                                 .offset = 20,
                                 .size = 2},
      KernelHiddenArgLayoutEntry{.kind = KernelHiddenArgKind::DynamicLdsSize,
                                 .kind_name = "hidden_dynamic_lds_size",
                                 .offset = 24,
                                 .size = 4},
  };

  KernelArgPack args;
  args.PushU64(0x1234);
  const auto bytes = BuildKernargImage(
      metadata, args,
      LaunchConfig{.grid_dim_x = 7,
                   .grid_dim_y = 1,
                   .grid_dim_z = 1,
                   .block_dim_x = 128,
                   .block_dim_y = 1,
                   .block_dim_z = 1,
                   .shared_memory_bytes = 512});
  EXPECT_EQ(LoadU64(bytes, 0), 0x1234u);
  EXPECT_EQ(LoadU32(bytes, 16), 7u);
  EXPECT_EQ(static_cast<uint16_t>(LoadU32(bytes, 20) & 0xffffu), 128u);
  EXPECT_EQ(LoadU32(bytes, 24), 512u);
}

TEST(KernargPackerTest, FallsBackToDefaultImplicitHiddenArgsWhenLayoutIsAbsent) {
  KernelLaunchMetadata metadata;
  metadata.arg_layout = {
      KernelArgLayoutEntry{.kind = KernelArgValueKind::GlobalBuffer, .kind_name = "global_buffer", .size = 8},
      KernelArgLayoutEntry{.kind = KernelArgValueKind::ByValue, .kind_name = "by_value", .size = 4},
  };

  KernelArgPack args;
  args.PushU64(0x99);
  args.PushU32(17);

  const auto bytes = BuildKernargImage(
      metadata, args,
      LaunchConfig{.grid_dim_x = 3,
                   .grid_dim_y = 2,
                   .grid_dim_z = 1,
                   .block_dim_x = 64,
                   .block_dim_y = 4,
                   .block_dim_z = 1});
  EXPECT_EQ(LoadU64(bytes, 0), 0x99u);
  EXPECT_EQ(LoadU32(bytes, 8), 17u);
  EXPECT_EQ(LoadU32(bytes, 16), 3u);
  EXPECT_EQ(LoadU32(bytes, 20), 2u);
  EXPECT_EQ(LoadU32(bytes, 24), 1u);
  EXPECT_EQ(static_cast<uint16_t>(LoadU32(bytes, 28) & 0xffffu), 64u);
  EXPECT_EQ(static_cast<uint16_t>(LoadU32(bytes, 30) & 0xffffu), 4u);
  EXPECT_EQ(static_cast<uint16_t>(LoadU32(bytes, 32) & 0xffffu), 1u);
}

}  // namespace
}  // namespace gpu_model
