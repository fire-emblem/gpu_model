#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstring>

#include "gpu_arch/memory/memory_pool.h"
#include "runtime/config/kernarg_packer.h"

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
      KernelArgLayoutEntry{
          .kind = KernelArgValueKind::GlobalBuffer, .kind_name = "global_buffer", .offset = std::nullopt, .size = 8},
      KernelArgLayoutEntry{
          .kind = KernelArgValueKind::ByValue, .kind_name = "by_value", .offset = std::nullopt, .size = 4},
      KernelArgLayoutEntry{
          .kind = KernelArgValueKind::ByValue, .kind_name = "by_value", .offset = std::nullopt, .size = 2},
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
      KernelArgLayoutEntry{
          .kind = KernelArgValueKind::GlobalBuffer, .kind_name = "global_buffer", .offset = std::nullopt, .size = 8},
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

TEST(KernargPackerTest, PacksThreeDimensionalTypedHiddenArgs) {
  KernelLaunchMetadata metadata;
  metadata.kernarg_segment_size = 64;
  metadata.hidden_arg_layout = {
      KernelHiddenArgLayoutEntry{.kind = KernelHiddenArgKind::BlockCountZ,
                                 .kind_name = "hidden_block_count_z",
                                 .offset = 8,
                                 .size = 4},
      KernelHiddenArgLayoutEntry{.kind = KernelHiddenArgKind::GroupSizeZ,
                                 .kind_name = "hidden_group_size_z",
                                 .offset = 12,
                                 .size = 2},
      KernelHiddenArgLayoutEntry{.kind = KernelHiddenArgKind::GridDims,
                                 .kind_name = "hidden_grid_dims",
                                 .offset = 16,
                                 .size = 2},
  };

  const auto bytes = BuildKernargImage(
      metadata, {},
      LaunchConfig{.grid_dim_x = 2,
                   .grid_dim_y = 3,
                   .grid_dim_z = 4,
                   .block_dim_x = 8,
                   .block_dim_y = 16,
                   .block_dim_z = 32});
  EXPECT_EQ(LoadU32(bytes, 8), 4u);
  EXPECT_EQ(static_cast<uint16_t>(LoadU32(bytes, 12) & 0xffffu), 32u);
  EXPECT_EQ(static_cast<uint16_t>(LoadU32(bytes, 16) & 0xffffu), 3u);
}

TEST(KernargPackerTest, PacksApertureBaseHiddenArgs) {
  KernelLaunchMetadata metadata;
  metadata.kernarg_segment_size = 32;
  metadata.hidden_arg_layout = {
      KernelHiddenArgLayoutEntry{.kind = KernelHiddenArgKind::PrivateBase,
                                 .kind_name = "hidden_private_base",
                                 .offset = 8,
                                 .size = 4},
      KernelHiddenArgLayoutEntry{.kind = KernelHiddenArgKind::SharedBase,
                                 .kind_name = "hidden_shared_base",
                                 .offset = 12,
                                 .size = 4},
  };

  const auto bytes = BuildKernargImage(metadata, {}, {});
  EXPECT_EQ(LoadU32(bytes, 8), MemoryPoolBaseUpper32(MemoryPoolKind::Private));
  EXPECT_EQ(LoadU32(bytes, 12), MemoryPoolBaseUpper32(MemoryPoolKind::Shared));
}

TEST(KernargPackerTest, FallsBackToDefaultImplicitHiddenArgsWhenLayoutIsAbsent) {
  KernelLaunchMetadata metadata;
  metadata.arg_layout = {
      KernelArgLayoutEntry{
          .kind = KernelArgValueKind::GlobalBuffer, .kind_name = "global_buffer", .offset = std::nullopt, .size = 8},
      KernelArgLayoutEntry{
          .kind = KernelArgValueKind::ByValue, .kind_name = "by_value", .offset = std::nullopt, .size = 4},
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

TEST(KernargPackerTest, FallsBackToThreeDimensionalImplicitHiddenArgs) {
  KernelLaunchMetadata metadata;
  metadata.arg_layout = {
      KernelArgLayoutEntry{
          .kind = KernelArgValueKind::GlobalBuffer, .kind_name = "global_buffer", .offset = std::nullopt, .size = 8},
  };

  KernelArgPack args;
  args.PushU64(0x88);

  const auto bytes = BuildKernargImage(
      metadata, args,
      LaunchConfig{.grid_dim_x = 2,
                   .grid_dim_y = 3,
                   .grid_dim_z = 4,
                   .block_dim_x = 8,
                   .block_dim_y = 16,
                   .block_dim_z = 32});
  EXPECT_EQ(LoadU64(bytes, 0), 0x88u);
  EXPECT_EQ(LoadU32(bytes, 8), 2u);
  EXPECT_EQ(LoadU32(bytes, 12), 3u);
  EXPECT_EQ(LoadU32(bytes, 16), 4u);
  EXPECT_EQ(static_cast<uint16_t>(LoadU32(bytes, 20) & 0xffffu), 8u);
  EXPECT_EQ(static_cast<uint16_t>(LoadU32(bytes, 22) & 0xffffu), 16u);
  EXPECT_EQ(static_cast<uint16_t>(LoadU32(bytes, 24) & 0xffffu), 32u);
}

TEST(KernargPackerTest, PacksByValueAggregateWithoutScalarOnlyRestriction) {
  KernelLaunchMetadata metadata;
  metadata.kernarg_segment_size = 48;
  metadata.arg_layout = {
      KernelArgLayoutEntry{
          .kind = KernelArgValueKind::GlobalBuffer, .kind_name = "global_buffer", .offset = std::nullopt, .size = 8},
      KernelArgLayoutEntry{
          .kind = KernelArgValueKind::ByValue, .kind_name = "by_value", .offset = std::nullopt, .size = 12},
      KernelArgLayoutEntry{
          .kind = KernelArgValueKind::ByValue, .kind_name = "by_value", .offset = std::nullopt, .size = 16},
  };

  const std::array<std::byte, 12> small_aggregate = {
      std::byte{0x01}, std::byte{0x02}, std::byte{0x03}, std::byte{0x04},
      std::byte{0x05}, std::byte{0x06}, std::byte{0x07}, std::byte{0x08},
      std::byte{0x09}, std::byte{0x0a}, std::byte{0x0b}, std::byte{0x0c},
  };
  const std::array<std::byte, 16> large_aggregate = {
      std::byte{0x10}, std::byte{0x11}, std::byte{0x12}, std::byte{0x13},
      std::byte{0x14}, std::byte{0x15}, std::byte{0x16}, std::byte{0x17},
      std::byte{0x18}, std::byte{0x19}, std::byte{0x1a}, std::byte{0x1b},
      std::byte{0x1c}, std::byte{0x1d}, std::byte{0x1e}, std::byte{0x1f},
  };

  KernelArgPack args;
  args.PushU64(0x1122334455667788ull);
  args.PushBytes(small_aggregate);
  args.PushBytes(large_aggregate);

  const auto bytes = BuildKernargImage(
      metadata, args,
      LaunchConfig{.grid_dim_x = 1, .grid_dim_y = 1, .grid_dim_z = 1, .block_dim_x = 64});
  EXPECT_EQ(LoadU64(bytes, 0), 0x1122334455667788ull);
  EXPECT_TRUE(std::equal(small_aggregate.begin(), small_aggregate.end(), bytes.begin() + 8));
  EXPECT_TRUE(std::equal(large_aggregate.begin(), large_aggregate.end(), bytes.begin() + 20));
}

TEST(KernargPackerTest, HonorsVisibleArgOffsetsForPaddedAggregateLayout) {
  KernelLaunchMetadata metadata;
  metadata.kernarg_segment_size = 32;
  metadata.arg_layout = {
      KernelArgLayoutEntry{
          .kind = KernelArgValueKind::GlobalBuffer, .kind_name = "global_buffer", .offset = std::nullopt, .size = 8},
      KernelArgLayoutEntry{
          .kind = KernelArgValueKind::ByValue, .kind_name = "by_value", .offset = 16, .size = 12},
  };

  const std::array<std::byte, 12> aggregate = {
      std::byte{0x21}, std::byte{0x22}, std::byte{0x23}, std::byte{0x24},
      std::byte{0x25}, std::byte{0x26}, std::byte{0x27}, std::byte{0x28},
      std::byte{0x29}, std::byte{0x2a}, std::byte{0x2b}, std::byte{0x2c},
  };

  KernelArgPack args;
  args.PushU64(0x0102030405060708ull);
  args.PushBytes(aggregate);

  const auto bytes = BuildKernargImage(
      metadata, args,
      LaunchConfig{.grid_dim_x = 1, .grid_dim_y = 1, .grid_dim_z = 1, .block_dim_x = 64});
  EXPECT_EQ(LoadU64(bytes, 0), 0x0102030405060708ull);
  for (size_t i = 8; i < 16; ++i) {
    EXPECT_EQ(bytes[i], std::byte{0});
  }
  EXPECT_TRUE(std::equal(aggregate.begin(), aggregate.end(), bytes.begin() + 16));
}

TEST(KernargPackerTest, FallsBackToActualVisibleArgSizesWhenLayoutIsAbsent) {
  KernelLaunchMetadata metadata;

  const std::array<std::byte, 12> aggregate = {
      std::byte{0x31}, std::byte{0x32}, std::byte{0x33}, std::byte{0x34},
      std::byte{0x35}, std::byte{0x36}, std::byte{0x37}, std::byte{0x38},
      std::byte{0x39}, std::byte{0x3a}, std::byte{0x3b}, std::byte{0x3c},
  };

  KernelArgPack args;
  args.PushU64(0x0102030405060708ull);
  args.PushBytes(aggregate);
  args.PushU64(0x1112131415161718ull);

  const auto bytes = BuildKernargImage(
      metadata, args,
      LaunchConfig{.grid_dim_x = 2,
                   .grid_dim_y = 3,
                   .grid_dim_z = 4,
                   .block_dim_x = 8,
                   .block_dim_y = 16,
                   .block_dim_z = 32});
  EXPECT_EQ(LoadU64(bytes, 0), 0x0102030405060708ull);
  EXPECT_TRUE(std::equal(aggregate.begin(), aggregate.end(), bytes.begin() + 8));
  EXPECT_EQ(LoadU64(bytes, 20), 0x1112131415161718ull);
}

TEST(KernargPackerTest, AlignsFallbackHiddenArgsAfterActualVisibleArgPayload) {
  KernelLaunchMetadata metadata;

  const std::array<std::byte, 12> aggregate = {
      std::byte{0x41}, std::byte{0x42}, std::byte{0x43}, std::byte{0x44},
      std::byte{0x45}, std::byte{0x46}, std::byte{0x47}, std::byte{0x48},
      std::byte{0x49}, std::byte{0x4a}, std::byte{0x4b}, std::byte{0x4c},
  };

  KernelArgPack args;
  args.PushU64(0x99);
  args.PushBytes(aggregate);

  const auto bytes = BuildKernargImage(
      metadata, args,
      LaunchConfig{.grid_dim_x = 5,
                   .grid_dim_y = 6,
                   .grid_dim_z = 7,
                   .block_dim_x = 64,
                   .block_dim_y = 2,
                   .block_dim_z = 1});
  EXPECT_EQ(LoadU64(bytes, 0), 0x99u);
  EXPECT_TRUE(std::equal(aggregate.begin(), aggregate.end(), bytes.begin() + 8));
  EXPECT_EQ(LoadU32(bytes, 24), 5u);
  EXPECT_EQ(LoadU32(bytes, 28), 6u);
  EXPECT_EQ(LoadU32(bytes, 32), 7u);
  EXPECT_EQ(static_cast<uint16_t>(LoadU32(bytes, 36) & 0xffffu), 64u);
  EXPECT_EQ(static_cast<uint16_t>(LoadU32(bytes, 38) & 0xffffu), 2u);
  EXPECT_EQ(static_cast<uint16_t>(LoadU32(bytes, 40) & 0xffffu), 1u);
}

TEST(KernargPackerTest, PacksTypedGlobalOffsetsFromLaunchConfig) {
  KernelLaunchMetadata metadata;
  metadata.kernarg_segment_size = 40;
  metadata.hidden_arg_layout = {
      KernelHiddenArgLayoutEntry{.kind = KernelHiddenArgKind::GlobalOffsetX,
                                 .kind_name = "hidden_global_offset_x",
                                 .offset = 8,
                                 .size = 8},
      KernelHiddenArgLayoutEntry{.kind = KernelHiddenArgKind::GlobalOffsetY,
                                 .kind_name = "hidden_global_offset_y",
                                 .offset = 16,
                                 .size = 8},
      KernelHiddenArgLayoutEntry{.kind = KernelHiddenArgKind::GlobalOffsetZ,
                                 .kind_name = "hidden_global_offset_z",
                                 .offset = 24,
                                 .size = 8},
  };

  const auto bytes = BuildKernargImage(
      metadata, {},
      LaunchConfig{.grid_dim_x = 2,
                   .grid_dim_y = 3,
                   .grid_dim_z = 4,
                   .block_dim_x = 8,
                   .block_dim_y = 16,
                   .block_dim_z = 32,
                   .global_offset_x = 0x1111222233334444ull,
                   .global_offset_y = 0x5555666677778888ull,
                   .global_offset_z = 0x9999aaaabbbbccccull});
  EXPECT_EQ(LoadU64(bytes, 8), 0x1111222233334444ull);
  EXPECT_EQ(LoadU64(bytes, 16), 0x5555666677778888ull);
  EXPECT_EQ(LoadU64(bytes, 24), 0x9999aaaabbbbccccull);
}

TEST(KernargPackerTest, PacksTypedQueuePointerFromLaunchConfig) {
  KernelLaunchMetadata metadata;
  metadata.kernarg_segment_size = 24;
  metadata.hidden_arg_layout = {
      KernelHiddenArgLayoutEntry{.kind = KernelHiddenArgKind::QueuePtr,
                                 .kind_name = "hidden_default_queue",
                                 .offset = 8,
                                 .size = 8},
  };

  const auto bytes = BuildKernargImage(
      metadata, {},
      LaunchConfig{.grid_dim_x = 1,
                   .block_dim_x = 64,
                   .queue_ptr = 0x123456789abcdef0ull});
  EXPECT_EQ(LoadU64(bytes, 8), 0x123456789abcdef0ull);
}

}  // namespace
}  // namespace gpu_model
