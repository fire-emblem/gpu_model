#include <gtest/gtest.h>

#include "gpu_model/loader/device_image_loader.h"

namespace gpu_model {
namespace {

TEST(DeviceImageLoaderTest, MaterializesSegmentsIntoMemorySystem) {
  DeviceLoadPlan plan;
  plan.required_shared_bytes = 64;
  plan.preferred_kernarg_bytes = 24;
  plan.segments.push_back(DeviceSegmentImage{
      .kind = DeviceSegmentKind::Code,
      .pool = MemoryPoolKind::Code,
      .mapping = MemoryMappingKind::Copy,
      .name = "code",
      .alignment = 16,
      .bytes = {std::byte{0xaa}, std::byte{0xbb}, std::byte{0xcc}},
      .required_bytes = 3,
  });
  plan.segments.push_back(DeviceSegmentImage{
      .kind = DeviceSegmentKind::ConstantData,
      .pool = MemoryPoolKind::Constant,
      .mapping = MemoryMappingKind::Copy,
      .name = "const",
      .alignment = 8,
      .bytes = {std::byte{0x11}, std::byte{0x22}},
      .required_bytes = 8,
  });

  MemorySystem memory;
  const auto result = DeviceImageLoader{}.Materialize(plan, memory);
  ASSERT_EQ(result.segments.size(), 2u);
  EXPECT_EQ(result.required_shared_bytes, 64u);
  EXPECT_EQ(result.preferred_kernarg_bytes, 24u);
  EXPECT_EQ(result.segments[0].allocation.pool, MemoryPoolKind::Code);
  EXPECT_EQ(result.segments[1].allocation.pool, MemoryPoolKind::Constant);
  EXPECT_EQ(memory.LoadValue<uint8_t>(MemoryPoolKind::Code,
                                      result.segments[0].allocation.range.base + 0),
            0xaau);
  EXPECT_EQ(memory.LoadValue<uint8_t>(MemoryPoolKind::Code,
                                      result.segments[0].allocation.range.base + 1),
            0xbbu);
  EXPECT_EQ(memory.LoadValue<uint8_t>(MemoryPoolKind::Constant,
                                      result.segments[1].allocation.range.base + 0),
            0x11u);
  EXPECT_EQ(memory.LoadValue<uint8_t>(MemoryPoolKind::Constant,
                                      result.segments[1].allocation.range.base + 1),
            0x22u);
  EXPECT_EQ(result.segments[1].allocation.range.size, 8u);
  EXPECT_EQ(memory.pool_memory_size(MemoryPoolKind::Global), 0u);
  EXPECT_GT(memory.pool_memory_size(MemoryPoolKind::Code), 0u);
  EXPECT_GT(memory.pool_memory_size(MemoryPoolKind::Constant), 0u);
}

TEST(DeviceImageLoaderTest, MaterializesZeroFillKernargIntoDedicatedPool) {
  DeviceLoadPlan plan;
  plan.preferred_kernarg_bytes = 24;
  plan.segments.push_back(DeviceSegmentImage{
      .kind = DeviceSegmentKind::KernargTemplate,
      .pool = MemoryPoolKind::Kernarg,
      .mapping = MemoryMappingKind::ZeroFill,
      .name = "kernarg",
      .alignment = 16,
      .bytes = {},
      .required_bytes = 24,
  });

  MemorySystem memory;
  const auto result = DeviceImageLoader{}.Materialize(plan, memory);
  ASSERT_EQ(result.segments.size(), 1u);
  EXPECT_EQ(result.segments[0].allocation.pool, MemoryPoolKind::Kernarg);
  EXPECT_EQ(result.segments[0].allocation.range.size, 24u);
  EXPECT_EQ(memory.pool_memory_size(MemoryPoolKind::Global), 0u);
  EXPECT_EQ(memory.pool_memory_size(MemoryPoolKind::Kernarg), 24u);
  for (uint64_t i = 0; i < 24u; ++i) {
    EXPECT_EQ(memory.LoadValue<uint8_t>(MemoryPoolKind::Kernarg, i), 0u);
  }
}

TEST(DeviceImageLoaderTest, MaterializesRawDataIntoRawDataPool) {
  DeviceLoadPlan plan;
  plan.segments.push_back(DeviceSegmentImage{
      .kind = DeviceSegmentKind::RawData,
      .pool = MemoryPoolKind::RawData,
      .mapping = MemoryMappingKind::Copy,
      .name = "raw_data",
      .alignment = 4,
      .bytes = {std::byte{0x91}, std::byte{0x92}, std::byte{0x93}},
      .required_bytes = 3,
  });

  MemorySystem memory;
  const auto result = DeviceImageLoader{}.Materialize(plan, memory);
  ASSERT_EQ(result.segments.size(), 1u);
  EXPECT_EQ(result.segments[0].allocation.pool, MemoryPoolKind::RawData);
  EXPECT_EQ(memory.pool_memory_size(MemoryPoolKind::RawData), 3u);
  EXPECT_EQ(memory.LoadValue<uint8_t>(MemoryPoolKind::RawData, 0), 0x91u);
  EXPECT_EQ(memory.LoadValue<uint8_t>(MemoryPoolKind::RawData, 2), 0x93u);
}

}  // namespace
}  // namespace gpu_model
