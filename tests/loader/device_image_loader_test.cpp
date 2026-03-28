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
  EXPECT_EQ(memory.LoadGlobalValue<uint8_t>(result.segments[0].allocation.range.base + 0), 0xaau);
  EXPECT_EQ(memory.LoadGlobalValue<uint8_t>(result.segments[0].allocation.range.base + 1), 0xbbu);
  EXPECT_EQ(memory.LoadGlobalValue<uint8_t>(result.segments[1].allocation.range.base + 0), 0x11u);
  EXPECT_EQ(memory.LoadGlobalValue<uint8_t>(result.segments[1].allocation.range.base + 1), 0x22u);
  EXPECT_EQ(result.segments[1].allocation.range.size, 8u);
}

}  // namespace
}  // namespace gpu_model
