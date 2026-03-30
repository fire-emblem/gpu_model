#include <gtest/gtest.h>

#include "gpu_model/loader/device_segment_image.h"
#include "gpu_model/program/program_object.h"

namespace gpu_model {
namespace {

TEST(DeviceSegmentImageTest, BuildsPlanForProgramObjectWithConstAndSharedMetadata) {
  ProgramObject image("kernel_a", "s_endpgm\n",
                      MetadataBlob{.values = {{"required_shared_bytes", "256"},
                                              {"arg_layout", "global_buffer:8,by_value:4"}}},
                      ConstSegment{.bytes = {std::byte{0x1}, std::byte{0x2}}},
                      RawDataSegment{.bytes = {std::byte{0x9}, std::byte{0xa}, std::byte{0xb}}});

  const auto plan = BuildDeviceLoadPlan(image);
  ASSERT_EQ(plan.segments.size(), 4u);
  EXPECT_EQ(plan.segments[0].pool, MemoryPoolKind::Code);
  EXPECT_EQ(plan.segments[1].pool, MemoryPoolKind::Constant);
  EXPECT_EQ(plan.segments[2].pool, MemoryPoolKind::RawData);
  EXPECT_EQ(plan.segments[3].pool, MemoryPoolKind::Kernarg);
  EXPECT_EQ(plan.segments[3].kind, DeviceSegmentKind::KernargTemplate);
  EXPECT_EQ(plan.segments[3].mapping, MemoryMappingKind::ZeroFill);
  EXPECT_EQ(plan.required_shared_bytes, 256u);
  EXPECT_EQ(plan.preferred_kernarg_bytes, 128u);
}

TEST(DeviceSegmentImageTest, BuildsPlanForEncodedProgramObject) {
  EncodedProgramObject image;
  image.kernel_name = "kernel_b";
  image.metadata.values["group_segment_fixed_size"] = "128";
  image.metadata.values["arg_layout"] = "global_buffer:8,global_buffer:8,by_value:4";
  image.code_bytes = {std::byte{0xde}, std::byte{0xad}, std::byte{0xbe}, std::byte{0xef}};

  const auto plan = BuildDeviceLoadPlan(image);
  ASSERT_EQ(plan.segments.size(), 2u);
  EXPECT_EQ(plan.segments[0].pool, MemoryPoolKind::Code);
  EXPECT_EQ(plan.segments[0].required_bytes, 4u);
  EXPECT_EQ(plan.segments[1].pool, MemoryPoolKind::Kernarg);
  EXPECT_EQ(plan.segments[1].required_bytes, 128u);
  EXPECT_EQ(plan.required_shared_bytes, 128u);
  EXPECT_EQ(plan.preferred_kernarg_bytes, 128u);
}

}  // namespace
}  // namespace gpu_model
