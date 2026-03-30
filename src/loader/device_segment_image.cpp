#include "gpu_model/loader/device_segment_image.h"

#include "gpu_model/isa/kernel_metadata.h"

namespace gpu_model {

namespace {

DeviceSegmentImage MakeCodeSegment(std::string name, std::vector<std::byte> bytes) {
  const uint64_t required_bytes = bytes.size();
  return DeviceSegmentImage{
      .kind = DeviceSegmentKind::Code,
      .pool = MemoryPoolKind::Code,
      .mapping = MemoryMappingKind::Copy,
      .name = std::move(name),
      .alignment = 256,
      .bytes = std::move(bytes),
      .required_bytes = required_bytes,
  };
}

DeviceSegmentImage MakeConstSegment(const ConstSegment& const_segment) {
  return DeviceSegmentImage{
      .kind = DeviceSegmentKind::ConstantData,
      .pool = MemoryPoolKind::Constant,
      .mapping = MemoryMappingKind::Copy,
      .name = "const_segment",
      .alignment = 16,
      .bytes = const_segment.bytes,
      .required_bytes = const_segment.bytes.size(),
  };
}

DeviceSegmentImage MakeRawDataSegment(const RawDataSegment& raw_data_segment) {
  return DeviceSegmentImage{
      .kind = DeviceSegmentKind::RawData,
      .pool = MemoryPoolKind::RawData,
      .mapping = MemoryMappingKind::Copy,
      .name = "raw_data_segment",
      .alignment = 16,
      .bytes = raw_data_segment.bytes,
      .required_bytes = raw_data_segment.bytes.size(),
  };
}

}  // namespace

DeviceLoadPlan BuildDeviceLoadPlan(const ProgramObject& image) {
  DeviceLoadPlan plan;
  const auto metadata = ParseKernelLaunchMetadata(image.metadata());
  const auto code_bytes = std::vector<std::byte>(
      reinterpret_cast<const std::byte*>(image.assembly_text().data()),
      reinterpret_cast<const std::byte*>(image.assembly_text().data()) + image.assembly_text().size());
  plan.segments.push_back(MakeCodeSegment(image.kernel_name() + ".asm", code_bytes));
  if (!image.const_segment().bytes.empty()) {
    plan.segments.push_back(MakeConstSegment(image.const_segment()));
  }
  if (!image.raw_data_segment().bytes.empty()) {
    plan.segments.push_back(MakeRawDataSegment(image.raw_data_segment()));
  }
  plan.required_shared_bytes = metadata.required_shared_bytes.value_or(0);
  plan.preferred_kernarg_bytes = RequiredKernargTemplateBytes(metadata);
  if (plan.preferred_kernarg_bytes != 0) {
    plan.segments.push_back(DeviceSegmentImage{
        .kind = DeviceSegmentKind::KernargTemplate,
        .pool = MemoryPoolKind::Kernarg,
        .mapping = MemoryMappingKind::ZeroFill,
        .name = image.kernel_name() + ".kernarg",
        .alignment = 16,
        .bytes = {},
        .required_bytes = plan.preferred_kernarg_bytes,
    });
  }
  return plan;
}

DeviceLoadPlan BuildDeviceLoadPlan(const EncodedProgramObject& image) {
  DeviceLoadPlan plan;
  const auto metadata = ParseKernelLaunchMetadata(image.metadata);
  plan.segments.push_back(MakeCodeSegment(image.kernel_name + ".text", image.code_bytes));
  plan.required_shared_bytes = metadata.required_shared_bytes.value_or(0);
  plan.preferred_kernarg_bytes = RequiredKernargTemplateBytes(metadata);
  if (plan.preferred_kernarg_bytes != 0) {
    plan.segments.push_back(DeviceSegmentImage{
        .kind = DeviceSegmentKind::KernargTemplate,
        .pool = MemoryPoolKind::Kernarg,
        .mapping = MemoryMappingKind::ZeroFill,
        .name = image.kernel_name + ".kernarg",
        .alignment = 16,
        .bytes = {},
        .required_bytes = plan.preferred_kernarg_bytes,
    });
  }
  return plan;
}

}  // namespace gpu_model
