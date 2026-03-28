#include "gpu_model/loader/device_segment_image.h"

#include <sstream>

namespace gpu_model {

namespace {

uint32_t ParseU32Metadata(const MetadataBlob& metadata, const std::string& key) {
  const auto it = metadata.values.find(key);
  if (it == metadata.values.end() || it->second.empty()) {
    return 0;
  }
  return static_cast<uint32_t>(std::stoul(it->second));
}

uint32_t EstimateKernargBytes(const MetadataBlob& metadata) {
  const auto it = metadata.values.find("arg_layout");
  if (it == metadata.values.end() || it->second.empty()) {
    return 0;
  }
  std::istringstream input(it->second);
  std::string token;
  uint32_t total = 0;
  while (std::getline(input, token, ',')) {
    const auto colon = token.rfind(':');
    if (colon == std::string::npos) {
      continue;
    }
    total += static_cast<uint32_t>(std::stoul(token.substr(colon + 1)));
  }
  return total;
}

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

DeviceLoadPlan BuildDeviceLoadPlan(const ProgramImage& image) {
  DeviceLoadPlan plan;
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
  plan.required_shared_bytes = ParseU32Metadata(image.metadata(), "required_shared_bytes");
  plan.preferred_kernarg_bytes = EstimateKernargBytes(image.metadata());
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

DeviceLoadPlan BuildDeviceLoadPlan(const AmdgpuCodeObjectImage& image) {
  DeviceLoadPlan plan;
  plan.segments.push_back(MakeCodeSegment(image.kernel_name + ".text", image.code_bytes));
  plan.required_shared_bytes = ParseU32Metadata(image.metadata, "group_segment_fixed_size");
  plan.preferred_kernarg_bytes = EstimateKernargBytes(image.metadata);
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
