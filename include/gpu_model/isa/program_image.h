#pragma once

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "gpu_model/isa/metadata.h"

namespace gpu_model {

struct ConstSegment {
  std::vector<std::byte> bytes;
};

struct RawDataSegment {
  std::vector<std::byte> bytes;
};

class ProgramImage {
 public:
  ProgramImage() = default;
  ProgramImage(std::string kernel_name,
               std::string assembly_text,
               MetadataBlob metadata = {},
               ConstSegment const_segment = {},
               RawDataSegment raw_data_segment = {})
      : kernel_name_(std::move(kernel_name)),
        assembly_text_(std::move(assembly_text)),
        metadata_(std::move(metadata)),
        const_segment_(std::move(const_segment)),
        raw_data_segment_(std::move(raw_data_segment)) {}

  const std::string& kernel_name() const { return kernel_name_; }
  const std::string& assembly_text() const { return assembly_text_; }
  const MetadataBlob& metadata() const { return metadata_; }
  const ConstSegment& const_segment() const { return const_segment_; }
  const RawDataSegment& raw_data_segment() const { return raw_data_segment_; }

 private:
  std::string kernel_name_;
  std::string assembly_text_;
  MetadataBlob metadata_;
  ConstSegment const_segment_;
  RawDataSegment raw_data_segment_;
};

}  // namespace gpu_model
