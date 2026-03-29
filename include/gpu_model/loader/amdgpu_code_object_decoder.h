#pragma once

#include <filesystem>
#include <optional>
#include <utility>

#include "gpu_model/program/encoded_program_object.h"

namespace gpu_model {

struct AmdgpuCodeObjectImage : EncodedProgramObject {
  AmdgpuCodeObjectImage() = default;
  AmdgpuCodeObjectImage(EncodedProgramObject&& object)
      : EncodedProgramObject(std::move(object)) {}
};

class AmdgpuCodeObjectDecoder {
 public:
  AmdgpuCodeObjectImage Decode(const std::filesystem::path& path,
                               std::optional<std::string> kernel_name = std::nullopt) const;
};

}  // namespace gpu_model
