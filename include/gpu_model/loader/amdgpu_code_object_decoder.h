#pragma once

#include <filesystem>
#include <optional>

#include "gpu_model/program/encoded_program_object.h"

namespace gpu_model {

class AmdgpuCodeObjectDecoder {
 public:
  EncodedProgramObject Decode(const std::filesystem::path& path,
                              std::optional<std::string> kernel_name = std::nullopt) const;
};

}  // namespace gpu_model
