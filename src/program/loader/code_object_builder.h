#pragma once

#include <filesystem>
#include <optional>
#include <string>

#include "program/program_object/program_object.h"

namespace gpu_model {

ProgramObject LoadEncodedProgramObject(const std::filesystem::path& path,
                                       std::optional<std::string> kernel_name = std::nullopt);

}  // namespace gpu_model
