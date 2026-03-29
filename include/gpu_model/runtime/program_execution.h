#pragma once

#include <optional>

#include "gpu_model/loader/amdgpu_code_object_decoder.h"
#include "gpu_model/runtime/launch_request.h"

namespace gpu_model {

struct PreparedProgramExecution {
  ProgramExecutionRoute resolved_route = ProgramExecutionRoute::AutoSelect;
  const ProgramImage* execution_image = nullptr;
  const AmdgpuCodeObjectImage* raw_code_object = nullptr;
  std::optional<ProgramImage> owned_program_image;
  std::optional<AmdgpuCodeObjectImage> owned_raw_code_object;
};

PreparedProgramExecution PrepareProgramExecution(const ProgramImage& image,
                                                 ProgramExecutionRoute requested_route);

}  // namespace gpu_model
