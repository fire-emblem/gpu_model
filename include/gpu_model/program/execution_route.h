#pragma once

#include <memory>

#include "gpu_model/program/program_execution_route.h"

namespace gpu_model {

class ProgramObject;
struct EncodedProgramObject;

using ExecutionRoute = ProgramExecutionRoute;

struct PreparedExecutionRoute {
  ExecutionRoute resolved_route = ExecutionRoute::AutoSelect;
  const ProgramObject* execution_image = nullptr;
  const EncodedProgramObject* raw_code_object = nullptr;
  std::shared_ptr<ProgramObject> owned_program_image;
  std::shared_ptr<EncodedProgramObject> owned_raw_code_object;
};

PreparedExecutionRoute PrepareExecutionRoute(const ProgramObject& image,
                                             ExecutionRoute requested_route);

}  // namespace gpu_model
