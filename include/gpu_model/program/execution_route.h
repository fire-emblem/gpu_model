#pragma once

#include <memory>

namespace gpu_model {

class ProgramObject;
struct EncodedProgramObject;

struct PreparedExecutionRoute {
  const ProgramObject* execution_image = nullptr;
  const EncodedProgramObject* raw_code_object = nullptr;
  std::shared_ptr<EncodedProgramObject> owned_raw_code_object;
};

PreparedExecutionRoute PrepareExecutionRoute(const ProgramObject& image);

}  // namespace gpu_model
