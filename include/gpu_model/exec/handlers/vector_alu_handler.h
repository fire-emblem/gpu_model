#pragma once

#include "gpu_model/exec/handlers/compute_handler_base.h"

namespace gpu_model {

class VectorAluHandler final : public ComputeHandlerBase {
 public:
  std::string_view name() const final { return "vector_alu"; }
};

}  // namespace gpu_model
