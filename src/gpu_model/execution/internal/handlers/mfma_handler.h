#pragma once

#include "gpu_model/execution/internal/handlers/compute_handler_base.h"

namespace gpu_model {

class MfmaHandler final : public ComputeHandlerBase {
 public:
  std::string_view name() const final { return "mfma"; }
};

}  // namespace gpu_model
