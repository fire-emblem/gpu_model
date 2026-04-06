#pragma once

#include "gpu_model/execution/internal/handlers/control_handler_base.h"

namespace gpu_model {

class BranchHandler final : public ControlHandlerBase {
 public:
  std::string_view name() const final { return "branch"; }
};

}  // namespace gpu_model
