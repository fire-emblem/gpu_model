#pragma once

#include "gpu_model/exec/handlers/control_handler_base.h"

namespace gpu_model {

class MaskHandler final : public ControlHandlerBase {
 public:
  std::string_view name() const final { return "mask"; }
};

}  // namespace gpu_model
