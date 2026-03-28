#pragma once

#include "gpu_model/exec/handlers/control_handler_base.h"

namespace gpu_model {

class BuiltinStateHandler final : public ControlHandlerBase {
 public:
  std::string_view name() const final { return "builtin_state"; }
};

}  // namespace gpu_model
