#pragma once

#include "gpu_model/execution/internal/handlers/exec_handler_base.h"

namespace gpu_model {

class ControlHandlerBase : public ExecHandlerBase {
 public:
  ExecDomain domain() const final { return ExecDomain::Control; }
};

}  // namespace gpu_model
