#pragma once

#include "gpu_model/execution/internal/handlers/exec_handler_base.h"

namespace gpu_model {

class ComputeHandlerBase : public ExecHandlerBase {
 public:
  ExecDomain domain() const final { return ExecDomain::Compute; }
};

}  // namespace gpu_model
