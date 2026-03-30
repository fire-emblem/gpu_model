#pragma once

#include "gpu_model/execution/internal/handlers/exec_handler_base.h"

namespace gpu_model {

class MemoryHandlerBase : public ExecHandlerBase {
 public:
  ExecDomain domain() const final { return ExecDomain::Memory; }
};

}  // namespace gpu_model
