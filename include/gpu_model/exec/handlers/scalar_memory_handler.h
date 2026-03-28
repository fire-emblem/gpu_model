#pragma once

#include "gpu_model/exec/handlers/memory_handler_base.h"

namespace gpu_model {

class ScalarMemoryHandler final : public MemoryHandlerBase {
 public:
  std::string_view name() const final { return "scalar_memory"; }
};

}  // namespace gpu_model
