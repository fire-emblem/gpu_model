#pragma once

#include "gpu_model/execution/internal/handlers/memory_handler_base.h"

namespace gpu_model {

class LdsMemoryHandler final : public MemoryHandlerBase {
 public:
  std::string_view name() const final { return "lds_memory"; }
};

}  // namespace gpu_model
