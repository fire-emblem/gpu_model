#pragma once

#include "gpu_model/execution/internal/handlers/memory_handler_base.h"

namespace gpu_model {

class VectorMemoryHandler final : public MemoryHandlerBase {
 public:
  std::string_view name() const final { return "vector_memory"; }
};

}  // namespace gpu_model
