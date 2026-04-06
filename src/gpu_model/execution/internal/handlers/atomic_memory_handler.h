#pragma once

#include "gpu_model/execution/internal/handlers/memory_handler_base.h"

namespace gpu_model {

class AtomicMemoryHandler final : public MemoryHandlerBase {
 public:
  std::string_view name() const final { return "atomic_memory"; }
};

}  // namespace gpu_model
