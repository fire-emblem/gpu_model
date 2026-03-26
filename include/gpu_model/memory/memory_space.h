#pragma once

namespace gpu_model {

enum class MemorySpace {
  Global,
  Constant,
  Shared,
  Private,
};

enum class AccessKind {
  Load,
  Store,
  Atomic,
  AsyncLoad,
  AsyncStore,
};

}  // namespace gpu_model
