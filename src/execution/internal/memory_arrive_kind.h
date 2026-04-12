#pragma once

namespace gpu_model {

/// Memory arrive kind for tracking pending memory operations.
/// Used by both execution state and trace layer.
enum class MemoryArriveKind {
  Load,
  Store,
  Shared,
  Private,
  ScalarBuffer,
};

}  // namespace gpu_model
