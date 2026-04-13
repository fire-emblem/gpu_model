#pragma once

namespace gpu_model {

/// Memory arrive kind for tracking pending memory operations.
/// This is a gpu_arch-level enum used by both execution state and trace layer.
/// Placed in gpu_arch/memory/ because it describes memory operation characteristics,
/// not execution behavior.
enum class MemoryArriveKind {
  Load,
  Store,
  Shared,
  Private,
  ScalarBuffer,
};

}  // namespace gpu_model
