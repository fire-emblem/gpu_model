#pragma once

#include <cstdint>

namespace gpu_model {

inline bool TryAcquireBarrierSlot(uint32_t capacity,
                                  uint32_t& slots_in_use,
                                  bool& barrier_slot_acquired) {
  if (barrier_slot_acquired) {
    return true;
  }
  if (capacity != 0 && slots_in_use >= capacity) {
    return false;
  }
  ++slots_in_use;
  barrier_slot_acquired = true;
  return true;
}

inline void ReleaseBarrierSlot(uint32_t& slots_in_use, bool& barrier_slot_acquired) {
  if (!barrier_slot_acquired) {
    return;
  }
  if (slots_in_use > 0) {
    --slots_in_use;
  }
  barrier_slot_acquired = false;
}

}  // namespace gpu_model
