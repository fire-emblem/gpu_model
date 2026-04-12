#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "gpu_model/state/wave/wave_runtime_state.h"
#include "gpu_model/memory/memory_pool.h"
#include "gpu_model/memory/memory_request.h"
#include "gpu_model/memory/memory_system.h"

namespace gpu_model {

namespace memory_ops {

uint64_t LoadPoolLaneValue(const MemorySystem& memory, MemoryPoolKind pool, const LaneAccess& lane);
uint64_t LoadGlobalLaneValue(const MemorySystem& memory, const LaneAccess& lane);
void StoreGlobalLaneValue(MemorySystem& memory, const LaneAccess& lane);

uint64_t LoadByteLaneValue(const std::vector<std::byte>& memory, const LaneAccess& lane);
void StoreByteLaneValue(std::vector<std::byte>& memory, const LaneAccess& lane);

uint64_t LoadPrivateLaneValue(std::array<std::vector<std::byte>, kWaveSize>& memory,
                              uint32_t lane_id,
                              const LaneAccess& lane,
                              bool extend_on_read);
uint64_t LoadPrivateLaneValue(const std::array<std::vector<std::byte>, kWaveSize>& memory,
                              uint32_t lane_id,
                              const LaneAccess& lane);
void StorePrivateLaneValue(std::array<std::vector<std::byte>, kWaveSize>& memory,
                           uint32_t lane_id,
                           const LaneAccess& lane);

}  // namespace memory_ops

}  // namespace gpu_model
