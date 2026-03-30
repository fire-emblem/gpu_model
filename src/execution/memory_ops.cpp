#include "gpu_model/execution/memory_ops.h"

#include <cstring>
#include <stdexcept>

namespace gpu_model::memory_ops {

uint64_t LoadPoolLaneValue(const MemorySystem& memory, MemoryPoolKind pool, const LaneAccess& lane) {
  switch (lane.bytes) {
    case 4: {
      const int32_t value = memory.LoadValue<int32_t>(pool, lane.addr);
      return static_cast<uint64_t>(static_cast<int64_t>(value));
    }
    case 8:
      return memory.LoadValue<uint64_t>(pool, lane.addr);
    default:
      throw std::invalid_argument("unsupported load width");
  }
}

uint64_t LoadGlobalLaneValue(const MemorySystem& memory, const LaneAccess& lane) {
  switch (lane.bytes) {
    case 4: {
      const int32_t value = memory.LoadGlobalValue<int32_t>(lane.addr);
      return static_cast<uint64_t>(static_cast<int64_t>(value));
    }
    case 8:
      return memory.LoadGlobalValue<uint64_t>(lane.addr);
    default:
      throw std::invalid_argument("unsupported load width");
  }
}

void StoreGlobalLaneValue(MemorySystem& memory, const LaneAccess& lane) {
  switch (lane.bytes) {
    case 4:
      memory.StoreGlobalValue<int32_t>(lane.addr, static_cast<int32_t>(lane.value));
      return;
    case 8:
      memory.StoreGlobalValue<uint64_t>(lane.addr, lane.value);
      return;
    default:
      throw std::invalid_argument("unsupported store width");
  }
}

uint64_t LoadByteLaneValue(const std::vector<std::byte>& memory, const LaneAccess& lane) {
  const size_t end = static_cast<size_t>(lane.addr) + lane.bytes;
  if (end > memory.size()) {
    throw std::out_of_range("byte-addressable memory load out of range");
  }
  switch (lane.bytes) {
    case 4: {
      int32_t value = 0;
      std::memcpy(&value, memory.data() + lane.addr, sizeof(value));
      return static_cast<uint64_t>(static_cast<int64_t>(value));
    }
    case 8: {
      uint64_t value = 0;
      std::memcpy(&value, memory.data() + lane.addr, sizeof(value));
      return value;
    }
    default:
      throw std::invalid_argument("unsupported load width");
  }
}

void StoreByteLaneValue(std::vector<std::byte>& memory, const LaneAccess& lane) {
  switch (lane.bytes) {
    case 4: {
      const int32_t value = static_cast<int32_t>(lane.value);
      std::memcpy(memory.data() + lane.addr, &value, sizeof(value));
      return;
    }
    case 8:
      std::memcpy(memory.data() + lane.addr, &lane.value, sizeof(lane.value));
      return;
    default:
      throw std::invalid_argument("unsupported store width");
  }
}

uint64_t LoadPrivateLaneValue(std::array<std::vector<std::byte>, kWaveSize>& memory,
                              uint32_t lane_id,
                              const LaneAccess& lane,
                              bool extend_on_read) {
  auto& lane_memory = memory.at(lane_id);
  const size_t end = static_cast<size_t>(lane.addr) + lane.bytes;
  if (lane_memory.size() < end) {
    if (!extend_on_read) {
      return 0;
    }
    lane_memory.resize(end, std::byte{0});
  }
  return LoadByteLaneValue(lane_memory, lane);
}

uint64_t LoadPrivateLaneValue(const std::array<std::vector<std::byte>, kWaveSize>& memory,
                              uint32_t lane_id,
                              const LaneAccess& lane) {
  const auto& lane_memory = memory.at(lane_id);
  const size_t end = static_cast<size_t>(lane.addr) + lane.bytes;
  if (lane_memory.size() < end) {
    return 0;
  }
  return LoadByteLaneValue(lane_memory, lane);
}

void StorePrivateLaneValue(std::array<std::vector<std::byte>, kWaveSize>& memory,
                           uint32_t lane_id,
                           const LaneAccess& lane) {
  auto& lane_memory = memory.at(lane_id);
  const size_t end = static_cast<size_t>(lane.addr) + lane.bytes;
  if (lane_memory.size() < end) {
    lane_memory.resize(end, std::byte{0});
  }
  StoreByteLaneValue(lane_memory, lane);
}

}  // namespace gpu_model::memory_ops
