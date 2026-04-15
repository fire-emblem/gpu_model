#include "runtime/model_runtime/model_runtime_memory_ops.h"

#include <cstring>
#include <vector>

#include "gpu_arch/memory/memory_system.h"

namespace gpu_model {

void RuntimeMemcpyDeviceToDevice(MemorySystem& memory,
                                 uint64_t dst_addr,
                                 uint64_t src_addr,
                                 size_t bytes) {
  std::vector<std::byte> buffer(bytes);
  memory.ReadGlobal(src_addr, std::span<std::byte>(buffer));
  memory.WriteGlobal(dst_addr, std::span<const std::byte>(buffer));
}

void RuntimeMemsetD8(MemorySystem& memory, uint64_t addr, uint8_t value, size_t bytes) {
  std::vector<std::byte> buffer(bytes, static_cast<std::byte>(value));
  memory.WriteGlobal(addr, std::span<const std::byte>(buffer));
}

void RuntimeMemsetD16(MemorySystem& memory, uint64_t addr, uint16_t value, size_t count) {
  std::vector<std::byte> buffer(count * sizeof(uint16_t));
  for (size_t i = 0; i < count; ++i) {
    std::memcpy(buffer.data() + i * sizeof(uint16_t), &value, sizeof(uint16_t));
  }
  memory.WriteGlobal(addr, std::span<const std::byte>(buffer));
}

void RuntimeMemsetD32(MemorySystem& memory, uint64_t addr, uint32_t value, size_t count) {
  std::vector<std::byte> buffer(count * sizeof(uint32_t));
  for (size_t i = 0; i < count; ++i) {
    std::memcpy(buffer.data() + i * sizeof(uint32_t), &value, sizeof(uint32_t));
  }
  memory.WriteGlobal(addr, std::span<const std::byte>(buffer));
}

}  // namespace gpu_model
