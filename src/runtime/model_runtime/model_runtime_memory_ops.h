#pragma once

#include <cstddef>
#include <cstdint>

namespace gpu_model {

class MemorySystem;

void RuntimeMemcpyDeviceToDevice(MemorySystem& memory,
                                 uint64_t dst_addr,
                                 uint64_t src_addr,
                                 size_t bytes);
void RuntimeMemsetD8(MemorySystem& memory, uint64_t addr, uint8_t value, size_t bytes);
void RuntimeMemsetD16(MemorySystem& memory, uint64_t addr, uint16_t value, size_t count);
void RuntimeMemsetD32(MemorySystem& memory, uint64_t addr, uint32_t value, size_t count);

}  // namespace gpu_model
