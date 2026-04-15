#pragma once

#include <cstddef>
#include <cstdint>

namespace gpu_model {

class DeviceMemoryManager;
class ModelRuntime;

void* AbiAllocateDevice(ModelRuntime& runtime, DeviceMemoryManager& manager, size_t bytes);
void* AbiAllocateManaged(ModelRuntime& runtime, DeviceMemoryManager& manager, size_t bytes);
bool AbiFreeDevice(ModelRuntime& runtime, DeviceMemoryManager& manager, void* device_ptr);

}  // namespace gpu_model
