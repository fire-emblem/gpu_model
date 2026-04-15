#pragma once

#include <cstddef>
#include <cstdint>

#include "runtime/model_runtime/compat/device_memory_manager.h"

namespace gpu_model {

class ModelRuntime;

void AbiMemcpyHostToDevice(ModelRuntime& runtime,
                           const DeviceMemoryManager& manager,
                           void* dst_device_ptr,
                           const void* src_host_ptr,
                           size_t bytes);
void AbiMemcpyDeviceToHost(const ModelRuntime& runtime,
                           const DeviceMemoryManager& manager,
                           void* dst_host_ptr,
                           const void* src_device_ptr,
                           size_t bytes);
void AbiMemcpyDeviceToDevice(ModelRuntime& runtime,
                             const DeviceMemoryManager& manager,
                             void* dst_device_ptr,
                             const void* src_device_ptr,
                             size_t bytes);
void AbiMemsetDevice(ModelRuntime& runtime,
                     const DeviceMemoryManager& manager,
                     void* device_ptr,
                     uint8_t value,
                     size_t bytes);
void AbiMemsetDeviceD16(ModelRuntime& runtime,
                        const DeviceMemoryManager& manager,
                        void* device_ptr,
                        uint16_t value,
                        size_t count);
void AbiMemsetDeviceD32(ModelRuntime& runtime,
                        const DeviceMemoryManager& manager,
                        void* device_ptr,
                        uint32_t value,
                        size_t count);

}  // namespace gpu_model
