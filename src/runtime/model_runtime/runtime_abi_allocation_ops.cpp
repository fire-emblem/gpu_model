#include "runtime/model_runtime/runtime_abi_allocation_ops.h"

#include "runtime/model_runtime/device_memory_manager.h"
#include "runtime/model_runtime/model_runtime.h"

namespace gpu_model {

void* AbiAllocateDevice(ModelRuntime& runtime, DeviceMemoryManager& manager, size_t bytes) {
  const uint64_t model_addr = runtime.Malloc(bytes);
  return manager.AllocateGlobal(bytes, model_addr);
}

void* AbiAllocateManaged(ModelRuntime& runtime, DeviceMemoryManager& manager, size_t bytes) {
  const uint64_t model_addr = runtime.MallocManaged(bytes);
  return manager.AllocateManaged(bytes, model_addr);
}

bool AbiFreeDevice(ModelRuntime& runtime, DeviceMemoryManager& manager, void* device_ptr) {
  const auto* allocation = manager.FindAllocation(device_ptr);
  if (allocation == nullptr || allocation->mapped_addr != device_ptr) {
    return false;
  }
  runtime.Free(allocation->model_addr);
  return manager.Free(device_ptr);
}

}  // namespace gpu_model
