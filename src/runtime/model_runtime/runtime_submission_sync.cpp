#include "runtime/model_runtime/runtime_submission_sync.h"

#include "runtime/model_runtime/device_memory_manager.h"
#include "runtime/model_runtime/model_runtime.h"

namespace gpu_model {

void SyncManagedHostToDevice(DeviceMemoryManager& manager) {
  manager.SyncManagedHostToDevice();
}

void SyncManagedDeviceToHost(DeviceMemoryManager& manager) {
  manager.SyncManagedDeviceToHost();
}

void DeviceSynchronizeWithManagedSync(ModelRuntime& runtime, DeviceMemoryManager& manager) {
  SyncManagedHostToDevice(manager);
  runtime.DeviceSynchronize();
  SyncManagedDeviceToHost(manager);
}

void StreamSynchronizeWithManagedSync(ModelRuntime& runtime,
                                      DeviceMemoryManager& manager,
                                      RuntimeSubmissionContext submission_context) {
  SyncManagedHostToDevice(manager);
  runtime.StreamSynchronize(submission_context);
  SyncManagedDeviceToHost(manager);
}

}  // namespace gpu_model
