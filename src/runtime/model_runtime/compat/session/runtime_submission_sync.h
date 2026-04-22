#pragma once

#include "runtime/model_runtime/compat/launch/runtime_submission_context.h"

namespace gpu_model {

class DeviceMemoryManager;
class ModelRuntime;

void SyncManagedHostToDevice(DeviceMemoryManager& manager);
void SyncManagedDeviceToHost(DeviceMemoryManager& manager);
void DeviceSynchronizeWithManagedSync(ModelRuntime& runtime, DeviceMemoryManager& manager);
void StreamSynchronizeWithManagedSync(ModelRuntime& runtime,
                                      DeviceMemoryManager& manager,
                                      RuntimeSubmissionContext submission_context);

}  // namespace gpu_model
