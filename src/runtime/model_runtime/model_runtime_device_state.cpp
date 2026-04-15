#include "runtime/model_runtime/model_runtime_device_state.h"

namespace gpu_model {

void ModelRuntimeDeviceState::Reset() {
  current_device_ = 0;
}

int ModelRuntimeDeviceState::GetDeviceCount() const {
  return 1;
}

int ModelRuntimeDeviceState::GetDevice() const {
  return current_device_;
}

bool ModelRuntimeDeviceState::SetDevice(int device_id) {
  if (device_id != 0) {
    return false;
  }
  current_device_ = device_id;
  return true;
}

}  // namespace gpu_model
