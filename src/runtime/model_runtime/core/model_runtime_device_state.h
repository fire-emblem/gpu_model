#pragma once

namespace gpu_model {

class ModelRuntimeDeviceState {
 public:
  void Reset();
  int GetDeviceCount() const;
  int GetDevice() const;
  bool SetDevice(int device_id);

 private:
  int current_device_ = 0;
};

}  // namespace gpu_model
