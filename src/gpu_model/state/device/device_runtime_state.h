#pragma once

#include <cstdint>
#include <vector>

namespace gpu_model {

/// DeviceState — Device 运行时状态
///
/// Device 是 GPU 设备的顶层抽象。
struct DeviceState {
  uint32_t device_id = 0;
  uint32_t dpc_count = 0;
};

}  // namespace gpu_model
