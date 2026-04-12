#pragma once

#include <cstdint>

namespace gpu_model {

/// DeviceDef — Device 纯定义
///
/// Device 是 GPU 设备的顶层抽象，包含多个 DPC。
/// 纯定义，不含运行时状态。

/// Device 常量定义（默认值）
inline constexpr uint32_t kDefaultDpcPerDevice = 8;

struct DeviceConfig {
  uint32_t device_id = 0;
  uint32_t dpc_count = kDefaultDpcPerDevice;
  uint32_t total_ap_count() const { return dpc_count * kDefaultApPerDpc; }
};

}  // namespace gpu_model
