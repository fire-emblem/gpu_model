#pragma once

#include <cstdint>

namespace gpu_model {

/// DpcDef — DPC (Dual Process Complex) 纯定义
///
/// DPC 包含多个 AP (Array Processor)，是设备拓扑的中间层级。
/// 纯定义，不含运行时状态。

/// DPC 常量定义（默认值）
inline constexpr uint32_t kDefaultApPerDpc = 13;

struct DpcConfig {
  uint32_t dpc_id = 0;
  uint32_t ap_count = kDefaultApPerDpc;
};

}  // namespace gpu_model
