#pragma once

#include <cstdint>
#include <vector>

namespace gpu_model {

/// DpcState — DPC 运行时状态
///
/// DPC (Dual Process Complex) 包含多个 AP。
struct DpcState {
  uint32_t dpc_id = 0;
  uint32_t ap_count = 0;
};

}  // namespace gpu_model
