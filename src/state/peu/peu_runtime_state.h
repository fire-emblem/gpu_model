#pragma once

#include <cstdint>
#include <vector>

#include "state/wave/wave_runtime_state.h"

namespace gpu_model {

/// PeuState — PEU 运行时状态
///
/// 包含 PEU 的 ID 和驻留 wave 列表。
struct PeuState {
  uint32_t peu_id = 0;
  std::vector<WaveContext> resident_waves;
};

}  // namespace gpu_model
