#pragma once

#include <cstdint>

namespace gpu_model {

/// PeuDef — PEU (Processing Element Unit) 纯定义
///
/// 包含 PEU 的结构属性，不含运行时状态。
/// 运行时状态见 state/peu/peu_runtime_state.h

/// PEU 常量定义
inline constexpr uint32_t kDefaultResidentWaveSlotsPerPeu = 10;

/// PEU 发射端口类型（架构定义）
enum class PeuIssuePort {
  Scalar,      // 标量发射端口
  Vector,      // 向量发射端口
  Memory,      // 内存发射端口
  Branch,      // 分支发射端口
};

}  // namespace gpu_model
