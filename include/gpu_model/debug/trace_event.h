#pragma once

#include <cstdint>
#include <string>

namespace gpu_model {

enum class TraceEventKind {
  Launch,
  BlockPlaced,
  WaveStep,
  ExecMaskUpdate,
  MemoryAccess,
  WaveExit,
  Stall,
  Arrive,
};

struct TraceEvent {
  TraceEventKind kind = TraceEventKind::Launch;
  uint64_t cycle = 0;
  uint32_t block_id = 0;
  uint32_t wave_id = 0;
  uint64_t pc = 0;
  std::string message;
};

}  // namespace gpu_model
