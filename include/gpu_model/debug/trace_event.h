#pragma once

#include <cstdint>
#include <string>

namespace gpu_model {

enum class TraceEventKind {
  Launch,
  BlockPlaced,
  BlockLaunch,
  WaveLaunch,
  WaveStats,
  WaveStep,
  Commit,
  ExecMaskUpdate,
  MemoryAccess,
  Barrier,
  WaveExit,
  Stall,
  Arrive,
};

struct TraceEvent {
  TraceEventKind kind = TraceEventKind::Launch;
  uint64_t cycle = 0;
  uint32_t dpc_id = 0;
  uint32_t ap_id = 0;
  uint32_t peu_id = 0;
  uint32_t slot_id = 0;
  uint32_t block_id = 0;
  uint32_t wave_id = 0;
  uint64_t pc = 0;
  std::string message;
};

}  // namespace gpu_model
