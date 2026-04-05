#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "gpu_model/debug/trace/event.h"

namespace gpu_model {

enum class RecorderProgramEventKind {
  Launch,
  BlockPlaced,
  BlockAdmit,
  BlockLaunch,
  BlockActivate,
  BlockRetire,
};

enum class RecorderEntryKind {
  WaveLaunch,
  WaveGenerate,
  WaveDispatch,
  SlotBind,
  WaveStats,
  InstructionIssue,
  Commit,
  ExecMaskUpdate,
  MemoryAccess,
  Barrier,
  WaveExit,
  Stall,
  Arrive,
};

struct RecorderProgramEvent {
  uint64_t sequence = 0;
  RecorderProgramEventKind kind = RecorderProgramEventKind::Launch;
  TraceEvent event;
  TraceSlotModelKind slot_model_kind = TraceSlotModelKind::None;
  TraceStallReason stall_reason = TraceStallReason::None;
  TraceBarrierKind barrier_kind = TraceBarrierKind::None;
  TraceArriveKind arrive_kind = TraceArriveKind::None;
  TraceLifecycleStage lifecycle_stage = TraceLifecycleStage::None;
  TraceWaitcntState waitcnt_state;
  std::string canonical_name;
  std::string presentation_name;
  std::string display_name;
  std::string category;
  std::string compatibility_message;
};

struct RecorderEntry {
  uint64_t sequence = 0;
  RecorderEntryKind kind = RecorderEntryKind::InstructionIssue;
  TraceEvent event;
  TraceSlotModelKind slot_model_kind = TraceSlotModelKind::None;
  TraceStallReason stall_reason = TraceStallReason::None;
  TraceBarrierKind barrier_kind = TraceBarrierKind::None;
  TraceArriveKind arrive_kind = TraceArriveKind::None;
  TraceLifecycleStage lifecycle_stage = TraceLifecycleStage::None;
  TraceWaitcntState waitcnt_state;
  std::string canonical_name;
  std::string presentation_name;
  std::string display_name;
  std::string category;
  std::string compatibility_message;
  uint64_t begin_cycle = 0;
  uint64_t end_cycle = 0;
  bool has_cycle_range = false;
};

struct RecorderWave {
  uint32_t dpc_id = 0;
  uint32_t ap_id = 0;
  uint32_t peu_id = 0;
  uint32_t slot_id = 0;
  uint32_t block_id = 0;
  uint32_t wave_id = 0;
  std::vector<RecorderEntry> entries;
};

class Recorder {
 public:
  void Record(const TraceEvent& event);

  const std::vector<TraceEvent>& events() const { return events_; }
  const std::vector<RecorderProgramEvent>& program_events() const { return program_events_; }
  const std::vector<RecorderWave>& waves() const { return waves_; }

 private:
  std::vector<TraceEvent> events_;
  std::vector<RecorderProgramEvent> program_events_;
  std::vector<RecorderWave> waves_;
  uint64_t next_sequence_ = 0;
};

}  // namespace gpu_model
