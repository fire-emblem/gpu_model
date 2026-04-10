#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "gpu_model/debug/trace/document.h"
#include "gpu_model/debug/trace/event.h"
#include "gpu_model/debug/trace/step_detail.h"

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
  ActivePromote,
  IssueSelect,
  WaveWait,
  WaveArrive,
  WaveResume,
  WaveSwitchAway,
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

  // Snapshot setters (producer-owned facts)
  void SetRunSnapshot(TraceRunSnapshot snapshot) { run_snapshot_ = std::move(snapshot); }
  void SetModelConfigSnapshot(TraceModelConfigSnapshot snapshot) { model_config_snapshot_ = std::move(snapshot); }
  void SetKernelSnapshot(TraceKernelSnapshot snapshot) { kernel_snapshot_ = std::move(snapshot); }
  void AddWaveInitSnapshot(TraceWaveInitSnapshot snapshot) { wave_init_snapshots_.push_back(std::move(snapshot)); }
  void SetSummarySnapshot(TraceSummarySnapshot snapshot) { summary_snapshot_ = std::move(snapshot); }
  void AddWarningSnapshot(TraceWarningSnapshot snapshot) { warning_snapshots_.push_back(std::move(snapshot)); }

  // Snapshot getters
  const std::optional<TraceRunSnapshot>& run_snapshot() const { return run_snapshot_; }
  const std::optional<TraceModelConfigSnapshot>& model_config_snapshot() const { return model_config_snapshot_; }
  const std::optional<TraceKernelSnapshot>& kernel_snapshot() const { return kernel_snapshot_; }
  const std::vector<TraceWaveInitSnapshot>& wave_init_snapshots() const { return wave_init_snapshots_; }
  const std::optional<TraceSummarySnapshot>& summary_snapshot() const { return summary_snapshot_; }
  const std::vector<TraceWarningSnapshot>& warning_snapshots() const { return warning_snapshots_; }

 private:
  std::vector<TraceEvent> events_;
  std::vector<RecorderProgramEvent> program_events_;
  std::vector<RecorderWave> waves_;
  uint64_t next_sequence_ = 0;

  // Document-level snapshots (producer-owned facts)
  std::optional<TraceRunSnapshot> run_snapshot_;
  std::optional<TraceModelConfigSnapshot> model_config_snapshot_;
  std::optional<TraceKernelSnapshot> kernel_snapshot_;
  std::vector<TraceWaveInitSnapshot> wave_init_snapshots_;
  std::optional<TraceSummarySnapshot> summary_snapshot_;
  std::vector<TraceWarningSnapshot> warning_snapshots_;
};

}  // namespace gpu_model
