#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include "debug/trace/step_detail.h"

namespace gpu_model {

enum class TraceSlotModelKind {
  None,
  LogicalUnbounded,
  ResidentFixed,
};

enum class TraceStallReason {
  None,
  WarpSwitch,
  WaitCntGlobal,
  WaitCntShared,
  WaitCntPrivate,
  WaitCntScalarBuffer,
  BarrierSlotUnavailable,
  Other,
};

enum class TraceBarrierKind {
  None,
  Wave,
  Arrive,
  Release,
};

enum class TraceArriveKind {
  None,
  Load,
  Store,
  Shared,
  Private,
  ScalarBuffer,
};

enum class TraceArriveProgressKind {
  None,
  StillBlocked,
  Resume,
};

enum class TraceLifecycleStage {
  None,
  Launch,
  Exit,
};

enum class TraceFlowPhase {
  None,
  Start,
  Finish,
};

enum class TraceEventKind {
  Launch,
  BlockPlaced,
  BlockAdmit,
  BlockLaunch,
  BlockActivate,
  BlockRetire,
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
  WaveStep,
  Commit,
  ExecMaskUpdate,
  MemoryAccess,
  Barrier,
  WaveExit,
  Stall,
  Arrive,
};

struct TraceWaitcntState {
  bool valid = false;
  uint32_t threshold_global = UINT32_MAX;
  uint32_t threshold_shared = UINT32_MAX;
  uint32_t threshold_private = UINT32_MAX;
  uint32_t threshold_scalar_buffer = UINT32_MAX;
  bool has_pending_before = false;
  uint32_t pending_before_global = 0;
  uint32_t pending_before_shared = 0;
  uint32_t pending_before_private = 0;
  uint32_t pending_before_scalar_buffer = 0;
  uint32_t pending_global = 0;
  uint32_t pending_shared = 0;
  uint32_t pending_private = 0;
  uint32_t pending_scalar_buffer = 0;
  bool blocked_global = false;
  bool blocked_shared = false;
  bool blocked_private = false;
  bool blocked_scalar_buffer = false;
};

struct TraceEvent {
  TraceEventKind kind = TraceEventKind::Launch;
  uint64_t cycle = 0;
  uint32_t dpc_id = 0;
  uint32_t ap_id = 0;
  uint32_t peu_id = 0;
  uint32_t slot_id = 0;
  TraceSlotModelKind slot_model_kind = TraceSlotModelKind::None;
  std::string slot_model;
  uint32_t block_id = 0;
  uint32_t wave_id = 0;
  uint64_t pc = 0;
  TraceStallReason stall_reason = TraceStallReason::None;
  TraceBarrierKind barrier_kind = TraceBarrierKind::None;
  TraceArriveKind arrive_kind = TraceArriveKind::None;
  TraceArriveProgressKind arrive_progress = TraceArriveProgressKind::None;
  TraceLifecycleStage lifecycle_stage = TraceLifecycleStage::None;
  TraceWaitcntState waitcnt_state;
  bool has_cycle_range = false;
  uint64_t range_end_cycle = 0;
  std::string semantic_canonical_name;
  std::string semantic_presentation_name;
  std::string semantic_category;
  std::string display_name;
  std::string message;
  uint64_t flow_id = 0;
  TraceFlowPhase flow_phase = TraceFlowPhase::None;
  // Structured step detail (producer-owned fact)
  std::optional<TraceWaveStepDetail> step_detail;
};

inline constexpr std::string_view kTraceStallReasonPrefix = "reason=";
inline constexpr std::string_view kTraceStallReasonWarpSwitch = "warp_switch";
inline constexpr std::string_view kTraceStallReasonWaitCntGlobal = "waitcnt_global";
inline constexpr std::string_view kTraceStallReasonWaitCntShared = "waitcnt_shared";
inline constexpr std::string_view kTraceStallReasonWaitCntPrivate = "waitcnt_private";
inline constexpr std::string_view kTraceStallReasonWaitCntScalarBuffer = "waitcnt_scalar_buffer";
inline constexpr std::string_view kTraceStallReasonBarrierSlotUnavailable =
    "barrier_slot_unavailable";

inline std::string_view TraceSlotModelName(TraceSlotModelKind kind) {
  switch (kind) {
    case TraceSlotModelKind::None:
      return "";
    case TraceSlotModelKind::LogicalUnbounded:
      return "logical_unbounded";
    case TraceSlotModelKind::ResidentFixed:
      return "resident_fixed";
  }
  return "";
}

inline TraceSlotModelKind TraceSlotModelKindFromName(std::string_view name) {
  if (name.empty()) {
    return TraceSlotModelKind::None;
  }
  if (name == "logical_unbounded") {
    return TraceSlotModelKind::LogicalUnbounded;
  }
  if (name == "resident_fixed") {
    return TraceSlotModelKind::ResidentFixed;
  }
  return TraceSlotModelKind::None;
}

inline std::string MakeTraceStallReasonMessage(std::string_view reason) {
  if (reason.starts_with(kTraceStallReasonPrefix)) {
    return std::string(reason);
  }
  return std::string(kTraceStallReasonPrefix) + std::string(reason);
}

inline std::string_view TraceStallReasonName(TraceStallReason reason) {
  switch (reason) {
    case TraceStallReason::None:
      return "";
    case TraceStallReason::WarpSwitch:
      return kTraceStallReasonWarpSwitch;
    case TraceStallReason::WaitCntGlobal:
      return kTraceStallReasonWaitCntGlobal;
    case TraceStallReason::WaitCntShared:
      return kTraceStallReasonWaitCntShared;
    case TraceStallReason::WaitCntPrivate:
      return kTraceStallReasonWaitCntPrivate;
    case TraceStallReason::WaitCntScalarBuffer:
      return kTraceStallReasonWaitCntScalarBuffer;
    case TraceStallReason::BarrierSlotUnavailable:
      return kTraceStallReasonBarrierSlotUnavailable;
    case TraceStallReason::Other:
      return "other";
  }
  return "";
}

inline std::string_view TraceBarrierKindName(TraceBarrierKind kind) {
  switch (kind) {
    case TraceBarrierKind::None:
      return "";
    case TraceBarrierKind::Wave:
      return "wave";
    case TraceBarrierKind::Arrive:
      return "arrive";
    case TraceBarrierKind::Release:
      return "release";
  }
  return "";
}

inline std::string_view TraceArriveKindName(TraceArriveKind kind) {
  switch (kind) {
    case TraceArriveKind::None:
      return "";
    case TraceArriveKind::Load:
      return "load";
    case TraceArriveKind::Store:
      return "store";
    case TraceArriveKind::Shared:
      return "shared";
    case TraceArriveKind::Private:
      return "private";
    case TraceArriveKind::ScalarBuffer:
      return "scalar_buffer";
  }
  return "";
}

inline std::string_view TraceArriveProgressKindName(TraceArriveProgressKind kind) {
  switch (kind) {
    case TraceArriveProgressKind::None:
      return "";
    case TraceArriveProgressKind::StillBlocked:
      return "still_blocked";
    case TraceArriveProgressKind::Resume:
      return "resume";
  }
  return "";
}

inline std::string_view TraceLifecycleStageName(TraceLifecycleStage stage) {
  switch (stage) {
    case TraceLifecycleStage::None:
      return "";
    case TraceLifecycleStage::Launch:
      return "launch";
    case TraceLifecycleStage::Exit:
      return "exit";
  }
  return "";
}

inline std::string_view TraceFlowPhaseName(TraceFlowPhase phase) {
  switch (phase) {
    case TraceFlowPhase::None:
      return "";
    case TraceFlowPhase::Start:
      return "start";
    case TraceFlowPhase::Finish:
      return "finish";
  }
  return "";
}

inline std::string_view TraceStallReasonPayload(std::string_view message) {
  if (!message.starts_with(kTraceStallReasonPrefix)) {
    return message;
  }
  return message.substr(kTraceStallReasonPrefix.size());
}

inline TraceStallReason TraceStallReasonFromMessage(std::string_view message) {
  const std::string_view payload = TraceStallReasonPayload(message);
  if (payload.empty()) {
    return TraceStallReason::None;
  }
  if (payload == kTraceStallReasonWarpSwitch) {
    return TraceStallReason::WarpSwitch;
  }
  if (payload == kTraceStallReasonWaitCntGlobal) {
    return TraceStallReason::WaitCntGlobal;
  }
  if (payload == kTraceStallReasonWaitCntShared) {
    return TraceStallReason::WaitCntShared;
  }
  if (payload == kTraceStallReasonWaitCntPrivate) {
    return TraceStallReason::WaitCntPrivate;
  }
  if (payload == kTraceStallReasonWaitCntScalarBuffer) {
    return TraceStallReason::WaitCntScalarBuffer;
  }
  if (payload == kTraceStallReasonBarrierSlotUnavailable) {
    return TraceStallReason::BarrierSlotUnavailable;
  }
  return TraceStallReason::Other;
}

inline TraceSlotModelKind TraceEffectiveSlotModelKind(const TraceEvent& event) {
  if (event.slot_model_kind != TraceSlotModelKind::None) {
    return event.slot_model_kind;
  }
  return TraceSlotModelKindFromName(event.slot_model);
}

inline bool TraceHasSlotModel(const TraceEvent& event, TraceSlotModelKind kind) {
  return TraceEffectiveSlotModelKind(event) == kind;
}

inline std::string_view TraceEffectiveSlotModelName(const TraceEvent& event) {
  const std::string_view typed = TraceSlotModelName(TraceEffectiveSlotModelKind(event));
  if (!typed.empty()) {
    return typed;
  }
  return event.slot_model;
}

inline TraceStallReason TraceEffectiveStallReason(const TraceEvent& event) {
  if (event.stall_reason != TraceStallReason::None) {
    return event.stall_reason;
  }
  if (event.kind != TraceEventKind::Stall) {
    return TraceStallReason::None;
  }
  return TraceStallReasonFromMessage(event.message);
}

inline bool TraceHasStallReason(const TraceEvent& event, TraceStallReason reason) {
  return event.kind == TraceEventKind::Stall && TraceEffectiveStallReason(event) == reason;
}

inline bool TraceHasWaitcntState(const TraceEvent& event) {
  return event.waitcnt_state.valid;
}

}  // namespace gpu_model
