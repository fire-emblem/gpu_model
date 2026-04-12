#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <string_view>

#include "gpu_model/debug/trace/event.h"
#include "gpu_model/execution/internal/memory_arrive_kind.h"

namespace gpu_model {

struct TraceWaveView {
  uint32_t dpc_id = 0;
  uint32_t ap_id = 0;
  uint32_t peu_id = 0;
  uint32_t slot_id = 0;
  uint32_t block_id = 0;
  uint32_t wave_id = 0;
  uint64_t pc = 0;
};

enum class TraceMemoryArriveKind {
  Load,
  Store,
  Shared,
  Private,
  ScalarBuffer,
};

/// Convert execution-layer MemoryArriveKind to trace-layer TraceMemoryArriveKind.
/// This allows execution state to use MemoryArriveKind without depending on trace.
inline TraceMemoryArriveKind ToTraceMemoryArriveKind(MemoryArriveKind kind) {
  switch (kind) {
    case MemoryArriveKind::Load:
      return TraceMemoryArriveKind::Load;
    case MemoryArriveKind::Store:
      return TraceMemoryArriveKind::Store;
    case MemoryArriveKind::Shared:
      return TraceMemoryArriveKind::Shared;
    case MemoryArriveKind::Private:
      return TraceMemoryArriveKind::Private;
    case MemoryArriveKind::ScalarBuffer:
      return TraceMemoryArriveKind::ScalarBuffer;
  }
  return TraceMemoryArriveKind::Load;
}

inline constexpr std::string_view kTraceWaveStartMessage = "wave_start";
inline constexpr std::string_view kTraceWaveEndMessage = "wave_end";
inline constexpr std::string_view kTraceCommitMessage = "commit";
inline constexpr std::string_view kTraceExitMessage = "exit";
inline constexpr std::string_view kTraceBarrierWaveMessage = "wave";
inline constexpr std::string_view kTraceBarrierArriveMessage = "arrive";
inline constexpr std::string_view kTraceBarrierReleaseMessage = "release";
inline constexpr std::string_view kTraceArriveLoadMessage = "load_arrive";
inline constexpr std::string_view kTraceArriveStoreMessage = "store_arrive";
inline constexpr std::string_view kTraceArriveSharedMessage = "shared_arrive";
inline constexpr std::string_view kTraceArrivePrivateMessage = "private_arrive";
inline constexpr std::string_view kTraceArriveScalarBufferMessage = "scalar_buffer_arrive";

inline std::string MakeTraceWaveStartMessage(std::string_view details = {}) {
  if (details.empty()) {
    return std::string(kTraceWaveStartMessage);
  }
  return std::string(kTraceWaveStartMessage) + " " + std::string(details);
}

inline std::string_view TraceArriveMessageForMemoryAccess(bool is_load) {
  return is_load ? kTraceArriveLoadMessage : kTraceArriveStoreMessage;
}

inline std::string_view TraceMemoryArriveMessage(TraceMemoryArriveKind kind) {
  switch (kind) {
    case TraceMemoryArriveKind::Load:
      return kTraceArriveLoadMessage;
    case TraceMemoryArriveKind::Store:
      return kTraceArriveStoreMessage;
    case TraceMemoryArriveKind::Shared:
      return kTraceArriveSharedMessage;
    case TraceMemoryArriveKind::Private:
      return kTraceArrivePrivateMessage;
    case TraceMemoryArriveKind::ScalarBuffer:
      return kTraceArriveScalarBufferMessage;
  }
  return {};
}

inline TraceArriveKind TraceArriveKindForMemoryAccess(TraceMemoryArriveKind kind) {
  switch (kind) {
    case TraceMemoryArriveKind::Load:
      return TraceArriveKind::Load;
    case TraceMemoryArriveKind::Store:
      return TraceArriveKind::Store;
    case TraceMemoryArriveKind::Shared:
      return TraceArriveKind::Shared;
    case TraceMemoryArriveKind::Private:
      return TraceArriveKind::Private;
    case TraceMemoryArriveKind::ScalarBuffer:
      return TraceArriveKind::ScalarBuffer;
  }
  return TraceArriveKind::None;
}

inline std::string MakeTraceStallDisplayName(TraceStallReason stall_reason) {
  const std::string_view reason_name = TraceStallReasonName(stall_reason);
  if (reason_name.empty()) {
    return "stall";
  }
  return std::string(reason_name);
}

inline std::string MakeTraceLifecycleDisplayName(TraceLifecycleStage stage) {
  const std::string_view stage_name = TraceLifecycleStageName(stage);
  return stage_name.empty() ? "lifecycle" : std::string(stage_name);
}

inline std::string MakeTraceBarrierDisplayName(TraceBarrierKind kind) {
  const std::string_view barrier_name = TraceBarrierKindName(kind);
  return barrier_name.empty() ? "barrier" : std::string(barrier_name);
}

inline std::string MakeTraceArriveDisplayName(TraceMemoryArriveKind kind) {
  return std::string(TraceArriveKindName(TraceArriveKindForMemoryAccess(kind)));
}

inline std::string MakeTraceWaveStepDisplayName(std::string_view detail) {
  constexpr std::string_view kOpPrefix = "op=";
  const size_t op_pos = detail.find(kOpPrefix);
  if (op_pos == std::string_view::npos) {
    return std::string(detail);
  }
  const size_t value_begin = op_pos + kOpPrefix.size();
  const size_t value_end = detail.find_first_of(" \n", value_begin);
  return std::string(detail.substr(value_begin, value_end - value_begin));
}

inline std::string WaitcntBlockedDomainSuffix(const TraceWaitcntState& state) {
  if (!state.valid) {
    return {};
  }
  std::string suffix;
  const auto append_domain = [&](std::string_view domain) {
    if (!suffix.empty()) {
      suffix += "_";
    }
    suffix += domain;
  };
  if (state.blocked_global) {
    append_domain("global");
  }
  if (state.blocked_shared) {
    append_domain("shared");
  }
  if (state.blocked_private) {
    append_domain("private");
  }
  if (state.blocked_scalar_buffer) {
    append_domain("scalar_buffer");
  }
  return suffix;
}

inline std::string CanonicalStallName(TraceStallReason stall_reason,
                                      const TraceWaitcntState& waitcnt_state) {
  const std::string blocked_suffix = WaitcntBlockedDomainSuffix(waitcnt_state);
  if (!blocked_suffix.empty()) {
    return "stall_waitcnt_" + blocked_suffix;
  }
  if (stall_reason == TraceStallReason::Other) {
    return "stall";
  }
  const std::string_view reason_name = TraceStallReasonName(stall_reason);
  if (reason_name.empty()) {
    return {};
  }
  return "stall_" + std::string(reason_name);
}

inline std::string StallCategory(TraceStallReason stall_reason,
                                 const TraceWaitcntState& waitcnt_state) {
  if (stall_reason == TraceStallReason::WarpSwitch) {
    return "wave/switch_away";
  }
  const std::string blocked_suffix = WaitcntBlockedDomainSuffix(waitcnt_state);
  if (!blocked_suffix.empty()) {
    return "stall/waitcnt_" + blocked_suffix;
  }
  if (stall_reason == TraceStallReason::None || stall_reason == TraceStallReason::Other) {
    return "stall";
  }
  return "stall/" + std::string(TraceStallReasonName(stall_reason));
}

inline void SetProducerSemanticFields(TraceEvent& event,
                                      std::string canonical_name,
                                      std::string category,
                                      std::string presentation_name = {}) {
  event.semantic_canonical_name = std::move(canonical_name);
  event.semantic_presentation_name =
      presentation_name.empty() ? event.semantic_canonical_name : std::move(presentation_name);
  event.semantic_category = std::move(category);
}

inline TraceEvent MakeTraceWaveEvent(const TraceWaveView& wave,
                                     TraceEventKind kind,
                                     uint64_t cycle,
                                     TraceSlotModelKind slot_model,
                                     std::string message,
                                     uint64_t pc = std::numeric_limits<uint64_t>::max()) {
  return TraceEvent{
      .kind = kind,
      .cycle = cycle,
      .dpc_id = wave.dpc_id,
      .ap_id = wave.ap_id,
      .peu_id = wave.peu_id,
      .slot_id = wave.slot_id,
      .slot_model_kind = slot_model,
      .slot_model = std::string(TraceSlotModelName(slot_model)),
      .block_id = wave.block_id,
      .wave_id = wave.wave_id,
      .pc = pc == std::numeric_limits<uint64_t>::max() ? wave.pc : pc,
      .stall_reason = kind == TraceEventKind::Stall ? TraceStallReasonFromMessage(message)
                                                    : TraceStallReason::None,
      .waitcnt_state = {},
      .has_cycle_range = false,
      .range_end_cycle = 0,
      .semantic_canonical_name = {},
      .semantic_presentation_name = {},
      .semantic_category = {},
      .display_name = {},
      .message = std::move(message),
      .step_detail = std::nullopt,
  };
}

inline TraceEvent MakeTraceEvent(TraceEventKind kind,
                                 uint64_t cycle,
                                 std::string message,
                                 TraceSlotModelKind slot_model = TraceSlotModelKind::None) {
  return TraceEvent{
      .kind = kind,
      .cycle = cycle,
      .slot_model_kind = slot_model,
      .slot_model = std::string(TraceSlotModelName(slot_model)),
      .stall_reason = kind == TraceEventKind::Stall ? TraceStallReasonFromMessage(message)
                                                    : TraceStallReason::None,
      .waitcnt_state = {},
      .has_cycle_range = false,
      .range_end_cycle = 0,
      .semantic_canonical_name = {},
      .semantic_presentation_name = {},
      .semantic_category = {},
      .display_name = {},
      .message = std::move(message),
      .step_detail = std::nullopt,
  };
}

inline TraceEvent MakeTraceRuntimeLaunchEvent(uint64_t cycle, std::string message) {
  return MakeTraceEvent(TraceEventKind::Launch, cycle, std::move(message));
}

inline TraceEvent MakeTraceBlockEvent(uint32_t dpc_id,
                                      uint32_t ap_id,
                                      uint32_t block_id,
                                      TraceEventKind kind,
                                      uint64_t cycle,
                                      std::string message,
                                      TraceSlotModelKind slot_model = TraceSlotModelKind::None) {
  return TraceEvent{
      .kind = kind,
      .cycle = cycle,
      .dpc_id = dpc_id,
      .ap_id = ap_id,
      .slot_model_kind = slot_model,
      .slot_model = std::string(TraceSlotModelName(slot_model)),
      .block_id = block_id,
      .stall_reason = kind == TraceEventKind::Stall ? TraceStallReasonFromMessage(message)
                                                    : TraceStallReason::None,
      .waitcnt_state = {},
      .has_cycle_range = false,
      .range_end_cycle = 0,
      .semantic_canonical_name = {},
      .semantic_presentation_name = {},
      .semantic_category = {},
      .display_name = {},
      .message = std::move(message),
      .step_detail = std::nullopt,
  };
}

inline TraceEvent MakeTraceBlockPlacedEvent(uint32_t dpc_id,
                                            uint32_t ap_id,
                                            uint32_t block_id,
                                            uint64_t cycle,
                                            std::string message) {
  return MakeTraceBlockEvent(dpc_id,
                             ap_id,
                             block_id,
                             TraceEventKind::BlockPlaced,
                             cycle,
                             std::move(message));
}

inline TraceEvent MakeTraceBlockAdmitEvent(uint32_t dpc_id,
                                           uint32_t ap_id,
                                           uint32_t block_id,
                                           uint64_t cycle,
                                           std::string message) {
  TraceEvent event = MakeTraceBlockEvent(
      dpc_id, ap_id, block_id, TraceEventKind::BlockAdmit, cycle, std::move(message));
  event.display_name = "block_admit";
  return event;
}

inline TraceEvent MakeTraceBlockActivateEvent(uint32_t dpc_id,
                                              uint32_t ap_id,
                                              uint32_t block_id,
                                              uint64_t cycle,
                                              std::string message) {
  TraceEvent event = MakeTraceBlockEvent(
      dpc_id, ap_id, block_id, TraceEventKind::BlockActivate, cycle, std::move(message));
  event.display_name = "block_activate";
  return event;
}

inline TraceEvent MakeTraceBlockRetireEvent(uint32_t dpc_id,
                                            uint32_t ap_id,
                                            uint32_t block_id,
                                            uint64_t cycle,
                                            std::string message) {
  TraceEvent event = MakeTraceBlockEvent(
      dpc_id, ap_id, block_id, TraceEventKind::BlockRetire, cycle, std::move(message));
  event.display_name = "block_retire";
  return event;
}

inline TraceEvent MakeTraceWaveLaunchEvent(const TraceWaveView& wave,
                                           uint64_t cycle,
                                           std::string detail,
                                           TraceSlotModelKind slot_model) {
  TraceEvent event = MakeTraceWaveEvent(wave, TraceEventKind::WaveLaunch, cycle, slot_model,
                                        MakeTraceWaveStartMessage(detail));
  event.lifecycle_stage = TraceLifecycleStage::Launch;
  event.display_name = MakeTraceLifecycleDisplayName(event.lifecycle_stage);
  return event;
}

inline TraceEvent MakeTraceWaveGenerateEvent(const TraceWaveView& wave,
                                             uint64_t cycle,
                                             TraceSlotModelKind slot_model) {
  TraceEvent event = MakeTraceWaveEvent(
      wave, TraceEventKind::WaveGenerate, cycle, slot_model, "wave_generate");
  event.display_name = "wave_generate";
  return event;
}

inline TraceEvent MakeTraceWaveDispatchEvent(const TraceWaveView& wave,
                                             uint64_t cycle,
                                             TraceSlotModelKind slot_model) {
  TraceEvent event = MakeTraceWaveEvent(
      wave, TraceEventKind::WaveDispatch, cycle, slot_model, "wave_dispatch");
  event.display_name = "wave_dispatch";
  return event;
}

inline TraceEvent MakeTraceSlotBindEvent(const TraceWaveView& wave,
                                         uint64_t cycle,
                                         TraceSlotModelKind slot_model) {
  TraceEvent event =
      MakeTraceWaveEvent(wave, TraceEventKind::SlotBind, cycle, slot_model, "slot_bind");
  event.display_name = "slot_bind";
  return event;
}

inline TraceEvent MakeTraceActivePromoteEvent(const TraceWaveView& wave,
                                              uint64_t cycle,
                                              TraceSlotModelKind slot_model) {
  TraceEvent event = MakeTraceWaveEvent(
      wave, TraceEventKind::ActivePromote, cycle, slot_model, "active_promote");
  event.display_name = "active_promote";
  return event;
}

inline TraceEvent MakeTraceIssueSelectEvent(const TraceWaveView& wave,
                                            uint64_t cycle,
                                            TraceSlotModelKind slot_model) {
  TraceEvent event =
      MakeTraceWaveEvent(wave, TraceEventKind::IssueSelect, cycle, slot_model, "issue_select");
  event.display_name = "issue_select";
  return event;
}

inline TraceEvent MakeTraceWaveStepEvent(const TraceWaveView& wave,
                                         uint64_t cycle,
                                         TraceSlotModelKind slot_model,
                                         std::string detail,
                                         uint64_t pc = std::numeric_limits<uint64_t>::max(),
                                         uint64_t issue_duration_cycles = 0) {
  TraceEvent event = MakeTraceWaveEvent(
      wave, TraceEventKind::WaveStep, cycle, slot_model, std::move(detail), pc);
  if (issue_duration_cycles > 0) {
    event.has_cycle_range = true;
    event.range_end_cycle = cycle + issue_duration_cycles;
  }
  event.display_name = MakeTraceWaveStepDisplayName(event.message);
  return event;
}

// Overload that accepts structured step detail
inline TraceEvent MakeTraceWaveStepEvent(const TraceWaveView& wave,
                                         uint64_t cycle,
                                         TraceSlotModelKind slot_model,
                                         std::string detail,
                                         TraceWaveStepDetail step_detail,
                                         uint64_t pc = std::numeric_limits<uint64_t>::max(),
                                         uint64_t issue_duration_cycles = 0) {
  TraceEvent event = MakeTraceWaveStepEvent(wave, cycle, slot_model, std::move(detail), pc, issue_duration_cycles);
  // Use asm_text from step_detail as display_name for better readability
  if (!step_detail.asm_text.empty()) {
    event.display_name = step_detail.asm_text;
  }
  event.step_detail = std::move(step_detail);
  return event;
}

inline TraceEvent MakeTraceCommitEvent(const TraceWaveView& wave,
                                       uint64_t cycle,
                                       TraceSlotModelKind slot_model,
                                       uint64_t pc = std::numeric_limits<uint64_t>::max()) {
  TraceEvent event = MakeTraceWaveEvent(wave, TraceEventKind::Commit, cycle, slot_model,
                                        std::string(kTraceCommitMessage), pc);
  event.display_name = "commit";
  return event;
}

inline TraceEvent MakeTraceWaveExitEvent(const TraceWaveView& wave,
                                         uint64_t cycle,
                                         TraceSlotModelKind slot_model,
                                         uint64_t pc = std::numeric_limits<uint64_t>::max()) {
  TraceEvent event = MakeTraceWaveEvent(wave, TraceEventKind::WaveExit, cycle, slot_model,
                                        std::string(kTraceWaveEndMessage), pc);
  event.lifecycle_stage = TraceLifecycleStage::Exit;
  event.display_name = MakeTraceLifecycleDisplayName(event.lifecycle_stage);
  return event;
}

inline TraceEvent MakeTraceBarrierWaveEvent(const TraceWaveView& wave,
                                            uint64_t cycle,
                                            TraceSlotModelKind slot_model,
                                            uint64_t pc = std::numeric_limits<uint64_t>::max()) {
  TraceEvent event = MakeTraceWaveEvent(wave, TraceEventKind::Barrier, cycle, slot_model,
                                        std::string(kTraceBarrierWaveMessage), pc);
  event.barrier_kind = TraceBarrierKind::Wave;
  event.display_name = MakeTraceBarrierDisplayName(event.barrier_kind);
  return event;
}

inline TraceEvent MakeTraceBarrierArriveEvent(const TraceWaveView& wave,
                                              uint64_t cycle,
                                              TraceSlotModelKind slot_model,
                                              uint64_t pc = std::numeric_limits<uint64_t>::max()) {
  TraceEvent event = MakeTraceWaveEvent(wave, TraceEventKind::Barrier, cycle, slot_model,
                                        std::string(kTraceBarrierArriveMessage), pc);
  event.barrier_kind = TraceBarrierKind::Arrive;
  event.display_name = MakeTraceBarrierDisplayName(event.barrier_kind);
  return event;
}

inline TraceEvent MakeTraceBarrierReleaseEvent(uint32_t dpc_id,
                                               uint32_t ap_id,
                                               uint32_t block_id,
                                               uint64_t cycle) {
  TraceEvent event = MakeTraceBlockEvent(dpc_id, ap_id, block_id, TraceEventKind::Barrier, cycle,
                                         std::string(kTraceBarrierReleaseMessage));
  event.barrier_kind = TraceBarrierKind::Release;
  event.display_name = MakeTraceBarrierDisplayName(event.barrier_kind);
  return event;
}

inline TraceEvent MakeTraceMemoryArriveEvent(const TraceWaveView& wave,
                                             uint64_t cycle,
                                             TraceMemoryArriveKind kind,
                                             TraceSlotModelKind slot_model,
                                             uint64_t pc = std::numeric_limits<uint64_t>::max()) {
  TraceEvent event = MakeTraceWaveEvent(wave, TraceEventKind::Arrive, cycle, slot_model,
                                        std::string(TraceMemoryArriveMessage(kind)), pc);
  event.arrive_kind = TraceArriveKindForMemoryAccess(kind);
  event.display_name = MakeTraceArriveDisplayName(kind);
  return event;
}

inline TraceEvent MakeTraceWaveWaitEvent(const TraceWaveView& wave,
                                         uint64_t cycle,
                                         TraceSlotModelKind slot_model,
                                         TraceStallReason stall_reason = TraceStallReason::None,
                                         uint64_t pc = std::numeric_limits<uint64_t>::max(),
                                         TraceWaitcntState waitcnt_state = {}) {
  TraceEvent event =
      MakeTraceWaveEvent(wave, TraceEventKind::WaveWait, cycle, slot_model, "wave_wait", pc);
  event.stall_reason = stall_reason;
  event.display_name = "wave_wait";
  event.waitcnt_state = waitcnt_state;
  if (stall_reason == TraceStallReason::None || stall_reason == TraceStallReason::Other) {
    SetProducerSemanticFields(event, "wave_wait", "wave/wait");
  } else {
    SetProducerSemanticFields(
        event, "wave_wait", "wave/wait/" + std::string(TraceStallReasonName(stall_reason)));
  }
  return event;
}

inline TraceEvent MakeTraceWaveArriveEvent(const TraceWaveView& wave,
                                           uint64_t cycle,
                                           TraceMemoryArriveKind kind,
                                           TraceSlotModelKind slot_model,
                                           TraceArriveProgressKind progress =
                                               TraceArriveProgressKind::None,
                                           uint64_t pc = std::numeric_limits<uint64_t>::max(),
                                           TraceWaitcntState waitcnt_state = {}) {
  TraceEvent event =
      MakeTraceWaveEvent(wave, TraceEventKind::WaveArrive, cycle, slot_model, "wave_arrive", pc);
  event.arrive_kind = TraceArriveKindForMemoryAccess(kind);
  event.arrive_progress = progress;
  event.display_name = "wave_arrive";
  event.waitcnt_state = waitcnt_state;
  return event;
}

inline TraceEvent MakeTraceWaveResumeEvent(const TraceWaveView& wave,
                                           uint64_t cycle,
                                           TraceSlotModelKind slot_model,
                                           uint64_t pc = std::numeric_limits<uint64_t>::max(),
                                           TraceWaitcntState waitcnt_state = {}) {
  TraceEvent event =
      MakeTraceWaveEvent(wave, TraceEventKind::WaveResume, cycle, slot_model, "wave_resume", pc);
  event.display_name = "wave_resume";
  event.waitcnt_state = waitcnt_state;
  return event;
}

inline TraceEvent MakeTraceWaitStallEvent(const TraceWaveView& wave,
                                          uint64_t cycle,
                                          TraceStallReason stall_reason,
                                          TraceSlotModelKind slot_model,
                                          uint64_t pc = std::numeric_limits<uint64_t>::max(),
                                          TraceWaitcntState waitcnt_state = {}) {
  TraceEvent event =
      MakeTraceWaveEvent(wave, TraceEventKind::Stall, cycle, slot_model,
                         MakeTraceStallReasonMessage(TraceStallReasonName(stall_reason)), pc);
  event.display_name = MakeTraceStallDisplayName(stall_reason);
  event.waitcnt_state = waitcnt_state;
  if (stall_reason == TraceStallReason::WarpSwitch) {
    SetProducerSemanticFields(event,
                              CanonicalStallName(stall_reason, waitcnt_state),
                              StallCategory(stall_reason, waitcnt_state),
                              "wave_switch_away");
  } else {
    SetProducerSemanticFields(event,
                              CanonicalStallName(stall_reason, waitcnt_state),
                              StallCategory(stall_reason, waitcnt_state));
  }
  return event;
}

inline TraceEvent MakeTraceBlockedStallEvent(const TraceWaveView& wave,
                                             uint64_t cycle,
                                             std::string_view reason,
                                             TraceSlotModelKind slot_model,
                                             uint64_t pc = std::numeric_limits<uint64_t>::max(),
                                             TraceWaitcntState waitcnt_state = {}) {
  const TraceStallReason stall_reason =
      TraceStallReasonFromMessage(MakeTraceStallReasonMessage(reason));
  if (stall_reason != TraceStallReason::Other) {
    return MakeTraceWaitStallEvent(wave, cycle, stall_reason, slot_model, pc, waitcnt_state);
  }

  TraceEvent event = MakeTraceWaveEvent(
      wave, TraceEventKind::Stall, cycle, slot_model, MakeTraceStallReasonMessage(reason), pc);
  event.stall_reason = TraceStallReason::Other;
  event.display_name = "stall";
  event.waitcnt_state = waitcnt_state;
  SetProducerSemanticFields(
      event, "stall_" + std::string(reason), "stall/" + std::string(reason));
  return event;
}

inline TraceEvent MakeTraceWaveSwitchStallEvent(const TraceWaveView& wave,
                                                uint64_t cycle,
                                                TraceSlotModelKind slot_model,
                                                uint64_t pc = std::numeric_limits<uint64_t>::max()) {
  return MakeTraceWaitStallEvent(wave, cycle, TraceStallReason::WarpSwitch, slot_model, pc);
}

inline TraceEvent MakeTraceWaveSwitchAwayEvent(const TraceWaveView& wave,
                                               uint64_t cycle,
                                               TraceSlotModelKind slot_model,
                                               uint64_t pc = std::numeric_limits<uint64_t>::max()) {
  TraceEvent event = MakeTraceWaveEvent(
      wave, TraceEventKind::WaveSwitchAway, cycle, slot_model, "wave_switch_away", pc);
  event.stall_reason = TraceStallReason::WarpSwitch;
  event.display_name = "wave_switch_away";
  SetProducerSemanticFields(event, "wave_switch_away", "wave/switch_away");
  return event;
}

}  // namespace gpu_model
