#pragma once

#include <cstdint>
#include <limits>
#include <string>

#include "gpu_model/debug/trace_event.h"

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

// Semantic trace factories are the canonical producer/test entry surface.
// New trace construction should use these helpers instead of raw semantic strings.
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
      .message = std::move(message),
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
      .message = std::move(message),
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
      .message = std::move(message),
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

inline TraceEvent MakeTraceWaveLaunchEvent(const TraceWaveView& wave,
                                           uint64_t cycle,
                                           std::string detail,
                                           TraceSlotModelKind slot_model) {
  return MakeTraceWaveEvent(wave, TraceEventKind::WaveLaunch, cycle, slot_model,
                            MakeTraceWaveStartMessage(detail));
}

inline TraceEvent MakeTraceWaveStepEvent(const TraceWaveView& wave,
                                         uint64_t cycle,
                                         TraceSlotModelKind slot_model,
                                         std::string detail,
                                         uint64_t pc = std::numeric_limits<uint64_t>::max()) {
  return MakeTraceWaveEvent(wave, TraceEventKind::WaveStep, cycle, slot_model, std::move(detail),
                            pc);
}

inline TraceEvent MakeTraceCommitEvent(const TraceWaveView& wave,
                                       uint64_t cycle,
                                       TraceSlotModelKind slot_model,
                                       uint64_t pc = std::numeric_limits<uint64_t>::max()) {
  return MakeTraceWaveEvent(wave, TraceEventKind::Commit, cycle, slot_model,
                            std::string(kTraceCommitMessage), pc);
}

inline TraceEvent MakeTraceWaveExitEvent(const TraceWaveView& wave,
                                         uint64_t cycle,
                                         TraceSlotModelKind slot_model,
                                         uint64_t pc = std::numeric_limits<uint64_t>::max()) {
  return MakeTraceWaveEvent(wave, TraceEventKind::WaveExit, cycle, slot_model,
                            std::string(kTraceWaveEndMessage), pc);
}

inline TraceEvent MakeTraceBarrierWaveEvent(const TraceWaveView& wave,
                                            uint64_t cycle,
                                            TraceSlotModelKind slot_model,
                                            uint64_t pc = std::numeric_limits<uint64_t>::max()) {
  return MakeTraceWaveEvent(wave, TraceEventKind::Barrier, cycle, slot_model,
                            std::string(kTraceBarrierWaveMessage), pc);
}

inline TraceEvent MakeTraceBarrierArriveEvent(const TraceWaveView& wave,
                                              uint64_t cycle,
                                              TraceSlotModelKind slot_model,
                                              uint64_t pc = std::numeric_limits<uint64_t>::max()) {
  return MakeTraceWaveEvent(wave, TraceEventKind::Barrier, cycle, slot_model,
                            std::string(kTraceBarrierArriveMessage), pc);
}

inline TraceEvent MakeTraceBarrierReleaseEvent(uint32_t dpc_id,
                                               uint32_t ap_id,
                                               uint32_t block_id,
                                               uint64_t cycle) {
  return MakeTraceBlockEvent(dpc_id, ap_id, block_id, TraceEventKind::Barrier, cycle,
                             std::string(kTraceBarrierReleaseMessage));
}

inline TraceEvent MakeTraceMemoryArriveEvent(const TraceWaveView& wave,
                                             uint64_t cycle,
                                             TraceMemoryArriveKind kind,
                                             TraceSlotModelKind slot_model,
                                             uint64_t pc = std::numeric_limits<uint64_t>::max()) {
  return MakeTraceWaveEvent(wave, TraceEventKind::Arrive, cycle, slot_model,
                            std::string(TraceMemoryArriveMessage(kind)), pc);
}

inline TraceEvent MakeTraceWaitStallEvent(const TraceWaveView& wave,
                                          uint64_t cycle,
                                          TraceStallReason stall_reason,
                                          TraceSlotModelKind slot_model,
                                          uint64_t pc = std::numeric_limits<uint64_t>::max()) {
  return MakeTraceWaveEvent(wave, TraceEventKind::Stall, cycle, slot_model,
                            MakeTraceStallReasonMessage(TraceStallReasonName(stall_reason)), pc);
}

inline TraceEvent MakeTraceWaveSwitchStallEvent(const TraceWaveView& wave,
                                                uint64_t cycle,
                                                TraceSlotModelKind slot_model,
                                                uint64_t pc = std::numeric_limits<uint64_t>::max()) {
  return MakeTraceWaitStallEvent(wave, cycle, TraceStallReason::WarpSwitch, slot_model, pc);
}

}  // namespace gpu_model
