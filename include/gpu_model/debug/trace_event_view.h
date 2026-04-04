#pragma once

#include <cstdint>
#include <string>

#include "gpu_model/debug/trace_event.h"

namespace gpu_model {

struct TraceEventView {
  TraceEventKind kind = TraceEventKind::Launch;
  uint64_t cycle = 0;
  uint32_t dpc_id = 0;
  uint32_t ap_id = 0;
  uint32_t peu_id = 0;
  uint32_t slot_id = 0;
  TraceSlotModelKind slot_model_kind = TraceSlotModelKind::None;
  uint32_t block_id = 0;
  uint32_t wave_id = 0;
  uint64_t pc = 0;
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
  bool used_legacy_fallback = false;
};

TraceEventView MakeTraceEventView(const TraceEvent& event);

}  // namespace gpu_model
