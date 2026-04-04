#include "gpu_model/debug/trace_event_export.h"

#include <string_view>

namespace gpu_model {

namespace {

std::string FormatWaitcntThreshold(uint32_t value) {
  return value == UINT32_MAX ? "*" : std::to_string(value);
}

std::string FormatWaitcntThresholds(const TraceWaitcntState& state) {
  if (!state.valid) {
    return {};
  }
  return "g=" + FormatWaitcntThreshold(state.threshold_global) + " s=" +
         FormatWaitcntThreshold(state.threshold_shared) + " p=" +
         FormatWaitcntThreshold(state.threshold_private) + " sb=" +
         FormatWaitcntThreshold(state.threshold_scalar_buffer);
}

std::string FormatWaitcntPending(const TraceWaitcntState& state) {
  if (!state.valid) {
    return {};
  }
  return "g=" + std::to_string(state.pending_global) + " s=" +
         std::to_string(state.pending_shared) + " p=" + std::to_string(state.pending_private) +
         " sb=" + std::to_string(state.pending_scalar_buffer);
}

std::string FormatWaitcntBlockedDomains(const TraceWaitcntState& state) {
  if (!state.valid) {
    return {};
  }
  std::string blocked;
  const auto append_domain = [&](std::string_view domain) {
    if (!blocked.empty()) {
      blocked += "|";
    }
    blocked += domain;
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
  return blocked;
}

}  // namespace

TraceEventExportFields MakeTraceEventExportFields(const TraceEventView& view) {
  return TraceEventExportFields{
      .slot_model = std::string(TraceSlotModelName(view.slot_model_kind)),
      .stall_reason = std::string(TraceStallReasonName(view.stall_reason)),
      .barrier_kind = std::string(TraceBarrierKindName(view.barrier_kind)),
      .arrive_kind = std::string(TraceArriveKindName(view.arrive_kind)),
      .lifecycle_stage = std::string(TraceLifecycleStageName(view.lifecycle_stage)),
      .waitcnt_thresholds = FormatWaitcntThresholds(view.waitcnt_state),
      .waitcnt_pending = FormatWaitcntPending(view.waitcnt_state),
      .waitcnt_blocked_domains = FormatWaitcntBlockedDomains(view.waitcnt_state),
      .canonical_name = view.canonical_name,
      .presentation_name = view.presentation_name,
      .display_name = view.display_name,
      .category = view.category,
      .compatibility_message = view.compatibility_message,
  };
}

}  // namespace gpu_model
