#include "gpu_model/debug/trace/event_export.h"

#include <sstream>
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

std::string FormatWaitcntPendingBefore(const TraceWaitcntState& state) {
  if (!state.valid || !state.has_pending_before) {
    return {};
  }
  return "g=" + std::to_string(state.pending_before_global) + " s=" +
         std::to_string(state.pending_before_shared) + " p=" +
         std::to_string(state.pending_before_private) + " sb=" +
         std::to_string(state.pending_before_scalar_buffer);
}

std::string FormatWaitcntPendingTransition(const TraceWaitcntState& state) {
  if (!state.valid || !state.has_pending_before) {
    return {};
  }
  return "g=" + std::to_string(state.pending_before_global) + "->" +
         std::to_string(state.pending_global) + " s=" +
         std::to_string(state.pending_before_shared) + "->" +
         std::to_string(state.pending_shared) + " p=" +
         std::to_string(state.pending_before_private) + "->" +
         std::to_string(state.pending_private) + " sb=" +
         std::to_string(state.pending_before_scalar_buffer) + "->" +
         std::to_string(state.pending_scalar_buffer);
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

std::string HexU64(uint64_t value) {
  std::ostringstream out;
  out << "0x" << std::hex << std::nouppercase << value;
  return out.str();
}

}  // namespace

TraceEventExportFields MakeTraceEventExportFields(const TraceEventView& view) {
  const bool has_flow =
      view.flow_phase != TraceFlowPhase::None || view.flow_id != 0;
  return TraceEventExportFields{
      .slot_model = std::string(TraceSlotModelName(view.slot_model_kind)),
      .stall_reason = std::string(TraceStallReasonName(view.stall_reason)),
      .barrier_kind = std::string(TraceBarrierKindName(view.barrier_kind)),
      .arrive_kind = std::string(TraceArriveKindName(view.arrive_kind)),
      .arrive_progress = std::string(TraceArriveProgressKindName(view.arrive_progress)),
      .lifecycle_stage = std::string(TraceLifecycleStageName(view.lifecycle_stage)),
      .waitcnt_thresholds = FormatWaitcntThresholds(view.waitcnt_state),
      .waitcnt_pending_before = FormatWaitcntPendingBefore(view.waitcnt_state),
      .waitcnt_pending = FormatWaitcntPending(view.waitcnt_state),
      .waitcnt_pending_transition = FormatWaitcntPendingTransition(view.waitcnt_state),
      .waitcnt_blocked_domains = FormatWaitcntBlockedDomains(view.waitcnt_state),
      .canonical_name = view.canonical_name,
      .presentation_name = view.presentation_name,
      .display_name = view.display_name,
      .category = view.category,
      .compatibility_message = view.compatibility_message,
      .has_flow = has_flow,
      .flow_id = has_flow ? HexU64(view.flow_id) : std::string(),
      .flow_phase = std::string(TraceFlowPhaseName(view.flow_phase)),
      .has_cycle_range = false,
      .begin_cycle = {},
      .end_cycle = {},
  };
}

TraceEventExportFields MakeTraceEventExportFields(const RecorderProgramEvent& event) {
  const bool has_flow =
      event.event.flow_phase != TraceFlowPhase::None || event.event.flow_id != 0;
  return TraceEventExportFields{
      .slot_model = std::string(TraceSlotModelName(event.slot_model_kind)),
      .stall_reason = std::string(TraceStallReasonName(event.stall_reason)),
      .barrier_kind = std::string(TraceBarrierKindName(event.barrier_kind)),
      .arrive_kind = std::string(TraceArriveKindName(event.arrive_kind)),
      .arrive_progress = std::string(TraceArriveProgressKindName(event.event.arrive_progress)),
      .lifecycle_stage = std::string(TraceLifecycleStageName(event.lifecycle_stage)),
      .waitcnt_thresholds = FormatWaitcntThresholds(event.waitcnt_state),
      .waitcnt_pending_before = FormatWaitcntPendingBefore(event.waitcnt_state),
      .waitcnt_pending = FormatWaitcntPending(event.waitcnt_state),
      .waitcnt_pending_transition = FormatWaitcntPendingTransition(event.waitcnt_state),
      .waitcnt_blocked_domains = FormatWaitcntBlockedDomains(event.waitcnt_state),
      .canonical_name = event.canonical_name,
      .presentation_name = event.presentation_name,
      .display_name = event.display_name,
      .category = event.category,
      .compatibility_message = event.compatibility_message,
      .has_flow = has_flow,
      .flow_id = has_flow ? HexU64(event.event.flow_id) : std::string(),
      .flow_phase = std::string(TraceFlowPhaseName(event.event.flow_phase)),
      .has_cycle_range = false,
      .begin_cycle = {},
      .end_cycle = {},
  };
}

TraceEventExportFields MakeTraceEventExportFields(const RecorderEntry& event) {
  const bool has_flow =
      event.event.flow_phase != TraceFlowPhase::None || event.event.flow_id != 0;
  return TraceEventExportFields{
      .slot_model = std::string(TraceSlotModelName(event.slot_model_kind)),
      .stall_reason = std::string(TraceStallReasonName(event.stall_reason)),
      .barrier_kind = std::string(TraceBarrierKindName(event.barrier_kind)),
      .arrive_kind = std::string(TraceArriveKindName(event.arrive_kind)),
      .arrive_progress = std::string(TraceArriveProgressKindName(event.event.arrive_progress)),
      .lifecycle_stage = std::string(TraceLifecycleStageName(event.lifecycle_stage)),
      .waitcnt_thresholds = FormatWaitcntThresholds(event.waitcnt_state),
      .waitcnt_pending_before = FormatWaitcntPendingBefore(event.waitcnt_state),
      .waitcnt_pending = FormatWaitcntPending(event.waitcnt_state),
      .waitcnt_pending_transition = FormatWaitcntPendingTransition(event.waitcnt_state),
      .waitcnt_blocked_domains = FormatWaitcntBlockedDomains(event.waitcnt_state),
      .canonical_name = event.canonical_name,
      .presentation_name = event.presentation_name,
      .display_name = event.display_name,
      .category = event.category,
      .compatibility_message = event.compatibility_message,
      .has_flow = has_flow,
      .flow_id = has_flow ? HexU64(event.event.flow_id) : std::string(),
      .flow_phase = std::string(TraceFlowPhaseName(event.event.flow_phase)),
      .has_cycle_range = event.has_cycle_range,
      .begin_cycle = event.has_cycle_range ? HexU64(event.begin_cycle) : std::string(),
      .end_cycle = event.has_cycle_range ? HexU64(event.end_cycle) : std::string(),
  };
}

CanonicalTraceEvent MakeCanonicalTraceEvent(const TraceEvent& event) {
  TraceEventView view = MakeTraceEventView(event);
  TraceEventExportFields fields = MakeTraceEventExportFields(view);
  return CanonicalTraceEvent{
      .event = &event,
      .view = std::move(view),
      .fields = std::move(fields),
  };
}

}  // namespace gpu_model
