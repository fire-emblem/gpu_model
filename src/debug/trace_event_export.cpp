#include "gpu_model/debug/trace_event_export.h"

namespace gpu_model {

TraceEventExportFields MakeTraceEventExportFields(const TraceEventView& view) {
  return TraceEventExportFields{
      .slot_model = std::string(TraceSlotModelName(view.slot_model_kind)),
      .stall_reason = std::string(TraceStallReasonName(view.stall_reason)),
      .barrier_kind = std::string(TraceBarrierKindName(view.barrier_kind)),
      .arrive_kind = std::string(TraceArriveKindName(view.arrive_kind)),
      .lifecycle_stage = std::string(TraceLifecycleStageName(view.lifecycle_stage)),
      .canonical_name = view.canonical_name,
      .presentation_name = view.presentation_name,
      .display_name = view.display_name,
      .category = view.category,
      .compatibility_message = view.compatibility_message,
  };
}

}  // namespace gpu_model
