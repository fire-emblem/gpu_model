#pragma once

#include <string>

#include "gpu_model/debug/recorder/recorder.h"
#include "gpu_model/debug/trace/event_view.h"

namespace gpu_model {

struct TraceEventExportFields {
  std::string slot_model;
  std::string stall_reason;
  std::string barrier_kind;
  std::string arrive_kind;
  std::string arrive_progress;
  std::string lifecycle_stage;
  std::string waitcnt_thresholds;
  std::string waitcnt_pending_before;
  std::string waitcnt_pending;
  std::string waitcnt_pending_transition;
  std::string waitcnt_blocked_domains;
  std::string canonical_name;
  std::string presentation_name;
  std::string display_name;
  std::string category;
  std::string compatibility_message;
  bool has_cycle_range = false;
  std::string begin_cycle;
  std::string end_cycle;
};

struct CanonicalTraceEvent {
  const TraceEvent* event = nullptr;
  TraceEventView view;
  TraceEventExportFields fields;
};

TraceEventExportFields MakeTraceEventExportFields(const TraceEventView& view);
TraceEventExportFields MakeTraceEventExportFields(const RecorderProgramEvent& event);
TraceEventExportFields MakeTraceEventExportFields(const RecorderEntry& event);
CanonicalTraceEvent MakeCanonicalTraceEvent(const TraceEvent& event);

}  // namespace gpu_model
