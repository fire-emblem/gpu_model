#pragma once

#include <string>

#include "gpu_model/debug/trace_event_view.h"

namespace gpu_model {

struct TraceEventExportFields {
  std::string slot_model;
  std::string stall_reason;
  std::string barrier_kind;
  std::string arrive_kind;
  std::string lifecycle_stage;
  std::string waitcnt_thresholds;
  std::string waitcnt_pending;
  std::string waitcnt_blocked_domains;
  std::string canonical_name;
  std::string presentation_name;
  std::string display_name;
  std::string category;
  std::string compatibility_message;
};

TraceEventExportFields MakeTraceEventExportFields(const TraceEventView& view);

}  // namespace gpu_model
