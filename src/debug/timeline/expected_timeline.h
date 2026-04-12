#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "debug/trace/event.h"

namespace gpu_model {

struct TimelineLaneKey {
  uint32_t dpc_id = 0;
  uint32_t ap_id = 0;
  uint32_t peu_id = 0;
  uint32_t slot_id = 0;
  uint32_t wave_id = 0;

  friend bool operator==(const TimelineLaneKey&, const TimelineLaneKey&) = default;
};

struct TimelineEventKey {
  TimelineLaneKey lane;
  uint64_t pc = 0;
  std::string name;

  friend bool operator==(const TimelineEventKey&, const TimelineEventKey&) = default;
};

struct ExpectedSlice {
  TimelineEventKey key;
  uint64_t begin_cycle = 0;
  uint64_t end_cycle = 0;
};

struct ExpectedMarker {
  TimelineEventKey key;
  uint64_t cycle = 0;
  std::optional<TraceStallReason> stall_reason;
  std::optional<TraceArriveProgressKind> arrive_progress;
};

struct OrderingConstraint {
  TimelineEventKey earlier;
  TimelineEventKey later;
};

struct ExpectedTimeline {
  std::vector<ExpectedSlice> required_slices;
  std::vector<ExpectedMarker> required_markers;
  std::vector<TimelineEventKey> forbidden_slices;
  std::vector<OrderingConstraint> ordering;
};

}  // namespace gpu_model
