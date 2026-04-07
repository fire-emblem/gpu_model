#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "gpu_model/debug/recorder/recorder.h"
#include "gpu_model/debug/timeline/expected_timeline.h"

namespace gpu_model {

struct ActualSlice {
  TimelineEventKey key;
  uint64_t begin_cycle = 0;
  uint64_t end_cycle = 0;
  uint64_t sequence = 0;
};

struct ActualMarker {
  TimelineEventKey key;
  uint64_t cycle = 0;
  uint64_t sequence = 0;
  TraceStallReason stall_reason = TraceStallReason::None;
  TraceArriveProgressKind arrive_progress = TraceArriveProgressKind::None;
};

struct ActualTimelineSnapshot {
  std::vector<ActualSlice> slices;
  std::vector<ActualMarker> markers;
};

ActualTimelineSnapshot BuildActualTimelineSnapshot(const Recorder& recorder);

}  // namespace gpu_model
