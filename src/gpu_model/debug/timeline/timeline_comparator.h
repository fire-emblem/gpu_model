#pragma once

#include <string>

#include "gpu_model/debug/timeline/actual_timeline_snapshot.h"
#include "gpu_model/debug/timeline/expected_timeline.h"

namespace gpu_model {

struct TimelineComparisonResult {
  bool ok = true;
  std::string message;
};

TimelineComparisonResult CompareTimeline(const ExpectedTimeline& expected,
                                         const ActualTimelineSnapshot& actual);

}  // namespace gpu_model
