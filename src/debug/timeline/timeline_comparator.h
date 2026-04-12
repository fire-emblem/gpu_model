#pragma once

#include <string>

#include "debug/timeline/actual_timeline_snapshot.h"
#include "debug/timeline/expected_timeline.h"

namespace gpu_model {

struct TimelineComparisonResult {
  bool ok = true;
  std::string message;
};

TimelineComparisonResult CompareTimeline(const ExpectedTimeline& expected,
                                         const ActualTimelineSnapshot& actual);

}  // namespace gpu_model
