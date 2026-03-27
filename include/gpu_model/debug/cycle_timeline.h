#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "gpu_model/debug/trace_event.h"

namespace gpu_model {

struct CycleTimelineOptions {
  uint32_t max_columns = 120;
  std::optional<uint64_t> cycle_begin;
  std::optional<uint64_t> cycle_end;
};

class CycleTimelineRenderer {
 public:
  static std::string RenderAscii(const std::vector<TraceEvent>& events,
                                 CycleTimelineOptions options = {});
};

}  // namespace gpu_model
