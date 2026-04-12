#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include "debug/recorder/recorder.h"

namespace gpu_model {

enum class CycleTimelineGroupBy {
  Wave,
  Block,
  Peu,
  Ap,
  Dpc,
};

enum class CycleTimelineMarkerDetail {
  Default,
  Full,
};

struct CycleTimelineOptions {
  uint32_t max_columns = 120;
  std::optional<uint64_t> cycle_begin;
  std::optional<uint64_t> cycle_end;
  CycleTimelineGroupBy group_by = CycleTimelineGroupBy::Wave;
  CycleTimelineMarkerDetail marker_detail = CycleTimelineMarkerDetail::Default;
};

class CycleTimelineRenderer {
 public:
  static std::string RenderGoogleTrace(const Recorder& recorder,
                                       CycleTimelineOptions options = {});
  static std::string RenderPerfettoTraceProto(const Recorder& recorder,
                                              CycleTimelineOptions options = {});
};

}  // namespace gpu_model
