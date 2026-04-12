#include "debug/timeline/timeline_comparator.h"

#include <algorithm>
#include <sstream>
#include <variant>

namespace gpu_model {

namespace {

std::string FormatLane(const TimelineLaneKey& lane) {
  std::ostringstream oss;
  oss << "DPC_" << lane.dpc_id << "/AP_" << lane.ap_id << "/PEU_" << lane.peu_id
      << "/WAVE_SLOT_" << lane.slot_id << "/WAVE_" << lane.wave_id;
  return oss.str();
}

std::string FormatKey(const TimelineEventKey& key) {
  std::ostringstream oss;
  oss << "lane=" << FormatLane(key.lane) << " pc=0x" << std::hex << key.pc << std::dec
      << " name=" << key.name;
  return oss.str();
}

bool Matches(const ExpectedSlice& expected, const ActualSlice& actual) {
  return expected.key == actual.key && expected.begin_cycle == actual.begin_cycle &&
         expected.end_cycle == actual.end_cycle;
}

bool Matches(const ExpectedMarker& expected, const ActualMarker& actual) {
  if (!(expected.key == actual.key) || expected.cycle != actual.cycle) {
    return false;
  }
  if (expected.stall_reason.has_value() && actual.stall_reason != *expected.stall_reason) {
    return false;
  }
  if (expected.arrive_progress.has_value() && actual.arrive_progress != *expected.arrive_progress) {
    return false;
  }
  return true;
}

struct ActualOccurrence {
  uint64_t sequence = 0;
};

std::optional<ActualOccurrence> FindOccurrence(const ActualTimelineSnapshot& actual,
                                               const TimelineEventKey& key) {
  for (const auto& marker : actual.markers) {
    if (marker.key == key) {
      return ActualOccurrence{.sequence = marker.sequence};
    }
  }
  for (const auto& slice : actual.slices) {
    if (slice.key == key) {
      return ActualOccurrence{.sequence = slice.sequence};
    }
  }
  return std::nullopt;
}

}  // namespace

TimelineComparisonResult CompareTimeline(const ExpectedTimeline& expected,
                                         const ActualTimelineSnapshot& actual) {
  for (const auto& slice : expected.required_slices) {
    const auto it =
        std::find_if(actual.slices.begin(), actual.slices.end(), [&](const ActualSlice& candidate) {
          return Matches(slice, candidate);
        });
    if (it == actual.slices.end()) {
      std::ostringstream oss;
      oss << "missing slice: " << FormatKey(slice.key) << " expected=[" << slice.begin_cycle << ","
          << slice.end_cycle << ")";
      return {.ok = false, .message = oss.str()};
    }
  }

  for (const auto& marker : expected.required_markers) {
    const auto it = std::find_if(actual.markers.begin(),
                                 actual.markers.end(),
                                 [&](const ActualMarker& candidate) { return Matches(marker, candidate); });
    if (it == actual.markers.end()) {
      std::ostringstream oss;
      oss << "missing marker: " << FormatKey(marker.key) << " cycle=" << marker.cycle;
      return {.ok = false, .message = oss.str()};
    }
  }

  for (const auto& forbidden : expected.forbidden_slices) {
    const auto it = std::find_if(actual.slices.begin(),
                                 actual.slices.end(),
                                 [&](const ActualSlice& candidate) { return candidate.key == forbidden; });
    if (it != actual.slices.end()) {
      std::ostringstream oss;
      oss << "unexpected slice: " << FormatKey(forbidden) << " actual=[" << it->begin_cycle << ","
          << it->end_cycle << ")";
      return {.ok = false, .message = oss.str()};
    }
  }

  for (const auto& ordering : expected.ordering) {
    const auto earlier = FindOccurrence(actual, ordering.earlier);
    const auto later = FindOccurrence(actual, ordering.later);
    if (!earlier.has_value()) {
      return {.ok = false, .message = "ordering missing earlier: " + FormatKey(ordering.earlier)};
    }
    if (!later.has_value()) {
      return {.ok = false, .message = "ordering missing later: " + FormatKey(ordering.later)};
    }
    if (earlier->sequence >= later->sequence) {
      return {.ok = false,
              .message = "ordering violation: " + ordering.earlier.name + " should appear before " +
                         ordering.later.name};
    }
  }

  return {.ok = true, .message = {}};
}

}  // namespace gpu_model
