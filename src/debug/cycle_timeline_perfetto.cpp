#include "cycle_timeline_internal.h"

#include <algorithm>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include "gpu_model/debug/trace_event_view.h"
#include "trace_perfetto_proto.h"

namespace gpu_model {

namespace {

uint64_t Fnv1a64(std::string_view text) {
  uint64_t hash = 1469598103934665603ull;
  for (const unsigned char ch : text) {
    hash ^= static_cast<uint64_t>(ch);
    hash *= 1099511628211ull;
  }
  return hash;
}

uint64_t MakeTrackUuid(std::string_view kind, std::string_view label) {
  const uint64_t hash = Fnv1a64(std::string(kind) + ":" + std::string(label));
  return hash == 0 ? 1 : hash;
}

std::vector<TrackDescriptorNode> BuildPerfettoTrackTree(const TimelineData& data) {
  std::set<TrackDescriptorNode> nodes;

  const uint64_t device_uuid = MakeTrackUuid("track", "Device");
  nodes.insert(TrackDescriptorNode{.uuid = device_uuid,
                                   .parent_uuid = std::nullopt,
                                   .name = "Device",
                                   .sibling_order_rank = std::nullopt,
                                   .child_ordering = 3u});

  const uint64_t runtime_uuid = MakeTrackUuid("track", "Runtime");
  nodes.insert(TrackDescriptorNode{.uuid = runtime_uuid,
                                   .parent_uuid = std::nullopt,
                                   .name = "Runtime",
                                   .sibling_order_rank = std::nullopt,
                                   .child_ordering = std::nullopt});

  auto append_slot_path = [&](const SlotKey& key) {
    const std::string dpc_name = DpcLabel(key);
    const uint64_t dpc_uuid = MakeTrackUuid("track", dpc_name);
    nodes.insert(TrackDescriptorNode{.uuid = dpc_uuid,
                                     .parent_uuid = device_uuid,
                                     .name = dpc_name,
                                     .sibling_order_rank = static_cast<int32_t>(key.dpc_id),
                                     .child_ordering = 3u});

    const std::string ap_path = dpc_name + "/" + ApLabel(key);
    const uint64_t ap_uuid = MakeTrackUuid("track", ap_path);
    nodes.insert(TrackDescriptorNode{.uuid = ap_uuid,
                                     .parent_uuid = dpc_uuid,
                                     .name = ApLabel(key),
                                     .sibling_order_rank = static_cast<int32_t>(key.ap_id),
                                     .child_ordering = 3u});

    const std::string peu_path = ap_path + "/" + PeuLabel(key);
    const uint64_t peu_uuid = MakeTrackUuid("track", peu_path);
    nodes.insert(TrackDescriptorNode{.uuid = peu_uuid,
                                     .parent_uuid = ap_uuid,
                                     .name = PeuLabel(key),
                                     .sibling_order_rank = static_cast<int32_t>(key.peu_id),
                                     .child_ordering = 3u});

    const std::string slot_path = peu_path + "/" + SlotLabel(key);
    const uint64_t slot_uuid = MakeTrackUuid("track", slot_path);
    nodes.insert(TrackDescriptorNode{.uuid = slot_uuid,
                                     .parent_uuid = peu_uuid,
                                     .name = SlotLabel(key),
                                     .sibling_order_rank = static_cast<int32_t>(key.slot_id),
                                     .child_ordering = std::nullopt});
  };

  for (const auto& [key, row_segments] : data.segments) {
    if (!row_segments.empty()) {
      append_slot_path(key);
    }
  }
  for (const auto& [key, row_markers] : data.markers) {
    if (!row_markers.empty()) {
      append_slot_path(key);
    }
  }
  return {nodes.begin(), nodes.end()};
}

uint64_t SlotTrackUuid(const SlotKey& key) {
  const std::string path = DpcLabel(key) + "/" + ApLabel(key) + "/" + PeuLabel(key) + "/" +
                           SlotLabel(key);
  return MakeTrackUuid("track", path);
}

}  // namespace

std::string RenderPerfettoTraceExport(const TimelineData& data,
                                      uint64_t begin,
                                      uint64_t end) {
  if (end < begin) {
    return {};
  }

  std::string trace;
  const std::vector<TrackDescriptorNode> tracks = BuildPerfettoTrackTree(data);
  for (const auto& track : tracks) {
    AppendTracePacket(EncodeTrackDescriptorPacket(track), trace);
  }

  const uint64_t runtime_track_uuid = MakeTrackUuid("track", "Runtime");
  for (const auto& runtime_event : data.runtime_events) {
    if (runtime_event.cycle < begin || runtime_event.cycle > end) {
      continue;
    }
    const TraceEventView view = MakeTraceEventView(runtime_event);
    AppendTracePacket(
        EncodeTrackEventPacket(runtime_event.cycle, runtime_track_uuid, 3u, view.presentation_name),
        trace);
  }

  for (const auto& [key, row_segments] : data.segments) {
    const uint64_t track_uuid = SlotTrackUuid(key);
    for (const auto& segment : row_segments) {
      if (segment.commit_cycle < begin || segment.issue_cycle > end) {
        continue;
      }
      const uint64_t clipped_begin = std::max(begin, segment.issue_cycle);
      const uint64_t clipped_end = std::min(end, segment.commit_cycle);
      AppendTracePacket(
          EncodeTrackEventPacket(clipped_begin, track_uuid, 1u, segment.op), trace);
      AppendTracePacket(EncodeTrackEventPacket(
                            clipped_end > clipped_begin ? clipped_end : clipped_begin + 1,
                            track_uuid,
                            2u,
                            std::nullopt),
                        trace);
    }
  }

  for (const auto& [key, row_markers] : data.markers) {
    const uint64_t track_uuid = SlotTrackUuid(key);
    for (const auto& marker : row_markers) {
      if (marker.cycle < begin || marker.cycle > end) {
        continue;
      }
      AppendTracePacket(
          EncodeTrackEventPacket(marker.cycle, track_uuid, 3u, MarkerEventName(marker)), trace);
    }
  }

  return trace;
}

}  // namespace gpu_model
