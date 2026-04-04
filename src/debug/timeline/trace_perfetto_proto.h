#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace gpu_model {

struct TrackDescriptorNode {
  uint64_t uuid = 0;
  std::optional<uint64_t> parent_uuid;
  std::string name;
  std::optional<int32_t> sibling_order_rank;
  std::optional<uint32_t> child_ordering;

  bool operator<(const TrackDescriptorNode& other) const;
};

std::string EncodeTrackDescriptorPacket(const TrackDescriptorNode& node);
std::string EncodeTrackEventPacket(uint64_t timestamp,
                                   uint64_t track_uuid,
                                   std::optional<uint32_t> type,
                                   std::optional<std::string_view> name);
void AppendTracePacket(std::string_view packet, std::string& out);

}  // namespace gpu_model
