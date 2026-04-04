#include "trace_perfetto_proto.h"

#include <tuple>

namespace gpu_model {

namespace {

void AppendVarint(uint64_t value, std::string& out) {
  while (value >= 0x80) {
    out.push_back(static_cast<char>((value & 0x7fu) | 0x80u));
    value >>= 7;
  }
  out.push_back(static_cast<char>(value));
}

void AppendKey(uint32_t field_number, uint32_t wire_type, std::string& out) {
  AppendVarint((static_cast<uint64_t>(field_number) << 3u) | wire_type, out);
}

void AppendLengthDelimited(uint32_t field_number, std::string_view payload, std::string& out) {
  AppendKey(field_number, 2u, out);
  AppendVarint(payload.size(), out);
  out.append(payload.data(), payload.size());
}

void AppendVarintField(uint32_t field_number, uint64_t value, std::string& out) {
  AppendKey(field_number, 0u, out);
  AppendVarint(value, out);
}

void AppendStringField(uint32_t field_number, std::string_view value, std::string& out) {
  AppendLengthDelimited(field_number, value, out);
}

}  // namespace

bool TrackDescriptorNode::operator<(const TrackDescriptorNode& other) const {
  return std::tie(uuid, parent_uuid, name, sibling_order_rank, child_ordering) <
         std::tie(other.uuid, other.parent_uuid, other.name, other.sibling_order_rank,
                  other.child_ordering);
}

std::string EncodeTrackDescriptorPacket(const TrackDescriptorNode& node) {
  std::string descriptor;
  AppendVarintField(1, node.uuid, descriptor);
  if (node.parent_uuid.has_value()) {
    AppendVarintField(5, *node.parent_uuid, descriptor);
  }
  AppendStringField(2, node.name, descriptor);
  if (node.child_ordering.has_value()) {
    AppendVarintField(11, *node.child_ordering, descriptor);
  }
  if (node.sibling_order_rank.has_value()) {
    AppendVarintField(12, static_cast<uint64_t>(static_cast<uint32_t>(*node.sibling_order_rank)),
                      descriptor);
  }

  std::string packet;
  AppendLengthDelimited(60, descriptor, packet);
  return packet;
}

std::string EncodeTrackEventPacket(uint64_t timestamp,
                                   uint64_t track_uuid,
                                   std::optional<uint32_t> type,
                                   std::optional<std::string_view> name) {
  std::string event;
  if (type.has_value()) {
    AppendVarintField(9, *type, event);
  }
  AppendVarintField(11, track_uuid, event);
  if (name.has_value()) {
    AppendStringField(23, *name, event);
  }

  std::string packet;
  AppendVarintField(8, timestamp, packet);
  AppendLengthDelimited(11, event, packet);
  return packet;
}

void AppendTracePacket(std::string_view packet, std::string& out) {
  AppendKey(1, 2u, out);
  AppendVarint(packet.size(), out);
  out.append(packet.data(), packet.size());
}

}  // namespace gpu_model
