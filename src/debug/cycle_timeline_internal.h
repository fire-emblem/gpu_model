#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "gpu_model/debug/trace_event.h"
#include "gpu_model/debug/cycle_timeline.h"

namespace gpu_model {

struct SlotKey {
  uint32_t dpc_id = 0;
  uint32_t ap_id = 0;
  uint32_t peu_id = 0;
  uint32_t slot_id = 0;

  bool operator<(const SlotKey& other) const;
};

struct RowDescriptor {
  uint32_t pid = 0;
  uint32_t tid = 0;
  int32_t process_sort_index = 0;
  int32_t thread_sort_index = 0;
  std::string process_name;
  std::string thread_name;

  bool operator<(const RowDescriptor& other) const;
};

struct Segment {
  uint64_t issue_cycle = 0;
  uint64_t commit_cycle = 0;
  std::string op;
  std::string slot_model;
  uint32_t block_id = 0;
  uint32_t wave_id = 0;
  uint64_t pc = 0;
};

struct Marker {
  uint64_t cycle = 0;
  char symbol = '.';
  TraceEventKind kind = TraceEventKind::Launch;
  TraceStallReason stall_reason = TraceStallReason::None;
  TraceBarrierKind barrier_kind = TraceBarrierKind::None;
  TraceArriveKind arrive_kind = TraceArriveKind::None;
  TraceLifecycleStage lifecycle_stage = TraceLifecycleStage::None;
  std::string canonical_name;
  std::string presentation_name;
  std::string display_name;
  std::string category;
  std::string message;
  std::string slot_model;
  std::string stall_reason_name;
  std::string barrier_kind_name;
  std::string arrive_kind_name;
  std::string lifecycle_stage_name;
  uint32_t block_id = 0;
  uint32_t wave_id = 0;
};

struct TimelineData {
  std::map<SlotKey, std::vector<Segment>> segments;
  std::map<SlotKey, std::vector<Marker>> markers;
  std::unordered_map<std::string, char> symbols;
  std::set<std::string> slot_models;
  std::vector<TraceEvent> runtime_events;
};

std::string SlotLabel(const SlotKey& key);
std::string PeuLabel(const SlotKey& key);
std::string ApLabel(const SlotKey& key);
std::string DpcLabel(const SlotKey& key);
std::string ProcessName(const SlotKey& key, CycleTimelineGroupBy group_by);
RowDescriptor DescribeRow(const SlotKey& key,
                          CycleTimelineGroupBy group_by,
                          std::optional<uint32_t> block_id = std::nullopt);

std::string MarkerEventName(const Marker& marker);

std::string RenderGoogleTraceExport(const TimelineData& data,
                                    uint64_t begin,
                                    uint64_t end,
                                    CycleTimelineGroupBy group_by);

std::string RenderPerfettoTraceExport(const TimelineData& data,
                                      uint64_t begin,
                                      uint64_t end);

std::string RenderAsciiTimelineExport(const TimelineData& data,
                                      uint64_t begin,
                                      uint64_t end,
                                      uint32_t max_columns,
                                      CycleTimelineGroupBy group_by);

}  // namespace gpu_model
