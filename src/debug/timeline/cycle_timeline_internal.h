#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "gpu_model/debug/recorder/recorder.h"
#include "gpu_model/debug/trace/event.h"
#include "gpu_model/debug/trace/event_export.h"
#include "gpu_model/debug/timeline/cycle_timeline.h"

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
  uint64_t render_duration_cycles = 0;
  std::string op;
  std::string slot_model;
  uint32_t block_id = 0;
  uint32_t wave_id = 0;
  uint64_t pc = 0;
};

struct TimelineSemanticEvent {
  TraceEventKind kind = TraceEventKind::Launch;
  uint64_t cycle = 0;
  uint32_t dpc_id = 0;
  uint32_t ap_id = 0;
  uint32_t peu_id = 0;
  uint32_t slot_id = 0;
  uint32_t block_id = 0;
  uint32_t wave_id = 0;
  uint64_t pc = 0;
  TraceSlotModelKind slot_model_kind = TraceSlotModelKind::None;
  TraceStallReason stall_reason = TraceStallReason::None;
  TraceBarrierKind barrier_kind = TraceBarrierKind::None;
  TraceArriveKind arrive_kind = TraceArriveKind::None;
  TraceArriveProgressKind arrive_progress = TraceArriveProgressKind::None;
  TraceLifecycleStage lifecycle_stage = TraceLifecycleStage::None;
  TraceWaitcntState waitcnt_state;
  TraceEventExportFields fields;
};

struct Marker {
  char symbol = '.';
  TimelineSemanticEvent semantic;
};

struct TimelineData {
  std::map<SlotKey, std::vector<Segment>> segments;
  std::map<SlotKey, std::vector<Marker>> markers;
  std::unordered_map<std::string, char> symbols;
  std::set<std::string> slot_models;
  std::vector<TimelineSemanticEvent> runtime_events;
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

}  // namespace gpu_model
