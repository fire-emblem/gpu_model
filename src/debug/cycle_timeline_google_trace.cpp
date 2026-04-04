#include "cycle_timeline_internal.h"

#include <algorithm>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <string_view>

#include "gpu_model/debug/trace_event_export.h"
#include "gpu_model/debug/trace_event_view.h"
#include "gpu_model/debug/trace_json_fields.h"
#include "gpu_model/execution/internal/tensor_op_utils.h"

namespace gpu_model {

namespace {

std::string RuntimeLabel() {
  return "Runtime";
}

std::string_view CanonicalizeReasonLabel(std::string_view reason) {
  std::string_view rest = reason;
  const size_t start = rest.find_first_not_of(" \t\n");
  if (start == std::string_view::npos) {
    return {};
  }
  rest.remove_prefix(start);
  const size_t end = rest.find_first_of(" \t\n");
  if (end == std::string_view::npos) {
    return rest;
  }
  return rest.substr(0, end);
}

std::string StallLabel(std::string_view message) {
  const std::string_view reason = CanonicalizeReasonLabel(TraceStallReasonPayload(message));
  if (!reason.empty()) {
    return std::string(reason);
  }
  return std::string(message);
}

std::string StallLabel(const Marker& marker) {
  if (marker.stall_reason != TraceStallReason::None) {
    return std::string(TraceStallReasonName(marker.stall_reason));
  }
  return StallLabel(marker.message);
}

std::string HexU64(uint64_t value) {
  std::ostringstream out;
  out << "0x" << std::hex << std::nouppercase << value;
  return out.str();
}

std::string MarkerNameImpl(const Marker& marker) {
  if (!marker.presentation_name.empty()) {
    return marker.presentation_name;
  }
  switch (marker.kind) {
    case TraceEventKind::Arrive:
      return marker.message.empty() ? "arrive" : marker.message;
    case TraceEventKind::Barrier:
      return marker.message.empty() ? "barrier" : "barrier_" + marker.message;
    case TraceEventKind::WaveExit:
      return "wave_exit";
    case TraceEventKind::Stall:
      if (StallLabel(marker) == kTraceStallReasonWarpSwitch) {
        return "wave_switch_away";
      }
      if (marker.message.empty() && marker.stall_reason == TraceStallReason::None) {
        return "stall";
      }
      return "stall_" + StallLabel(marker);
    case TraceEventKind::WaveLaunch:
      return "wave_launch";
    case TraceEventKind::BlockLaunch:
      return "block_launch";
    default:
      return marker.message.empty() ? "event" : marker.message;
  }
}

std::string MarkerCategory(const Marker& marker) {
  if (!marker.category.empty()) {
    return marker.category;
  }
  switch (marker.kind) {
    case TraceEventKind::Arrive:
      return marker.canonical_name.empty() ? "memory/arrive" : "memory/" + marker.canonical_name;
    case TraceEventKind::Barrier:
      return "sync/barrier";
    case TraceEventKind::WaveExit:
      return "control/exit";
    case TraceEventKind::Stall:
      if (StallLabel(marker) == kTraceStallReasonWarpSwitch) {
        return "wave/switch_away";
      }
      if (marker.message.empty() && marker.stall_reason == TraceStallReason::None) {
        return "stall";
      }
      return "stall/" + StallLabel(marker);
    case TraceEventKind::WaveLaunch:
      return "launch/wave";
    case TraceEventKind::BlockLaunch:
      return "launch/block";
    default:
      return "marker";
  }
}

std::string SegmentArgs(const SlotKey& key, const Segment& segment) {
  std::ostringstream out;
  out << "\"dpc\":" << key.dpc_id << ",\"ap\":" << key.ap_id << ",\"peu\":" << key.peu_id
      << ",\"slot\":" << key.slot_id << ",\"block\":" << segment.block_id << ",\"wave\":"
      << segment.wave_id;
  if (!segment.slot_model.empty()) {
    out << ",\"slot_model\":\"" << EscapeTraceJson(segment.slot_model) << "\"";
  }
  if (segment.pc != 0) {
    out << ",\"pc\":\"" << EscapeTraceJson(HexU64(segment.pc)) << "\"";
  }
  out << ",\"issue_cycle\":" << segment.issue_cycle << ",\"commit_cycle\":"
      << segment.commit_cycle;
  return out.str();
}

std::string RuntimeArgs(const TraceEventExportFields& fields) {
  std::ostringstream out;
  out << "\"message\":\"" << EscapeTraceJson(fields.compatibility_message) << "\"";
  AppendTraceExportJsonFields(out, fields, {});
  return out.str();
}

TraceEventExportFields MarkerFields(const Marker& marker) {
  return TraceEventExportFields{.slot_model = marker.slot_model,
                                .stall_reason = marker.stall_reason_name,
                                .barrier_kind = marker.barrier_kind_name,
                                .arrive_kind = marker.arrive_kind_name,
                                .lifecycle_stage = marker.lifecycle_stage_name,
                                .waitcnt_thresholds = {},
                                .waitcnt_pending = {},
                                .waitcnt_blocked_domains = {},
                                .canonical_name = marker.canonical_name,
                                .presentation_name = marker.presentation_name,
                                .display_name = marker.display_name,
                                .category = marker.category,
                                .compatibility_message = marker.message};
}

std::string MarkerArgs(const SlotKey& key, const Marker& marker) {
  std::ostringstream out;
  out << "\"dpc\":" << key.dpc_id << ",\"ap\":" << key.ap_id << ",\"peu\":" << key.peu_id
      << ",\"slot\":" << key.slot_id << ",\"block\":" << marker.block_id << ",\"wave\":"
      << marker.wave_id << ",\"cycle\":" << marker.cycle;
  AppendTraceExportJsonFields(out, MarkerFields(marker));
  return out.str();
}

std::string MetadataJson(const TimelineData& data, std::optional<std::string_view> error = std::nullopt) {
  std::ostringstream out;
  out << "{\"time_unit\":\"cycle\",\"slot_models\":[";
  bool first = true;
  for (const auto& slot_model : data.slot_models) {
    if (!first) {
      out << ',';
    }
    first = false;
    out << "\"" << EscapeTraceJson(slot_model) << "\"";
  }
  out << "],\"hierarchy_levels\":[\"Device\",\"DPC\",\"AP\",\"PEU\",\"WAVE_SLOT\"]"
      << ",\"label_style\":\"numeric\""
      << ",\"track_layout\":\"flattened_path_process_plus_slot_thread\""
      << ",\"perfetto_format\":\"chrome_json\""
      << ",\"perfetto_hierarchy_note\":\"chrome_json is limited to process/thread tracks; "
         "DPC/AP/PEU/WAVE_SLOT are encoded as flattened path labels until a native "
         "TrackDescriptor exporter is added\"";
  if (error.has_value()) {
    out << ",\"error\":\"" << EscapeTraceJson(*error) << "\"";
  }
  out << "}";
  return out.str();
}

std::set<RowDescriptor> CollectDeclaredRows(const TimelineData& data, CycleTimelineGroupBy group_by) {
  std::set<RowDescriptor> declared_rows;
  for (const auto& [key, row_segments] : data.segments) {
    for (const auto& segment : row_segments) {
      declared_rows.insert(DescribeRow(key, group_by, segment.block_id));
    }
  }
  for (const auto& [key, row_markers] : data.markers) {
    for (const auto& marker : row_markers) {
      declared_rows.insert(DescribeRow(key, group_by, marker.block_id));
    }
  }
  return declared_rows;
}

}  // namespace

std::string MarkerEventName(const Marker& marker) {
  return MarkerNameImpl(marker);
}

std::string RenderGoogleTraceExport(const TimelineData& data,
                                    uint64_t begin,
                                    uint64_t end,
                                    CycleTimelineGroupBy group_by) {
  if (end < begin) {
    return "{\"traceEvents\":[],\"metadata\":" + MetadataJson(data, "invalid_range") + "}\n";
  }

  std::ostringstream out;
  out << "{\"traceEvents\":[";

  bool first = true;
  auto append = [&](const std::string& text) {
    if (!first) {
      out << ',';
    }
    first = false;
    out << text;
  };

  append("{\"name\":\"process_name\",\"ph\":\"M\",\"pid\":0,\"tid\":0,\"args\":{\"name\":\"" +
         EscapeTraceJson(RuntimeLabel()) + "\"}}");
  for (const auto& runtime_event : data.runtime_events) {
    if (runtime_event.cycle < begin || runtime_event.cycle > end) {
      continue;
    }
    const TraceEventView view = MakeTraceEventView(runtime_event);
    const TraceEventExportFields fields = MakeTraceEventExportFields(view);
    append("{\"name\":\"" + EscapeTraceJson(view.presentation_name) +
           "\",\"cat\":\"" + EscapeTraceJson(view.category) +
           "\",\"ph\":\"i\",\"s\":\"g\",\"pid\":0,\"tid\":0,\"ts\":" +
           std::to_string(runtime_event.cycle) + ",\"args\":{" + RuntimeArgs(fields) + "}}");
  }

  const std::set<RowDescriptor> declared_rows = CollectDeclaredRows(data, group_by);
  std::set<uint32_t> declared_processes;
  for (const auto& row : declared_rows) {
    if (declared_processes.insert(row.pid).second) {
      append("{\"name\":\"process_name\",\"ph\":\"M\",\"pid\":" + std::to_string(row.pid) +
             ",\"tid\":0,\"args\":{\"name\":\"" + EscapeTraceJson(row.process_name) + "\"}}");
      append("{\"name\":\"process_sort_index\",\"ph\":\"M\",\"pid\":" + std::to_string(row.pid) +
             ",\"tid\":0,\"args\":{\"sort_index\":" +
             std::to_string(row.process_sort_index) + "}}");
    }
    append("{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":" + std::to_string(row.pid) +
           ",\"tid\":" + std::to_string(row.tid) + ",\"args\":{\"name\":\"" +
           EscapeTraceJson(row.thread_name) + "\"}}");
    append("{\"name\":\"thread_sort_index\",\"ph\":\"M\",\"pid\":" + std::to_string(row.pid) +
           ",\"tid\":" + std::to_string(row.tid) + ",\"args\":{\"sort_index\":" +
           std::to_string(row.thread_sort_index) + "}}");
  }

  for (const auto& [key, row_segments] : data.segments) {
    for (const auto& segment : row_segments) {
      const RowDescriptor row = DescribeRow(key, group_by, segment.block_id);
      if (segment.commit_cycle < begin || segment.issue_cycle > end) {
        continue;
      }
      const uint64_t clipped_begin = std::max(begin, segment.issue_cycle);
      const uint64_t clipped_end = std::min(end, segment.commit_cycle);
      const uint64_t duration = clipped_end > clipped_begin ? clipped_end - clipped_begin : 1;
      const std::string category = IsTensorMnemonic(segment.op) ? "tensor" : "instruction";
      append("{\"name\":\"" + EscapeTraceJson(segment.op) +
             "\",\"cat\":\"" + category + "\",\"ph\":\"X\",\"pid\":" +
             std::to_string(row.pid) + ",\"tid\":" + std::to_string(row.tid) + ",\"ts\":" +
             std::to_string(clipped_begin) + ",\"dur\":" + std::to_string(duration) +
             ",\"args\":{" + SegmentArgs(key, segment) + "}}");
    }
  }

  for (const auto& [key, row_markers] : data.markers) {
    for (const auto& marker : row_markers) {
      const RowDescriptor row = DescribeRow(key, group_by, marker.block_id);
      if (marker.cycle < begin || marker.cycle > end) {
        continue;
      }
      append("{\"name\":\"" + EscapeTraceJson(MarkerEventName(marker)) +
             "\",\"cat\":\"" + EscapeTraceJson(MarkerCategory(marker)) +
             "\",\"ph\":\"i\",\"s\":\"t\",\"pid\":" + std::to_string(row.pid) +
             ",\"tid\":" + std::to_string(row.tid) + ",\"ts\":" +
             std::to_string(marker.cycle) + ",\"args\":{" + MarkerArgs(key, marker) + "}}");
    }
  }

  out << "],\"metadata\":" << MetadataJson(data) << "}\n";
  return out.str();
}

}  // namespace gpu_model
