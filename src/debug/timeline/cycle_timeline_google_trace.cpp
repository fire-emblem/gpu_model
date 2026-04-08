#include "cycle_timeline_internal.h"

#include <algorithm>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <string_view>

#include "gpu_model/debug/trace/event_export.h"
#include "../trace_json_fields_internal.h"
#include "gpu_model/execution/internal/tensor_op_utils.h"

namespace gpu_model {

namespace {

std::string RuntimeLabel() {
  return "Runtime";
}

std::string HexU64(uint64_t value) {
  std::ostringstream out;
  out << "0x" << std::hex << std::nouppercase << value;
  return out.str();
}

std::string MarkerNameImpl(const Marker& marker) {
  const auto& fields = marker.semantic.fields;
  if (!fields.presentation_name.empty()) {
    return fields.presentation_name;
  }
  if (!fields.canonical_name.empty()) {
    return fields.canonical_name;
  }
  if (!fields.display_name.empty()) {
    return fields.display_name;
  }
  if (!fields.compatibility_message.empty()) {
    return fields.compatibility_message;
  }
  return "event";
}

std::string MarkerCategory(const Marker& marker) {
  const auto& fields = marker.semantic.fields;
  if (!fields.category.empty()) {
    return fields.category;
  }
  return "marker";
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
      << segment.commit_cycle << ",\"render_duration_cycles\":"
      << segment.render_duration_cycles;
  return out.str();
}

std::string RuntimeArgs(const TraceEventExportFields& fields) {
  std::ostringstream out;
  out << "\"message\":\"" << EscapeTraceJson(fields.compatibility_message) << "\"";
  AppendTraceExportJsonFields(out, fields, {});
  return out.str();
}

TraceEventExportFields MarkerFields(const Marker& marker) {
  return marker.semantic.fields;
}

std::string MarkerArgs(const SlotKey& key, const Marker& marker) {
  std::ostringstream out;
  out << "\"dpc\":" << key.dpc_id << ",\"ap\":" << key.ap_id << ",\"peu\":" << key.peu_id
      << ",\"slot\":" << key.slot_id << ",\"block\":" << marker.semantic.block_id << ",\"wave\":"
      << marker.semantic.wave_id << ",\"cycle\":" << marker.semantic.cycle;
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
      declared_rows.insert(DescribeRow(key, group_by, marker.semantic.block_id));
    }
  }
  for (const auto& [key, endpoints] : data.async_memory_flow_endpoints) {
    for (const auto& endpoint : endpoints) {
      declared_rows.insert(DescribeRow(key, group_by, endpoint.semantic.block_id));
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
    append("{\"name\":\"" + EscapeTraceJson(runtime_event.fields.presentation_name) +
           "\",\"cat\":\"" + EscapeTraceJson(runtime_event.fields.category) +
           "\",\"ph\":\"i\",\"s\":\"g\",\"pid\":0,\"tid\":0,\"ts\":" +
           std::to_string(runtime_event.cycle) + ",\"args\":{" +
           RuntimeArgs(runtime_event.fields) + "}}");
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
      const uint64_t render_end_cycle = segment.issue_cycle + segment.render_duration_cycles;
      if (render_end_cycle < begin || segment.issue_cycle > end) {
        continue;
      }
      const uint64_t clipped_begin = std::max(begin, segment.issue_cycle);
      const uint64_t clipped_end = std::min(end, render_end_cycle);
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
      const RowDescriptor row = DescribeRow(key, group_by, marker.semantic.block_id);
      if (marker.semantic.cycle < begin || marker.semantic.cycle > end) {
        continue;
      }
      append("{\"name\":\"" + EscapeTraceJson(MarkerEventName(marker)) +
           "\",\"cat\":\"" + EscapeTraceJson(MarkerCategory(marker)) +
           "\",\"ph\":\"i\",\"s\":\"t\",\"pid\":" + std::to_string(row.pid) +
           ",\"tid\":" + std::to_string(row.tid) + ",\"ts\":" +
           std::to_string(marker.semantic.cycle) + ",\"args\":{" + MarkerArgs(key, marker) + "}}");
    }
  }

  for (const auto& [key, endpoints] : data.async_memory_flow_endpoints) {
    for (const auto& endpoint : endpoints) {
      const auto& fields = endpoint.semantic.fields;
      if (!IsAsyncMemoryFlowSemanticEvent(endpoint.semantic)) {
        continue;
      }
      if (endpoint.semantic.cycle < begin || endpoint.semantic.cycle > end) {
        continue;
      }

      const char phase =
          fields.flow_phase == "start"   ? 's'
          : fields.flow_phase == "finish" ? 'f'
                                         : '\0';
      if (phase == '\0') {
        continue;
      }

      const RowDescriptor row = DescribeRow(key, group_by, endpoint.semantic.block_id);
      append("{\"name\":\"async_memory\",\"cat\":\"flow/async_memory\",\"ph\":\"" +
             std::string(1, phase) + "\",\"pid\":" + std::to_string(row.pid) +
             ",\"tid\":" + std::to_string(row.tid) + ",\"ts\":" +
             std::to_string(endpoint.semantic.cycle) + ",\"id\":\"" +
             EscapeTraceJson(fields.flow_id) +
             (phase == 'f' ? "\",\"bp\":\"e\"}" : "\"}"));
    }
  }

  out << "],\"metadata\":" << MetadataJson(data) << "}\n";
  return out.str();
}

}  // namespace gpu_model
