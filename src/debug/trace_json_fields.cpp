#include "gpu_model/debug/trace_json_fields.h"

namespace gpu_model {

std::string EscapeTraceJson(std::string_view text) {
  std::string escaped;
  escaped.reserve(text.size());
  for (const char ch : text) {
    switch (ch) {
      case '\\':
        escaped += "\\\\";
        break;
      case '"':
        escaped += "\\\"";
        break;
      case '\n':
        escaped += "\\n";
        break;
      default:
        escaped.push_back(ch);
        break;
    }
  }
  return escaped;
}

void AppendOptionalTraceJsonStringField(std::ostringstream& out,
                                        std::string_view key,
                                        std::string_view value) {
  if (value.empty()) {
    return;
  }
  out << ",\"" << key << "\":\"" << EscapeTraceJson(value) << "\"";
}

void AppendTracePresentationJsonFields(std::ostringstream& out,
                                       std::string_view canonical_name,
                                       std::string_view presentation_name,
                                       std::string_view display_name,
                                       std::string_view category,
                                       std::string_view compatibility_message) {
  AppendOptionalTraceJsonStringField(out, "canonical_name", canonical_name);
  AppendOptionalTraceJsonStringField(out, "presentation_name", presentation_name);
  AppendOptionalTraceJsonStringField(out, "display_name", display_name);
  AppendOptionalTraceJsonStringField(out, "category", category);
  AppendOptionalTraceJsonStringField(out, "message", compatibility_message);
}

void AppendTraceExportJsonFields(std::ostringstream& out,
                                 const TraceEventExportFields& fields,
                                 std::string_view message_key) {
  AppendOptionalTraceJsonStringField(out, "slot_model", fields.slot_model);
  AppendOptionalTraceJsonStringField(out, "stall_reason", fields.stall_reason);
  AppendOptionalTraceJsonStringField(out, "barrier_kind", fields.barrier_kind);
  AppendOptionalTraceJsonStringField(out, "arrive_kind", fields.arrive_kind);
  AppendOptionalTraceJsonStringField(out, "lifecycle_stage", fields.lifecycle_stage);
  AppendOptionalTraceJsonStringField(out, "waitcnt_thresholds", fields.waitcnt_thresholds);
  AppendOptionalTraceJsonStringField(out, "waitcnt_pending", fields.waitcnt_pending);
  AppendOptionalTraceJsonStringField(out, "waitcnt_blocked_domains",
                                     fields.waitcnt_blocked_domains);
  AppendOptionalTraceJsonStringField(out, "canonical_name", fields.canonical_name);
  AppendOptionalTraceJsonStringField(out, "presentation_name", fields.presentation_name);
  AppendOptionalTraceJsonStringField(out, "display_name", fields.display_name);
  AppendOptionalTraceJsonStringField(out, "category", fields.category);
  if (!message_key.empty()) {
    AppendOptionalTraceJsonStringField(out, message_key, fields.compatibility_message);
  }
}

}  // namespace gpu_model
