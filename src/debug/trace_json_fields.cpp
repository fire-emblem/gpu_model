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

}  // namespace gpu_model
