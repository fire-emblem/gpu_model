#pragma once

#include <sstream>
#include <string>
#include <string_view>

namespace gpu_model {

std::string EscapeTraceJson(std::string_view text);

void AppendOptionalTraceJsonStringField(std::ostringstream& out,
                                        std::string_view key,
                                        std::string_view value);

void AppendTracePresentationJsonFields(std::ostringstream& out,
                                       std::string_view canonical_name,
                                       std::string_view presentation_name,
                                       std::string_view display_name,
                                       std::string_view category,
                                       std::string_view compatibility_message);

}  // namespace gpu_model
