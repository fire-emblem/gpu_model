#pragma once

#include <string>
#include <string_view>

#include "gpu_model/debug/recorder/recorder.h"
#include "gpu_model/debug/trace/event.h"

namespace gpu_model {

std::string_view TraceEventKindName(TraceEventKind kind);
std::string FormatTextTraceEventLine(const TraceEvent& event);
std::string FormatJsonTraceEventLine(const TraceEvent& event);
std::string FormatTextTraceEventLine(const RecorderProgramEvent& event);
std::string FormatTextTraceEventLine(const RecorderEntry& event);
std::string FormatJsonTraceEventLine(const RecorderProgramEvent& event);
std::string FormatJsonTraceEventLine(const RecorderEntry& event);

}  // namespace gpu_model
