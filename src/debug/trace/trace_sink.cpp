#include "gpu_model/debug/trace/sink.h"

#include <stdexcept>

#include "../trace_format_internal.h"

namespace gpu_model {

void NullTraceSink::OnEvent(const TraceEvent& event) {
  (void)event;
}

void CollectingTraceSink::OnEvent(const TraceEvent& event) {
  events_.push_back(event);
}

FileTraceSink::FileTraceSink(const std::filesystem::path& path) : out_(path) {
  if (!out_) {
    throw std::runtime_error("failed to open trace file");
  }
}

void FileTraceSink::OnEvent(const TraceEvent& event) {
  out_ << FormatTextTraceEventLine(event);
}

JsonTraceSink::JsonTraceSink(const std::filesystem::path& path) : out_(path) {
  if (!out_) {
    throw std::runtime_error("failed to open json trace file");
  }
}

void JsonTraceSink::OnEvent(const TraceEvent& event) {
  out_ << FormatJsonTraceEventLine(event);
}

}  // namespace gpu_model
