#include "gpu_model/debug/trace_sink.h"

#include <stdexcept>
#include <string_view>

namespace gpu_model {

namespace {

std::string_view KindToString(TraceEventKind kind) {
  switch (kind) {
    case TraceEventKind::Launch:
      return "Launch";
    case TraceEventKind::BlockPlaced:
      return "BlockPlaced";
    case TraceEventKind::WaveStep:
      return "WaveStep";
    case TraceEventKind::ExecMaskUpdate:
      return "ExecMaskUpdate";
    case TraceEventKind::MemoryAccess:
      return "MemoryAccess";
    case TraceEventKind::Barrier:
      return "Barrier";
    case TraceEventKind::WaveExit:
      return "WaveExit";
    case TraceEventKind::Stall:
      return "Stall";
    case TraceEventKind::Arrive:
      return "Arrive";
  }
  return "Unknown";
}

}  // namespace

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
  out_ << "cycle=" << event.cycle << " kind=" << KindToString(event.kind)
       << " block=" << event.block_id << " wave=" << event.wave_id << " pc=" << event.pc
       << " msg=" << event.message << '\n';
}

JsonTraceSink::JsonTraceSink(const std::filesystem::path& path) : out_(path) {
  if (!out_) {
    throw std::runtime_error("failed to open json trace file");
  }
}

void JsonTraceSink::OnEvent(const TraceEvent& event) {
  out_ << "{\"cycle\":" << event.cycle << ",\"kind\":\"" << KindToString(event.kind)
       << "\",\"block_id\":" << event.block_id << ",\"wave_id\":" << event.wave_id
       << ",\"pc\":" << event.pc << ",\"message\":\"" << EscapeJson(event.message) << "\"}\n";
}

std::string JsonTraceSink::EscapeJson(const std::string& text) {
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

}  // namespace gpu_model
