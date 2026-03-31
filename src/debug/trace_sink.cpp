#include "gpu_model/debug/trace_sink.h"

#include <iomanip>
#include <stdexcept>
#include <string_view>

namespace gpu_model {

namespace {

std::string HexU64(uint64_t value) {
  std::ostringstream out;
  out << "0x" << std::hex << std::nouppercase << value;
  return out.str();
}

std::string_view KindToString(TraceEventKind kind) {
  switch (kind) {
    case TraceEventKind::Launch:
      return "Launch";
    case TraceEventKind::BlockPlaced:
      return "BlockPlaced";
    case TraceEventKind::BlockLaunch:
      return "BlockLaunch";
    case TraceEventKind::WaveLaunch:
      return "WaveLaunch";
    case TraceEventKind::WaveStats:
      return "WaveStats";
    case TraceEventKind::WaveStep:
      return "WaveStep";
    case TraceEventKind::Commit:
      return "Commit";
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
  out_ << "pc=" << HexU64(event.pc) << " cycle=" << HexU64(event.cycle)
       << " dpc=" << HexU64(event.dpc_id) << " ap=" << HexU64(event.ap_id)
       << " peu=" << HexU64(event.peu_id)
       << " kind=" << KindToString(event.kind) << " block=" << HexU64(event.block_id)
       << " wave=" << HexU64(event.wave_id)
       << " msg=" << event.message << '\n';
}

JsonTraceSink::JsonTraceSink(const std::filesystem::path& path) : out_(path) {
  if (!out_) {
    throw std::runtime_error("failed to open json trace file");
  }
}

void JsonTraceSink::OnEvent(const TraceEvent& event) {
  out_ << "{\"pc\":\"" << HexU64(event.pc) << "\",\"cycle\":\"" << HexU64(event.cycle)
       << "\",\"dpc_id\":\"" << HexU64(event.dpc_id) << "\",\"ap_id\":\""
       << HexU64(event.ap_id) << "\",\"peu_id\":\"" << HexU64(event.peu_id)
       << "\",\"kind\":\"" << KindToString(event.kind) << "\",\"block_id\":\""
       << HexU64(event.block_id) << "\",\"wave_id\":\"" << HexU64(event.wave_id)
       << "\",\"message\":\"" << EscapeJson(event.message) << "\"}\n";
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
