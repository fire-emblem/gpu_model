#include "gpu_model/debug/trace_sink.h"

#include <iomanip>
#include <stdexcept>
#include <string_view>

#include "gpu_model/debug/trace_event_export.h"
#include "gpu_model/debug/trace_json_fields.h"
#include "gpu_model/debug/trace_event_view.h"

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
  const TraceEventView view = MakeTraceEventView(event);
  const TraceEventExportFields fields = MakeTraceEventExportFields(view);
  out_ << "pc=" << HexU64(view.pc) << " cycle=" << HexU64(view.cycle)
       << " dpc=" << HexU64(view.dpc_id) << " ap=" << HexU64(view.ap_id)
       << " peu=" << HexU64(view.peu_id) << " slot=" << HexU64(view.slot_id)
       << " slot_model=" << fields.slot_model
       << " slot_model_kind=" << fields.slot_model
       << " kind=" << KindToString(view.kind) << " block=" << HexU64(view.block_id)
       << " wave=" << HexU64(view.wave_id)
       << " canonical_name=" << fields.canonical_name
       << " presentation_name=" << fields.presentation_name
       << " display_name=" << fields.display_name
       << " category=" << fields.category
       << " stall_reason=" << fields.stall_reason
       << " barrier_kind=" << fields.barrier_kind
       << " arrive_kind=" << fields.arrive_kind
       << " lifecycle_stage=" << fields.lifecycle_stage
       << " waitcnt_thresholds=" << fields.waitcnt_thresholds
       << " waitcnt_pending=" << fields.waitcnt_pending
       << " waitcnt_blocked_domains=" << fields.waitcnt_blocked_domains
       << " msg=" << fields.compatibility_message << '\n';
}

JsonTraceSink::JsonTraceSink(const std::filesystem::path& path) : out_(path) {
  if (!out_) {
    throw std::runtime_error("failed to open json trace file");
  }
}

void JsonTraceSink::OnEvent(const TraceEvent& event) {
  const TraceEventView view = MakeTraceEventView(event);
  const TraceEventExportFields fields = MakeTraceEventExportFields(view);
  std::ostringstream args;
  AppendTraceExportJsonFields(args, fields);
  out_ << "{\"pc\":\"" << HexU64(view.pc) << "\",\"cycle\":\"" << HexU64(view.cycle)
       << "\",\"dpc_id\":\"" << HexU64(view.dpc_id) << "\",\"ap_id\":\""
       << HexU64(view.ap_id) << "\",\"peu_id\":\"" << HexU64(view.peu_id)
       << "\",\"slot_id\":\"" << HexU64(view.slot_id) << "\",\"slot_model_kind\":\""
       << EscapeTraceJson(fields.slot_model) << "\",\"kind\":\""
       << KindToString(view.kind) << "\",\"block_id\":\"" << HexU64(view.block_id)
       << "\",\"wave_id\":\"" << HexU64(view.wave_id) << "\"" << args.str() << "}\n";
}

}  // namespace gpu_model
