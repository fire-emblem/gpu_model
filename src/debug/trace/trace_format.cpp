#include "../trace_format_internal.h"

#include <iomanip>
#include <sstream>

#include "gpu_model/debug/trace/event_export.h"
#include "../trace_json_fields_internal.h"

namespace gpu_model {

namespace {

std::string HexU64(uint64_t value) {
  std::ostringstream out;
  out << "0x" << std::hex << std::nouppercase << value;
  return out.str();
}

std::string FormatTextTraceLineFromFields(const TraceEvent& event,
                                         const TraceEventExportFields& fields) {
  std::ostringstream out;
  // Format: [cycle]   kind   w{block}.{slot}  pc   details
  // Example: [000000]   wave_generate  w0.0  0x100   block=0 slot=0

  // Cycle in brackets, 6 digits zero-padded
  out << "[" << std::setfill('0') << std::setw(6) << event.cycle << "]   ";

  // Event kind (canonical name), left-aligned in 16-char field with space padding
  out << std::setfill(' ') << std::setw(16) << std::left << fields.canonical_name << "  ";

  // Wave identifier: w{block}.{slot} or just "global" for program events
  if (event.wave_id != 0 || event.block_id != 0) {
    out << "w" << event.block_id << "." << event.slot_id << "  ";
  } else {
    out << "global  ";
  }

  // PC in hex
  out << HexU64(event.pc) << "   ";

  // Display name / message as details
  out << fields.display_name;

  out << '\n';
  return out.str();
}

std::string FormatJsonTraceLineFromFields(const TraceEvent& event,
                                         const TraceEventExportFields& fields) {
  std::ostringstream args;
  AppendTraceExportJsonFields(args, fields);

  std::ostringstream out;
  out << "{\"pc\":\"" << HexU64(event.pc) << "\",\"cycle\":\"" << HexU64(event.cycle)
      << "\",\"dpc_id\":\"" << HexU64(event.dpc_id) << "\",\"ap_id\":\""
      << HexU64(event.ap_id) << "\",\"peu_id\":\"" << HexU64(event.peu_id)
      << "\",\"slot_id\":\"" << HexU64(event.slot_id) << "\",\"slot_model_kind\":\""
      << EscapeTraceJson(fields.slot_model) << "\",\"kind\":\""
      << TraceEventKindName(event.kind) << "\",\"block_id\":\"" << HexU64(event.block_id)
      << "\",\"wave_id\":\"" << HexU64(event.wave_id) << "\""
      << ",\"has_cycle_range\":" << (fields.has_cycle_range ? "true" : "false");
  if (fields.has_cycle_range) {
    out << ",\"begin_cycle\":\"" << EscapeTraceJson(fields.begin_cycle) << "\""
        << ",\"end_cycle\":\"" << EscapeTraceJson(fields.end_cycle) << "\"";
  }
  if (fields.has_flow) {
    out << ",\"has_flow\":true"
        << ",\"flow_id\":\"" << EscapeTraceJson(fields.flow_id) << "\""
        << ",\"flow_phase\":\"" << EscapeTraceJson(fields.flow_phase) << "\"";
  }
  out << args.str() << "}\n";
  return out.str();
}

}  // namespace

std::string_view TraceEventKindName(TraceEventKind kind) {
  switch (kind) {
    case TraceEventKind::Launch:
      return "Launch";
    case TraceEventKind::BlockPlaced:
      return "BlockPlaced";
    case TraceEventKind::BlockAdmit:
      return "BlockAdmit";
    case TraceEventKind::BlockLaunch:
      return "BlockLaunch";
    case TraceEventKind::BlockActivate:
      return "BlockActivate";
    case TraceEventKind::BlockRetire:
      return "BlockRetire";
    case TraceEventKind::WaveLaunch:
      return "WaveLaunch";
    case TraceEventKind::WaveGenerate:
      return "WaveGenerate";
    case TraceEventKind::WaveDispatch:
      return "WaveDispatch";
    case TraceEventKind::SlotBind:
      return "SlotBind";
    case TraceEventKind::ActivePromote:
      return "ActivePromote";
    case TraceEventKind::IssueSelect:
      return "IssueSelect";
    case TraceEventKind::WaveWait:
      return "WaveWait";
    case TraceEventKind::WaveArrive:
      return "WaveArrive";
    case TraceEventKind::WaveResume:
      return "WaveResume";
    case TraceEventKind::WaveSwitchAway:
      return "WaveSwitchAway";
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

std::string FormatTextTraceEventLine(const TraceEvent& event) {
  const CanonicalTraceEvent canonical = MakeCanonicalTraceEvent(event);
  const auto& fields = canonical.fields;
  return FormatTextTraceLineFromFields(event, fields);
}

std::string FormatJsonTraceEventLine(const TraceEvent& event) {
  const CanonicalTraceEvent canonical = MakeCanonicalTraceEvent(event);
  const auto& fields = canonical.fields;
  return FormatJsonTraceLineFromFields(event, fields);
}

namespace {

}  // namespace

std::string FormatTextTraceEventLine(const RecorderProgramEvent& event) {
  return FormatTextTraceLineFromFields(event.event, MakeTraceEventExportFields(event));
}

std::string FormatTextTraceEventLine(const RecorderEntry& event) {
  return FormatTextTraceLineFromFields(event.event, MakeTraceEventExportFields(event));
}

std::string FormatJsonTraceEventLine(const RecorderProgramEvent& event) {
  return FormatJsonTraceLineFromFields(event.event, MakeTraceEventExportFields(event));
}

std::string FormatJsonTraceEventLine(const RecorderEntry& event) {
  return FormatJsonTraceLineFromFields(event.event, MakeTraceEventExportFields(event));
}

}  // namespace gpu_model
