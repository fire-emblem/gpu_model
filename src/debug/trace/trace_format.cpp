#include "../trace_format_internal.h"

#include <iomanip>
#include <sstream>

#include "gpu_model/debug/trace/event_export.h"
#include "../trace_json_fields_internal.h"

namespace gpu_model {

namespace {

// Returns true if the event kind is wave-specific (should show wave identifier)
bool IsWaveSpecificEvent(TraceEventKind kind) {
  switch (kind) {
    case TraceEventKind::WaveLaunch:
    case TraceEventKind::WaveGenerate:
    case TraceEventKind::WaveDispatch:
    case TraceEventKind::SlotBind:
    case TraceEventKind::ActivePromote:
    case TraceEventKind::IssueSelect:
    case TraceEventKind::WaveWait:
    case TraceEventKind::WaveArrive:
    case TraceEventKind::WaveResume:
    case TraceEventKind::WaveSwitchAway:
    case TraceEventKind::WaveStats:
    case TraceEventKind::WaveStep:
    case TraceEventKind::Commit:
    case TraceEventKind::ExecMaskUpdate:
    case TraceEventKind::MemoryAccess:
    case TraceEventKind::Barrier:
    case TraceEventKind::WaveExit:
    case TraceEventKind::Stall:
    case TraceEventKind::Arrive:
      return true;
    default:
      return false;
  }
}

std::string HexU64(uint64_t value) {
  std::ostringstream out;
  out << "0x" << std::hex << std::nouppercase << value;
  return out.str();
}

std::string FormatWaveStepDetailBlock(const TraceWaveStepDetail& detail) {
  std::ostringstream out;

  // rw: block
  if (!detail.scalar_reads.empty() || !detail.vector_reads.empty() ||
      !detail.scalar_writes.empty() || !detail.vector_writes.empty()) {
    out << "  rw:\n";

    // Reads
    if (!detail.scalar_reads.empty() || !detail.vector_reads.empty()) {
      out << "    R:\n";
      if (!detail.scalar_reads.empty()) {
        out << "      scalar: ";
        for (size_t i = 0; i < detail.scalar_reads.size(); ++i) {
          if (i > 0) out << " ";
          out << detail.scalar_reads[i];
        }
        out << "\n";
      }
      for (const auto& vr : detail.vector_reads) {
        out << "      " << vr << "\n";
      }
    }

    // Writes
    if (!detail.scalar_writes.empty() || !detail.vector_writes.empty()) {
      out << "    W:\n";
      if (!detail.scalar_writes.empty()) {
        out << "      scalar: ";
        for (size_t i = 0; i < detail.scalar_writes.size(); ++i) {
          if (i > 0) out << " ";
          out << detail.scalar_writes[i];
        }
        out << "\n";
      }
      for (const auto& vw : detail.vector_writes) {
        out << "      " << vw << "\n";
      }
    }
  }

  // mem: block
  if (!detail.mem_summary.empty() && detail.mem_summary != "none") {
    out << "  mem: " << detail.mem_summary << "\n";
  }

  // mask: block
  if (!detail.exec_before.empty() || !detail.exec_after.empty()) {
    out << "  mask: exec_before=" << detail.exec_before;
    if (!detail.exec_after.empty()) {
      out << " exec_after=" << detail.exec_after;
    }
    out << "\n";
  }

  // timing: block
  if (detail.issue_cycle > 0 || detail.commit_cycle > 0 || detail.duration_cycles > 0) {
    out << "  timing: issue=" << detail.issue_cycle;
    if (detail.commit_cycle > 0) {
      out << " commit=" << detail.commit_cycle;
    }
    if (detail.duration_cycles > 0) {
      out << " dur=" << detail.duration_cycles;
    }
    out << "\n";
  }

  // state: block
  if (!detail.state_summary.empty()) {
    out << "  state: " << detail.state_summary << "\n";
  }

  return out.str();
}

std::string FormatTextTraceLineFromFields(const TraceEvent& event,
                                         const TraceEventExportFields& fields,
                                         uint64_t /*sequence*/ = 0) {
  std::ostringstream out;
  // Format: [cycle:XXXXXX] [pc:0xXXX] [wave:0xXXX] [event:kind] details
  // Example: [cycle:000000] [pc:0x1900] [wave:0x0] [event:wave_step] s_load_dword s0, s[4:5], 0x2c

  // [cycle:XXXXXX] - 6 digits zero-padded
  out << "[cycle:" << std::setfill('0') << std::setw(6) << event.cycle << "] ";

  // [pc:0xXXX]
  out << "[pc:" << HexU64(event.pc) << "] ";

  // [wave:0xXXX] or [wave:global]
  if (IsWaveSpecificEvent(event.kind)) {
    uint64_t stable_wave_id =
        (static_cast<uint64_t>(event.block_id) << 32u) | static_cast<uint64_t>(event.wave_id);
    out << "[wave:" << HexU64(stable_wave_id) << "] ";
  } else {
    out << "[wave:global] ";
  }

  // [event:kind]
  out << "[event:" << fields.canonical_name << "] ";

  // Display name / message as details
  out << fields.display_name;

  out << '\n';

  // WaveStep expanded block
  if (event.kind == TraceEventKind::WaveStep && event.step_detail.has_value()) {
    out << FormatWaveStepDetailBlock(*event.step_detail);
  }

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
  return FormatTextTraceLineFromFields(event, fields, 0);
}

std::string FormatJsonTraceEventLine(const TraceEvent& event) {
  const CanonicalTraceEvent canonical = MakeCanonicalTraceEvent(event);
  const auto& fields = canonical.fields;
  return FormatJsonTraceLineFromFields(event, fields);
}

namespace {

}  // namespace

std::string FormatTextTraceEventLine(const RecorderProgramEvent& event) {
  return FormatTextTraceLineFromFields(event.event, MakeTraceEventExportFields(event), event.sequence);
}

std::string FormatTextTraceEventLine(const RecorderEntry& event) {
  return FormatTextTraceLineFromFields(event.event, MakeTraceEventExportFields(event), event.sequence);
}

std::string FormatJsonTraceEventLine(const RecorderProgramEvent& event) {
  return FormatJsonTraceLineFromFields(event.event, MakeTraceEventExportFields(event));
}

std::string FormatJsonTraceEventLine(const RecorderEntry& event) {
  return FormatJsonTraceLineFromFields(event.event, MakeTraceEventExportFields(event));
}

}  // namespace gpu_model
