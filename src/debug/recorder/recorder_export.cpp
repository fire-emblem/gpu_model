#include "gpu_model/debug/recorder/export.h"

#include <algorithm>
#include <string>
#include <vector>

#include "../trace_format_internal.h"

namespace gpu_model {

namespace {

struct OrderedRecordedEvent {
  uint64_t sequence = 0;
  const RecorderProgramEvent* program_event = nullptr;
  const RecorderEntry* entry = nullptr;
};

std::vector<OrderedRecordedEvent> CollectOrderedRecordedEvents(const Recorder& recorder) {
  std::vector<OrderedRecordedEvent> ordered;
  ordered.reserve(recorder.program_events().size() + recorder.events().size());

  for (const auto& program_event : recorder.program_events()) {
    ordered.push_back(OrderedRecordedEvent{
        .sequence = program_event.sequence,
        .program_event = &program_event,
    });
  }
  for (const auto& wave : recorder.waves()) {
    for (const auto& entry : wave.entries) {
      ordered.push_back(OrderedRecordedEvent{
          .sequence = entry.sequence,
          .entry = &entry,
      });
    }
  }

  std::sort(ordered.begin(), ordered.end(), [](const OrderedRecordedEvent& lhs,
                                               const OrderedRecordedEvent& rhs) {
    return lhs.sequence < rhs.sequence;
  });
  return ordered;
}

}  // namespace

std::filesystem::path TextRecorderSerializer::DefaultArtifactPath() const {
  return "trace.txt";
}

std::string TextRecorderSerializer::Serialize(const Recorder& recorder) const {
  return RenderRecorderTextTrace(recorder);
}

std::filesystem::path JsonRecorderSerializer::DefaultArtifactPath() const {
  return "trace.jsonl";
}

std::string JsonRecorderSerializer::Serialize(const Recorder& recorder) const {
  return RenderRecorderJsonTrace(recorder);
}

std::unique_ptr<RecorderSerializer> MakeTextRecorderSerializer() {
  return std::make_unique<TextRecorderSerializer>();
}

std::unique_ptr<RecorderSerializer> MakeJsonRecorderSerializer() {
  return std::make_unique<JsonRecorderSerializer>();
}

std::string RenderRecorderTextTrace(const Recorder& recorder) {
  std::string text;

  // Header
  text += "GPU_MODEL TRACE\n";
  text += "================\n\n";

  // [RUN] section
  if (recorder.run_snapshot().has_value()) {
    const auto& run = *recorder.run_snapshot();
    text += "[RUN]\n";
    if (!run.invocation.empty()) {
      text += run.invocation + "\n";
    }
    text += "execution_model=" + run.execution_model + "\n";
    text += "trace_time_basis=" + run.trace_time_basis + "\n";
    text += "trace_cycle_is_physical_time=" + std::string(run.trace_cycle_is_physical_time ? "true" : "false") + "\n";
    text += "\n";
  }

  // [MODEL_CONFIG] section
  if (recorder.model_config_snapshot().has_value()) {
    const auto& config = *recorder.model_config_snapshot();
    text += "[MODEL_CONFIG]\n";
    text += "num_dpcs=" + std::to_string(config.num_dpcs) + "\n";
    text += "num_aps_per_dpc=" + std::to_string(config.num_aps_per_dpc) + "\n";
    text += "num_peus_per_ap=" + std::to_string(config.num_peus_per_ap) + "\n";
    text += "num_slots_per_peu=" + std::to_string(config.num_slots_per_peu) + "\n";
    text += "slot_model=" + config.slot_model + "\n";
    text += "\n";
  }

  // [KERNEL] section
  if (recorder.kernel_snapshot().has_value()) {
    const auto& kernel = *recorder.kernel_snapshot();
    text += "[KERNEL]\n";
    text += "kernel_name=" + kernel.kernel_name + "\n";
    text += "grid_dim=" + std::to_string(kernel.grid_dim_x) + "," +
             std::to_string(kernel.grid_dim_y) + "," +
             std::to_string(kernel.grid_dim_z) + "\n";
    text += "block_dim=" + std::to_string(kernel.block_dim_x) + "," +
             std::to_string(kernel.block_dim_y) + "," +
             std::to_string(kernel.block_dim_z) + "\n";
    text += "\n";
  }

  // [WAVE_INIT] section
  if (!recorder.wave_init_snapshots().empty()) {
    text += "[WAVE_INIT]\n";
    for (const auto& wave_init : recorder.wave_init_snapshots()) {
      // Format: wave=w{block}.{slot} loc=dpc{dpc}/ap{ap}/peu{peu}/slot{slot} slot_model={model} start_pc=0x{pc}
      text += "wave=w" + std::to_string(wave_init.block_id) + "." + std::to_string(wave_init.slot_id) +
              " block=" + std::to_string(wave_init.block_id) +
              " dpc=" + std::to_string(wave_init.dpc_id) +
              " ap=" + std::to_string(wave_init.ap_id) +
              " peu=" + std::to_string(wave_init.peu_id) +
              " slot=" + std::to_string(wave_init.slot_id) +
              " slot_model=" + wave_init.slot_model +
              " start_pc=0x" + std::to_string(wave_init.start_pc) + "\n";
    }
    text += "\n";
  }

  // [EVENTS] section
  text += "[EVENTS]\n";
  for (const auto& recorded : CollectOrderedRecordedEvents(recorder)) {
    if (recorded.program_event != nullptr) {
      text += FormatTextTraceEventLine(*recorded.program_event);
    } else if (recorded.entry != nullptr) {
      text += FormatTextTraceEventLine(*recorded.entry);
    }
  }
  text += "\n";

  // [SUMMARY] section
  if (recorder.summary_snapshot().has_value()) {
    const auto& summary = *recorder.summary_snapshot();
    text += "[SUMMARY]\n";
    text += "kernel_status=" + summary.kernel_status + "\n";
    text += "gpu_tot_sim_cycle=" + std::to_string(summary.gpu_tot_sim_cycle) + "\n";
    text += "gpu_tot_sim_insn=" + std::to_string(summary.gpu_tot_sim_insn) + "\n";
    text += "gpu_tot_ipc=" + std::to_string(summary.gpu_tot_ipc) + "\n";
    text += "gpu_tot_wave_exits=" + std::to_string(summary.gpu_tot_wave_exits) + "\n";
    text += "stall_waitcnt=" + std::to_string(summary.stall_waitcnt_global) + "\n";
    text += "stall_warp_switch=" + std::to_string(summary.stall_warp_switch) + "\n";
    text += "stall_barrier=" + std::to_string(summary.stall_barrier_slot) + "\n";
    text += "\n";
  }

  // [WARNINGS] section
  if (!recorder.warning_snapshots().empty()) {
    text += "[WARNINGS]\n";
    for (const auto& warning : recorder.warning_snapshots()) {
      text += "kind=" + warning.warning_kind + " message=" + warning.message + "\n";
    }
    text += "\n";
  }

  return text;
}

std::string RenderRecorderJsonTrace(const Recorder& recorder) {
  std::string text;

  // Emit run snapshot as first JSON object
  if (recorder.run_snapshot().has_value()) {
    const auto& run = *recorder.run_snapshot();
    text += "{\"type\":\"run_snapshot\"";
    if (!run.invocation.empty()) {
      // Escape quotes and backslashes in invocation string
      std::string escaped;
      for (char c : run.invocation) {
        if (c == '"' || c == '\\') {
          escaped += '\\';
        }
        escaped += c;
      }
      text += ",\"invocation\":\"" + escaped + "\"";
    }
    text += ",\"execution_model\":\"" + run.execution_model +
            "\",\"trace_time_basis\":\"" + run.trace_time_basis +
            "\",\"trace_cycle_is_physical_time\":" + (run.trace_cycle_is_physical_time ? "true" : "false") + "}\n";
  }

  // Emit model config snapshot
  if (recorder.model_config_snapshot().has_value()) {
    const auto& config = *recorder.model_config_snapshot();
    text += "{\"type\":\"model_config_snapshot\",\"num_dpcs\":" + std::to_string(config.num_dpcs) +
            ",\"num_aps_per_dpc\":" + std::to_string(config.num_aps_per_dpc) +
            ",\"num_peus_per_ap\":" + std::to_string(config.num_peus_per_ap) +
            ",\"num_slots_per_peu\":" + std::to_string(config.num_slots_per_peu) +
            ",\"slot_model\":\"" + config.slot_model + "\"}\n";
  }

  // Emit kernel snapshot
  if (recorder.kernel_snapshot().has_value()) {
    const auto& kernel = *recorder.kernel_snapshot();
    text += "{\"type\":\"kernel_snapshot\",\"kernel_name\":\"" + kernel.kernel_name +
            "\",\"grid_dim\":[" + std::to_string(kernel.grid_dim_x) + "," +
            std::to_string(kernel.grid_dim_y) + "," + std::to_string(kernel.grid_dim_z) + "]" +
            ",\"block_dim\":[" + std::to_string(kernel.block_dim_x) + "," +
            std::to_string(kernel.block_dim_y) + "," + std::to_string(kernel.block_dim_z) + "]}\n";
  }

  // Emit wave init snapshots
  for (const auto& wave_init : recorder.wave_init_snapshots()) {
    text += "{\"type\":\"wave_init_snapshot\",\"wave\":\"w" + std::to_string(wave_init.block_id) + "." + std::to_string(wave_init.slot_id) + "\"" +
            ",\"block_id\":" + std::to_string(wave_init.block_id) +
            ",\"dpc_id\":" + std::to_string(wave_init.dpc_id) +
            ",\"ap_id\":" + std::to_string(wave_init.ap_id) +
            ",\"peu_id\":" + std::to_string(wave_init.peu_id) +
            ",\"slot_id\":" + std::to_string(wave_init.slot_id) +
            ",\"slot_model\":\"" + wave_init.slot_model + "\"" +
            ",\"start_pc\":" + std::to_string(wave_init.start_pc) + "}\n";
  }

  // Emit events
  for (const auto& recorded : CollectOrderedRecordedEvents(recorder)) {
    if (recorded.program_event != nullptr) {
      text += FormatJsonTraceEventLine(*recorded.program_event);
    } else if (recorded.entry != nullptr) {
      text += FormatJsonTraceEventLine(*recorded.entry);
    }
  }

  // Emit summary snapshot
  if (recorder.summary_snapshot().has_value()) {
    const auto& summary = *recorder.summary_snapshot();
    text += "{\"type\":\"summary_snapshot\",\"kernel_status\":\"" + summary.kernel_status + "\"" +
            ",\"gpu_tot_sim_cycle\":" + std::to_string(summary.gpu_tot_sim_cycle) +
            ",\"gpu_tot_sim_insn\":" + std::to_string(summary.gpu_tot_sim_insn) +
            ",\"gpu_tot_ipc\":" + std::to_string(summary.gpu_tot_ipc) +
            ",\"gpu_tot_wave_exits\":" + std::to_string(summary.gpu_tot_wave_exits) +
            ",\"stall_waitcnt\":" + std::to_string(summary.stall_waitcnt_global) +
            ",\"stall_warp_switch\":" + std::to_string(summary.stall_warp_switch) +
            ",\"stall_barrier\":" + std::to_string(summary.stall_barrier_slot) + "}\n";
  }

  // Emit warning snapshots
  for (const auto& warning : recorder.warning_snapshots()) {
    text += "{\"type\":\"warning_snapshot\",\"warning_kind\":\"" + warning.warning_kind +
            "\",\"message\":\"" + warning.message + "\"" +
            ",\"cycle\":" + std::to_string(warning.cycle) + "}\n";
  }

  return text;
}

}  // namespace gpu_model
