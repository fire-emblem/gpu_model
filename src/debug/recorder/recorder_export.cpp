#include "debug/recorder/export.h"

#include <algorithm>
#include <cstdio>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "../trace_format_internal.h"

namespace gpu_model {

namespace {

std::string HexU64(uint64_t value) {
  std::ostringstream out;
  out << "0x" << std::hex << std::nouppercase << value;
  return out.str();
}

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
    if (!run.functional_mode.empty()) {
      text += "functional_mode=" + run.functional_mode + "\n";
    }
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
    text += "launch_index=" + std::to_string(kernel.launch_index) + "\n";
    text += "grid_dim=" + std::to_string(kernel.grid_dim_x) + "," +
             std::to_string(kernel.grid_dim_y) + "," +
             std::to_string(kernel.grid_dim_z) + "\n";
    text += "block_dim=" + std::to_string(kernel.block_dim_x) + "," +
             std::to_string(kernel.block_dim_y) + "," +
             std::to_string(kernel.block_dim_z) + "\n";
    if (kernel.theoretical_max_waves_per_peu > 0) {
      text += "theoretical_max_waves_per_peu=" +
              std::to_string(kernel.theoretical_max_waves_per_peu) + "\n";
      text += "theoretical_max_blocks_per_ap=" +
              std::to_string(kernel.theoretical_max_blocks_per_ap) + "\n";
      text += "theoretical_occupancy_pct=" +
              std::to_string(kernel.theoretical_occupancy_pct) + "\n";
      if (!kernel.occupancy_wave_limiter.empty()) {
        text += "occupancy_wave_limiter=" + kernel.occupancy_wave_limiter + "\n";
      }
      if (!kernel.occupancy_block_limiter.empty()) {
        text += "occupancy_block_limiter=" + kernel.occupancy_block_limiter + "\n";
      }
      if (kernel.kernel_vgpr_count > 0) {
        text += "kernel_vgpr_count=" + std::to_string(kernel.kernel_vgpr_count) + "\n";
      }
      if (kernel.kernel_sgpr_count > 0) {
        text += "kernel_sgpr_count=" + std::to_string(kernel.kernel_sgpr_count) + "\n";
      }
      if (kernel.kernel_agpr_count > 0) {
        text += "kernel_agpr_count=" + std::to_string(kernel.kernel_agpr_count) + "\n";
      }
      if (kernel.kernel_shared_memory_bytes > 0) {
        text += "kernel_shared_memory_bytes=" + std::to_string(kernel.kernel_shared_memory_bytes) + "\n";
      }
    }
    text += "\n";
  }

  // [WAVE_INIT] section
  if (!recorder.wave_init_snapshots().empty()) {
    text += "[WAVE_INIT]\n";
    for (const auto& wave_init : recorder.wave_init_snapshots()) {
      // Format: wave=0x{stable_wave_id} block={block_id} loc=dpc{dpc}/ap{ap}/peu{peu}/slot{slot} slot_model={model} start_pc=0x{pc}
      // stable_wave_id = (block_id << 32) | wave_id, hex format for readability
      text += "wave=" + HexU64(wave_init.stable_wave_id) +
              " block=" + std::to_string(wave_init.block_id) +
              " loc=dpc" + std::to_string(wave_init.dpc_id) +
              "/ap" + std::to_string(wave_init.ap_id) +
              "/peu" + std::to_string(wave_init.peu_id) +
              "/slot" + std::to_string(wave_init.slot_id) +
              " slot_model=" + wave_init.slot_model +
              " start_pc=0x" + std::to_string(wave_init.start_pc) + "\n";
    }
    text += "\n";
  }

  // [EVENTS] section
  // Filter to keep only wave_step and wave_exit events for cleaner trace output.
  // Other events (issue_select, commit, wave_wait, etc.) are available in trace.jsonl.
  text += "[EVENTS]\n";
  for (const auto& recorded : CollectOrderedRecordedEvents(recorder)) {
    const TraceEvent* event = nullptr;
    if (recorded.program_event != nullptr) {
      event = &recorded.program_event->event;
    } else if (recorded.entry != nullptr) {
      event = &recorded.entry->event;
    }
    if (event != nullptr) {
      // Only include wave_step (instruction execution) and wave_exit (lifecycle end)
      if (event->kind == TraceEventKind::WaveStep || event->kind == TraceEventKind::WaveExit) {
        if (recorded.program_event != nullptr) {
          text += FormatTextTraceEventLine(*recorded.program_event);
        } else if (recorded.entry != nullptr) {
          text += FormatTextTraceEventLine(*recorded.entry);
        }
      }
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
    text += "\n";

    // [STALL_SUMMARY] section
    text += "[STALL_SUMMARY]\n";
    text += "stall_waitcnt=" + std::to_string(summary.stall_waitcnt_global) + "\n";
    text += "stall_warp_switch=" + std::to_string(summary.stall_warp_switch) + "\n";
    text += "stall_barrier=" + std::to_string(summary.stall_barrier_slot) + "\n";
    text += "stall_other=" + std::to_string(summary.stall_other) + "\n";
    text += "\n";

    // [MEMORY_AND_RESOURCES] section
    text += "[MEMORY_AND_RESOURCES]\n";
    text += "global_loads=" + std::to_string(summary.global_loads) + "\n";
    text += "global_stores=" + std::to_string(summary.global_stores) + "\n";
    text += "shared_loads=" + std::to_string(summary.shared_loads) + "\n";
    text += "shared_stores=" + std::to_string(summary.shared_stores) + "\n";
    text += "private_loads=" + std::to_string(summary.private_loads) + "\n";
    text += "private_stores=" + std::to_string(summary.private_stores) + "\n";
    text += "scalar_loads=" + std::to_string(summary.scalar_loads) + "\n";
    text += "scalar_stores=" + std::to_string(summary.scalar_stores) + "\n";
    text += "\n";

    // [PERF] section - Instruction mix with percentages
    text += "[PERF]\n";
    const uint64_t total_insts = summary.gpu_tot_sim_insn;
    auto pct = [total_insts](uint64_t count) -> std::string {
      if (total_insts == 0) return "0.00";
      char buf[32];
      snprintf(buf, sizeof(buf), "%.2f", 100.0 * static_cast<double>(count) / static_cast<double>(total_insts));
      return std::string(buf);
    };
    text += "instruction_mix_total=" + std::to_string(total_insts) + "\n";
    text += "scalar_alu=" + std::to_string(summary.scalar_alu_insts) + " (" + pct(summary.scalar_alu_insts) + "%)\n";
    text += "scalar_mem=" + std::to_string(summary.scalar_mem_insts) + " (" + pct(summary.scalar_mem_insts) + "%)\n";
    text += "vector_alu=" + std::to_string(summary.vector_alu_insts) + " (" + pct(summary.vector_alu_insts) + "%)\n";
    text += "vector_mem=" + std::to_string(summary.vector_mem_insts) + " (" + pct(summary.vector_mem_insts) + "%)\n";
    text += "branch=" + std::to_string(summary.branch_insts) + " (" + pct(summary.branch_insts) + "%)\n";
    text += "sync=" + std::to_string(summary.sync_insts) + " (" + pct(summary.sync_insts) + "%)\n";
    text += "tensor=" + std::to_string(summary.tensor_insts) + " (" + pct(summary.tensor_insts) + "%)\n";
    text += "other=" + std::to_string(summary.other_insts) + " (" + pct(summary.other_insts) + "%)\n";
    text += "\n";

    // Wave statistics
    text += "waves_launched=" + std::to_string(summary.waves_launched) + "\n";
    text += "waves_completed=" + std::to_string(summary.waves_completed) + "\n";
    text += "max_concurrent_waves=" + std::to_string(summary.max_concurrent_waves) + "\n";
    text += "active_utilization_pct=" + std::to_string(summary.active_utilization_pct) + "\n";
    text += "\n";

    // [OCCUPANCY] section
    if (summary.theoretical_max_waves_per_peu > 0) {
      text += "[OCCUPANCY]\n";
      text += "theoretical_max_waves_per_peu=" +
              std::to_string(summary.theoretical_max_waves_per_peu) + "\n";
      text += "theoretical_max_blocks_per_ap=" +
              std::to_string(summary.theoretical_max_blocks_per_ap) + "\n";
      text += "theoretical_occupancy_pct=" +
              std::to_string(summary.theoretical_occupancy_pct) + "\n";
      if (!summary.occupancy_wave_limiter.empty()) {
        text += "occupancy_wave_limiter=" + summary.occupancy_wave_limiter + "\n";
      }
      if (!summary.occupancy_block_limiter.empty()) {
        text += "occupancy_block_limiter=" + summary.occupancy_block_limiter + "\n";
      }
      text += "\n";
    }

    // [PERF_OPT] section - Performance optimization metrics
    text += "[PERF_OPT]\n";
    text += "total_flops=" + std::to_string(summary.total_flops) + "\n";
    text += "total_bytes=" + std::to_string(summary.total_bytes) + "\n";
    text += "arithmetic_intensity=" + std::to_string(summary.arithmetic_intensity) + "\n";
    text += "bound_classification=" + summary.bound_classification + "\n";
    text += "bytes_per_cycle=" + std::to_string(summary.bytes_per_cycle) + "\n";
    text += "flops_per_cycle=" + std::to_string(summary.flops_per_cycle) + "\n";
    text += "memory_intensity=" + std::to_string(summary.memory_intensity) + "\n";
    text += "compute_intensity=" + std::to_string(summary.compute_intensity) + "\n";
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
            std::to_string(kernel.block_dim_y) + "," + std::to_string(kernel.block_dim_z) + "]";
    if (kernel.theoretical_max_waves_per_peu > 0) {
      text += ",\"theoretical_max_waves_per_peu\":" + std::to_string(kernel.theoretical_max_waves_per_peu) +
              ",\"theoretical_max_blocks_per_ap\":" + std::to_string(kernel.theoretical_max_blocks_per_ap) +
              ",\"theoretical_occupancy_pct\":" + std::to_string(kernel.theoretical_occupancy_pct);
      if (!kernel.occupancy_wave_limiter.empty()) {
        text += ",\"occupancy_wave_limiter\":\"" + kernel.occupancy_wave_limiter + "\"";
      }
      if (!kernel.occupancy_block_limiter.empty()) {
        text += ",\"occupancy_block_limiter\":\"" + kernel.occupancy_block_limiter + "\"";
      }
      if (kernel.kernel_vgpr_count > 0) {
        text += ",\"kernel_vgpr_count\":" + std::to_string(kernel.kernel_vgpr_count);
      }
      if (kernel.kernel_sgpr_count > 0) {
        text += ",\"kernel_sgpr_count\":" + std::to_string(kernel.kernel_sgpr_count);
      }
      if (kernel.kernel_agpr_count > 0) {
        text += ",\"kernel_agpr_count\":" + std::to_string(kernel.kernel_agpr_count);
      }
      if (kernel.kernel_shared_memory_bytes > 0) {
        text += ",\"kernel_shared_memory_bytes\":" + std::to_string(kernel.kernel_shared_memory_bytes);
      }
    }
    text += "}\n";
  }

  // Emit wave init snapshots
  for (const auto& wave_init : recorder.wave_init_snapshots()) {
    text += "{\"type\":\"wave_init_snapshot\",\"wave\":\"" + std::to_string(wave_init.stable_wave_id) + "\"" +
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
            ",\"stall_barrier\":" + std::to_string(summary.stall_barrier_slot);
    if (summary.theoretical_max_waves_per_peu > 0) {
      text += ",\"theoretical_max_waves_per_peu\":" + std::to_string(summary.theoretical_max_waves_per_peu) +
              ",\"theoretical_max_blocks_per_ap\":" + std::to_string(summary.theoretical_max_blocks_per_ap) +
              ",\"theoretical_occupancy_pct\":" + std::to_string(summary.theoretical_occupancy_pct);
      if (!summary.occupancy_wave_limiter.empty()) {
        text += ",\"occupancy_wave_limiter\":\"" + summary.occupancy_wave_limiter + "\"";
      }
      if (!summary.occupancy_block_limiter.empty()) {
        text += ",\"occupancy_block_limiter\":\"" + summary.occupancy_block_limiter + "\"";
      }
    }
    text += "}\n";
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
