#include "runtime/exec_engine/launch_trace_emitter.h"

#include <sstream>
#include <string>

#include "debug/trace/event_factory.h"
#include "program/program_object/program_object.h"
#include "utils/config/invocation.h"
#include "utils/logging/log_macros.h"

namespace gpu_model {

void EmitLaunchTracePreamble(TraceSink& trace,
                             const LaunchRequest& request,
                             const GpuArchSpec& spec,
                             std::string_view kernel_name,
                             bool use_program_object_payload,
                             uint64_t submit_cycle,
                             const ProgramObject* program_object,
                             const PlacementMap& placement,
                             const std::optional<OccupancyResult>& occupancy,
                             const KernelResourceUsage& resource_usage) {
  TraceRunSnapshot run_snapshot{
      .invocation = CaptureInvocationLine(),
      .execution_model = request.mode == ExecutionMode::Cycle ? "cycle" : "functional",
      .functional_mode = request.functional_mode,
      .trace_time_basis = "modeled_cycle",
      .trace_cycle_is_physical_time = false,
  };
  trace.OnRunSnapshot(run_snapshot);

  TraceModelConfigSnapshot model_config_snapshot{
      .num_dpcs = spec.dpc_count,
      .num_aps_per_dpc = spec.ap_per_dpc,
      .num_peus_per_ap = spec.peu_per_ap,
      .num_slots_per_peu = spec.cycle_resources.resident_wave_slots_per_peu,
      .slot_model = "resident_fixed",
  };
  trace.OnModelConfigSnapshot(model_config_snapshot);

  TraceKernelSnapshot kernel_snapshot;
  kernel_snapshot.kernel_name = std::string(kernel_name);
  kernel_snapshot.kernel_launch_uid = 0;
  kernel_snapshot.launch_index = request.launch_index;
  kernel_snapshot.grid_dim_x = request.config.grid_dim_x;
  kernel_snapshot.grid_dim_y = request.config.grid_dim_y;
  kernel_snapshot.grid_dim_z = request.config.grid_dim_z;
  kernel_snapshot.block_dim_x = request.config.block_dim_x;
  kernel_snapshot.block_dim_y = request.config.block_dim_y;
  kernel_snapshot.block_dim_z = request.config.block_dim_z;
  if (occupancy.has_value()) {
    const auto& occ = *occupancy;
    kernel_snapshot.theoretical_max_waves_per_peu = occ.max_waves_per_peu;
    kernel_snapshot.theoretical_max_blocks_per_ap = occ.max_blocks_per_ap;
    kernel_snapshot.theoretical_occupancy_pct = occ.occupancy_ratio * 100.0f;
    kernel_snapshot.occupancy_wave_limiter = occ.wave_limiting_factor;
    kernel_snapshot.occupancy_block_limiter = occ.block_limiting_factor;
  }
  kernel_snapshot.kernel_vgpr_count = resource_usage.vgpr_count;
  kernel_snapshot.kernel_sgpr_count = resource_usage.sgpr_count;
  kernel_snapshot.kernel_agpr_count = resource_usage.agpr_count;
  kernel_snapshot.kernel_shared_memory_bytes = resource_usage.shared_memory_bytes;
  trace.OnKernelSnapshot(kernel_snapshot);

  std::ostringstream launch_message;
  launch_message << "kernel=" << kernel_name << " arch=" << spec.name;
  if (use_program_object_payload && program_object != nullptr &&
      (program_object->kernel_descriptor().agpr_count != 0 ||
       program_object->kernel_descriptor().accum_offset != 0)) {
    launch_message << " agpr_count=" << program_object->kernel_descriptor().agpr_count
                   << " accum_offset=" << program_object->kernel_descriptor().accum_offset;
  }
  trace.OnEvent(MakeTraceRuntimeLaunchEvent(submit_cycle, launch_message.str()));

  for (const auto& block : placement.blocks) {
    std::ostringstream message;
    message << "block=" << block.block_id << " block_xyz=(" << block.block_idx_x << ","
            << block.block_idx_y << "," << block.block_idx_z << ") dpc=" << block.dpc_id
            << " ap=" << block.ap_id << " waves=" << block.waves.size();
    trace.OnEvent(MakeTraceBlockPlacedEvent(block.dpc_id,
                                            block.ap_id,
                                            block.block_id,
                                            submit_cycle,
                                            message.str()));
  }
}

void EmitLaunchTraceSummary(TraceSink& trace,
                            const LaunchRequest& request,
                            const LaunchResult& result,
                            const std::optional<OccupancyResult>& occupancy) {
  const auto& pcs = result.program_cycle_stats;
  TraceSummarySnapshot summary_snapshot{
      .kernel_status = result.ok ? "PASS" : "FAIL",
      .launch_index = request.launch_index,
      .submit_cycle = result.submit_cycle,
      .begin_cycle = result.begin_cycle,
      .end_cycle = result.end_cycle,
      .gpu_tot_sim_cycle = result.total_cycles,
      .gpu_tot_sim_insn = pcs.has_value() ? pcs->instructions_executed : 0,
      .gpu_tot_ipc = pcs.has_value() && result.total_cycles > 0
                         ? static_cast<double>(pcs->instructions_executed) /
                               static_cast<double>(result.total_cycles)
                         : 0.0,
      .gpu_tot_wave_exits = pcs.has_value() ? pcs->waves_completed : 0,
      .stall_waitcnt_global = pcs.has_value() ? pcs->stall_waitcnt : 0,
      .stall_waitcnt_shared = 0,
      .stall_waitcnt_private = 0,
      .stall_warp_switch = pcs.has_value() ? pcs->stall_switch_away : 0,
      .stall_barrier_slot = pcs.has_value() ? pcs->stall_barrier : 0,
      .stall_other = pcs.has_value() ? pcs->stall_resource + pcs->stall_dependency : 0,
      .scalar_alu_insts = pcs.has_value() ? pcs->scalar_alu_insts : 0,
      .scalar_mem_insts = pcs.has_value() ? pcs->scalar_mem_insts : 0,
      .vector_alu_insts = pcs.has_value() ? pcs->vector_alu_insts : 0,
      .vector_mem_insts = pcs.has_value() ? pcs->vector_mem_insts : 0,
      .branch_insts = pcs.has_value() ? pcs->branch_insts : 0,
      .sync_insts = pcs.has_value() ? pcs->sync_insts : 0,
      .tensor_insts = pcs.has_value() ? pcs->tensor_insts : 0,
      .other_insts = pcs.has_value() ? pcs->other_insts : 0,
      .global_loads = pcs.has_value() ? pcs->global_loads : 0,
      .global_stores = pcs.has_value() ? pcs->global_stores : 0,
      .shared_loads = pcs.has_value() ? pcs->shared_loads : 0,
      .shared_stores = pcs.has_value() ? pcs->shared_stores : 0,
      .private_loads = pcs.has_value() ? pcs->private_loads : 0,
      .private_stores = pcs.has_value() ? pcs->private_stores : 0,
      .scalar_loads = pcs.has_value() ? pcs->scalar_loads : 0,
      .scalar_stores = pcs.has_value() ? pcs->scalar_stores : 0,
      .global_load_bytes = pcs.has_value() ? pcs->global_load_bytes : 0,
      .global_store_bytes = pcs.has_value() ? pcs->global_store_bytes : 0,
      .shared_load_bytes = pcs.has_value() ? pcs->shared_load_bytes : 0,
      .shared_store_bytes = pcs.has_value() ? pcs->shared_store_bytes : 0,
      .fp32_ops = pcs.has_value() ? pcs->fp32_ops : 0,
      .fp64_ops = pcs.has_value() ? pcs->fp64_ops : 0,
      .int32_ops = pcs.has_value() ? pcs->int32_ops : 0,
      .tensor_ops = pcs.has_value() ? pcs->tensor_ops : 0,
      .waves_launched = pcs.has_value() ? pcs->waves_launched : 0,
      .waves_completed = pcs.has_value() ? pcs->waves_completed : 0,
      .max_concurrent_waves = pcs.has_value() ? pcs->max_concurrent_waves : 0,
      .theoretical_max_waves_per_peu = occupancy.has_value() ? occupancy->max_waves_per_peu : 0,
      .theoretical_max_blocks_per_ap = occupancy.has_value() ? occupancy->max_blocks_per_ap : 0,
      .theoretical_occupancy_pct = occupancy.has_value() ? occupancy->occupancy_ratio * 100.0f : 0.0f,
      .occupancy_wave_limiter = occupancy.has_value() ? occupancy->wave_limiting_factor : "",
      .occupancy_block_limiter = occupancy.has_value() ? occupancy->block_limiting_factor : "",
      .active_utilization_pct = pcs.has_value() && result.total_cycles > 0
                                    ? pcs->ActiveUtilization() * 100.0
                                    : 0.0,
      .total_flops = pcs.has_value() ? pcs->TotalFLOPs() : 0,
      .total_bytes = pcs.has_value() ? pcs->TotalBytes() : 0,
      .arithmetic_intensity = pcs.has_value() ? pcs->ArithmeticIntensity() : 0.0,
      .bound_classification = pcs.has_value() ? pcs->BoundClassification() : "unknown",
      .bytes_per_cycle = pcs.has_value() && result.total_cycles > 0
                             ? pcs->BytesPerCycle() : 0.0,
      .flops_per_cycle = pcs.has_value() && result.total_cycles > 0
                             ? pcs->FLOPsPerCycle() : 0.0,
      .memory_intensity = pcs.has_value() ? pcs->MemoryIntensity() : 0.0,
      .compute_intensity = pcs.has_value() ? pcs->ComputeIntensity() : 0.0,
  };
  trace.OnSummarySnapshot(summary_snapshot);
}

}  // namespace gpu_model
