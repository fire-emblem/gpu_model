#include "gpu_model/runtime/exec_engine.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <thread>

#include "gpu_model/arch/arch_registry.h"
#include "gpu_model/debug/trace/event_factory.h"
#include "gpu_model/execution/cycle_exec_engine.h"
#include "gpu_model/execution/functional_exec_engine.h"
#include "gpu_model/execution/program_object_exec_engine.h"
#include "gpu_model/execution/internal/cycle_issue_policy.h"
#include "gpu_model/isa/kernel_metadata.h"
#include "gpu_model/loader/device_image_loader.h"
#include "gpu_model/loader/asm_parser.h"
#include "gpu_model/runtime/runtime_config.h"
#include "gpu_model/util/logging.h"
#include "gpu_model/util/invocation.h"

namespace gpu_model {

class ExecEngineImpl {
 public:
  explicit ExecEngineImpl(TraceSink* default_trace = nullptr);

  MemorySystem& memory() { return memory_; }
  const MemorySystem& memory() const { return memory_; }
  void SetFixedGlobalMemoryLatency(uint64_t latency);
  void SetGlobalMemoryLatencyProfile(uint64_t dram_latency,
                                     uint64_t l2_hit_latency,
                                     uint64_t l1_hit_latency);
  void SetSharedBankConflictModel(uint32_t bank_count, uint32_t bank_width_bytes);
  void SetLaunchTimingProfile(uint64_t kernel_launch_gap_cycles,
                              uint64_t kernel_launch_cycles,
                              uint64_t block_launch_cycles,
                              uint64_t wave_generation_cycles,
                              uint64_t wave_dispatch_cycles,
                              uint64_t wave_launch_cycles,
                              uint64_t warp_switch_cycles,
                              uint64_t arg_load_cycles);
  void SetIssueCycleClassOverrides(const IssueCycleClassOverridesSpec& overrides);
  void SetIssueCycleOpOverrides(const IssueCycleOpOverridesSpec& overrides);
  void SetCycleIssueLimits(const ArchitecturalIssueLimits& limits) {
    issue_limits_override_ = limits;
  }
  void SetCycleIssuePolicy(const ArchitecturalIssuePolicy& policy) {
    issue_policy_override_ = policy;
  }
  void SetFunctionalExecutionConfig(FunctionalExecutionConfig config) {
    functional_execution_config_ = config;
  }
  void SetFunctionalExecutionMode(FunctionalExecutionMode mode) {
    functional_execution_config_.mode = mode;
  }
  const FunctionalExecutionConfig& functional_execution_config() const {
    return functional_execution_config_;
  }
  uint64_t device_cycle() const { return device_cycle_; }
  void ResetDeviceCycle() { device_cycle_ = 0; has_cycle_launch_history_ = false; }

  LaunchResult Launch(const LaunchRequest& request);

 private:
  CycleTimingConfig ResolveCycleTimingConfig(const GpuArchSpec& spec) const;
  TraceSink& ResolveTraceSink(TraceSink* request_trace);

  MemorySystem memory_;
  NullTraceSink null_trace_sink_;
  TraceSink* default_trace_ = nullptr;
  std::optional<uint64_t> flat_global_latency_override_;
  std::optional<uint64_t> dram_latency_override_;
  std::optional<uint64_t> l2_hit_latency_override_;
  std::optional<uint64_t> l1_hit_latency_override_;
  std::optional<uint32_t> shared_bank_count_override_;
  std::optional<uint32_t> shared_bank_width_override_;
  std::optional<uint64_t> kernel_launch_gap_cycles_override_;
  std::optional<uint64_t> kernel_launch_cycles_override_;
  std::optional<uint64_t> block_launch_cycles_override_;
  std::optional<uint64_t> wave_generation_cycles_override_;
  std::optional<uint64_t> wave_dispatch_cycles_override_;
  std::optional<uint64_t> wave_launch_cycles_override_;
  std::optional<uint64_t> warp_switch_cycles_override_;
  std::optional<uint64_t> arg_load_cycles_override_;
  std::optional<IssueCycleClassOverridesSpec> issue_cycle_class_overrides_;
  std::optional<IssueCycleOpOverridesSpec> issue_cycle_op_overrides_;
  std::optional<ArchitecturalIssueLimits> issue_limits_override_;
  std::optional<ArchitecturalIssuePolicy> issue_policy_override_;
  FunctionalExecutionConfig functional_execution_config_{};
  bool disable_trace_ = false;
  uint64_t device_cycle_ = 0;
  bool has_cycle_launch_history_ = false;
  // Must remain monotonic across launches on a long-lived ExecEngine/recorder.
  std::atomic<uint64_t> next_trace_flow_id_{1};
};

namespace {

const char* ToEnvModeName(FunctionalExecutionMode mode) {
  switch (mode) {
    case FunctionalExecutionMode::SingleThreaded:
      return "st";
    case FunctionalExecutionMode::MultiThreaded:
      return "mt";
  }
  return "unknown";
}

}  // namespace

ExecEngineImpl::ExecEngineImpl(TraceSink* default_trace) : default_trace_(default_trace) {
  logging::EnsureInitialized();
  const auto& config = GetRuntimeConfig();
  disable_trace_ = config.disable_trace;
  if (config.functional.mode != FunctionalExecutionMode::SingleThreaded ||
      config.functional.worker_threads > 0) {
    functional_execution_config_ = config.functional;
    GPU_MODEL_LOG_INFO("runtime",
                       "functional_mode=%s workers=%u",
                       ToEnvModeName(functional_execution_config_.mode),
                       functional_execution_config_.worker_threads);
  }
}

void ExecEngineImpl::SetFixedGlobalMemoryLatency(uint64_t latency) {
  flat_global_latency_override_ = latency;
  dram_latency_override_.reset();
  l2_hit_latency_override_.reset();
  l1_hit_latency_override_.reset();
}

void ExecEngineImpl::SetGlobalMemoryLatencyProfile(uint64_t dram_latency,
                                                   uint64_t l2_hit_latency,
                                                   uint64_t l1_hit_latency) {
  flat_global_latency_override_.reset();
  dram_latency_override_ = dram_latency;
  l2_hit_latency_override_ = l2_hit_latency;
  l1_hit_latency_override_ = l1_hit_latency;
}

void ExecEngineImpl::SetSharedBankConflictModel(uint32_t bank_count, uint32_t bank_width_bytes) {
  shared_bank_count_override_ = bank_count;
  shared_bank_width_override_ = bank_width_bytes;
}

void ExecEngineImpl::SetLaunchTimingProfile(uint64_t kernel_launch_gap_cycles,
                                            uint64_t kernel_launch_cycles,
                                            uint64_t block_launch_cycles,
                                            uint64_t wave_generation_cycles,
                                            uint64_t wave_dispatch_cycles,
                                            uint64_t wave_launch_cycles,
                                            uint64_t warp_switch_cycles,
                                            uint64_t arg_load_cycles) {
  kernel_launch_gap_cycles_override_ = kernel_launch_gap_cycles;
  kernel_launch_cycles_override_ = kernel_launch_cycles;
  block_launch_cycles_override_ = block_launch_cycles;
  wave_generation_cycles_override_ = wave_generation_cycles;
  wave_dispatch_cycles_override_ = wave_dispatch_cycles;
  wave_launch_cycles_override_ = wave_launch_cycles;
  warp_switch_cycles_override_ = warp_switch_cycles;
  arg_load_cycles_override_ = arg_load_cycles;
}

void ExecEngineImpl::SetIssueCycleClassOverrides(const IssueCycleClassOverridesSpec& overrides) {
  issue_cycle_class_overrides_ = overrides;
}

void ExecEngineImpl::SetIssueCycleOpOverrides(const IssueCycleOpOverridesSpec& overrides) {
  issue_cycle_op_overrides_ = overrides;
}

void ExecEngine::SetCycleIssueLimits(const ArchitecturalIssueLimits& limits) {
  impl_->SetCycleIssueLimits(limits);
}

void ExecEngine::SetCycleIssuePolicy(const ArchitecturalIssuePolicy& policy) {
  impl_->SetCycleIssuePolicy(policy);
}

LaunchResult ExecEngineImpl::Launch(const LaunchRequest& request) {
  LaunchResult result;

  std::string arch_name = request.arch_name;
  if (arch_name.empty() && request.program_object != nullptr) {
    const auto it = request.program_object->metadata().values.find("arch");
    if (it != request.program_object->metadata().values.end()) {
      arch_name = it->second;
    }
  }
  if (arch_name.empty()) {
    arch_name = "c500";
  }

  const auto spec = ArchRegistry::Get(arch_name);
  if (!spec) {
    result.error_message = "unknown architecture: " + arch_name;
    return result;
  }

  ExecutableKernel parsed_kernel;
  const ExecutableKernel* kernel = request.kernel;
  const ProgramObject* program_object = request.program_object;
  const bool use_program_object_payload =
      program_object != nullptr && program_object->has_encoded_payload();
  if (!use_program_object_payload && kernel == nullptr && program_object != nullptr) {
    parsed_kernel = AsmParser{}.Parse(*request.program_object);
    kernel = &parsed_kernel;
  }
  if (kernel == nullptr && !use_program_object_payload) {
    result.error_message = "launch request missing kernel or program object";
    return result;
  }
  if (request.config.grid_dim_x == 0 || request.config.grid_dim_y == 0 ||
      request.config.grid_dim_z == 0 || request.config.block_dim_x == 0 ||
      request.config.block_dim_y == 0 || request.config.block_dim_z == 0) {
    result.error_message = "grid and block dimensions must be non-zero";
    return result;
  }

  // Initialize with request config; may be adjusted for static shared memory
  LaunchConfig adjusted_config = request.config;

  GPU_MODEL_LOG_INFO("runtime",
                     "launch begin mode=%s program_payload=%d arch=%s grid=(%u,%u,%u) block=(%u,%u,%u)",
                     request.mode == ExecutionMode::Cycle ? "cycle" : "functional",
                     use_program_object_payload ? 1 : 0,
                     arch_name.c_str(),
                     request.config.grid_dim_x,
                     request.config.grid_dim_y,
                     request.config.grid_dim_z,
                     request.config.block_dim_x,
                     request.config.block_dim_y,
                     request.config.block_dim_z);

  const MetadataBlob& launch_metadata_source =
      use_program_object_payload ? program_object->metadata()
                                 : (request.program_object != nullptr && kernel == nullptr
                                        ? request.program_object->metadata()
                                        : kernel->metadata());

  try {
    const auto launch_metadata = ParseKernelLaunchMetadata(launch_metadata_source);
    if (launch_metadata.arch.has_value() && *launch_metadata.arch != spec->name) {
      result.error_message =
          "kernel metadata arch does not match selected architecture";
      return result;
    }
    const std::string kernel_name =
        use_program_object_payload ? program_object->kernel_name() : kernel->name();
    if (launch_metadata.entry.has_value() && *launch_metadata.entry != kernel_name) {
      result.error_message = "kernel metadata entry does not match kernel name";
      return result;
    }
    if (!launch_metadata.module_kernels.empty()) {
      const bool found = std::find(launch_metadata.module_kernels.begin(),
                                   launch_metadata.module_kernels.end(),
                                   kernel_name) != launch_metadata.module_kernels.end();
      if (!found) {
        result.error_message = "kernel name is not present in module_kernels metadata";
        return result;
      }
    }
    if (launch_metadata.arg_count.has_value() &&
        request.args.size() != *launch_metadata.arg_count) {
      result.error_message = "kernel argument count does not match metadata";
      return result;
    }
    const uint32_t statically_loaded_shared_bytes =
        request.device_load != nullptr ? request.device_load->required_shared_bytes : 0u;
    const uint32_t available_shared_bytes =
        std::max({request.config.shared_memory_bytes,
                  launch_metadata.group_segment_fixed_size.value_or(0u),
                  statically_loaded_shared_bytes});
    if (launch_metadata.required_shared_bytes.has_value() &&
        available_shared_bytes < *launch_metadata.required_shared_bytes) {
      result.error_message = "shared memory launch size is smaller than metadata requirement";
      return result;
    }
    // Adjust config with proper shared memory size that includes static shared memory
    adjusted_config.shared_memory_bytes = available_shared_bytes;
    GPU_MODEL_LOG_INFO("runtime", "shared memory: launch=%u group_segment=%u static_load=%u available=%u",
                       request.config.shared_memory_bytes,
                       launch_metadata.group_segment_fixed_size.value_or(0u),
                       statically_loaded_shared_bytes,
                       available_shared_bytes);
    if (launch_metadata.block_dim_multiple.has_value() &&
        request.config.block_dim_x % *launch_metadata.block_dim_multiple != 0) {
      result.error_message = "block_dim_x does not satisfy metadata multiple requirement";
      return result;
    }
    if (launch_metadata.max_block_dim.has_value() &&
        request.config.block_dim_x > *launch_metadata.max_block_dim) {
      result.error_message = "block_dim_x exceeds metadata maximum";
      return result;
    }
  } catch (const std::exception& ex) {
    result.error_message = std::string("failed to parse kernel metadata: ") + ex.what();
    return result;
  }

  auto& trace = ResolveTraceSink(request.trace);
  try {
    // Emit run snapshot
    TraceRunSnapshot run_snapshot{
        .invocation = CaptureInvocationLine(),
        .execution_model = request.mode == ExecutionMode::Cycle ? "cycle" : "functional",
        .functional_mode = request.functional_mode,
        .trace_time_basis = "modeled_cycle",
        .trace_cycle_is_physical_time = false,
    };
    trace.OnRunSnapshot(run_snapshot);

    // Emit model config snapshot
    TraceModelConfigSnapshot model_config_snapshot{
        .num_dpcs = spec->dpc_count,
        .num_aps_per_dpc = spec->ap_per_dpc,
        .num_peus_per_ap = spec->peu_per_ap,
        .num_slots_per_peu = spec->cycle_resources.resident_wave_slots_per_peu,
        .slot_model = "resident_fixed",
    };
    trace.OnModelConfigSnapshot(model_config_snapshot);

    // Emit kernel snapshot
    TraceKernelSnapshot kernel_snapshot{
        .kernel_name = use_program_object_payload ? program_object->kernel_name() : kernel->name(),
        .kernel_launch_uid = 0,
        .launch_index = request.launch_index,
        .grid_dim_x = request.config.grid_dim_x,
        .grid_dim_y = request.config.grid_dim_y,
        .grid_dim_z = request.config.grid_dim_z,
        .block_dim_x = request.config.block_dim_x,
        .block_dim_y = request.config.block_dim_y,
        .block_dim_z = request.config.block_dim_z,
    };
    trace.OnKernelSnapshot(kernel_snapshot);

    if (request.mode == ExecutionMode::Cycle) {
      result.submit_cycle =
          has_cycle_launch_history_ ? device_cycle_ + spec->launch_timing.kernel_launch_gap_cycles
                                    : 0;
      result.begin_cycle = result.submit_cycle + spec->launch_timing.kernel_launch_cycles;
    }
    std::ostringstream launch_message;
    launch_message << "kernel="
                   << (use_program_object_payload ? program_object->kernel_name() : kernel->name())
                   << " arch=" << spec->name;
    if (use_program_object_payload &&
        (program_object->kernel_descriptor().agpr_count != 0 ||
         program_object->kernel_descriptor().accum_offset != 0)) {
      launch_message << " agpr_count=" << program_object->kernel_descriptor().agpr_count
                     << " accum_offset=" << program_object->kernel_descriptor().accum_offset;
    }
    trace.OnEvent(MakeTraceRuntimeLaunchEvent(result.submit_cycle, launch_message.str()));

    result.placement = Mapper::Place(*spec, request.config);
    for (const auto& block : result.placement.blocks) {
      std::ostringstream message;
      message << "block=" << block.block_id << " block_xyz=(" << block.block_idx_x << ","
              << block.block_idx_y << "," << block.block_idx_z << ") dpc=" << block.dpc_id << " ap=" << block.ap_id
              << " waves=" << block.waves.size();
      trace.OnEvent(MakeTraceBlockPlacedEvent(block.dpc_id,
                                              block.ap_id,
                                              block.block_id,
                                              result.submit_cycle,
                                              message.str()));
    }
    if (request.mode == ExecutionMode::Functional) {
        if (use_program_object_payload) {
          const auto raw_result =
            ProgramObjectExecEngine{}.Run(*program_object, *spec,
                                    ResolveCycleTimingConfig(*spec),
                                    request.config,
                                    ExecutionMode::Functional,
                                    functional_execution_config_,
                                    request.args,
                                    request.device_load, memory_, trace, &next_trace_flow_id_);
        result.ok = raw_result.ok;
        result.error_message = raw_result.error_message;
        result.total_cycles = raw_result.total_cycles;
        result.end_cycle = raw_result.end_cycle;
        result.stats = raw_result.stats;
        result.program_cycle_stats = raw_result.program_cycle_stats;
      } else {
        ExecutionContext context{
            .spec = *spec,
            .kernel = *kernel,
            .launch_config = adjusted_config,
            .args = request.args,
            .placement = result.placement,
            .device_load = request.device_load,
            .memory = memory_,
            .trace = trace,
            .trace_flow_id_source = &next_trace_flow_id_,
            .stats = &result.stats,
            .global_memory_latency_cycles =
                ResolveCycleTimingConfig(*spec).cache_model.dram_latency,
            .arg_load_cycles = spec->launch_timing.arg_load_cycles,
            .issue_cycle_class_overrides = ResolveCycleTimingConfig(*spec).issue_cycle_class_overrides,
            .issue_cycle_op_overrides = ResolveCycleTimingConfig(*spec).issue_cycle_op_overrides,
        };
        if (functional_execution_config_.mode == FunctionalExecutionMode::MultiThreaded) {
          FunctionalExecEngine executor(context);
          const uint32_t workers =
              functional_execution_config_.worker_threads == 0
                  ? DefaultMtWorkerThreadCountForEnv()
                  : functional_execution_config_.worker_threads;
          result.total_cycles = executor.RunParallelBlocks(workers);
          result.program_cycle_stats = executor.TakeProgramCycleStats();
        } else {
          FunctionalExecEngine executor(context);
          result.total_cycles = executor.RunSequential();
          result.program_cycle_stats = executor.TakeProgramCycleStats();
        }
        result.end_cycle = result.begin_cycle + result.total_cycles;
      }
    } else if (request.mode == ExecutionMode::Cycle) {
      if (use_program_object_payload) {
        const auto raw_result =
            ProgramObjectExecEngine{}.Run(*program_object, *spec,
                                    ResolveCycleTimingConfig(*spec),
                                    request.config,
                                    ExecutionMode::Cycle,
                                    FunctionalExecutionConfig{},
                                    request.args,
                                    request.device_load, memory_, trace, &next_trace_flow_id_);
        result.ok = raw_result.ok;
        result.error_message = raw_result.error_message;
        result.stats = raw_result.stats;
        result.program_cycle_stats = raw_result.program_cycle_stats;
        result.total_cycles = raw_result.program_cycle_stats.has_value()
                                  ? raw_result.program_cycle_stats->total_cycles
                                  : raw_result.total_cycles;
        result.end_cycle = result.begin_cycle + result.total_cycles;
      } else {
        ExecutionContext context{
            .spec = *spec,
            .kernel = *kernel,
            .launch_config = adjusted_config,
            .args = request.args,
            .placement = result.placement,
            .device_load = request.device_load,
            .memory = memory_,
            .trace = trace,
            .trace_flow_id_source = &next_trace_flow_id_,
            .stats = &result.stats,
            .arg_load_cycles = spec->launch_timing.arg_load_cycles,
            .issue_cycle_class_overrides = ResolveCycleTimingConfig(*spec).issue_cycle_class_overrides,
            .issue_cycle_op_overrides = ResolveCycleTimingConfig(*spec).issue_cycle_op_overrides,
        };
        context.cycle = result.begin_cycle;
        CycleExecEngine executor(ResolveCycleTimingConfig(*spec));
        result.end_cycle = executor.Run(context);
        result.total_cycles = result.end_cycle - result.begin_cycle;
        result.program_cycle_stats = executor.TakeProgramCycleStats();
      }
      device_cycle_ = result.end_cycle;
      has_cycle_launch_history_ = true;
    } else {
      result.error_message = "requested execution mode is not implemented";
      return result;
    }

    if (!use_program_object_payload || result.error_message.empty()) {
      result.ok = true;
    }

    // Emit summary snapshot
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
        .stall_other = pcs.has_value()
                           ? pcs->stall_resource + pcs->stall_dependency
                           : 0,
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
        .waves_launched = pcs.has_value() ? pcs->waves_launched : 0,
        .waves_completed = pcs.has_value() ? pcs->waves_completed : 0,
        .max_concurrent_waves = pcs.has_value() ? pcs->max_concurrent_waves : 0,
        .active_utilization_pct = pcs.has_value() && pcs->total_cycles > 0
                                      ? pcs->ActiveUtilization() * 100.0
                                      : 0.0,
    };
    trace.OnSummarySnapshot(summary_snapshot);
  } catch (const std::exception& ex) {
    result.error_message = ex.what();
    result.ok = false;
  }

  return result;
}

CycleTimingConfig ExecEngineImpl::ResolveCycleTimingConfig(const GpuArchSpec& spec) const {
  CycleTimingConfig config;
  config.cache_model = spec.cache_model;
  config.shared_bank_model = spec.shared_bank_model;
  config.launch_timing = spec.launch_timing;
  config.issue_cycle_class_overrides = spec.issue_cycle_class_overrides;
  config.issue_cycle_op_overrides = spec.issue_cycle_op_overrides;
  config.issue_limits = CycleIssueLimitsForSpec(spec);
  config.issue_policy = CycleIssuePolicyForSpec(spec);
  config.eligible_wave_selection_policy = CycleEligibleWaveSelectionPolicyForSpec(spec);
  const bool has_issue_policy_override = issue_policy_override_.has_value();

  if (flat_global_latency_override_.has_value()) {
    config.cache_model.enabled = false;
    config.cache_model.dram_latency = *flat_global_latency_override_;
    config.cache_model.l2_hit_latency = *flat_global_latency_override_;
    config.cache_model.l1_hit_latency = *flat_global_latency_override_;
  } else {
    if (dram_latency_override_.has_value()) {
      config.cache_model.dram_latency = *dram_latency_override_;
    }
    if (l2_hit_latency_override_.has_value()) {
      config.cache_model.l2_hit_latency = *l2_hit_latency_override_;
    }
    if (l1_hit_latency_override_.has_value()) {
      config.cache_model.l1_hit_latency = *l1_hit_latency_override_;
    }
  }

  if (shared_bank_count_override_.has_value()) {
    config.shared_bank_model.enabled = true;
    config.shared_bank_model.bank_count = *shared_bank_count_override_;
  }
  if (shared_bank_width_override_.has_value()) {
    config.shared_bank_model.enabled = true;
    config.shared_bank_model.bank_width_bytes = *shared_bank_width_override_;
  }

  if (kernel_launch_gap_cycles_override_.has_value()) {
    config.launch_timing.kernel_launch_gap_cycles = *kernel_launch_gap_cycles_override_;
  }
  if (kernel_launch_cycles_override_.has_value()) {
    config.launch_timing.kernel_launch_cycles = *kernel_launch_cycles_override_;
  }
  if (block_launch_cycles_override_.has_value()) {
    config.launch_timing.block_launch_cycles = *block_launch_cycles_override_;
  }
  if (wave_generation_cycles_override_.has_value()) {
    config.launch_timing.wave_generation_cycles = *wave_generation_cycles_override_;
  }
  if (wave_dispatch_cycles_override_.has_value()) {
    config.launch_timing.wave_dispatch_cycles = *wave_dispatch_cycles_override_;
  }
  if (wave_launch_cycles_override_.has_value()) {
    config.launch_timing.wave_launch_cycles = *wave_launch_cycles_override_;
  }
  if (warp_switch_cycles_override_.has_value()) {
    config.launch_timing.warp_switch_cycles = *warp_switch_cycles_override_;
  }
  if (arg_load_cycles_override_.has_value()) {
    config.launch_timing.arg_load_cycles = *arg_load_cycles_override_;
  }
  if (issue_cycle_class_overrides_.has_value()) {
    config.issue_cycle_class_overrides = *issue_cycle_class_overrides_;
  }
  if (issue_cycle_op_overrides_.has_value()) {
    config.issue_cycle_op_overrides = *issue_cycle_op_overrides_;
  }
  if (has_issue_policy_override) {
    config.issue_policy = *issue_policy_override_;
    config.issue_limits = issue_policy_override_->type_limits;
  }
  if (issue_limits_override_.has_value()) {
    config.issue_limits = *issue_limits_override_;
  }
  if (issue_limits_override_.has_value() && !has_issue_policy_override) {
    config.issue_policy = CycleIssuePolicyWithLimits(*config.issue_policy, config.issue_limits);
  } else if (config.issue_policy.has_value()) {
    config.issue_policy->type_limits = config.issue_limits;
  }

  return config;
}

TraceSink& ExecEngineImpl::ResolveTraceSink(TraceSink* request_trace) {
  // If a TraceSink is explicitly passed in request, always use it
  if (request_trace != nullptr) {
    return *request_trace;
  }
  // If a TraceSink was passed to constructor, always use it
  if (default_trace_ != nullptr) {
    return *default_trace_;
  }
  // Otherwise, check if trace is disabled
  if (disable_trace_) {
    return null_trace_sink_;
  }
  return null_trace_sink_;
}

ExecEngine::ExecEngine(TraceSink* default_trace)
    : impl_(std::make_unique<ExecEngineImpl>(default_trace)) {}

ExecEngine::~ExecEngine() = default;
ExecEngine::ExecEngine(ExecEngine&& other) noexcept = default;
ExecEngine& ExecEngine::operator=(ExecEngine&& other) noexcept = default;

MemorySystem& ExecEngine::memory() {
  return impl_->memory();
}

const MemorySystem& ExecEngine::memory() const {
  return impl_->memory();
}

void ExecEngine::SetFixedGlobalMemoryLatency(uint64_t latency) {
  impl_->SetFixedGlobalMemoryLatency(latency);
}

void ExecEngine::SetGlobalMemoryLatencyProfile(uint64_t dram_latency,
                                               uint64_t l2_hit_latency,
                                               uint64_t l1_hit_latency) {
  impl_->SetGlobalMemoryLatencyProfile(dram_latency, l2_hit_latency, l1_hit_latency);
}

void ExecEngine::SetSharedBankConflictModel(uint32_t bank_count, uint32_t bank_width_bytes) {
  impl_->SetSharedBankConflictModel(bank_count, bank_width_bytes);
}

void ExecEngine::SetLaunchTimingProfile(uint64_t kernel_launch_gap_cycles,
                                        uint64_t kernel_launch_cycles,
                                        uint64_t block_launch_cycles,
                                        uint64_t wave_launch_cycles,
                                        uint64_t warp_switch_cycles,
                                        uint64_t arg_load_cycles) {
  impl_->SetLaunchTimingProfile(kernel_launch_gap_cycles,
                                kernel_launch_cycles,
                                block_launch_cycles,
                                /*wave_generation_cycles=*/0,
                                /*wave_dispatch_cycles=*/0,
                                wave_launch_cycles,
                                warp_switch_cycles,
                                arg_load_cycles);
}

void ExecEngine::SetLaunchTimingProfile(uint64_t kernel_launch_gap_cycles,
                                        uint64_t kernel_launch_cycles,
                                        uint64_t block_launch_cycles,
                                        uint64_t wave_generation_cycles,
                                        uint64_t wave_dispatch_cycles,
                                        uint64_t wave_launch_cycles,
                                        uint64_t warp_switch_cycles,
                                        uint64_t arg_load_cycles) {
  impl_->SetLaunchTimingProfile(kernel_launch_gap_cycles,
                                kernel_launch_cycles,
                                block_launch_cycles,
                                wave_generation_cycles,
                                wave_dispatch_cycles,
                                wave_launch_cycles,
                                warp_switch_cycles,
                                arg_load_cycles);
}

void ExecEngine::SetIssueCycleClassOverrides(const IssueCycleClassOverridesSpec& overrides) {
  impl_->SetIssueCycleClassOverrides(overrides);
}

void ExecEngine::SetIssueCycleOpOverrides(const IssueCycleOpOverridesSpec& overrides) {
  impl_->SetIssueCycleOpOverrides(overrides);
}

void ExecEngine::SetFunctionalExecutionConfig(FunctionalExecutionConfig config) {
  impl_->SetFunctionalExecutionConfig(config);
}

void ExecEngine::SetFunctionalExecutionMode(FunctionalExecutionMode mode) {
  impl_->SetFunctionalExecutionMode(mode);
}

const FunctionalExecutionConfig& ExecEngine::functional_execution_config() const {
  return impl_->functional_execution_config();
}

uint64_t ExecEngine::device_cycle() const {
  return impl_->device_cycle();
}

void ExecEngine::ResetDeviceCycle() {
  impl_->ResetDeviceCycle();
}

LaunchResult ExecEngine::Launch(const LaunchRequest& request) {
  return impl_->Launch(request);
}

}  // namespace gpu_model
