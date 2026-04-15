#include "runtime/exec_engine/launch_dispatcher.h"

#include "execution/cycle/cycle_exec_engine.h"
#include "execution/encoded/program_object_exec_engine.h"
#include "execution/functional/functional_exec_engine.h"
#include "runtime/config/runtime_env_config.h"

namespace gpu_model {

bool DispatchLaunch(const LaunchRequest& request,
                    const ValidatedLaunchRequest& prepared,
                    const CycleTimingConfig& timing_config,
                    const FunctionalExecutionConfig& functional_execution_config,
                    MemorySystem& memory,
                    TraceSink& trace,
                    std::atomic<uint64_t>* trace_flow_id_source,
                    LaunchResult& result) {
  const auto* spec = prepared.spec;
  const auto* program_object = prepared.program_object;
  const bool use_program_object_payload = prepared.use_program_object_payload;

  if (request.mode == ExecutionMode::Functional) {
    if (use_program_object_payload) {
      const auto raw_result =
          ProgramObjectExecEngine{}.Run(*program_object,
                                        *spec,
                                        timing_config,
                                        request.config,
                                        ExecutionMode::Functional,
                                        functional_execution_config,
                                        request.args,
                                        request.device_load,
                                        memory,
                                        trace,
                                        trace_flow_id_source);
      result.ok = raw_result.ok;
      result.error_message = raw_result.error_message;
      result.total_cycles = raw_result.total_cycles;
      result.end_cycle = raw_result.end_cycle;
      result.stats = raw_result.stats;
      result.program_cycle_stats = raw_result.program_cycle_stats;
      return true;
    }

    ExecutionContext context{
        .spec = *spec,
        .kernel = prepared.kernel_ref(),
        .launch_config = prepared.adjusted_config,
        .args = request.args,
        .placement = result.placement,
        .device_load = request.device_load,
        .memory = memory,
        .trace = trace,
        .trace_flow_id_source = trace_flow_id_source,
        .stats = &result.stats,
        .global_memory_latency_cycles = timing_config.cache_model.dram_latency,
        .arg_load_cycles = spec->launch_timing.arg_load_cycles,
        .issue_cycle_class_overrides = timing_config.issue_cycle_class_overrides,
        .issue_cycle_op_overrides = timing_config.issue_cycle_op_overrides,
    };
    if (functional_execution_config.mode == FunctionalExecutionMode::MultiThreaded) {
      FunctionalExecEngine executor(context);
      const uint32_t workers =
          functional_execution_config.worker_threads == 0
              ? DefaultMtWorkerThreadCountForEnv()
              : functional_execution_config.worker_threads;
      result.total_cycles = executor.RunParallelBlocks(workers);
      result.program_cycle_stats = executor.TakeProgramCycleStats();
    } else {
      FunctionalExecEngine executor(context);
      result.total_cycles = executor.RunSequential();
      result.program_cycle_stats = executor.TakeProgramCycleStats();
    }
    result.end_cycle = result.begin_cycle + result.total_cycles;
    return true;
  }

  if (request.mode == ExecutionMode::Cycle) {
    if (use_program_object_payload) {
      const auto raw_result =
          ProgramObjectExecEngine{}.Run(*program_object,
                                        *spec,
                                        timing_config,
                                        request.config,
                                        ExecutionMode::Cycle,
                                        FunctionalExecutionConfig{},
                                        request.args,
                                        request.device_load,
                                        memory,
                                        trace,
                                        trace_flow_id_source);
      result.ok = raw_result.ok;
      result.error_message = raw_result.error_message;
      result.stats = raw_result.stats;
      result.program_cycle_stats = raw_result.program_cycle_stats;
      result.total_cycles = raw_result.program_cycle_stats.has_value()
                                ? raw_result.program_cycle_stats->total_cycles
                                : raw_result.total_cycles;
      result.end_cycle = result.begin_cycle + result.total_cycles;
      return true;
    }

    ExecutionContext context{
        .spec = *spec,
        .kernel = prepared.kernel_ref(),
        .launch_config = prepared.adjusted_config,
        .args = request.args,
        .placement = result.placement,
        .device_load = request.device_load,
        .memory = memory,
        .trace = trace,
        .trace_flow_id_source = trace_flow_id_source,
        .stats = &result.stats,
        .arg_load_cycles = spec->launch_timing.arg_load_cycles,
        .issue_cycle_class_overrides = timing_config.issue_cycle_class_overrides,
        .issue_cycle_op_overrides = timing_config.issue_cycle_op_overrides,
    };
    context.cycle = result.begin_cycle;
    CycleExecEngine executor(timing_config);
    result.end_cycle = executor.Run(context);
    result.total_cycles = result.end_cycle - result.begin_cycle;
    result.program_cycle_stats = executor.TakeProgramCycleStats();
    return true;
  }

  result.error_message = "requested execution mode is not implemented";
  return false;
}

}  // namespace gpu_model
