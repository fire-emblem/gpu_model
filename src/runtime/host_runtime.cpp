#include "gpu_model/runtime/host_runtime.h"

#include <sstream>
#include <stdexcept>

#include "gpu_model/arch/arch_registry.h"
#include "gpu_model/exec/cycle_executor.h"
#include "gpu_model/exec/functional_executor.h"
#include "gpu_model/isa/kernel_program.h"

namespace gpu_model {

HostRuntime::HostRuntime(TraceSink* default_trace) : default_trace_(default_trace) {}

LaunchResult HostRuntime::Launch(const LaunchRequest& request) {
  LaunchResult result;

  const auto spec = ArchRegistry::Get(request.arch_name);
  if (!spec) {
    result.error_message = "unknown architecture: " + request.arch_name;
    return result;
  }
  if (request.kernel == nullptr) {
    result.error_message = "launch request missing kernel";
    return result;
  }
  if (request.config.grid_dim_x == 0 || request.config.block_dim_x == 0) {
    result.error_message = "grid_dim_x and block_dim_x must be non-zero";
    return result;
  }

  auto& trace = ResolveTraceSink(request.trace);
  trace.OnEvent(TraceEvent{
      .kind = TraceEventKind::Launch,
      .message = "kernel=" + request.kernel->name() + " arch=" + spec->name,
  });

  try {
    result.placement = Mapper::Place(*spec, request.config);
    for (const auto& block : result.placement.blocks) {
      std::ostringstream message;
      message << "block=" << block.block_id << " dpc=" << block.dpc_id << " ap=" << block.ap_id
              << " waves=" << block.waves.size();
      trace.OnEvent(TraceEvent{
          .kind = TraceEventKind::BlockPlaced,
          .block_id = block.block_id,
          .message = message.str(),
      });
    }

    ExecutionContext context{
        .spec = *spec,
        .kernel = *request.kernel,
        .launch_config = request.config,
        .args = request.args,
        .placement = result.placement,
        .memory = memory_,
        .trace = trace,
    };

    if (request.mode == ExecutionMode::Functional) {
      FunctionalExecutor executor;
      result.total_cycles = executor.Run(context);
    } else if (request.mode == ExecutionMode::Cycle) {
      CycleExecutor executor(fixed_global_memory_latency_);
      result.total_cycles = executor.Run(context);
    } else {
      result.error_message = "requested execution mode is not implemented";
      return result;
    }

    result.ok = true;
  } catch (const std::exception& ex) {
    result.error_message = ex.what();
    result.ok = false;
  }

  return result;
}

TraceSink& HostRuntime::ResolveTraceSink(TraceSink* request_trace) {
  if (request_trace != nullptr) {
    return *request_trace;
  }
  if (default_trace_ != nullptr) {
    return *default_trace_;
  }
  return null_trace_sink_;
}

}  // namespace gpu_model
