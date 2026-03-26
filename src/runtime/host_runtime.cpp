#include "gpu_model/runtime/host_runtime.h"

#include <sstream>
#include <stdexcept>

#include "gpu_model/arch/arch_registry.h"
#include "gpu_model/exec/cycle_executor.h"
#include "gpu_model/exec/functional_executor.h"
#include "gpu_model/isa/kernel_program.h"
#include "gpu_model/loader/asm_parser.h"

namespace gpu_model {

HostRuntime::HostRuntime(TraceSink* default_trace) : default_trace_(default_trace) {}

void HostRuntime::SetFixedGlobalMemoryLatency(uint64_t latency) {
  flat_global_latency_override_ = latency;
  dram_latency_override_.reset();
  l2_hit_latency_override_.reset();
  l1_hit_latency_override_.reset();
}

void HostRuntime::SetGlobalMemoryLatencyProfile(uint64_t dram_latency,
                                                uint64_t l2_hit_latency,
                                                uint64_t l1_hit_latency) {
  flat_global_latency_override_.reset();
  dram_latency_override_ = dram_latency;
  l2_hit_latency_override_ = l2_hit_latency;
  l1_hit_latency_override_ = l1_hit_latency;
}

void HostRuntime::SetSharedBankConflictModel(uint32_t bank_count, uint32_t bank_width_bytes) {
  shared_bank_count_override_ = bank_count;
  shared_bank_width_override_ = bank_width_bytes;
}

LaunchResult HostRuntime::Launch(const LaunchRequest& request) {
  LaunchResult result;

  std::string arch_name = request.arch_name;
  if (arch_name.empty() && request.program_image != nullptr) {
    const auto it = request.program_image->metadata().values.find("arch");
    if (it != request.program_image->metadata().values.end()) {
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

  KernelProgram parsed_kernel;
  const KernelProgram* kernel = request.kernel;
  if (kernel == nullptr && request.program_image != nullptr) {
    parsed_kernel = AsmParser{}.Parse(*request.program_image);
    kernel = &parsed_kernel;
  }
  if (kernel == nullptr) {
    result.error_message = "launch request missing kernel or program image";
    return result;
  }
  if (request.config.grid_dim_x == 0 || request.config.block_dim_x == 0) {
    result.error_message = "grid_dim_x and block_dim_x must be non-zero";
    return result;
  }

  auto& trace = ResolveTraceSink(request.trace);
  trace.OnEvent(TraceEvent{
      .kind = TraceEventKind::Launch,
      .message = "kernel=" + kernel->name() + " arch=" + spec->name,
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
        .kernel = *kernel,
        .launch_config = request.config,
        .args = request.args,
        .placement = result.placement,
        .memory = memory_,
        .trace = trace,
        .stats = &result.stats,
    };

    if (request.mode == ExecutionMode::Functional) {
      FunctionalExecutor executor;
      result.total_cycles = executor.Run(context);
    } else if (request.mode == ExecutionMode::Cycle) {
      CycleExecutor executor(ResolveCycleTimingConfig(*spec));
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

CycleTimingConfig HostRuntime::ResolveCycleTimingConfig(const GpuArchSpec& spec) const {
  CycleTimingConfig config;
  config.cache_model = spec.cache_model;
  config.shared_bank_model = spec.shared_bank_model;

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

  return config;
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
