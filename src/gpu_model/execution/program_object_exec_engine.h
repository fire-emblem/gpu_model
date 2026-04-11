#pragma once

#include <atomic>

#include "gpu_model/gpu_arch/chip_config/gpu_arch_spec.h"
#include "gpu_model/execution/cycle_exec_engine.h"
#include "gpu_model/debug/trace/sink.h"
#include "gpu_model/memory/memory_system.h"
#include "gpu_model/program/program_object.h"
#include "gpu_model/runtime/kernel_arg_pack.h"
#include "gpu_model/runtime/launch_config.h"
#include "gpu_model/runtime/launch_request.h"
#include "gpu_model/runtime/mapper.h"
#include "gpu_model/utils/config/execution_mode.h"

namespace gpu_model {

class ProgramObjectExecEngine {
 public:
  LaunchResult Run(const ProgramObject& image,
                   const GpuArchSpec& spec,
                   const CycleTimingConfig& timing_config,
                   const LaunchConfig& config,
                   ExecutionMode execution_mode,
                   FunctionalExecutionConfig functional_execution_config,
                   const KernelArgPack& args,
                   const DeviceLoadResult* device_load,
                   MemorySystem& memory,
                   TraceSink& trace,
                   std::atomic<uint64_t>* trace_flow_id_source) const;
};

}  // namespace gpu_model
