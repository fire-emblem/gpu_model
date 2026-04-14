#pragma once

#include <atomic>

#include "gpu_arch/chip_config/gpu_arch_spec.h"
#include "execution/cycle/cycle_exec_engine.h"
#include "debug/trace/sink.h"
#include "gpu_arch/memory/memory_system.h"
#include "program/program_object/program_object.h"
#include "runtime/config/kernel_arg_pack.h"
#include "runtime/config/launch_config.h"
#include "runtime/config/launch_request.h"
#include "runtime/model_runtime/mapper.h"
#include "utils/config/execution_mode.h"

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
