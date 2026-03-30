#pragma once

#include "gpu_model/arch/gpu_arch_spec.h"
#include "gpu_model/debug/trace_sink.h"
#include "gpu_model/memory/memory_system.h"
#include "gpu_model/program/encoded_program_object.h"
#include "gpu_model/runtime/kernel_arg_pack.h"
#include "gpu_model/runtime/launch_config.h"
#include "gpu_model/runtime/launch_request.h"
#include "gpu_model/runtime/mapper.h"

namespace gpu_model {

class EncodedExecEngine {
 public:
  LaunchResult Run(const EncodedProgramObject& image,
                   const GpuArchSpec& spec,
                   const LaunchConfig& config,
                   const KernelArgPack& args,
                   const DeviceLoadResult* device_load,
                   MemorySystem& memory,
                   TraceSink& trace) const;
};

}  // namespace gpu_model
