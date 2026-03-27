#pragma once

#include "gpu_model/arch/gpu_arch_spec.h"
#include "gpu_model/debug/trace_sink.h"
#include "gpu_model/loader/amdgpu_code_object_decoder.h"
#include "gpu_model/memory/memory_system.h"
#include "gpu_model/runtime/kernel_arg_pack.h"
#include "gpu_model/runtime/launch_config.h"
#include "gpu_model/runtime/launch_request.h"
#include "gpu_model/runtime/mapper.h"

namespace gpu_model {

class RawGcnExecutor {
 public:
  LaunchResult Run(const AmdgpuCodeObjectImage& image,
                   const GpuArchSpec& spec,
                   const LaunchConfig& config,
                   const KernelArgPack& args,
                   MemorySystem& memory,
                   TraceSink& trace) const;
};

}  // namespace gpu_model
