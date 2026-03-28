#pragma once

#include <array>
#include <bitset>
#include <cstdint>

#include "gpu_model/arch/gpu_arch_spec.h"
#include "gpu_model/debug/trace_sink.h"
#include "gpu_model/exec/op_plan.h"
#include "gpu_model/isa/kernel_program.h"
#include "gpu_model/memory/memory_system.h"
#include "gpu_model/runtime/kernel_arg_pack.h"
#include "gpu_model/runtime/launch_config.h"
#include "gpu_model/runtime/launch_request.h"
#include "gpu_model/runtime/mapper.h"
#include "gpu_model/state/wave_state.h"

namespace gpu_model {

struct DeviceLoadResult;

struct ExecutionContext {
  const GpuArchSpec& spec;
  const KernelProgram& kernel;
  const LaunchConfig& launch_config;
  const KernelArgPack& args;
  const PlacementMap& placement;
  const DeviceLoadResult* device_load = nullptr;
  MemorySystem& memory;
  TraceSink& trace;
  ExecutionStats* stats = nullptr;
  uint64_t cycle = 0;
  uint64_t arg_load_cycles = 4;
  IssueCycleClassOverridesSpec issue_cycle_class_overrides;
  IssueCycleOpOverridesSpec issue_cycle_op_overrides;
};

class Semantics {
 public:
  OpPlan BuildPlan(const Instruction& instruction,
                   const WaveState& wave,
                   const ExecutionContext& context) const;
};

}  // namespace gpu_model
