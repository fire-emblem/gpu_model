#pragma once

#include <array>
#include <atomic>
#include <bitset>
#include <cstdint>

#include "gpu_arch/chip_config/gpu_arch_spec.h"
#include "debug/trace/sink.h"
#include "execution/internal/op_plan.h"
#include "gpu_arch/memory/memory_system.h"
#include "program/executable/executable_kernel.h"
#include "runtime/kernel_arg_pack.h"
#include "runtime/launch_config.h"
#include "runtime/launch_request.h"
#include "runtime/mapper.h"
#include "state/wave/wave_runtime_state.h"

namespace gpu_model {

struct DeviceLoadResult;

struct ExecutionContext {
  const GpuArchSpec& spec;
  const ExecutableKernel& kernel;
  const LaunchConfig& launch_config;
  const KernelArgPack& args;
  const PlacementMap& placement;
  const DeviceLoadResult* device_load = nullptr;
  MemorySystem& memory;
  TraceSink& trace;
  // Shared flow-id allocator; must be monotonic over the recorder/engine lifetime.
  std::atomic<uint64_t>* trace_flow_id_source = nullptr;
  ExecutionStats* stats = nullptr;
  uint64_t cycle = 0;
  uint64_t global_memory_latency_cycles = 0;
  uint64_t arg_load_cycles = 4;
  IssueCycleClassOverridesSpec issue_cycle_class_overrides;
  IssueCycleOpOverridesSpec issue_cycle_op_overrides;

  uint64_t AllocateTraceFlowId() const {
    if (trace_flow_id_source == nullptr) {
      return 0;
    }
    return trace_flow_id_source->fetch_add(1, std::memory_order_relaxed);
  }
};

class Semantics {
 public:
  OpPlan BuildPlan(const Instruction& instruction,
                   const WaveContext& wave,
                   const ExecutionContext& context) const;
};

}  // namespace gpu_model
