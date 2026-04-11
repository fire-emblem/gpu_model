#pragma once

#include <array>
#include <atomic>
#include <bitset>
#include <cstdint>

#include "gpu_model/gpu_arch/chip_config/gpu_arch_spec.h"
#include "gpu_model/debug/trace/sink.h"
#include "gpu_model/execution/internal/op_plan.h"
#include "gpu_model/memory/memory_system.h"
#include "gpu_model/program/executable_kernel.h"
#include "gpu_model/runtime/kernel_arg_pack.h"
#include "gpu_model/runtime/launch_config.h"
#include "gpu_model/runtime/launch_request.h"
#include "gpu_model/runtime/mapper.h"
#include "gpu_model/execution/wave_context.h"

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
