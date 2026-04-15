#pragma once

#include <atomic>
#include <cstdint>

#include "debug/trace/sink.h"
#include "gpu_arch/device/gpu_arch_spec.h"
#include "state/memory/memory_system.h"
#include "program/executable/executable_kernel.h"
#include "runtime/config/kernel_arg_pack.h"
#include "runtime/config/launch_config.h"
#include "runtime/config/launch_request.h"
#include "runtime/model_runtime/mapper.h"
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

}  // namespace gpu_model
