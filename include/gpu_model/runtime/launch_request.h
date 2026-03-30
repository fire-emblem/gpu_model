#pragma once

#include <cstdint>
#include <string>

#include "gpu_model/program/execution_route.h"
#include "gpu_model/runtime/kernel_arg_pack.h"
#include "gpu_model/runtime/launch_config.h"
#include "gpu_model/runtime/mapper.h"

namespace gpu_model {

class ExecutableKernel;
class ProgramObject;
class TraceSink;
struct EncodedProgramObject;
struct DeviceLoadResult;

enum class ExecutionMode {
  Functional,
  Cycle,
};

struct ExecutionStats {
  uint64_t wave_steps = 0;
  uint64_t instructions_issued = 0;
  uint64_t memory_ops = 0;
  uint64_t global_loads = 0;
  uint64_t global_stores = 0;
  uint64_t shared_loads = 0;
  uint64_t shared_stores = 0;
  uint64_t private_loads = 0;
  uint64_t private_stores = 0;
  uint64_t constant_loads = 0;
  uint64_t barriers = 0;
  uint64_t wave_exits = 0;
  uint64_t l1_hits = 0;
  uint64_t l2_hits = 0;
  uint64_t cache_misses = 0;
  uint64_t shared_bank_conflict_penalty_cycles = 0;
};

struct LaunchRequest {
  std::string arch_name = "c500";
  const ExecutableKernel* kernel = nullptr;
  const ProgramObject* program_image = nullptr;
  const EncodedProgramObject* raw_code_object = nullptr;
  ExecutionRoute program_execution_route = ExecutionRoute::AutoSelect;
  const DeviceLoadResult* device_load = nullptr;
  LaunchConfig config;
  KernelArgPack args;
  ExecutionMode mode = ExecutionMode::Functional;
  TraceSink* trace = nullptr;
};

struct LaunchResult {
  bool ok = false;
  std::string error_message;
  uint64_t submit_cycle = 0;
  uint64_t begin_cycle = 0;
  uint64_t end_cycle = 0;
  uint64_t total_cycles = 0;
  PlacementMap placement;
  ExecutionStats stats;
};

}  // namespace gpu_model
