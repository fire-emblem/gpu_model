#pragma once

#include <cstdint>
#include <string>

#include "gpu_model/isa/program_image.h"
#include "gpu_model/runtime/kernel_arg_pack.h"
#include "gpu_model/runtime/launch_config.h"
#include "gpu_model/runtime/mapper.h"

namespace gpu_model {

class KernelProgram;
class TraceSink;
struct DeviceLoadResult;
struct AmdgpuCodeObjectImage;

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
  const KernelProgram* kernel = nullptr;
  const ProgramImage* program_image = nullptr;
  const AmdgpuCodeObjectImage* raw_code_object = nullptr;
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
