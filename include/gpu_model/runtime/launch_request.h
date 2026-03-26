#pragma once

#include <cstdint>
#include <string>

#include "gpu_model/runtime/kernel_arg_pack.h"
#include "gpu_model/runtime/launch_config.h"
#include "gpu_model/runtime/mapper.h"

namespace gpu_model {

class KernelProgram;
class TraceSink;

enum class ExecutionMode {
  Functional,
  Cycle,
};

struct LaunchRequest {
  std::string arch_name = "c500";
  const KernelProgram* kernel = nullptr;
  LaunchConfig config;
  KernelArgPack args;
  ExecutionMode mode = ExecutionMode::Functional;
  TraceSink* trace = nullptr;
};

struct LaunchResult {
  bool ok = false;
  std::string error_message;
  uint64_t total_cycles = 0;
  PlacementMap placement;
};

}  // namespace gpu_model
