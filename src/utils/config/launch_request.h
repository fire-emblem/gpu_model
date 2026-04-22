#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include "utils/config/kernel_arg_pack.h"
#include "utils/config/launch_config.h"
#include "utils/config/mapper.h"
#include "execution/stats/program_cycle_stats.h"
#include "runtime/model_runtime/compat/launch/runtime_submission_context.h"
#include "state/execution_stats.h"
#include "utils/config/execution_mode.h"

namespace gpu_model {

class ExecutableKernel;
class ProgramObject;
class TraceSink;
struct DeviceLoadResult;

struct LaunchRequest {
  std::string arch_name = "mac500";
  const ExecutableKernel* kernel = nullptr;
  const ProgramObject* program_object = nullptr;
  const DeviceLoadResult* device_load = nullptr;
  RuntimeSubmissionContext submission_context;
  LaunchConfig config;
  KernelArgPack args;
  ExecutionMode mode = ExecutionMode::Functional;
  TraceSink* trace = nullptr;
  // Launch context for trace output
  uint64_t launch_index = 0;
  std::string functional_mode;  // "st", "mt", or empty for cycle mode
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
  std::optional<ProgramCycleStats> program_cycle_stats;
};

}  // namespace gpu_model
