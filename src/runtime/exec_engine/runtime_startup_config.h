#pragma once

#include "utils/config/runtime_config.h"

namespace gpu_model {

struct ExecEngineStartupConfig {
  FunctionalExecutionConfig functional{};
  bool disable_trace = false;
  bool should_log_functional_config = false;
};

ExecEngineStartupConfig ResolveExecEngineStartupConfig(const RuntimeConfig& config);
const char* ToExecEngineFunctionalModeName(FunctionalExecutionMode mode);

}  // namespace gpu_model
