#pragma once

#include "utils/config/execution_mode.h"

namespace gpu_model {

struct RuntimeEnvConfig {
  bool has_functional_mode = false;
  FunctionalExecutionConfig functional{};
  bool disable_trace = false;
};

RuntimeEnvConfig LoadRuntimeEnvConfig();
uint32_t DefaultMtWorkerThreadCountForEnv();

}  // namespace gpu_model
