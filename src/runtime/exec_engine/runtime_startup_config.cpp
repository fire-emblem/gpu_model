#include "runtime/exec_engine/runtime_startup_config.h"

namespace gpu_model {

ExecEngineStartupConfig ResolveExecEngineStartupConfig(const RuntimeConfig& config) {
  ExecEngineStartupConfig startup;
  startup.functional = config.functional;
  startup.disable_trace = config.disable_trace;
  startup.should_log_functional_config =
      config.functional.mode != FunctionalExecutionMode::SingleThreaded ||
      config.functional.worker_threads > 0;
  return startup;
}

const char* ToExecEngineFunctionalModeName(FunctionalExecutionMode mode) {
  switch (mode) {
    case FunctionalExecutionMode::SingleThreaded:
      return "st";
    case FunctionalExecutionMode::MultiThreaded:
      return "mt";
  }
  return "unknown";
}

}  // namespace gpu_model
