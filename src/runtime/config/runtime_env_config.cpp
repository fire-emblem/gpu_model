#include "gpu_model/runtime/runtime_env_config.h"

#include <cstdlib>
#include <thread>
#include <string_view>

namespace gpu_model {

uint32_t DefaultMtWorkerThreadCountForEnv() {
  const uint32_t cpu_count = std::max(1u, std::thread::hardware_concurrency());
  return std::max(1u, (cpu_count * 9u) / 10u);
}

RuntimeEnvConfig LoadRuntimeEnvConfig() {
  RuntimeEnvConfig config;
  if (const char* disable_trace_env = std::getenv("GPU_MODEL_DISABLE_TRACE");
      disable_trace_env != nullptr && disable_trace_env[0] != '\0' &&
      std::string_view(disable_trace_env) != "0") {
    config.disable_trace = true;
  }
  const char* mode_env = std::getenv("GPU_MODEL_FUNCTIONAL_MODE");
  if (mode_env == nullptr) {
    return config;
  }

  config.has_functional_mode = true;
  const std::string_view mode(mode_env);
  if (mode == "mt" || mode == "parallel" || mode == "multi_threaded") {
    config.functional.mode = FunctionalExecutionMode::MultiThreaded;
  } else {
    config.functional.mode = FunctionalExecutionMode::SingleThreaded;
  }

  if (const char* workers_env = std::getenv("GPU_MODEL_FUNCTIONAL_WORKERS");
      workers_env != nullptr && workers_env[0] != '\0') {
    config.functional.worker_threads = static_cast<uint32_t>(std::strtoul(workers_env, nullptr, 10));
  } else if (config.functional.mode == FunctionalExecutionMode::MultiThreaded) {
    config.functional.worker_threads = DefaultMtWorkerThreadCountForEnv();
  }

  return config;
}

}  // namespace gpu_model
