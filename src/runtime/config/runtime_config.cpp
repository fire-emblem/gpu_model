#include "utils/config/runtime_config.h"

#include <cstdlib>
#include <cstdio>
#include <string_view>
#include <thread>
#include <algorithm>
#include <cctype>

namespace gpu_model {

namespace {

std::string ToLower(std::string_view sv) {
  std::string result(sv);
  std::transform(result.begin(), result.end(), result.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return result;
}

const char* ExecutionModeName(ExecutionMode mode) {
  switch (mode) {
    case ExecutionMode::Functional: return "functional";
    case ExecutionMode::Cycle: return "cycle";
  }
  return "unknown";
}

const char* FunctionalModeName(FunctionalExecutionMode mode) {
  switch (mode) {
    case FunctionalExecutionMode::SingleThreaded: return "st";
    case FunctionalExecutionMode::MultiThreaded: return "mt";
  }
  return "unknown";
}

}  // namespace

uint32_t DefaultMtWorkerThreadCountForEnv() {
  const uint32_t cpu_count = std::max(1u, std::thread::hardware_concurrency());
  return std::max(1u, (cpu_count * 9u) / 10u);
}

RuntimeConfigManager& RuntimeConfigManager::Instance() {
  static RuntimeConfigManager instance;
  return instance;
}

RuntimeConfigManager::RuntimeConfigManager() {
  ReloadFromEnv();
}

void RuntimeConfigManager::SetConfig(const RuntimeConfig& config) {
  // Log changes
  if (config_.execution_mode != config.execution_mode) {
    fprintf(stderr, "[RuntimeConfig] execution_mode: %s -> %s\n",
            ExecutionModeName(config_.execution_mode),
            ExecutionModeName(config.execution_mode));
  }
  if (config_.functional.mode != config.functional.mode) {
    fprintf(stderr, "[RuntimeConfig] functional_mode: %s -> %s\n",
            FunctionalModeName(config_.functional.mode),
            FunctionalModeName(config.functional.mode));
  }
  if (config_.disable_trace != config.disable_trace) {
    fprintf(stderr, "[RuntimeConfig] disable_trace: %s -> %s\n",
            config_.disable_trace ? "true" : "false",
            config.disable_trace ? "true" : "false");
  }
  if (config_.disable_loguru != config.disable_loguru) {
    fprintf(stderr, "[RuntimeConfig] disable_loguru: %s -> %s\n",
            config_.disable_loguru ? "true" : "false",
            config.disable_loguru ? "true" : "false");
  }
  config_ = config;
}

void RuntimeConfigManager::SetExecutionMode(ExecutionMode mode) {
  if (config_.execution_mode != mode) {
    fprintf(stderr, "[RuntimeConfig] execution_mode: %s -> %s\n",
            ExecutionModeName(config_.execution_mode),
            ExecutionModeName(mode));
    config_.execution_mode = mode;
  }
}

void RuntimeConfigManager::SetFunctionalMode(FunctionalExecutionMode mode) {
  if (config_.functional.mode != mode) {
    fprintf(stderr, "[RuntimeConfig] functional_mode: %s -> %s\n",
            FunctionalModeName(config_.functional.mode),
            FunctionalModeName(mode));
    config_.functional.mode = mode;
  }
}

void RuntimeConfigManager::SetWorkerThreads(uint32_t workers) {
  if (config_.functional.worker_threads != workers) {
    fprintf(stderr, "[RuntimeConfig] worker_threads: %u -> %u\n",
            config_.functional.worker_threads, workers);
    config_.functional.worker_threads = workers;
  }
}

void RuntimeConfigManager::SetDisableTrace(bool disable) {
  if (config_.disable_trace != disable) {
    fprintf(stderr, "[RuntimeConfig] disable_trace: %s -> %s\n",
            config_.disable_trace ? "true" : "false",
            disable ? "true" : "false");
    config_.disable_trace = disable;
  }
}

void RuntimeConfigManager::SetTraceDir(const std::string& dir) {
  if (config_.trace_dir != dir) {
    fprintf(stderr, "[RuntimeConfig] trace_dir: '%s' -> '%s'\n",
            config_.trace_dir.c_str(), dir.c_str());
    config_.trace_dir = dir;
  }
}

void RuntimeConfigManager::SetDisableLoguru(bool disable) {
  if (config_.disable_loguru != disable) {
    fprintf(stderr, "[RuntimeConfig] disable_loguru: %s -> %s\n",
            config_.disable_loguru ? "true" : "false",
            disable ? "true" : "false");
    config_.disable_loguru = disable;
  }
}

void RuntimeConfigManager::ReloadFromEnv() {
  // Execution mode - only override if explicitly set
  const char* exec_mode_env = std::getenv("GPU_MODEL_EXECUTION_MODE");
  if (exec_mode_env != nullptr && exec_mode_env[0] != '\0') {
    const std::string exec_mode = ToLower(exec_mode_env);
    if (exec_mode == "cycle") {
      config_.execution_mode = ExecutionMode::Cycle;
    } else {
      config_.execution_mode = ExecutionMode::Functional;
    }
  }

  // Functional mode - only override if explicitly set
  const char* func_mode_env = std::getenv("GPU_MODEL_FUNCTIONAL_MODE");
  if (func_mode_env != nullptr && func_mode_env[0] != '\0') {
    const std::string func_mode = ToLower(func_mode_env);
    if (func_mode == "mt" || func_mode == "multi_threaded" || func_mode == "parallel") {
      config_.functional.mode = FunctionalExecutionMode::MultiThreaded;
    } else if (func_mode == "st" || func_mode == "single_threaded") {
      config_.functional.mode = FunctionalExecutionMode::SingleThreaded;
    }
  }

  // Worker threads - only override if explicitly set
  const char* workers_env = std::getenv("GPU_MODEL_FUNCTIONAL_WORKERS");
  if (workers_env != nullptr && workers_env[0] != '\0') {
    config_.functional.worker_threads = static_cast<uint32_t>(std::strtoul(workers_env, nullptr, 10));
  } else if (config_.functional.mode == FunctionalExecutionMode::MultiThreaded) {
    config_.functional.worker_threads = DefaultMtWorkerThreadCountForEnv();
  }

  // Trace configuration - env var explicitly enables trace (default is disabled)
  // GPU_MODEL_DISABLE_TRACE=0 enables trace, GPU_MODEL_DISABLE_TRACE=1 keeps it disabled
  const char* disable_trace_env = std::getenv("GPU_MODEL_DISABLE_TRACE");
  if (disable_trace_env != nullptr && disable_trace_env[0] != '\0') {
    std::string_view sv(disable_trace_env);
    // "0" explicitly enables trace
    config_.disable_trace = !(sv == "0");
  }
  // Default is already disable_trace = true

  const char* trace_dir_env = std::getenv("GPU_MODEL_TRACE_DIR");
  if (trace_dir_env != nullptr && trace_dir_env[0] != '\0') {
    config_.trace_dir = trace_dir_env;
  }

  // Logging configuration - env var explicitly enables logging (default is disabled)
  const char* disable_loguru_env = std::getenv("GPU_MODEL_DISABLE_LOGURU");
  if (disable_loguru_env != nullptr && disable_loguru_env[0] != '\0') {
    std::string_view sv(disable_loguru_env);
    // "0" explicitly enables loguru
    config_.disable_loguru = !(sv == "0");
  }
  // Default is already disable_loguru = true

  const char* log_level_env = std::getenv("GPU_MODEL_LOG_LEVEL");
  if (log_level_env != nullptr && log_level_env[0] != '\0') {
    config_.log_level = log_level_env;
  }

  const char* log_file_level_env = std::getenv("GPU_MODEL_LOG_FILE_LEVEL");
  if (log_file_level_env != nullptr && log_file_level_env[0] != '\0') {
    config_.log_file_level = log_file_level_env;
  }

  const char* log_file_env = std::getenv("GPU_MODEL_LOG_FILE");
  if (log_file_env != nullptr && log_file_env[0] != '\0') {
    config_.log_file = log_file_env;
  }

  const char* log_modules_env = std::getenv("GPU_MODEL_LOG_MODULES");
  if (log_modules_env != nullptr && log_modules_env[0] != '\0') {
    config_.log_modules = log_modules_env;
  }

  // Debug configuration
  const char* encoded_exec_debug_env = std::getenv("GPU_MODEL_ENCODED_EXEC_DEBUG");
  if (encoded_exec_debug_env != nullptr && encoded_exec_debug_env[0] != '\0') {
    std::string_view sv(encoded_exec_debug_env);
    config_.encoded_exec_debug = (sv == "1" || sv == "true" || sv == "TRUE");
  }

  // Test configuration
  const char* test_profile_env = std::getenv("GPU_MODEL_TEST_PROFILE");
  if (test_profile_env != nullptr && test_profile_env[0] != '\0') {
    const std::string test_profile = ToLower(test_profile_env);
    config_.full_test_matrix = (test_profile == "full" || test_profile == "all" || test_profile == "1" || test_profile == "true");
    config_.phase1_compat_gate = (test_profile == "phase1-compat" || test_profile == "compat");
  }

  // Print config on first load
  if (!printed_init_) {
    PrintConfig();
    printed_init_ = true;
  }
}

void RuntimeConfigManager::PrintConfig() const {
  fprintf(stderr, "[RuntimeConfig] === Configuration ===\n");
  fprintf(stderr, "[RuntimeConfig] execution_mode: %s\n", ExecutionModeName(config_.execution_mode));
  fprintf(stderr, "[RuntimeConfig] functional_mode: %s\n", FunctionalModeName(config_.functional.mode));
  fprintf(stderr, "[RuntimeConfig] worker_threads: %u\n", config_.functional.worker_threads);
  fprintf(stderr, "[RuntimeConfig] disable_trace: %s\n", config_.disable_trace ? "true" : "false");
  if (!config_.trace_dir.empty()) {
    fprintf(stderr, "[RuntimeConfig] trace_dir: %s\n", config_.trace_dir.c_str());
  }
  fprintf(stderr, "[RuntimeConfig] disable_loguru: %s\n", config_.disable_loguru ? "true" : "false");
  if (!config_.log_level.empty()) {
    fprintf(stderr, "[RuntimeConfig] log_level: %s\n", config_.log_level.c_str());
  }
  if (!config_.log_file.empty()) {
    fprintf(stderr, "[RuntimeConfig] log_file: %s\n", config_.log_file.c_str());
  }
  if (!config_.log_modules.empty()) {
    fprintf(stderr, "[RuntimeConfig] log_modules: %s\n", config_.log_modules.c_str());
  }
  fprintf(stderr, "[RuntimeConfig] encoded_exec_debug: %s\n", config_.encoded_exec_debug ? "true" : "false");
  fprintf(stderr, "[RuntimeConfig] full_test_matrix: %s\n", config_.full_test_matrix ? "true" : "false");
  fprintf(stderr, "[RuntimeConfig] =====================\n");
}

}  // namespace gpu_model
