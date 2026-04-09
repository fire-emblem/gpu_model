#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include "gpu_model/execution/functional_execution_mode.h"
#include "gpu_model/runtime/exec_engine.h"

namespace gpu_model {

// Unified runtime configuration.
// All configuration is centralized here for consistent management.
// Configuration can be loaded from environment variables or set programmatically.
//
// Environment Variables:
//   GPU_MODEL_EXECUTION_MODE     - "functional" (default) or "cycle"
//   GPU_MODEL_FUNCTIONAL_MODE    - "mt" (default) or "st"
//   GPU_MODEL_FUNCTIONAL_WORKERS - Number of worker threads (default: 90% of CPU cores)
//   GPU_MODEL_DISABLE_TRACE      - "0" to enable trace, "1" to disable (default: disabled)
//   GPU_MODEL_TRACE_DIR          - Directory for trace output files
//   GPU_MODEL_DISABLE_LOGURU     - "0" to enable logging, "1" to disable (default: disabled)
//   GPU_MODEL_LOG_LEVEL          - "error", "warning", "info", "debug", "trace"
//   GPU_MODEL_LOG_FILE_LEVEL     - Same levels as LOG_LEVEL
//   GPU_MODEL_LOG_FILE           - Path to log file
//   GPU_MODEL_LOG_MODULES        - Comma-separated list of modules to enable
//   GPU_MODEL_ENCODED_EXEC_DEBUG - "1" to enable encoded exec debug
//   GPU_MODEL_TEST_PROFILE       - "full" for full test matrix
//
// Usage:
//   const auto& config = GetRuntimeConfig();  // Get current config
//   RuntimeConfigManager::Instance().SetDisableTrace(false);  // Override for testing

struct RuntimeConfig {
  // Execution mode configuration
  ExecutionMode execution_mode = ExecutionMode::Functional;
  FunctionalExecutionConfig functional{.mode = FunctionalExecutionMode::MultiThreaded};

  // Trace configuration (disabled by default for faster testing)
  bool disable_trace = true;
  std::string trace_dir;

  // Logging configuration (disabled by default for faster testing)
  std::string log_level;      // "error", "warning", "info", "debug", "trace"
  std::string log_file_level; // Same levels as log_level
  std::string log_file;       // Path to log file
  std::string log_modules;    // Comma-separated list of modules to enable
  bool disable_loguru = true;

  // Debug configuration
  bool encoded_exec_debug = false;

  // Test configuration
  bool full_test_matrix = false;
  bool phase1_compat_gate = false;
};

// Global runtime configuration singleton.
// Loads from environment variables on first access.
// Can be overridden programmatically for testing.
class RuntimeConfigManager {
 public:
  // Get the singleton instance
  static RuntimeConfigManager& Instance();

  // Get the current configuration
  const RuntimeConfig& config() const { return config_; }

  // Override configuration programmatically (for testing)
  void SetConfig(const RuntimeConfig& config);

  // Override specific fields (logs the change)
  void SetExecutionMode(ExecutionMode mode);
  void SetFunctionalMode(FunctionalExecutionMode mode);
  void SetWorkerThreads(uint32_t workers);
  void SetDisableTrace(bool disable);
  void SetTraceDir(const std::string& dir);
  void SetDisableLoguru(bool disable);

  // Reload configuration from environment variables
  void ReloadFromEnv();

  // Print current configuration to stderr
  void PrintConfig() const;

 private:
  RuntimeConfigManager();
  RuntimeConfig config_{};
  bool printed_init_ = false;
};

// Convenience function to get the current config
inline const RuntimeConfig& GetRuntimeConfig() {
  return RuntimeConfigManager::Instance().config();
}

// Convenience function to override config (for testing)
inline void SetRuntimeConfig(const RuntimeConfig& config) {
  RuntimeConfigManager::Instance().SetConfig(config);
}

// Get default worker thread count for MT mode
uint32_t DefaultMtWorkerThreadCountForEnv();

}  // namespace gpu_model
