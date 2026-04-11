#pragma once

#include <cstdint>
#include <string>

#include "gpu_model/utils/config/execution_mode.h"

namespace gpu_model {

struct RuntimeConfig {
  ExecutionMode execution_mode = ExecutionMode::Functional;
  FunctionalExecutionConfig functional{.mode = FunctionalExecutionMode::MultiThreaded};

  bool disable_trace = true;
  std::string trace_dir;

  std::string log_level;
  std::string log_file_level;
  std::string log_file;
  std::string log_modules;
  bool disable_loguru = true;

  bool encoded_exec_debug = false;

  bool full_test_matrix = false;
  bool phase1_compat_gate = false;
};

class RuntimeConfigManager {
 public:
  static RuntimeConfigManager& Instance();

  const RuntimeConfig& config() const { return config_; }

  void SetConfig(const RuntimeConfig& config);
  void SetExecutionMode(ExecutionMode mode);
  void SetFunctionalMode(FunctionalExecutionMode mode);
  void SetWorkerThreads(uint32_t workers);
  void SetDisableTrace(bool disable);
  void SetTraceDir(const std::string& dir);
  void SetDisableLoguru(bool disable);
  void ReloadFromEnv();
  void PrintConfig() const;

 private:
  RuntimeConfigManager();
  RuntimeConfig config_{};
  bool printed_init_ = false;
};

inline const RuntimeConfig& GetRuntimeConfig() {
  return RuntimeConfigManager::Instance().config();
}

inline void SetRuntimeConfig(const RuntimeConfig& config) {
  RuntimeConfigManager::Instance().SetConfig(config);
}

uint32_t DefaultMtWorkerThreadCountForEnv();

}  // namespace gpu_model
