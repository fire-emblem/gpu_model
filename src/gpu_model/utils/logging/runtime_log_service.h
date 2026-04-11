#pragma once

#include <string_view>

#include <loguru.hpp>

namespace gpu_model {

class RuntimeLogService {
 public:
  void EnsureInitialized();
  bool IsInitialized() const;
  bool ShouldLog(std::string_view module, int verbosity);
  void LogMessage(int verbosity, std::string_view module, const char* fmt, ...);
  void LogMessageForced(int verbosity, std::string_view module, const char* fmt, ...);
};

RuntimeLogService& GetRuntimeLogService();

namespace logging {

inline void EnsureInitialized() {
  GetRuntimeLogService().EnsureInitialized();
}

inline bool IsInitialized() {
  return GetRuntimeLogService().IsInitialized();
}

inline bool ShouldLog(std::string_view module, int verbosity) {
  return GetRuntimeLogService().ShouldLog(module, verbosity);
}

void LogMessage(int verbosity, std::string_view module, const char* fmt, ...);
void LogMessageForced(int verbosity, std::string_view module, const char* fmt, ...);

}  // namespace logging

}  // namespace gpu_model
