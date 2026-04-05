#pragma once

#include <string_view>

#include <loguru.hpp>

namespace gpu_model::logging {

void EnsureInitialized();
bool IsInitialized();
bool ShouldLog(std::string_view module, int verbosity);
void LogMessage(int verbosity, std::string_view module, const char* fmt, ...);

}  // namespace gpu_model::logging

#define GPU_MODEL_LOG_INFO(module, fmt, ...) \
  ::gpu_model::logging::LogMessage(loguru::Verbosity_INFO, module, fmt, ##__VA_ARGS__)

#define GPU_MODEL_LOG_DEBUG(module, fmt, ...) \
  ::gpu_model::logging::LogMessage(1, module, fmt, ##__VA_ARGS__)

#define GPU_MODEL_LOG_WARNING(module, fmt, ...) \
  ::gpu_model::logging::LogMessage(loguru::Verbosity_WARNING, module, fmt, ##__VA_ARGS__)

#define GPU_MODEL_LOG_ERROR(module, fmt, ...) \
  ::gpu_model::logging::LogMessage(loguru::Verbosity_ERROR, module, fmt, ##__VA_ARGS__)
