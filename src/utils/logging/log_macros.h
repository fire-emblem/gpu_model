#pragma once

#include "utils/logging/runtime_log_service.h"

#define GPU_MODEL_LOG_INFO(module, fmt, ...) \
  ::gpu_model::logging::LogMessage(loguru::Verbosity_INFO, module, fmt, ##__VA_ARGS__)

#define GPU_MODEL_LOG_DEBUG(module, fmt, ...) \
  ::gpu_model::logging::LogMessage(1, module, fmt, ##__VA_ARGS__)

#define GPU_MODEL_LOG_WARNING(module, fmt, ...) \
  ::gpu_model::logging::LogMessage(loguru::Verbosity_WARNING, module, fmt, ##__VA_ARGS__)

#define GPU_MODEL_LOG_ERROR(module, fmt, ...) \
  ::gpu_model::logging::LogMessage(loguru::Verbosity_ERROR, module, fmt, ##__VA_ARGS__)

#define GPU_MODEL_LOG_INFO_FORCED(module, fmt, ...) \
  ::gpu_model::logging::LogMessageForced(loguru::Verbosity_INFO, module, fmt, ##__VA_ARGS__)
