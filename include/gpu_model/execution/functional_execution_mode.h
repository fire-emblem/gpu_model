#pragma once

#include <cstdint>

namespace gpu_model {

enum class FunctionalExecutionMode {
  SingleThreaded,
  MarlParallel,
};

struct FunctionalExecutionConfig {
  FunctionalExecutionMode mode = FunctionalExecutionMode::SingleThreaded;
  uint32_t worker_threads = 0;
};

}  // namespace gpu_model
