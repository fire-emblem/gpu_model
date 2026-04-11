#pragma once

#include <cstdint>

namespace gpu_model {

enum class ExecutionMode {
  Functional,
  Cycle,
};

enum class FunctionalExecutionMode {
  SingleThreaded,
  MultiThreaded,
};

struct FunctionalExecutionConfig {
  FunctionalExecutionMode mode = FunctionalExecutionMode::MultiThreaded;
  uint32_t worker_threads = 0;
};

}  // namespace gpu_model
