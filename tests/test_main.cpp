#include <gtest/gtest.h>

#include "gpu_model/utils/logging/log_macros.h"
#include "gpu_model/utils/config/runtime_config.h"

int main(int argc, char** argv) {
  // Configure test defaults:
  // - MT mode by default (faster for most tests)
  // - Trace disabled by default (tests that need trace should enable it)
  // - Logging disabled by default (tests that need logging should enable it)
  // These can be overridden via environment variables:
  //   GPU_MODEL_DISABLE_TRACE=0 to enable trace
  //   GPU_MODEL_DISABLE_LOGURU=0 to enable logging
  //   GPU_MODEL_FUNCTIONAL_MODE=st to use single-threaded mode

  if (argc > 0 && argv != nullptr && argv[0] != nullptr) {
    setenv("GPU_MODEL_LOG_PROGRAM", argv[0], 0);
  }
  gpu_model::logging::EnsureInitialized();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
