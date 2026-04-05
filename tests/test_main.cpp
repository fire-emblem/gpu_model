#include <gtest/gtest.h>

#include "gpu_model/util/logging.h"

int main(int argc, char** argv) {
  if (argc > 0 && argv != nullptr && argv[0] != nullptr) {
    setenv("GPU_MODEL_LOG_PROGRAM", argv[0], 0);
  }
  gpu_model::logging::EnsureInitialized();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
