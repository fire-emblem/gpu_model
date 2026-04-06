#pragma once

#include <cstdint>

namespace gpu_model {

struct RuntimeSubmissionContext {
  int device_id = 0;
  uint64_t context_id = 0;
  uint64_t stream_id = 0;
};

}  // namespace gpu_model
