#pragma once

#include "gpu_model/runtime/hip_runtime.h"

namespace gpu_model {

class ModelRuntime : public HipRuntime {
 public:
  using HipRuntime::HipRuntime;

  HipRuntime& hooks() { return *this; }
  const HipRuntime& hooks() const { return *this; }
};

}  // namespace gpu_model
