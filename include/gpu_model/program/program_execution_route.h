#pragma once

namespace gpu_model {

enum class ProgramExecutionRoute {
  AutoSelect,
  EncodedRaw,
  LoweredModeled,
};

}  // namespace gpu_model
