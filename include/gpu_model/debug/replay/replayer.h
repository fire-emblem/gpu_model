#pragma once

namespace gpu_model {

// Placeholder for future execution-state replay / restore support.
class Replayer {
 public:
  virtual ~Replayer() = default;
};

}  // namespace gpu_model
