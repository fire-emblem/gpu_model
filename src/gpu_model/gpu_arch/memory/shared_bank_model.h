#pragma once

#include <cstdint>

#include "gpu_model/gpu_arch/chip_config/gpu_arch_spec.h"
#include "gpu_model/gpu_arch/memory/memory_request.h"

namespace gpu_model {

class SharedBankModel {
 public:
  explicit SharedBankModel(SharedBankModelSpec spec = {}) : spec_(spec) {}

  uint64_t ConflictPenalty(const MemoryRequest& request) const;

 private:
  SharedBankModelSpec spec_;
};

}  // namespace gpu_model
