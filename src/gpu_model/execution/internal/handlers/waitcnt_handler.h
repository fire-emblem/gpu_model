#pragma once

#include "gpu_model/execution/internal/handlers/sync_handler_base.h"

namespace gpu_model {

class WaitcntHandler final : public SyncHandlerBase {
 public:
  std::string_view name() const final { return "waitcnt"; }
};

}  // namespace gpu_model
