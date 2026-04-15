#include "runtime/model_runtime/core/model_runtime_reset_helper.h"

#include "runtime/exec_engine/exec_engine.h"
#include "runtime/model_runtime/core/model_runtime_device_state.h"
#include "runtime/model_runtime/module/module_registry.h"

namespace gpu_model {

void ResetModelRuntimeState(ModelRuntimeResetContext context) {
  if (context.owns_runtime) {
    context.owned_runtime = ExecEngine{};
    context.runtime_engine = &context.owned_runtime;
  } else {
    context.runtime_engine->ResetDeviceCycle();
  }
  context.device_state.Reset();
  context.module_registry.Reset();
  context.last_load_result.reset();
}

}  // namespace gpu_model
