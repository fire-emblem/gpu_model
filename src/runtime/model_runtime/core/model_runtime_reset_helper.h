#pragma once

#include <optional>

#include "program/loader/device_image_loader.h"

namespace gpu_model {

class ExecEngine;
class ModelRuntimeDeviceState;
class RuntimeModuleRegistry;

struct ModelRuntimeResetContext {
  ExecEngine& owned_runtime;
  ExecEngine*& runtime_engine;
  bool owns_runtime = false;
  ModelRuntimeDeviceState& device_state;
  RuntimeModuleRegistry& module_registry;
  std::optional<DeviceLoadResult>& last_load_result;
};

void ResetModelRuntimeState(ModelRuntimeResetContext context);

}  // namespace gpu_model
