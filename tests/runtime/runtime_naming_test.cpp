#include <gtest/gtest.h>

#include <type_traits>

#include "gpu_model/runtime/hip_runtime.h"
#include "gpu_model/runtime/model_runtime.h"
#include "gpu_model/runtime/runtime_engine.h"

namespace gpu_model {
namespace {

TEST(RuntimeNamingTest, NewRuntimeNamesAliasLegacyTypes) {
  static_assert(std::is_same_v<ModelRuntime, ModelRuntimeApi>);
  static_assert(std::is_same_v<HipRuntime, RuntimeHooks>);
  static_assert(std::is_same_v<RuntimeEngine, HostRuntime>);

  RuntimeEngine engine;
  HipRuntime hip(&engine);
  ModelRuntime model(&engine);
  EXPECT_EQ(hip.GetDeviceCount(), 1);
  EXPECT_EQ(model.GetDeviceCount(), 1);
}

}  // namespace
}  // namespace gpu_model
