#include <gtest/gtest.h>

#include <type_traits>

#include "gpu_model/execution/encoded_exec_engine.h"
#include "gpu_model/program/encoded_program_object.h"
#include "gpu_model/program/executable_kernel.h"
#include "gpu_model/program/program_object.h"
#include "gpu_model/runtime/hip_runtime.h"
#include "gpu_model/runtime/model_runtime.h"
#include "gpu_model/runtime/runtime_engine.h"

namespace gpu_model {
namespace {

TEST(RuntimeNamingTest, NewRuntimeTypesAreConcreteAndUsable) {
  static_assert(std::is_class_v<ModelRuntime>);
  static_assert(std::is_class_v<HipRuntime>);
  static_assert(std::is_class_v<RuntimeEngine>);
  static_assert(std::is_constructible_v<HipRuntime, RuntimeEngine*>);
  static_assert(std::is_constructible_v<ModelRuntime, RuntimeEngine*>);
  static_assert(std::is_default_constructible_v<EncodedProgramObject>);
  static_assert(std::is_same_v<EncodedExecEngine, RawGcnExecutor>);

  RuntimeEngine engine;
  HipRuntime hip(&engine);
  ModelRuntime model(&engine);
  EXPECT_EQ(hip.GetDeviceCount(), 1);
  EXPECT_EQ(model.GetDeviceCount(), 1);
}

TEST(RuntimeNamingTest, ResetReinitializesOwnedRuntimeState) {
  HipRuntime hip;
  const uint64_t hip_first = hip.Malloc(16);
  (void)hip.Malloc(16);
  hip.Reset();
  EXPECT_EQ(hip.Malloc(16), hip_first);

  ModelRuntime model;
  const uint64_t model_first = model.Malloc(16);
  (void)model.Malloc(16);
  model.Reset();
  EXPECT_EQ(model.Malloc(16), model_first);
}

TEST(RuntimeNamingTest, ResetKeepsInjectedRuntimeBinding) {
  RuntimeEngine shared_runtime;
  HipRuntime hip(&shared_runtime);
  ModelRuntime model(&shared_runtime);

  const uint64_t hip_first = hip.Malloc(16);
  hip.Reset();
  const uint64_t hip_second = hip.Malloc(16);
  EXPECT_GT(hip_second, hip_first);

  const uint64_t model_first = model.Malloc(16);
  model.Reset();
  const uint64_t model_second = model.Malloc(16);
  EXPECT_GT(model_second, model_first);
}

}  // namespace
}  // namespace gpu_model
