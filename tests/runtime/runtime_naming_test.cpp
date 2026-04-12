#include <gtest/gtest.h>

#include <type_traits>

#include "program/executable/executable_kernel.h"
#include "program/program_object/program_object.h"
#include "runtime/exec_engine.h"
#include "runtime/hip_runtime.h"
#include "runtime/model_runtime.h"

namespace gpu_model {
namespace {

TEST(RuntimeNamingTest, NewRuntimeTypesAreConcreteAndUsable) {
  static_assert(std::is_class_v<ModelRuntime>);
  static_assert(std::is_class_v<HipRuntime>);
  static_assert(std::is_class_v<ExecEngine>);
  static_assert(std::is_constructible_v<HipRuntime, ExecEngine*>);
  static_assert(std::is_constructible_v<ModelRuntime, ExecEngine*>);
  static_assert(std::is_default_constructible_v<ProgramObject>);

  ExecEngine engine;
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
  ExecEngine shared_runtime;
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
