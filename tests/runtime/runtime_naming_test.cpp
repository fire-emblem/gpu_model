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

TEST(RuntimeNamingTest, NewRuntimeNamesAliasLegacyTypes) {
  static_assert(std::is_same_v<ModelRuntime, ModelRuntimeApi>);
  static_assert(std::is_same_v<HipRuntime, RuntimeHooks>);
  static_assert(std::is_same_v<RuntimeEngine, HostRuntime>);
  static_assert(std::is_same_v<ProgramObject, ProgramImage>);
  static_assert(std::is_same_v<ExecutableKernel, KernelProgram>);
  static_assert(std::is_base_of_v<EncodedProgramObject, AmdgpuCodeObjectImage>);
  static_assert(std::is_same_v<EncodedExecEngine, RawGcnExecutor>);

  RuntimeEngine engine;
  HipRuntime hip(&engine);
  ModelRuntime model(&engine);
  EXPECT_EQ(hip.GetDeviceCount(), 1);
  EXPECT_EQ(model.GetDeviceCount(), 1);
}

}  // namespace
}  // namespace gpu_model
