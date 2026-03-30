#include <gtest/gtest.h>

#include <type_traits>

#include "gpu_model/execution/encoded_exec_engine.h"
#include "gpu_model/exec/encoded/executor/raw_gcn_executor.h"
#include "gpu_model/isa/kernel_program.h"
#include "gpu_model/isa/program_image.h"
#include "gpu_model/program/executable_kernel.h"
#include "gpu_model/program/program_object.h"
#include "gpu_model/runtime/hip_runtime.h"
#include "gpu_model/runtime/model_runtime.h"
#include "gpu_model/runtime/model_runtime_api.h"
#include "gpu_model/runtime/runtime_engine.h"
#include "gpu_model/runtime/runtime_hooks.h"

namespace gpu_model {
namespace {

TEST(CompatibilityAliasTest, OldAndNewNamesRemainEquivalentInPhase1) {
  static_assert(std::is_same_v<ModelRuntime, ModelRuntimeApi>);
  static_assert(std::is_same_v<ProgramObject, ProgramImage>);
  static_assert(std::is_same_v<EncodedExecEngine, RawGcnExecutor>);
  static_assert(std::is_same_v<RuntimeEngine, HostRuntime>);
  static_assert(std::is_same_v<HipRuntime, RuntimeHooks>);
  static_assert(std::is_same_v<ExecutableKernel, KernelProgram>);
}

}  // namespace
}  // namespace gpu_model
