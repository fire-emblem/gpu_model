#include <gtest/gtest.h>

#include <type_traits>

#include "gpu_model/program/encoded_program_object.h"
#include "gpu_model/program/executable_kernel.h"
#include "gpu_model/program/execution_route.h"
#include "gpu_model/program/object_reader.h"
#include "gpu_model/program/program_object.h"

namespace gpu_model {
namespace {

TEST(ProgramNamingTest, NewProgramTypeNamesAliasLegacyTypes) {
  static_assert(std::is_same_v<ProgramObject, ProgramImage>);
  static_assert(std::is_same_v<ExecutableKernel, KernelProgram>);
  static_assert(std::is_same_v<ObjectReader, ProgramFileLoader>);
}

TEST(ProgramNamingTest, ExecutionRouteRemainsLightweightEnumAlias) {
  static_assert(std::is_same_v<ExecutionRoute, ProgramExecutionRoute>);
  static_assert(std::is_enum_v<ExecutionRoute>);
  static_assert(!std::is_class_v<ExecutionRoute>);
  static_assert(static_cast<int>(ExecutionRoute::AutoSelect) ==
                static_cast<int>(ProgramExecutionRoute::AutoSelect));
}

TEST(ProgramNamingTest, EncodedProgramObjectHeaderProvidesDataObject) {
  static_assert(std::is_default_constructible_v<EncodedProgramObject>);
  static_assert(std::is_same_v<decltype(EncodedProgramObject{}.kernel_descriptor),
                               AmdgpuKernelDescriptor>);
}

}  // namespace
}  // namespace gpu_model
