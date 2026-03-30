#include <gtest/gtest.h>

#include <filesystem>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "gpu_model/program/encoded_program_object.h"
#include "gpu_model/program/executable_kernel.h"
#include "gpu_model/program/execution_route.h"
#include "gpu_model/program/object_reader.h"
#include "gpu_model/program/program_object.h"

namespace gpu_model {
namespace {

TEST(ProgramNamingTest, NewProgramTypesExposeConcreteInterfaces) {
  static_assert(std::is_class_v<ProgramObject>);
  static_assert(std::is_class_v<ExecutableKernel>);
  static_assert(std::is_class_v<ObjectReader>);
  static_assert(std::is_default_constructible_v<ProgramObject>);
  static_assert(std::is_default_constructible_v<ExecutableKernel>);
  static_assert(std::is_same_v<decltype(ProgramObject{}.kernel_name()), const std::string&>);
  static_assert(std::is_same_v<decltype(ExecutableKernel{}.name()), const std::string&>);
  static_assert(std::is_same_v<
                decltype(std::declval<const ObjectReader&>().LoadFromStem(
                    std::declval<const std::filesystem::path&>())),
                ProgramObject>);
  static_assert(std::is_same_v<
                decltype(std::declval<const ObjectReader&>().LoadFromObject(
                    std::declval<const std::filesystem::path&>(),
                    std::declval<std::optional<std::string>>())),
                ProgramObject>);
  static_assert(std::is_same_v<
                decltype(std::declval<const ObjectReader&>().LoadEncodedObject(
                    std::declval<const std::filesystem::path&>(),
                    std::declval<std::optional<std::string>>())),
                EncodedProgramObject>);
}

TEST(ProgramNamingTest, ExecutionRouteRemainsLightweightEnum) {
  static_assert(std::is_enum_v<ExecutionRoute>);
  static_assert(!std::is_class_v<ExecutionRoute>);
  static_assert(static_cast<int>(ExecutionRoute::AutoSelect) == 0);
  static_assert(static_cast<int>(ExecutionRoute::EncodedRaw) == 1);
  static_assert(static_cast<int>(ExecutionRoute::LoweredModeled) == 2);
}

TEST(ProgramNamingTest, EncodedProgramObjectHeaderProvidesDataObject) {
  static_assert(std::is_default_constructible_v<EncodedProgramObject>);
  static_assert(std::is_same_v<decltype(EncodedProgramObject{}.kernel_descriptor),
                               AmdgpuKernelDescriptor>);
}

TEST(ProgramNamingTest, ExecutableKernelCanConstructAndResolveLabel) {
  ExecutableKernel kernel(
      "kernel", {}, {{"entry", 7}}, MetadataBlob{}, ConstSegment{});
  EXPECT_EQ(kernel.ResolveLabel("entry"), 7u);
  EXPECT_THROW(static_cast<void>(kernel.ResolveLabel("missing")), std::out_of_range);
}

}  // namespace
}  // namespace gpu_model
