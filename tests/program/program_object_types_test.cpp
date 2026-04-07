#include <gtest/gtest.h>

#include <filesystem>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "gpu_model/program/executable_kernel.h"
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
  static_assert(std::is_same_v<decltype(std::declval<const ProgramObject&>().kernel_name()),
                               const std::string&>);
  static_assert(std::is_same_v<decltype(ExecutableKernel{}.name()), const std::string&>);
  static_assert(std::is_same_v<
                decltype(std::declval<const ObjectReader&>().LoadFromStem(
                    std::declval<const std::filesystem::path&>())),
                ProgramObject>);
  static_assert(std::is_same_v<
                decltype(std::declval<const ObjectReader&>().LoadProgramObject(
                    std::declval<const std::filesystem::path&>(),
                    std::declval<std::optional<std::string>>())),
                ProgramObject>);
  static_assert(std::is_same_v<decltype(std::declval<const ProgramObject&>().kernel_name()),
                               const std::string&>);
}

TEST(ProgramNamingTest, ExecutableKernelCanConstructAndResolveLabel) {
  ExecutableKernel kernel(
      "kernel", {}, {{"entry", 7}}, MetadataBlob{}, ConstSegment{});
  EXPECT_EQ(kernel.ResolveLabel("entry"), 7u);
  EXPECT_THROW(static_cast<void>(kernel.ResolveLabel("missing")), std::out_of_range);
}

}  // namespace
}  // namespace gpu_model
