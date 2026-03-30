#include <gtest/gtest.h>

#include <type_traits>

#include "gpu_model/execution/encoded_exec_engine.h"

namespace gpu_model {
namespace {

TEST(RuntimeProgramCompatibilityAliasTest, NonRuntimeProgramAliasesRemainEquivalentInPhase2) {
  static_assert(std::is_class_v<EncodedExecEngine>);
  static_assert(std::is_default_constructible_v<EncodedExecEngine>);
}

}  // namespace
}  // namespace gpu_model
