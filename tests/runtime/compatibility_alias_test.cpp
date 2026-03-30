#include <gtest/gtest.h>

#include <type_traits>

#include "gpu_model/execution/encoded_exec_engine.h"
#include "gpu_model/exec/encoded/executor/raw_gcn_executor.h"

namespace gpu_model {
namespace {

TEST(RuntimeProgramCompatibilityAliasTest, NonRuntimeProgramAliasesRemainEquivalentInPhase2) {
  static_assert(std::is_same_v<EncodedExecEngine, RawGcnExecutor>);
}

}  // namespace
}  // namespace gpu_model
