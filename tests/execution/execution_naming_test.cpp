#include <gtest/gtest.h>

#include <type_traits>

#include "gpu_model/execution/cycle_exec_engine.h"
#include "gpu_model/execution/encoded_exec_engine.h"
#include "gpu_model/execution/functional_exec_engine.h"
#include "gpu_model/execution/memory_ops.h"
#include "gpu_model/execution/plan_apply.h"
#include "gpu_model/execution/sync_ops.h"
#include "gpu_model/execution/wave_context.h"
#include "gpu_model/execution/wave_context_builder.h"

namespace gpu_model {
namespace {

TEST(ExecutionNamingTest, NewExecutionNamesAliasLegacyTypes) {
  static_assert(std::is_same_v<FunctionalExecEngine, FunctionalExecutionCore>);
  static_assert(std::is_same_v<CycleExecEngine, CycleExecutor>);
  static_assert(std::is_same_v<EncodedExecEngine, RawGcnExecutor>);
  static_assert(std::is_same_v<WaveContext, WaveState>);
}

TEST(ExecutionNamingTest, NewExecutionUtilityNamesAliasLegacyUtilities) {
  using BuildWaveContextBlocksSignature = std::vector<ExecutionBlockState> (*)(
      const PlacementMap&, const LaunchConfig&);
  static_assert(std::is_same_v<WaveContextBuilder, BuildWaveContextBlocksSignature>);
  static_assert(std::is_same_v<decltype(&BuildWaveContextBlocks), BuildWaveContextBlocksSignature>);

  static_assert(std::is_same_v<decltype(&memory_ops::LoadByteLaneValue),
                               decltype(&execution_memory_ops::LoadByteLaneValue)>);
  static_assert(std::is_same_v<decltype(&sync_ops::MarkWaveAtBarrier),
                               decltype(&execution_sync_ops::MarkWaveAtBarrier)>);
  static_assert(std::is_same_v<decltype(&ApplyExecutionPlanRegisterWrites),
                               decltype(&ApplyPlanRegisterWrites)>);

  using MaybeFormatExecutionMaskUpdateSignature = std::optional<std::string> (*)(
      const OpPlan&, const WaveContext&);
  static_assert(std::is_same_v<decltype(&MaybeFormatExecutionMaskUpdate),
                               MaybeFormatExecutionMaskUpdateSignature>);
  static_assert(std::is_same_v<decltype(&MaybeFormatExecutionMaskUpdate),
                               decltype(&MaybeFormatExecMaskUpdate)>);

  using ApplyExecutionPlanControlFlowSignature = void (*)(
      const OpPlan&, WaveContext&, bool, bool);
  static_assert(std::is_same_v<decltype(&ApplyExecutionPlanControlFlow),
                               ApplyExecutionPlanControlFlowSignature>);
  static_assert(std::is_same_v<decltype(&ApplyExecutionPlanControlFlow),
                               decltype(&ApplyPlanControlFlow)>);
}

}  // namespace
}  // namespace gpu_model
