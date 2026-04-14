#include <gtest/gtest.h>

#include <type_traits>
#include <utility>

#include "execution/cycle/cycle_exec_engine.h"
#include "execution/functional/functional_exec_engine.h"
#include "execution/internal/commit_logic/memory_ops.h"
#include "execution/internal/commit_logic/plan_apply.h"
#include "execution/internal/sync_ops/sync_ops.h"
#include "state/wave/barrier_state.h"
#include "execution/internal/block_schedule/wave_context_builder.h"
#include "runtime/exec_engine/exec_engine.h"
#include "state/ap/ap_runtime_state.h"
#include "state/wave/wave_runtime_state.h"

namespace gpu_model {
namespace {

TEST(ExecutionNamingTest, ExecutionHeadersDeclarePrimaryTypes) {
  using FunctionalRunSequentialSignature = uint64_t (FunctionalExecEngine::*)();
  using FunctionalRunParallelSignature = uint64_t (FunctionalExecEngine::*)(uint32_t);
  static_assert(std::is_constructible_v<FunctionalExecEngine, ExecutionContext&>);
  static_assert(std::is_same_v<decltype(&FunctionalExecEngine::RunSequential),
                               FunctionalRunSequentialSignature>);
  static_assert(std::is_same_v<decltype(&FunctionalExecEngine::RunParallelBlocks),
                               FunctionalRunParallelSignature>);

  static_assert(std::is_base_of_v<IExecutionEngine, CycleExecEngine>);
  static_assert(std::is_same_v<decltype(CycleTimingConfig{}.cache_model), CacheModelSpec>);
  static_assert(std::is_same_v<decltype(CycleTimingConfig{}.shared_bank_model), SharedBankModelSpec>);
  static_assert(std::is_same_v<decltype(CycleTimingConfig{}.issue_policy),
                               std::optional<ArchitecturalIssuePolicy>>);
  using CycleRunSignature = uint64_t (CycleExecEngine::*)(ExecutionContext&);
  static_assert(std::is_same_v<decltype(&CycleExecEngine::Run), CycleRunSignature>);
  using RuntimeSetIssuePolicySignature =
      void (ExecEngine::*)(const ArchitecturalIssuePolicy&);
  static_assert(std::is_same_v<decltype(&ExecEngine::SetCycleIssuePolicy),
                               RuntimeSetIssuePolicySignature>);

  static_assert(std::is_same_v<decltype(std::declval<WaveContext>().status), WaveStatus>);
  static_assert(std::is_same_v<decltype(std::declval<WaveContext>().private_memory),
                               std::array<std::vector<std::byte>, kWaveSize>>);
  static_assert(std::is_same_v<decltype(std::declval<ApState>().barrier), BarrierState>);
}

TEST(ExecutionNamingTest, ExecutionHeadersDeclarePrimaryUtilities) {
  using BuildWaveContextBlocksSignature = std::vector<ExecutionBlockState> (*)(
      const PlacementMap&, const LaunchConfig&);
  static_assert(std::is_same_v<WaveContextBuilder, BuildWaveContextBlocksSignature>);
  static_assert(std::is_same_v<decltype(&BuildWaveContextBlocks), BuildWaveContextBlocksSignature>);

  using LoadByteLaneValueSignature = uint64_t (*)(const std::vector<std::byte>&, const LaneAccess&);
  static_assert(std::is_same_v<decltype(&memory_ops::LoadByteLaneValue), LoadByteLaneValueSignature>);

  using MarkWaveAtBarrierSignature = void (*)(WaveContext&, uint64_t, uint32_t&, bool);
  static_assert(std::is_same_v<decltype(&MarkWaveAtBarrier), MarkWaveAtBarrierSignature>);

  using ApplyExecutionPlanRegisterWritesSignature = void (*)(const OpPlan&, WaveContext&);
  static_assert(std::is_same_v<decltype(&ApplyExecutionPlanRegisterWrites),
                               ApplyExecutionPlanRegisterWritesSignature>);

  using MaybeFormatExecutionMaskUpdateSignature = std::optional<std::string> (*)(
      const OpPlan&, const WaveContext&);
  static_assert(std::is_same_v<decltype(&MaybeFormatExecutionMaskUpdate),
                               MaybeFormatExecutionMaskUpdateSignature>);

  using ApplyExecutionPlanControlFlowSignature = void (*)(
      const ExecutableKernel&, const OpPlan&, WaveContext&, bool, bool);
  static_assert(std::is_same_v<decltype(&ApplyExecutionPlanControlFlow),
                               ApplyExecutionPlanControlFlowSignature>);
}

}  // namespace
}  // namespace gpu_model
