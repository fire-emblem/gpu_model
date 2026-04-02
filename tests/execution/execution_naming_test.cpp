#include <gtest/gtest.h>

#include <type_traits>
#include <utility>

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
  using CycleRunSignature = uint64_t (CycleExecEngine::*)(ExecutionContext&);
  static_assert(std::is_same_v<decltype(&CycleExecEngine::Run), CycleRunSignature>);

  using EncodedRunSignature = LaunchResult (EncodedExecEngine::*)(const EncodedProgramObject&,
                                                                  const GpuArchSpec&,
                                                                  const LaunchConfig&,
                                                                  ExecutionMode,
                                                                  FunctionalExecutionConfig,
                                                                  const KernelArgPack&,
                                                                  const DeviceLoadResult*,
                                                                  MemorySystem&,
                                                                  TraceSink&) const;
  static_assert(std::is_same_v<decltype(&EncodedExecEngine::Run), EncodedRunSignature>);

  static_assert(std::is_same_v<decltype(std::declval<WaveContext>().status), WaveStatus>);
  static_assert(std::is_same_v<decltype(std::declval<WaveContext>().private_memory),
                               std::array<std::vector<std::byte>, kWaveSize>>);
}

TEST(ExecutionNamingTest, ExecutionHeadersDeclarePrimaryUtilities) {
  using BuildWaveContextBlocksSignature = std::vector<ExecutionBlockState> (*)(
      const PlacementMap&, const LaunchConfig&);
  static_assert(std::is_same_v<WaveContextBuilder, BuildWaveContextBlocksSignature>);
  static_assert(std::is_same_v<decltype(&BuildWaveContextBlocks), BuildWaveContextBlocksSignature>);

  using LoadByteLaneValueSignature = uint64_t (*)(const std::vector<std::byte>&, const LaneAccess&);
  static_assert(std::is_same_v<decltype(&memory_ops::LoadByteLaneValue), LoadByteLaneValueSignature>);

  using MarkWaveAtBarrierSignature = void (*)(WaveContext&, uint64_t, uint32_t&, bool);
  static_assert(std::is_same_v<decltype(&sync_ops::MarkWaveAtBarrier), MarkWaveAtBarrierSignature>);

  using ApplyExecutionPlanRegisterWritesSignature = void (*)(const OpPlan&, WaveContext&);
  static_assert(std::is_same_v<decltype(&ApplyExecutionPlanRegisterWrites),
                               ApplyExecutionPlanRegisterWritesSignature>);

  using MaybeFormatExecutionMaskUpdateSignature = std::optional<std::string> (*)(
      const OpPlan&, const WaveContext&);
  static_assert(std::is_same_v<decltype(&MaybeFormatExecutionMaskUpdate),
                               MaybeFormatExecutionMaskUpdateSignature>);

  using ApplyExecutionPlanControlFlowSignature = void (*)(
      const OpPlan&, WaveContext&, bool, bool);
  static_assert(std::is_same_v<decltype(&ApplyExecutionPlanControlFlow),
                               ApplyExecutionPlanControlFlowSignature>);
}

}  // namespace
}  // namespace gpu_model
