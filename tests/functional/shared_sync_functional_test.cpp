#include <gtest/gtest.h>

#include <cstdint>

#include "gpu_model/isa/instruction_builder.h"
#include "gpu_model/runtime/runtime_engine.h"

namespace gpu_model {
namespace {

ExecutableKernel BuildSharedAtomicReductionKernel() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SysLaneId("v0");
  builder.SMov("s1", 1);

  builder.VCmpLtCmask("v0", "s1");
  builder.MaskSaveExec("s10");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("after_init");
  builder.VMov("v1", 0);
  builder.VMov("v2", 0);
  builder.MStoreShared("v1", "v2", 4);
  builder.Label("after_init");
  builder.MaskRestoreExec("s10");

  builder.SyncWaveBarrier();
  builder.SyncBarrier();

  builder.VMov("v1", 0);
  builder.VMov("v2", 1);
  builder.MAtomicAddShared("v1", "v2", 4);
  builder.SyncBarrier();

  builder.VCmpLtCmask("v0", "s1");
  builder.MaskSaveExec("s11");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("exit");
  builder.MLoadShared("v3", "v1", 4);
  builder.VMov("v4", 0);
  builder.MStoreGlobal("s0", "v4", "v3", 4);
  builder.Label("exit");
  builder.MaskRestoreExec("s11");
  builder.BExit();
  return builder.Build("shared_atomic_reduce");
}

TEST(SharedSyncFunctionalTest, SharedAtomicReductionAcrossTwoWavesIsCorrect) {
  RuntimeEngine runtime;
  const auto kernel = BuildSharedAtomicReductionKernel();
  const uint64_t out_addr = runtime.memory().AllocateGlobal(sizeof(int32_t));
  runtime.memory().StoreGlobalValue<int32_t>(out_addr, -1);

  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 128;
  request.config.shared_memory_bytes = 4;
  request.args.PushU64(out_addr);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  EXPECT_EQ(runtime.memory().LoadGlobalValue<int32_t>(out_addr), 128);
  EXPECT_GE(result.stats.shared_stores, 2u);
  EXPECT_GE(result.stats.barriers, 2u);
}

}  // namespace
}  // namespace gpu_model
