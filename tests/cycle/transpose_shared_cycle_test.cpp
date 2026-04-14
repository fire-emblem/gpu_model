#include <gtest/gtest.h>

#include <cstdint>
#include <string_view>

#include "debug/trace/sink.h"
#include "instruction/isa/instruction_builder.h"
#include "runtime/exec_engine/exec_engine.h"

namespace gpu_model {
namespace {

ExecutableKernel BuildSharedTransposeKernel() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SLoadArg("s1", 1);
  builder.SysGlobalIdX("v0");
  builder.SysGlobalIdY("v1");
  builder.SysLocalIdX("v2");
  builder.SysLocalIdY("v3");
  builder.SysBlockDimX("s2");
  builder.SysBlockDimY("s3");
  builder.SysGridDimX("s4");
  builder.SysGridDimY("s5");

  builder.SMul("s6", "s2", "s4");
  builder.SMul("s7", "s3", "s5");
  builder.VMul("v4", "v1", "s6");
  builder.VAdd("v5", "v4", "v0");
  builder.VMul("v6", "v3", "s2");
  builder.VAdd("v7", "v6", "v2");
  builder.MLoadGlobal("v8", "s0", "v5", 4);
  builder.SWaitCnt(/*global_count=*/0, /*shared_count=*/UINT32_MAX,
                   /*private_count=*/UINT32_MAX, /*scalar_buffer_count=*/UINT32_MAX);
  builder.MStoreShared("v7", "v8", 4);
  builder.SyncBarrier();
  builder.VMul("v9", "v2", "s3");
  builder.VAdd("v10", "v9", "v3");
  builder.MLoadShared("v11", "v10", 4);
  builder.SWaitCnt(/*global_count=*/UINT32_MAX, /*shared_count=*/0,
                   /*private_count=*/UINT32_MAX, /*scalar_buffer_count=*/UINT32_MAX);
  builder.MStoreGlobal("s1", "v5", "v11", 4);
  builder.BExit();
  return builder.Build("shared_transpose_2d_cycle");
}

TEST(TransposeSharedCycleTest, TransposeWorksInCycleModeAndUsesBarrier) {
  CollectingTraceSink trace;
  ExecEngine runtime(&trace);
  runtime.SetFixedGlobalMemoryLatency(8);
  const auto kernel = BuildSharedTransposeKernel();

  constexpr uint32_t grid_x = 1;
  constexpr uint32_t grid_y = 1;
  constexpr uint32_t block_x = 16;
  constexpr uint32_t block_y = 16;
  constexpr uint32_t width = grid_x * block_x;
  constexpr uint32_t height = grid_y * block_y;
  constexpr uint32_t total = width * height;

  const uint64_t in_addr = runtime.memory().AllocateGlobal(total * sizeof(int32_t));
  const uint64_t out_addr = runtime.memory().AllocateGlobal(total * sizeof(int32_t));
  for (uint32_t y = 0; y < height; ++y) {
    for (uint32_t x = 0; x < width; ++x) {
      const uint32_t index = y * width + x;
      runtime.memory().StoreGlobalValue<int32_t>(in_addr + index * sizeof(int32_t),
                                                 static_cast<int32_t>(1000 * y + x));
    }
  }

  LaunchRequest request;
  request.kernel = &kernel;
  request.mode = ExecutionMode::Cycle;
  request.config.grid_dim_x = grid_x;
  request.config.grid_dim_y = grid_y;
  request.config.block_dim_x = block_x;
  request.config.block_dim_y = block_y;
  request.config.shared_memory_bytes = total * sizeof(int32_t);
  request.args.PushU64(in_addr);
  request.args.PushU64(out_addr);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  EXPECT_GT(result.total_cycles, 0u);

  bool saw_barrier = false;
  for (const auto& event : trace.events()) {
    if (event.kind == TraceEventKind::Barrier) {
      saw_barrier = true;
      break;
    }
  }
  EXPECT_TRUE(saw_barrier);

  for (uint32_t y = 0; y < height; ++y) {
    for (uint32_t x = 0; x < width; ++x) {
      const uint32_t out_index = y * width + x;
      EXPECT_EQ(runtime.memory().LoadGlobalValue<int32_t>(out_addr + out_index * sizeof(int32_t)),
                static_cast<int32_t>(1000 * x + y));
    }
  }
}

}  // namespace
}  // namespace gpu_model
