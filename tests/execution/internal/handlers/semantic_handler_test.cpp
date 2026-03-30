#include <gtest/gtest.h>

#include "gpu_model/arch/arch_registry.h"
#include "gpu_model/execution/internal/semantic_handler.h"
#include "gpu_model/execution/internal/semantics.h"
#include "gpu_model/isa/instruction_builder.h"

namespace gpu_model {
namespace {

TEST(SemanticHandlerTest, DispatchesBuiltinAndScalarAluFamilies) {
  const auto spec = ArchRegistry::Get("c500");
  ASSERT_NE(spec, nullptr);

  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SMov("s1", 7);
  const auto kernel = builder.Build("semantic_handler_smoke");

  KernelArgPack args;
  args.PushU64(123);
  MemorySystem memory;
  NullTraceSink trace;
  PlacementMap placement;

  ExecutionContext context{
      .spec = *spec,
      .kernel = kernel,
      .launch_config = LaunchConfig{.grid_dim_x = 1, .block_dim_x = 64},
      .args = args,
      .placement = placement,
      .memory = memory,
      .trace = trace,
      .stats = nullptr,
      .cycle = 0,
      .arg_load_cycles = 4,
      .issue_cycle_class_overrides = {},
      .issue_cycle_op_overrides = {},
  };

  WaveContext wave;
  wave.thread_count = 64;
  wave.ResetInitialExec();

  Semantics semantics;
  const auto load_plan = semantics.BuildPlan(kernel.instructions().at(0), wave, context);
  ASSERT_EQ(load_plan.scalar_writes.size(), 1u);
  EXPECT_EQ(load_plan.scalar_writes[0].reg_index, 0u);

  const auto mov_plan = semantics.BuildPlan(kernel.instructions().at(1), wave, context);
  ASSERT_EQ(mov_plan.scalar_writes.size(), 1u);
  EXPECT_EQ(mov_plan.scalar_writes[0].value, 7u);

  EXPECT_EQ(SemanticHandlerRegistry::Get(SemanticFamily::Builtin).family(),
            SemanticFamily::Builtin);
  EXPECT_EQ(SemanticHandlerRegistry::Get(SemanticFamily::ScalarAlu).family(),
            SemanticFamily::ScalarAlu);
}

}  // namespace
}  // namespace gpu_model
