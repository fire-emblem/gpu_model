#include <gtest/gtest.h>

#include "gpu_arch/chip_config/arch_registry.h"
#include "execution/internal/plan/semantic_handler.h"
#include "execution/internal/plan/semantics.h"
#include "instruction/isa/instruction_builder.h"

namespace gpu_model {
namespace {

TEST(SemanticHandlerTest, DispatchesBuiltinAndScalarAluFamilies) {
  const auto spec = ArchRegistry::Get("mac500");
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

TEST(SemanticHandlerTest, BuildsScalarFirstBitSetPlan) {
  const auto spec = ArchRegistry::Get("mac500");
  ASSERT_NE(spec, nullptr);

  InstructionBuilder builder;
  builder.SFF1I32B32("s2", "s1");
  const auto kernel = builder.Build("semantic_handler_sff1_i32_b32");

  KernelArgPack args;
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
  wave.sgpr.Write(1, 0x20u);
  const auto set_plan = semantics.BuildPlan(kernel.instructions().at(0), wave, context);
  ASSERT_EQ(set_plan.scalar_writes.size(), 1u);
  EXPECT_EQ(set_plan.scalar_writes[0].reg_index, 2u);
  EXPECT_EQ(set_plan.scalar_writes[0].value, 5u);

  wave.sgpr.Write(1, 0u);
  const auto zero_plan = semantics.BuildPlan(kernel.instructions().at(0), wave, context);
  ASSERT_EQ(zero_plan.scalar_writes.size(), 1u);
  EXPECT_EQ(zero_plan.scalar_writes[0].reg_index, 2u);
  EXPECT_EQ(zero_plan.scalar_writes[0].value, 0xffffffffu);
}

TEST(SemanticHandlerTest, BuildsUnsignedScalarMinMaxPlans) {
  const auto spec = ArchRegistry::Get("mac500");
  ASSERT_NE(spec, nullptr);

  InstructionBuilder builder;
  builder.SMinU32("s2", "s1", "s3");
  builder.SMaxU32("s4", "s1", "s3");
  const auto kernel = builder.Build("semantic_handler_unsigned_min_max");

  KernelArgPack args;
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
  wave.sgpr.Write(1, 9u);
  wave.sgpr.Write(3, 0xfffffff0u);

  Semantics semantics;
  const auto min_plan = semantics.BuildPlan(kernel.instructions().at(0), wave, context);
  ASSERT_EQ(min_plan.scalar_writes.size(), 1u);
  EXPECT_EQ(min_plan.scalar_writes[0].reg_index, 2u);
  EXPECT_EQ(min_plan.scalar_writes[0].value, 9u);

  const auto max_plan = semantics.BuildPlan(kernel.instructions().at(1), wave, context);
  ASSERT_EQ(max_plan.scalar_writes.size(), 1u);
  EXPECT_EQ(max_plan.scalar_writes[0].reg_index, 4u);
  EXPECT_EQ(max_plan.scalar_writes[0].value, 0xfffffff0u);
}

}  // namespace
}  // namespace gpu_model
