#include <gtest/gtest.h>

#include "instruction/isa/instruction_builder.h"
#include "instruction/isa/kernel_metadata.h"
#include "runtime/exec_engine/exec_engine.h"

namespace gpu_model {
namespace {

TEST(KernelMetadataTest, ParsesStructuredLaunchMetadata) {
  MetadataBlob metadata{.values = {
                            {"arch", "mac500"},
                            {"entry", "meta_kernel"},
                            {"module_kernels", "meta_kernel,other_kernel"},
                            {"arg_count", "2"},
                            {"required_shared_bytes", "256"},
                            {"block_dim_multiple", "64"},
                            {"max_block_dim", "256"},
                        }};

  const auto parsed = ParseKernelLaunchMetadata(metadata);
  ASSERT_TRUE(parsed.arch.has_value());
  ASSERT_TRUE(parsed.entry.has_value());
  ASSERT_TRUE(parsed.arg_count.has_value());
  ASSERT_TRUE(parsed.required_shared_bytes.has_value());
  ASSERT_TRUE(parsed.block_dim_multiple.has_value());
  ASSERT_TRUE(parsed.max_block_dim.has_value());
  EXPECT_EQ(*parsed.arch, "mac500");
  EXPECT_EQ(*parsed.entry, "meta_kernel");
  ASSERT_EQ(parsed.module_kernels.size(), 2u);
  EXPECT_EQ(parsed.module_kernels[0], "meta_kernel");
  EXPECT_EQ(parsed.module_kernels[1], "other_kernel");
  EXPECT_EQ(*parsed.arg_count, 2u);
  EXPECT_EQ(*parsed.required_shared_bytes, 256u);
  EXPECT_EQ(*parsed.block_dim_multiple, 64u);
  EXPECT_EQ(*parsed.max_block_dim, 256u);
}

TEST(KernelMetadataTest, ParsesTypedArgAndHiddenArgLayouts) {
  MetadataBlob metadata{.values = {
                            {"arg_layout", "global_buffer:8,by_value:16:4,global_buffer:20:8"},
                            {"hidden_arg_layout",
                             "hidden_block_count_x:16:4,hidden_group_size_x:24:2"},
                            {"kernarg_segment_size", "64"},
                            {"group_segment_fixed_size", "512"},
                        }};

  const auto parsed = ParseKernelLaunchMetadata(metadata);
  ASSERT_EQ(parsed.arg_layout.size(), 3u);
  EXPECT_EQ(parsed.arg_layout[0].kind, KernelArgValueKind::GlobalBuffer);
  EXPECT_EQ(parsed.arg_layout[0].kind_name, "global_buffer");
  EXPECT_EQ(parsed.arg_layout[0].size, 8u);
  EXPECT_FALSE(parsed.arg_layout[0].offset.has_value());
  EXPECT_EQ(parsed.arg_layout[1].kind, KernelArgValueKind::ByValue);
  EXPECT_EQ(parsed.arg_layout[1].kind_name, "by_value");
  EXPECT_EQ(parsed.arg_layout[1].size, 4u);
  ASSERT_TRUE(parsed.arg_layout[1].offset.has_value());
  EXPECT_EQ(*parsed.arg_layout[1].offset, 16u);
  EXPECT_EQ(parsed.arg_layout[2].kind, KernelArgValueKind::GlobalBuffer);
  EXPECT_EQ(parsed.arg_layout[2].kind_name, "global_buffer");
  EXPECT_EQ(parsed.arg_layout[2].size, 8u);
  ASSERT_TRUE(parsed.arg_layout[2].offset.has_value());
  EXPECT_EQ(*parsed.arg_layout[2].offset, 20u);

  ASSERT_EQ(parsed.hidden_arg_layout.size(), 2u);
  EXPECT_EQ(parsed.hidden_arg_layout[0].kind, KernelHiddenArgKind::BlockCountX);
  EXPECT_EQ(parsed.hidden_arg_layout[0].kind_name, "hidden_block_count_x");
  EXPECT_EQ(parsed.hidden_arg_layout[0].offset, 16u);
  EXPECT_EQ(parsed.hidden_arg_layout[0].size, 4u);
  EXPECT_EQ(parsed.hidden_arg_layout[1].kind, KernelHiddenArgKind::GroupSizeX);
  EXPECT_EQ(parsed.hidden_arg_layout[1].kind_name, "hidden_group_size_x");
  EXPECT_EQ(parsed.hidden_arg_layout[1].offset, 24u);
  EXPECT_EQ(parsed.hidden_arg_layout[1].size, 2u);

  ASSERT_TRUE(parsed.kernarg_segment_size.has_value());
  ASSERT_TRUE(parsed.group_segment_fixed_size.has_value());
  EXPECT_EQ(*parsed.kernarg_segment_size, 64u);
  EXPECT_EQ(*parsed.group_segment_fixed_size, 512u);
  EXPECT_EQ(*parsed.required_shared_bytes, 512u);
  EXPECT_EQ(EstimateVisibleKernargBytes(parsed), 28u);
  EXPECT_EQ(RequiredKernargTemplateBytes(parsed), 64u);
  EXPECT_EQ(ToString(parsed.arg_layout[0].kind), "global_buffer");
  EXPECT_EQ(ToString(parsed.hidden_arg_layout[0].kind), "hidden_block_count_x");
}

TEST(KernelMetadataTest, RejectsLaunchesThatViolateMetadataConstraints) {
  InstructionBuilder builder;
  builder.BExit();
  const auto kernel = builder.Build(
      "validated_kernel",
      MetadataBlob{.values = {
                       {"arch", "mac500"},
                       {"entry", "validated_kernel"},
                       {"arg_count", "1"},
                       {"required_shared_bytes", "128"},
                       {"block_dim_multiple", "64"},
                       {"max_block_dim", "128"},
                   }});

  ExecEngine runtime;

  LaunchRequest wrong_args;
  wrong_args.kernel = &kernel;
  wrong_args.config.grid_dim_x = 1;
  wrong_args.config.block_dim_x = 64;
  wrong_args.config.shared_memory_bytes = 128;
  const auto wrong_arg_result = runtime.Launch(wrong_args);
  EXPECT_FALSE(wrong_arg_result.ok);
  EXPECT_EQ(wrong_arg_result.error_message, "kernel argument count does not match metadata");

  LaunchRequest wrong_shared;
  wrong_shared.kernel = &kernel;
  wrong_shared.config.grid_dim_x = 1;
  wrong_shared.config.block_dim_x = 64;
  wrong_shared.config.shared_memory_bytes = 64;
  wrong_shared.args.PushU32(1);
  const auto wrong_shared_result = runtime.Launch(wrong_shared);
  EXPECT_FALSE(wrong_shared_result.ok);
  EXPECT_EQ(wrong_shared_result.error_message,
            "shared memory launch size is smaller than metadata requirement");

  LaunchRequest wrong_multiple;
  wrong_multiple.kernel = &kernel;
  wrong_multiple.config.grid_dim_x = 1;
  wrong_multiple.config.block_dim_x = 96;
  wrong_multiple.config.shared_memory_bytes = 128;
  wrong_multiple.args.PushU32(1);
  const auto wrong_multiple_result = runtime.Launch(wrong_multiple);
  EXPECT_FALSE(wrong_multiple_result.ok);
  EXPECT_EQ(wrong_multiple_result.error_message,
            "block_dim_x does not satisfy metadata multiple requirement");

  LaunchRequest wrong_max;
  wrong_max.kernel = &kernel;
  wrong_max.config.grid_dim_x = 1;
  wrong_max.config.block_dim_x = 192;
  wrong_max.config.shared_memory_bytes = 128;
  wrong_max.args.PushU32(1);
  const auto wrong_max_result = runtime.Launch(wrong_max);
  EXPECT_FALSE(wrong_max_result.ok);
  EXPECT_EQ(wrong_max_result.error_message, "block_dim_x exceeds metadata maximum");
}

TEST(KernelMetadataTest, RejectsKernelNameNotPresentInModuleKernelList) {
  InstructionBuilder builder;
  builder.BExit();
  const auto kernel = builder.Build(
      "actual_kernel",
      MetadataBlob{.values = {
                       {"arch", "mac500"},
                       {"entry", "actual_kernel"},
                       {"module_name", "demo_mod"},
                       {"module_kernels", "other_kernel,third_kernel"},
                   }});

  ExecEngine runtime;
  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;

  const auto result = runtime.Launch(request);
  EXPECT_FALSE(result.ok);
  EXPECT_EQ(result.error_message, "kernel name is not present in module_kernels metadata");
}

}  // namespace
}  // namespace gpu_model
