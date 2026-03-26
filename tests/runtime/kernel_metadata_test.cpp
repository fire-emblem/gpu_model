#include <gtest/gtest.h>

#include "gpu_model/isa/instruction_builder.h"
#include "gpu_model/isa/kernel_metadata.h"
#include "gpu_model/runtime/host_runtime.h"

namespace gpu_model {
namespace {

TEST(KernelMetadataTest, ParsesStructuredLaunchMetadata) {
  MetadataBlob metadata{.values = {
                            {"arch", "c500"},
                            {"entry", "meta_kernel"},
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
  EXPECT_EQ(*parsed.arch, "c500");
  EXPECT_EQ(*parsed.entry, "meta_kernel");
  EXPECT_EQ(*parsed.arg_count, 2u);
  EXPECT_EQ(*parsed.required_shared_bytes, 256u);
  EXPECT_EQ(*parsed.block_dim_multiple, 64u);
  EXPECT_EQ(*parsed.max_block_dim, 256u);
}

TEST(KernelMetadataTest, RejectsLaunchesThatViolateMetadataConstraints) {
  InstructionBuilder builder;
  builder.BExit();
  const auto kernel = builder.Build(
      "validated_kernel",
      MetadataBlob{.values = {
                       {"arch", "c500"},
                       {"entry", "validated_kernel"},
                       {"arg_count", "1"},
                       {"required_shared_bytes", "128"},
                       {"block_dim_multiple", "64"},
                       {"max_block_dim", "128"},
                   }});

  HostRuntime runtime;

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

}  // namespace
}  // namespace gpu_model
