#include <gtest/gtest.h>

#include "gpu_arch/chip_config/amdgpu_target_config.h"

namespace gpu_model {
namespace {

TEST(AmdgpuTargetConfigTest, NormalizesMcpuFromTriplesAndWhitespace) {
  EXPECT_EQ(NormalizeAmdgpuMcpu("amdgcn-amd-amdhsa--gfx90a"), "gfx90a");
  EXPECT_EQ(NormalizeAmdgpuMcpu("  GFX90A:sramecc+"), "gfx90a");
}

TEST(AmdgpuTargetConfigTest, ReportsProjectTargetCompatibility) {
  EXPECT_TRUE(IsProjectAmdgpuMcpu("gfx90a"));
  EXPECT_FALSE(IsProjectAmdgpuMcpu("gfx940"));
  EXPECT_NE(ProjectAmdgpuTargetErrorMessage("gfx940").find("gfx90a"), std::string::npos);
}

}  // namespace
}  // namespace gpu_model
