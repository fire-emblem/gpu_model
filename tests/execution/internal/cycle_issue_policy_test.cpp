#include <gtest/gtest.h>

#include "gpu_model/arch/arch_registry.h"
#include "gpu_model/execution/internal/cycle_issue_policy.h"

namespace gpu_model {
namespace {

TEST(CycleIssuePolicyTest, ReturnsConfiguredIssueLimitsFromSpec) {
  const auto spec = ArchRegistry::Get("c500");
  ASSERT_NE(spec, nullptr);

  const auto limits = CycleIssueLimitsForSpec(*spec);
  EXPECT_EQ(limits.branch, 1u);
  EXPECT_EQ(limits.scalar_alu_or_memory, 1u);
  EXPECT_EQ(limits.vector_alu, 1u);
  EXPECT_EQ(limits.vector_memory, 1u);
  EXPECT_EQ(limits.local_data_share, 1u);
  EXPECT_EQ(limits.global_data_share_or_export, 1u);
  EXPECT_EQ(limits.special, 1u);
}

}  // namespace
}  // namespace gpu_model
