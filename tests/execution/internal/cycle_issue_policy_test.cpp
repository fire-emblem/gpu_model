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

TEST(CycleIssuePolicyTest, ReturnsConfiguredIssuePolicyFromSpec) {
  const auto spec = ArchRegistry::Get("c500");
  ASSERT_NE(spec, nullptr);

  const auto policy = CycleIssuePolicyForSpec(*spec);
  EXPECT_EQ(policy.type_limits.branch, 1u);
  EXPECT_EQ(policy.group_limits[0], 1u);
  EXPECT_EQ(policy.group_limits[6], 1u);
  EXPECT_EQ(policy.type_to_group[0], 0u);
  EXPECT_EQ(policy.type_to_group[6], 6u);
}

}  // namespace
}  // namespace gpu_model
