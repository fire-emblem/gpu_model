#include <gtest/gtest.h>

#include "gpu_arch/chip_config/gpu_arch_spec.h"

namespace gpu_model {
namespace {

TEST(GpuArchSpecTest, ComputesAggregateApAndPeuCounts) {
  GpuArchSpec spec;
  spec.dpc_count = 8;
  spec.ap_per_dpc = 13;
  spec.peu_per_ap = 4;

  EXPECT_EQ(spec.total_ap_count(), 104u);
  EXPECT_EQ(spec.total_peu_count(), 416u);
}

TEST(GpuArchSpecTest, DefaultsIssuePolicyFromArchitectureLayer) {
  GpuArchSpec spec;
  EXPECT_EQ(spec.cycle_resources.issue_limits.branch, 1u);
  EXPECT_EQ(spec.cycle_resources.issue_policy.type_limits.branch, 1u);
  EXPECT_EQ(spec.cycle_resources.issue_policy.group_limits[0], 1u);
}

}  // namespace
}  // namespace gpu_model
