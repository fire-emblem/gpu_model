#include <gtest/gtest.h>

#include "gpu_arch/chip_config/arch_registry.h"
#include "gpu_arch/chip_config/gpu_arch_spec.h"

namespace gpu_model {
namespace {

TEST(ArchRegistryTest, Mac500SpecExists) {
  const auto spec = ArchRegistry::Get("mac500");
  EXPECT_NE(spec, nullptr);
}

TEST(ArchRegistryTest, Mac500FieldsMatchDesign) {
  const auto spec = ArchRegistry::Get("mac500");
  ASSERT_NE(spec, nullptr);
  EXPECT_EQ(spec->wave_size, 64u);
  EXPECT_EQ(spec->dpc_count, 8u);
  EXPECT_EQ(spec->ap_per_dpc, 13u);
  EXPECT_EQ(spec->total_ap_count(), 104u);
  EXPECT_EQ(spec->peu_per_ap, 4u);
  EXPECT_EQ(spec->total_peu_count(), 416u);
  EXPECT_EQ(spec->max_resident_waves, 8u);
  EXPECT_EQ(spec->max_issuable_waves, 4u);
  EXPECT_EQ(spec->cycle_resources.resident_wave_slots_per_peu, 8u);
  EXPECT_EQ(spec->cycle_resources.barrier_slots_per_ap, 16u);
  EXPECT_EQ(spec->cycle_resources.issue_limits.branch, 1u);
  EXPECT_EQ(spec->cycle_resources.issue_limits.scalar_alu_or_memory, 1u);
  EXPECT_EQ(spec->cycle_resources.issue_limits.vector_alu, 1u);
  EXPECT_EQ(spec->cycle_resources.issue_limits.vector_memory, 1u);
  EXPECT_EQ(spec->cycle_resources.issue_limits.local_data_share, 1u);
  EXPECT_EQ(spec->cycle_resources.issue_limits.global_data_share_or_export, 1u);
  EXPECT_EQ(spec->cycle_resources.issue_limits.special, 1u);
  EXPECT_EQ(spec->cycle_resources.issue_policy.type_limits.branch, 1u);
  EXPECT_EQ(spec->cycle_resources.issue_policy.group_limits[0], 1u);
  EXPECT_EQ(spec->cycle_resources.issue_policy.group_limits[6], 1u);
  EXPECT_EQ(spec->cycle_resources.issue_policy.type_to_group[0], 0u);
  EXPECT_EQ(spec->cycle_resources.issue_policy.type_to_group[6], 0u);
  EXPECT_EQ(spec->default_issue_cycles, 4u);
}

TEST(ArchRegistryTest, LegacyGpuArchSpecHeaderRemainsAvailableAsBridge) {
  GpuArchSpec spec;
  spec.dpc_count = 2;
  spec.ap_per_dpc = 3;
  spec.peu_per_ap = 4;

  EXPECT_EQ(spec.total_ap_count(), 6u);
  EXPECT_EQ(spec.total_peu_count(), 24u);
  EXPECT_EQ(spec.cycle_resources.issue_policy.group_limits[0], 1u);
}

}  // namespace
}  // namespace gpu_model
