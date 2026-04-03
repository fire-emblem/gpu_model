#include <gtest/gtest.h>

#include "gpu_model/arch/arch_registry.h"

namespace gpu_model {
namespace {

TEST(ArchRegistryTest, C500SpecExists) {
  const auto spec = ArchRegistry::Get("c500");
  EXPECT_NE(spec, nullptr);
}

TEST(ArchRegistryTest, C500FieldsMatchDesign) {
  const auto spec = ArchRegistry::Get("c500");
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
  EXPECT_EQ(spec->default_issue_cycles, 4u);
}

}  // namespace
}  // namespace gpu_model
