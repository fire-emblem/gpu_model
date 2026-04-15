#include <gtest/gtest.h>

#include "gpu_arch/device/device_def.h"
#include "gpu_arch/dpc/dpc_def.h"
#include "gpu_arch/peu/peu_def.h"

namespace gpu_model {
namespace {

TEST(TopologyDefaultsTest, DpcDefaultMatchesMac500Topology) {
  DpcConfig dpc;
  EXPECT_EQ(kDefaultApPerDpc, 13u);
  EXPECT_EQ(dpc.ap_count, 13u);
}

TEST(TopologyDefaultsTest, DeviceDefaultAggregatesApCountFromDpcDefault) {
  DeviceConfig device;
  EXPECT_EQ(device.dpc_count, 8u);
  EXPECT_EQ(device.total_ap_count(), 104u);
}

TEST(TopologyDefaultsTest, PeuResidentSlotDefaultMatchesCycleTopology) {
  EXPECT_EQ(kDefaultResidentWaveSlotsPerPeu, 8u);
}

}  // namespace
}  // namespace gpu_model
