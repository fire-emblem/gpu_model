#include <gtest/gtest.h>

#include "gpu_arch/chip_config/arch_config/arch_config.h"
#include "gpu_arch/chip_config/arch_registry.h"
#include "gpu_arch/occupancy/occupancy_calculator.h"
#include "gpu_arch/occupancy/occupancy_report.h"

namespace gpu_model {
namespace {

KernelResourceUsage MakeLightKernel() {
  return {.vgpr_count = 16,
          .sgpr_count = 16,
          .agpr_count = 0,
          .shared_memory_bytes = 0,
          .private_memory_bytes = 0,
          .block_size = 256,
          .uses_barrier = false};
}

KernelResourceUsage MakeHeavyVgprKernel() {
  return {.vgpr_count = 128,
          .sgpr_count = 32,
          .agpr_count = 0,
          .shared_memory_bytes = 0,
          .private_memory_bytes = 0,
          .block_size = 256,
          .uses_barrier = false};
}

KernelResourceUsage MakeBarrierKernel() {
  return {.vgpr_count = 16,
          .sgpr_count = 16,
          .agpr_count = 0,
          .shared_memory_bytes = 16384,
          .private_memory_bytes = 0,
          .block_size = 256,
          .uses_barrier = true};
}

class OccupancyCalculatorTest : public ::testing::Test {
 protected:
  std::shared_ptr<const GpuArchSpec> spec_ = ArchRegistry::Get("mac500");
  OccupancyCalculator calc_{*spec_};
};

TEST_F(OccupancyCalculatorTest, LightKernelFullOccupancy) {
  auto result = calc_.Calculate(MakeLightKernel());
  // 256 VGPR total / (16 aligned to 16) = 16, capped at hw_max=8
  EXPECT_EQ(result.max_waves_per_peu, 8);
  EXPECT_FLOAT_EQ(result.occupancy_ratio, 1.0f);
  EXPECT_EQ(result.wave_limiting_factor, "hardware");
  // 8 waves / 4 waves_per_block = 2 blocks
  EXPECT_EQ(result.max_blocks_per_ap, 2);
}

TEST_F(OccupancyCalculatorTest, HeavyVgprReducesOccupancy) {
  auto result = calc_.Calculate(MakeHeavyVgprKernel());
  // 256 VGPR / 128 = 2 waves
  EXPECT_EQ(result.max_waves_per_peu, 2);
  EXPECT_EQ(result.wave_limiting_factor, "vgpr");
  // 2 waves < 4 waves_per_block, so 0 blocks
  EXPECT_EQ(result.max_blocks_per_ap, 0);
}

TEST_F(OccupancyCalculatorTest, BarrierKernelLimits) {
  auto result = calc_.Calculate(MakeBarrierKernel());
  EXPECT_EQ(result.max_waves_per_peu, 8);
  // Shared mem: 65536 / 16384 = 4 blocks
  // Barrier: 16 slots / 4 waves_per_block = 4 blocks
  // Wave: 8 waves / 4 = 2 blocks
  // HW: 2 blocks
  // Min = 2
  EXPECT_EQ(result.max_blocks_per_ap, 2);
}

TEST_F(OccupancyCalculatorTest, SgprLimiting) {
  KernelResourceUsage k = MakeLightKernel();
  k.sgpr_count = 128;  // 256 / 128 = 2 waves
  auto result = calc_.Calculate(k);
  EXPECT_EQ(result.max_waves_per_peu, 2);
  EXPECT_EQ(result.wave_limiting_factor, "sgpr");
}

TEST_F(OccupancyCalculatorTest, AgprCountsTowardVgprLimit) {
  KernelResourceUsage k = MakeLightKernel();
  k.vgpr_count = 64;
  k.agpr_count = 64;  // Total = 128, 256 / 128 = 2 waves
  auto result = calc_.Calculate(k);
  EXPECT_EQ(result.max_waves_per_peu, 2);
  EXPECT_EQ(result.wave_limiting_factor, "vgpr");
}

TEST_F(OccupancyCalculatorTest, VgprAllocGranule) {
  KernelResourceUsage k = MakeLightKernel();
  k.vgpr_count = 17;  // Aligned to granule=8 => 24, 256/24 = 10, capped at 8
  auto result = calc_.Calculate(k);
  EXPECT_EQ(result.max_waves_per_peu, 8);  // hw limit still wins
}

TEST_F(OccupancyCalculatorTest, SharedMemLimiting) {
  KernelResourceUsage k = MakeLightKernel();
  k.shared_memory_bytes = 49152;  // 65536 / 49152 = 1 block
  auto result = calc_.Calculate(k);
  EXPECT_EQ(result.max_blocks_per_ap, 1);
  EXPECT_EQ(result.block_limiting_factor, "shared_memory");
}

TEST_F(OccupancyCalculatorTest, PrivateMemoryLimiting) {
  KernelResourceUsage k = MakeLightKernel();
  k.private_memory_bytes = 2048;  // 4096 / 2048 = 2 waves
  auto result = calc_.Calculate(k);
  EXPECT_EQ(result.max_waves_per_peu, 2);
  EXPECT_EQ(result.wave_limiting_factor, "private_memory");
}

TEST_F(OccupancyCalculatorTest, DeviceOccupancy) {
  auto dev = calc_.CalculateDevice(MakeLightKernel());
  EXPECT_EQ(dev.total_ap, 8 * 13);   // 104
  EXPECT_EQ(dev.total_peu, 104 * 4);  // 416
  EXPECT_EQ(dev.total_active_waves, 8 * 416);
  EXPECT_EQ(dev.total_active_blocks, 2 * 104);
}

TEST_F(OccupancyCalculatorTest, SweepVgprCount) {
  auto results = calc_.SweepVgprCount(MakeLightKernel(), 8, 64);
  EXPECT_FALSE(results.empty());
  // At VGPR=8, should be hw-limited (8 waves)
  EXPECT_EQ(results[0].second.max_waves_per_peu, 8);
}

TEST_F(OccupancyCalculatorTest, SweepBlockSize) {
  auto results = calc_.SweepBlockSize(MakeLightKernel(), 64, 256);
  EXPECT_FALSE(results.empty());
  // Block size 256 => 4 waves/block
  auto& last = results.back();
  EXPECT_EQ(last.second.waves_per_block, 4);
}

TEST_F(OccupancyCalculatorTest, SweepSharedMemory) {
  auto results = calc_.SweepSharedMemory(MakeBarrierKernel(), 0, 65536, 8192);
  EXPECT_FALSE(results.empty());
  // At 0 shared memory, blocks limited by waves/hw
  EXPECT_EQ(results[0].second.max_blocks_per_ap, 2);
}

TEST_F(OccupancyCalculatorTest, ReportOutput) {
  std::ostringstream oss;
  OccupancyReportConfig config;
  PrintOccupancyReport(oss, *spec_, MakeLightKernel(), "test_kernel", config);
  std::string report = oss.str();
  EXPECT_NE(report.find("Occupancy Analysis"), std::string::npos);
  EXPECT_NE(report.find("test_kernel"), std::string::npos);
  EXPECT_NE(report.find("mac500"), std::string::npos);
  EXPECT_NE(report.find("100.0%"), std::string::npos);
}

TEST_F(OccupancyCalculatorTest, JsonOutput) {
  std::string json = OccupancyReportJson(*spec_, MakeLightKernel(), "test_kernel");
  EXPECT_NE(json.find("\"kernel\": \"test_kernel\""), std::string::npos);
  EXPECT_NE(json.find("\"arch\": \"mac500\""), std::string::npos);
  EXPECT_NE(json.find("\"max_waves_per_peu\": 8"), std::string::npos);
}

// ArchConfig loading tests
TEST(ArchConfigTest, LoadMac500FromJson) {
  auto spec = LoadArchConfig("configs/gpu_arch/mac500.json");
  ASSERT_NE(spec, nullptr);
  EXPECT_EQ(spec->name, "mac500");
  EXPECT_EQ(spec->dpc_count, 8);
  EXPECT_EQ(spec->ap_per_dpc, 13);
  EXPECT_EQ(spec->peu_per_ap, 4);
  EXPECT_EQ(spec->max_resident_waves_per_peu, 8);
}

TEST(ArchConfigTest, LoadFromJsonString) {
  const char* json_str = R"({
    "name": "test_arch",
    "wave_size": 64,
    "topology": {"dpc_count": 4, "ap_per_dpc": 10, "peu_per_ap": 2},
    "wave_slots": {"max_resident_waves_per_peu": 6, "max_issuable_waves_per_peu": 3},
    "registers": {"vgpr_count_per_peu": 512, "sgpr_count_per_peu": 256}
  })";
  auto spec = LoadArchConfigFromString(json_str);
  ASSERT_NE(spec, nullptr);
  EXPECT_EQ(spec->name, "test_arch");
  EXPECT_EQ(spec->dpc_count, 4);
  EXPECT_EQ(spec->vgpr_count_per_peu, 512);
}

TEST(ArchConfigTest, InvalidJsonReturnsNull) {
  auto spec = LoadArchConfigFromString("{invalid}");
  EXPECT_EQ(spec, nullptr);
}

TEST(ArchConfigTest, MissingFileReturnsNull) {
  auto spec = LoadArchConfig("/nonexistent/path.json");
  EXPECT_EQ(spec, nullptr);
}

TEST(ArchConfigTest, JsonConfigMatchesBuiltIn) {
  auto from_file = LoadArchConfig("configs/gpu_arch/mac500.json");
  auto from_registry = ArchRegistry::Get("mac500");
  ASSERT_NE(from_file, nullptr);
  ASSERT_NE(from_registry, nullptr);
  EXPECT_EQ(from_file->name, from_registry->name);
  EXPECT_EQ(from_file->dpc_count, from_registry->dpc_count);
  EXPECT_EQ(from_file->ap_per_dpc, from_registry->ap_per_dpc);
  EXPECT_EQ(from_file->peu_per_ap, from_registry->peu_per_ap);
  EXPECT_EQ(from_file->max_resident_waves_per_peu,
            from_registry->max_resident_waves_per_peu);
  EXPECT_EQ(from_file->vgpr_count_per_peu, from_registry->vgpr_count_per_peu);
  EXPECT_EQ(from_file->cache_model.l1_hit_latency,
            from_registry->cache_model.l1_hit_latency);
}

TEST(ArchConfigTest, PartialConfigUsesDefaults) {
  const char* json_str = R"({
    "name": "minimal",
    "topology": {"dpc_count": 2, "ap_per_dpc": 4, "peu_per_ap": 2}
  })";
  auto spec = LoadArchConfigFromString(json_str);
  ASSERT_NE(spec, nullptr);
  EXPECT_EQ(spec->name, "minimal");
  EXPECT_EQ(spec->dpc_count, 2);
  // Unspecified fields should have defaults
  EXPECT_EQ(spec->vgpr_count_per_peu, 256);
  EXPECT_EQ(spec->shared_load_latency, 4);
}

TEST(ArchConfigTest, OccupancyWithLoadedConfig) {
  auto spec = LoadArchConfig("configs/gpu_arch/mac500.json");
  ASSERT_NE(spec, nullptr);
  OccupancyCalculator calc(*spec);
  auto result = calc.Calculate(MakeLightKernel());
  EXPECT_EQ(result.max_waves_per_peu, 8);
  EXPECT_FLOAT_EQ(result.occupancy_ratio, 1.0f);
}

TEST(ArchConfigTest, Gfx90aConfig) {
  auto spec = LoadArchConfig("configs/gpu_arch/gfx90a.json");
  ASSERT_NE(spec, nullptr);
  EXPECT_EQ(spec->name, "gfx90a");
  EXPECT_EQ(spec->dpc_count, 8);
  EXPECT_EQ(spec->ap_per_dpc, 14);
  EXPECT_EQ(spec->max_resident_waves_per_peu, 10);
  EXPECT_EQ(spec->sgpr_count_per_peu, 512);
}

TEST(ArchConfigTest, Gfx90aOccupancy) {
  auto spec = LoadArchConfig("configs/gpu_arch/gfx90a.json");
  ASSERT_NE(spec, nullptr);
  OccupancyCalculator calc(*spec);
  // gfx90a: 10 wave slots, 2 blocks * 4 waves/block = 8 active waves
  // occupancy = 8/10 = 0.8
  auto result = calc.Calculate(MakeLightKernel());
  EXPECT_EQ(result.max_waves_per_peu, 10);
  EXPECT_EQ(result.active_waves_per_peu, 8);
  EXPECT_FLOAT_EQ(result.occupancy_ratio, 0.8f);
}

}  // namespace
}  // namespace gpu_model
