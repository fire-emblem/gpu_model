#include <gtest/gtest.h>

#include "gpu_model/exec/execution_memory_ops.h"

namespace gpu_model {
namespace {

TEST(ExecutionMemoryOpsTest, LoadsAndStoresByteAddressableMemory) {
  std::vector<std::byte> memory(16, std::byte{0});
  execution_memory_ops::StoreByteLaneValue(memory, LaneAccess{
                                                       .active = true,
                                                       .addr = 4,
                                                       .bytes = 4,
                                                       .value = 0x11223344u,
                                                   });
  EXPECT_EQ(execution_memory_ops::LoadByteLaneValue(memory, LaneAccess{
                                                               .active = true,
                                                               .addr = 4,
                                                               .bytes = 4,
                                                           }),
            0x11223344u);
}

TEST(ExecutionMemoryOpsTest, PrivateLaneLoadCanOptionallyExtendMissingStorage) {
  std::array<std::vector<std::byte>, kWaveSize> private_memory;

  EXPECT_EQ(execution_memory_ops::LoadPrivateLaneValue(private_memory, 0, LaneAccess{
                                                                            .active = true,
                                                                            .addr = 8,
                                                                            .bytes = 4,
                                                                        }, false),
            0u);
  EXPECT_TRUE(private_memory[0].empty());

  EXPECT_EQ(execution_memory_ops::LoadPrivateLaneValue(private_memory, 0, LaneAccess{
                                                                            .active = true,
                                                                            .addr = 8,
                                                                            .bytes = 4,
                                                                        }, true),
            0u);
  EXPECT_EQ(private_memory[0].size(), 12u);
}

}  // namespace
}  // namespace gpu_model
