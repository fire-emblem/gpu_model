#include <gtest/gtest.h>

#include "gpu_model/execution/internal/tensor_op_utils.h"

namespace gpu_model {
namespace {

TEST(TensorOpUtilsTest, DetectsTensorMnemonics) {
  EXPECT_TRUE(IsTensorMnemonic("v_mfma_f32_16x16x4f32"));
  EXPECT_TRUE(IsTensorMnemonic("v_accvgpr_read_b32"));
  EXPECT_FALSE(IsTensorMnemonic("v_add_f32_e32"));
}

TEST(TensorOpUtilsTest, MirrorPolicyWritesBothVgprAndAgpr) {
  WaveContext wave;

  WriteTensorResultRange(
      wave, 8, 2, 3, 0x12345678u, TensorResultStoragePolicy::MirrorToVgprAndAgpr);

  EXPECT_EQ(wave.vgpr.Read(8, 3), 0x12345678u);
  EXPECT_EQ(wave.vgpr.Read(9, 3), 0x12345678u);
  EXPECT_EQ(wave.agpr.Read(8, 3), 0x12345678u);
  EXPECT_EQ(wave.agpr.Read(9, 3), 0x12345678u);
}

TEST(TensorOpUtilsTest, AgprOnlyPolicyLeavesVgprUntouched) {
  WaveContext wave;
  wave.vgpr.Write(4, 1, 0xaabbccddu);

  WriteTensorResultRange(
      wave, 4, 1, 1, 0x55667788u, TensorResultStoragePolicy::AgprOnly);

  EXPECT_EQ(wave.vgpr.Read(4, 1), 0xaabbccddu);
  EXPECT_EQ(wave.agpr.Read(4, 1), 0x55667788u);
}

}  // namespace
}  // namespace gpu_model
