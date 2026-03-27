#include <gtest/gtest.h>

#include "gpu_model/decode/gcn_inst_decoder.h"

namespace gpu_model {
namespace {

TEST(GcnInstDecoderTest, DecodesRepresentativeScalarMemoryInstruction) {
  RawGcnInstruction raw{
      .pc = 0x1900,
      .size_bytes = 8,
      .words = {0xc0020002u, 0x0000002cu},
      .format_class = GcnInstFormatClass::Smrd,
      .mnemonic = "s_load_dword",
  };

  const auto decoded = GcnInstDecoder{}.Decode(raw);
  EXPECT_EQ(decoded.encoding_id, 2u);
  EXPECT_EQ(decoded.mnemonic, "s_load_dword");
  EXPECT_EQ(decoded.format_class, GcnInstFormatClass::Smrd);
  ASSERT_EQ(decoded.operands.size(), 3u);
}

TEST(GcnInstDecoderTest, DecodesRepresentativeBranchInstruction) {
  RawGcnInstruction raw{
      .pc = 0x192c,
      .size_bytes = 4,
      .words = {0xbf880019u},
      .format_class = GcnInstFormatClass::Sopp,
      .mnemonic = "s_cbranch_execz",
  };

  const auto decoded = GcnInstDecoder{}.Decode(raw);
  EXPECT_EQ(decoded.encoding_id, 10u);
  EXPECT_EQ(decoded.mnemonic, "s_cbranch_execz");
  ASSERT_EQ(decoded.operands.size(), 1u);
  EXPECT_EQ(decoded.operands[0].text, "25");
}

TEST(GcnInstDecoderTest, DecodesRepresentativeWaitcntInstruction) {
  RawGcnInstruction raw{
      .pc = 0x1910,
      .size_bytes = 4,
      .words = {0xbf8cc07fu},
      .format_class = GcnInstFormatClass::Sopp,
      .mnemonic = "s_waitcnt",
  };

  const auto decoded = GcnInstDecoder{}.Decode(raw);
  EXPECT_EQ(decoded.encoding_id, 12u);
  EXPECT_EQ(decoded.mnemonic, "s_waitcnt");
  ASSERT_EQ(decoded.operands.size(), 1u);
  EXPECT_EQ(decoded.operands[0].text, "lgkmcnt(64)");
}

TEST(GcnInstDecoderTest, DecodesRepresentativeVectorMoveInstruction) {
  RawGcnInstruction raw{
      .pc = 0x1950,
      .size_bytes = 4,
      .words = {0x7e060203u},
      .format_class = GcnInstFormatClass::Vop1,
      .mnemonic = "v_mov_b32_e32",
  };

  const auto decoded = GcnInstDecoder{}.Decode(raw);
  EXPECT_EQ(decoded.encoding_id, 13u);
  EXPECT_EQ(decoded.mnemonic, "v_mov_b32_e32");
  ASSERT_EQ(decoded.operands.size(), 2u);
  EXPECT_EQ(decoded.operands[0].text, "v3");
}

}  // namespace
}  // namespace gpu_model
