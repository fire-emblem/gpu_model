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

}  // namespace
}  // namespace gpu_model
