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
      .operands = "",
      .decoded_operands = {},
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
      .operands = "",
      .decoded_operands = {},
  };

  const auto decoded = GcnInstDecoder{}.Decode(raw);
  EXPECT_EQ(decoded.encoding_id, 10u);
  EXPECT_EQ(decoded.mnemonic, "s_cbranch_execz");
  ASSERT_EQ(decoded.operands.size(), 1u);
  EXPECT_EQ(decoded.operands[0].text, "25");
  EXPECT_EQ(decoded.operands[0].kind, DecodedGcnOperandKind::BranchTarget);
  EXPECT_TRUE(decoded.operands[0].info.has_immediate);
  EXPECT_EQ(decoded.operands[0].info.immediate, 25);
}

TEST(GcnInstDecoderTest, DecodesRepresentativeWaitcntInstruction) {
  RawGcnInstruction raw{
      .pc = 0x1910,
      .size_bytes = 4,
      .words = {0xbf8cc07fu},
      .format_class = GcnInstFormatClass::Sopp,
      .mnemonic = "s_waitcnt",
      .operands = "",
      .decoded_operands = {},
  };

  const auto decoded = GcnInstDecoder{}.Decode(raw);
  EXPECT_EQ(decoded.encoding_id, 12u);
  EXPECT_EQ(decoded.mnemonic, "s_waitcnt");
  ASSERT_EQ(decoded.operands.size(), 1u);
  EXPECT_EQ(decoded.operands[0].text, "lgkmcnt(0)");
  EXPECT_TRUE(decoded.operands[0].info.has_waitcnt);
  EXPECT_EQ(decoded.operands[0].info.wait_lgkmcnt, 0u);
  EXPECT_EQ(decoded.operands[0].info.wait_vmcnt, 15u);
  EXPECT_EQ(decoded.operands[0].info.wait_expcnt, 7u);
}

TEST(GcnInstDecoderTest, DecodesRepresentativeVectorMoveInstruction) {
  RawGcnInstruction raw{
      .pc = 0x1950,
      .size_bytes = 4,
      .words = {0x7e060203u},
      .format_class = GcnInstFormatClass::Vop1,
      .mnemonic = "v_mov_b32_e32",
      .operands = "",
      .decoded_operands = {},
  };

  const auto decoded = GcnInstDecoder{}.Decode(raw);
  EXPECT_EQ(decoded.encoding_id, 13u);
  EXPECT_EQ(decoded.mnemonic, "v_mov_b32_e32");
  ASSERT_EQ(decoded.operands.size(), 2u);
  EXPECT_EQ(decoded.operands[0].text, "v3");
}

TEST(GcnInstDecoderTest, DecodesRepresentativeScalarMoveLiteralInstruction) {
  RawGcnInstruction raw{
      .pc = 0x1954,
      .size_bytes = 8,
      .words = {0xbe8000ffu, 0x0000002au},
      .format_class = GcnInstFormatClass::Sop1,
      .mnemonic = "s_mov_b32",
      .operands = "",
      .decoded_operands = {},
  };

  const auto decoded = GcnInstDecoder{}.Decode(raw);
  EXPECT_EQ(decoded.encoding_id, 54u);
  EXPECT_EQ(decoded.mnemonic, "s_mov_b32");
  ASSERT_EQ(decoded.operands.size(), 2u);
  EXPECT_EQ(decoded.operands[0].text, "s0");
  EXPECT_EQ(decoded.operands[1].text, "0x2a");
}

TEST(GcnInstDecoderTest, DecodesRepresentativeScalarMovkInstruction) {
  RawGcnInstruction raw{
      .pc = 0x1958,
      .size_bytes = 4,
      .words = {0xb000002au},
      .format_class = GcnInstFormatClass::Sopk,
      .mnemonic = "s_movk_i32",
      .operands = "",
      .decoded_operands = {},
  };

  const auto decoded = GcnInstDecoder{}.Decode(raw);
  EXPECT_EQ(decoded.encoding_id, 78u);
  EXPECT_EQ(decoded.mnemonic, "s_movk_i32");
  ASSERT_EQ(decoded.operands.size(), 2u);
  EXPECT_EQ(decoded.operands[0].text, "s0");
  EXPECT_EQ(decoded.operands[1].text, "42");
}

TEST(GcnInstDecoderTest, DecodesRepresentativeGlobalLoadInstruction) {
  RawGcnInstruction raw{
      .pc = 0x1968,
      .size_bytes = 8,
      .words = {0xdc508000u, 0x067f0004u},
      .format_class = GcnInstFormatClass::Flat,
      .mnemonic = "global_load_dword",
      .operands = "",
      .decoded_operands = {},
  };

  const auto decoded = GcnInstDecoder{}.Decode(raw);
  EXPECT_EQ(decoded.encoding_id, 18u);
  EXPECT_EQ(decoded.mnemonic, "global_load_dword");
  ASSERT_EQ(decoded.operands.size(), 3u);
  EXPECT_EQ(decoded.operands[1].kind, DecodedGcnOperandKind::VectorRegRange);
  EXPECT_EQ(decoded.operands[1].text, "v[4:5]");
  EXPECT_EQ(decoded.operands[1].info.reg_first, 4u);
  EXPECT_EQ(decoded.operands[1].info.reg_count, 2u);
  EXPECT_EQ(decoded.operands[0].text, "v6");
  EXPECT_EQ(decoded.operands[2].text, "off");
  EXPECT_TRUE(decoded.operands[2].info.has_immediate);
  EXPECT_EQ(decoded.operands[2].info.immediate, 0);
}

TEST(GcnInstDecoderTest, DecodesRepresentativeGlobalStoreInstruction) {
  RawGcnInstruction raw{
      .pc = 0x1970,
      .size_bytes = 8,
      .words = {0xdc708000u, 0x047f0405u},
      .format_class = GcnInstFormatClass::Flat,
      .mnemonic = "global_store_dword",
      .operands = "",
      .decoded_operands = {},
  };

  const auto decoded = GcnInstDecoder{}.Decode(raw);
  EXPECT_EQ(decoded.encoding_id, 19u);
  EXPECT_EQ(decoded.mnemonic, "global_store_dword");
  ASSERT_EQ(decoded.operands.size(), 3u);
  EXPECT_EQ(decoded.operands[0].kind, DecodedGcnOperandKind::VectorRegRange);
  EXPECT_EQ(decoded.operands[0].text, "v[5:6]");
  EXPECT_EQ(decoded.operands[1].text, "v4");
  EXPECT_EQ(decoded.operands[2].text, "off");
}

TEST(GcnInstDecoderTest, DecodesRepresentativeVop3aFmaInstruction) {
  RawGcnInstruction raw{
      .pc = 0x1a00,
      .size_bytes = 8,
      .words = {0xd1cb0002u, 0x04140503u},
      .format_class = GcnInstFormatClass::Vop3a,
      .mnemonic = "v_fma_f32",
      .operands = "",
      .decoded_operands = {},
  };

  const auto decoded = GcnInstDecoder{}.Decode(raw);
  EXPECT_EQ(decoded.encoding_id, 25u);
  EXPECT_EQ(decoded.mnemonic, "v_fma_f32");
  ASSERT_EQ(decoded.operands.size(), 4u);
  EXPECT_EQ(decoded.operands[0].text, "v2");
  EXPECT_EQ(decoded.operands[1].text, "v3");
  EXPECT_EQ(decoded.operands[2].text, "s2");
  EXPECT_EQ(decoded.operands[3].text, "v5");
}

TEST(GcnInstDecoderTest, DecodesRepresentativeCarryProducingVectorAddInstruction) {
  RawGcnInstruction raw{
      .pc = 0x1a20,
      .size_bytes = 4,
      .words = {0x32060401u},
      .format_class = GcnInstFormatClass::Vop2,
      .mnemonic = "v_add_co_u32_e32",
      .operands = "",
      .decoded_operands = {},
  };

  const auto decoded = GcnInstDecoder{}.Decode(raw);
  EXPECT_EQ(decoded.encoding_id, 15u);
  EXPECT_EQ(decoded.mnemonic, "v_add_co_u32_e32");
  ASSERT_EQ(decoded.operands.size(), 4u);
  EXPECT_EQ(decoded.operands[0].text, "v3");
  EXPECT_EQ(decoded.operands[1].text, "vcc");
  EXPECT_EQ(decoded.operands[2].text, "s1");
  EXPECT_EQ(decoded.operands[3].text, "v2");
}

TEST(GcnInstDecoderTest, DecodesRepresentativeCarryConsumingVectorAddInstruction) {
  RawGcnInstruction raw{
      .pc = 0x1a24,
      .size_bytes = 4,
      .words = {0x38060401u},
      .format_class = GcnInstFormatClass::Vop2,
      .mnemonic = "v_addc_co_u32_e32",
      .operands = "",
      .decoded_operands = {},
  };

  const auto decoded = GcnInstDecoder{}.Decode(raw);
  EXPECT_EQ(decoded.encoding_id, 16u);
  EXPECT_EQ(decoded.mnemonic, "v_addc_co_u32_e32");
  ASSERT_EQ(decoded.operands.size(), 5u);
  EXPECT_EQ(decoded.operands[0].text, "v3");
  EXPECT_EQ(decoded.operands[1].text, "vcc");
  EXPECT_EQ(decoded.operands[2].text, "s1");
  EXPECT_EQ(decoded.operands[3].text, "v2");
  EXPECT_EQ(decoded.operands[4].text, "vcc");
}

TEST(GcnInstDecoderTest, DecodesRepresentativeVop3CarryE64Instruction) {
  RawGcnInstruction raw{
      .pc = 0x1a30,
      .size_bytes = 8,
      .words = {0xd1180402u, 0x00000a03u},
      .format_class = GcnInstFormatClass::Vop3a,
      .mnemonic = "v_add_co_u32_e64",
      .operands = "",
      .decoded_operands = {},
  };

  const auto decoded = GcnInstDecoder{}.Decode(raw);
  EXPECT_EQ(decoded.encoding_id, 35u);
  EXPECT_EQ(decoded.mnemonic, "v_add_co_u32_e64");
  ASSERT_EQ(decoded.operands.size(), 4u);
  EXPECT_EQ(decoded.operands[0].text, "v2");
  EXPECT_EQ(decoded.operands[1].text, "s[4:5]");
  EXPECT_EQ(decoded.operands[2].text, "s3");
  EXPECT_EQ(decoded.operands[3].text, "s5");
}

TEST(GcnInstDecoderTest, DecodesRepresentativeVop3CndmaskE64Instruction) {
  RawGcnInstruction raw{
      .pc = 0x1a38,
      .size_bytes = 8,
      .words = {0xd1000001u, 0x00100002u},
      .format_class = GcnInstFormatClass::Vop3a,
      .mnemonic = "v_cndmask_b32_e64",
      .operands = "",
      .decoded_operands = {},
  };

  const auto decoded = GcnInstDecoder{}.Decode(raw);
  EXPECT_EQ(decoded.encoding_id, 59u);
  EXPECT_EQ(decoded.mnemonic, "v_cndmask_b32_e64");
  ASSERT_EQ(decoded.operands.size(), 4u);
  EXPECT_EQ(decoded.operands[0].text, "v1");
  EXPECT_EQ(decoded.operands[1].text, "s2");
  EXPECT_EQ(decoded.operands[2].text, "s0");
  EXPECT_EQ(decoded.operands[3].text, "s[4:5]");
}

}  // namespace
}  // namespace gpu_model
