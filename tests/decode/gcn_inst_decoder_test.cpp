#include <gtest/gtest.h>

#include "gpu_model/decode/gcn_inst_decoder.h"

namespace gpu_model {
namespace {

uint32_t EncodeSoppWord(uint32_t opcode, int16_t simm16) {
  return 0xbf800000u | (opcode << 16u) | static_cast<uint16_t>(simm16);
}

uint32_t EncodeViSmrdWord(uint32_t opcode, uint32_t sdst_first, uint32_t sbase_first) {
  return 0xc0000000u | (((opcode >> 5u) & 0x3u) << 18u) | (1u << 17u) | (sdst_first << 6u) |
         ((sbase_first >> 1u) & 0x3fu);
}

uint32_t EncodeVop2Word(uint32_t opcode, uint32_t vdst, uint32_t src0, uint32_t vsrc1) {
  return (opcode << 25u) | (vdst << 17u) | (vsrc1 << 9u) | src0;
}

uint32_t EncodeVop1Word(uint32_t opcode, uint32_t vdst, uint32_t src0) {
  return 0x7e000000u | (vdst << 17u) | (opcode << 9u) | src0;
}

uint32_t EncodeVopcWord(uint32_t opcode, uint32_t src0, uint32_t vsrc1) {
  return 0x7c000000u | (opcode << 17u) | (vsrc1 << 9u) | src0;
}

uint32_t EncodeDsWord0(uint32_t opcode, uint32_t offset0 = 0, uint32_t offset1 = 0) {
  return 0xd8000000u | (opcode << 17u) | (offset1 << 8u) | offset0;
}

uint32_t EncodeDsWord1(uint32_t addr, uint32_t data0, uint32_t data1, uint32_t vdst) {
  return (vdst << 24u) | (data1 << 16u) | (data0 << 8u) | addr;
}

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

TEST(GcnInstDecoderTest, DecodesNoOperandTerminationInstruction) {
  RawGcnInstruction raw{
      .pc = 0x1904,
      .size_bytes = 4,
      .words = {0xbf810000u},
      .format_class = GcnInstFormatClass::Sopp,
      .mnemonic = "s_endpgm",
      .operands = "",
      .decoded_operands = {},
  };

  const auto decoded = GcnInstDecoder{}.Decode(raw);
  EXPECT_EQ(decoded.encoding_id, 1u);
  EXPECT_EQ(decoded.mnemonic, "s_endpgm");
  EXPECT_TRUE(decoded.operands.empty());
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

TEST(GcnInstDecoderTest, DecodesRepresentativeScalarShiftLeftB64Instruction) {
  RawGcnInstruction raw{
      .pc = 0x195c,
      .size_bytes = 4,
      .words = {0x8e848101u},
      .format_class = GcnInstFormatClass::Sop2,
      .mnemonic = "s_lshl_b64",
      .operands = "",
      .decoded_operands = {},
  };

  const auto decoded = GcnInstDecoder{}.Decode(raw);
  EXPECT_EQ(decoded.encoding_id, 73u);
  EXPECT_EQ(decoded.mnemonic, "s_lshl_b64");
  ASSERT_EQ(decoded.operands.size(), 3u);
  EXPECT_EQ(decoded.operands[0].text, "s[4:5]");
  EXPECT_EQ(decoded.operands[1].text, "s[1:2]");
  EXPECT_EQ(decoded.operands[2].text, "1");
}

TEST(GcnInstDecoderTest, DecodesRepresentativeScalarMemoryRangeInstructions) {
  {
    RawGcnInstruction raw{
        .pc = 0x1960,
        .size_bytes = 8,
        .words = {EncodeViSmrdWord(/*opcode=*/32, /*sdst_first=*/6, /*sbase_first=*/4), 0x10u},
        .format_class = GcnInstFormatClass::Smrd,
        .mnemonic = "s_load_dwordx2",
        .operands = "",
        .decoded_operands = {},
    };

    const auto decoded = GcnInstDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 3u);
    EXPECT_EQ(decoded.mnemonic, "s_load_dwordx2");
    ASSERT_EQ(decoded.operands.size(), 3u);
    EXPECT_EQ(decoded.operands[0].text, "s[6:7]");
    EXPECT_EQ(decoded.operands[1].text, "s[4:5]");
    EXPECT_EQ(decoded.operands[2].text, "0x10");
  }

  {
    RawGcnInstruction raw{
        .pc = 0x1968,
        .size_bytes = 8,
        .words = {EncodeViSmrdWord(/*opcode=*/64, /*sdst_first=*/0, /*sbase_first=*/4), 0x20u},
        .format_class = GcnInstFormatClass::Smrd,
        .mnemonic = "s_load_dwordx4",
        .operands = "",
        .decoded_operands = {},
    };

    const auto decoded = GcnInstDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 4u);
    EXPECT_EQ(decoded.mnemonic, "s_load_dwordx4");
    ASSERT_EQ(decoded.operands.size(), 3u);
    EXPECT_EQ(decoded.operands[0].text, "s[0:3]");
    EXPECT_EQ(decoded.operands[1].text, "s[4:5]");
    EXPECT_EQ(decoded.operands[2].text, "0x20");
  }
}

TEST(GcnInstDecoderTest, DecodesAdditionalScalarControlInstructions) {
  {
    RawGcnInstruction raw{
        .pc = 0x1978,
        .size_bytes = 4,
        .words = {EncodeSoppWord(/*opcode=*/2, /*simm16=*/5)},
        .format_class = GcnInstFormatClass::Sopp,
        .mnemonic = "s_branch",
        .operands = "",
        .decoded_operands = {},
    };

    const auto decoded = GcnInstDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 27u);
    EXPECT_EQ(decoded.mnemonic, "s_branch");
    ASSERT_EQ(decoded.operands.size(), 1u);
    EXPECT_EQ(decoded.operands[0].text, "5");
  }

  {
    RawGcnInstruction raw{
        .pc = 0x197c,
        .size_bytes = 4,
        .words = {EncodeSoppWord(/*opcode=*/5, /*simm16=*/3)},
        .format_class = GcnInstFormatClass::Sopp,
        .mnemonic = "s_cbranch_scc1",
        .operands = "",
        .decoded_operands = {},
    };

    const auto decoded = GcnInstDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 22u);
    EXPECT_EQ(decoded.mnemonic, "s_cbranch_scc1");
    ASSERT_EQ(decoded.operands.size(), 1u);
    EXPECT_EQ(decoded.operands[0].text, "3");
  }

  {
    RawGcnInstruction raw{
        .pc = 0x1980,
        .size_bytes = 4,
        .words = {EncodeSoppWord(/*opcode=*/10, /*simm16=*/0)},
        .format_class = GcnInstFormatClass::Sopp,
        .mnemonic = "s_barrier",
        .operands = "",
        .decoded_operands = {},
    };

    const auto decoded = GcnInstDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 29u);
    EXPECT_EQ(decoded.mnemonic, "s_barrier");
    ASSERT_EQ(decoded.operands.size(), 1u);
    EXPECT_EQ(decoded.operands[0].text, "barrier");
  }
}

TEST(GcnInstDecoderTest, DecodesAdditionalScalarCompareInstructions) {
  {
    RawGcnInstruction raw{
        .pc = 0x1984,
        .size_bytes = 4,
        .words = {0xbf060201u},
        .format_class = GcnInstFormatClass::Sopc,
        .mnemonic = "s_cmp_eq_u32",
        .operands = "",
        .decoded_operands = {},
    };

    const auto decoded = GcnInstDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 24u);
    EXPECT_EQ(decoded.mnemonic, "s_cmp_eq_u32");
    ASSERT_EQ(decoded.operands.size(), 2u);
    EXPECT_EQ(decoded.operands[0].text, "s1");
    EXPECT_EQ(decoded.operands[1].text, "s2");
  }

  {
    RawGcnInstruction raw{
        .pc = 0x1988,
        .size_bytes = 4,
        .words = {0xbf080201u},
        .format_class = GcnInstFormatClass::Sopc,
        .mnemonic = "s_cmp_gt_u32",
        .operands = "",
        .decoded_operands = {},
    };

    const auto decoded = GcnInstDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 39u);
    EXPECT_EQ(decoded.mnemonic, "s_cmp_gt_u32");
    ASSERT_EQ(decoded.operands.size(), 2u);
    EXPECT_EQ(decoded.operands[0].text, "s1");
    EXPECT_EQ(decoded.operands[1].text, "s2");
  }

  {
    RawGcnInstruction raw{
        .pc = 0x198c,
        .size_bytes = 4,
        .words = {0xbf0a0201u},
        .format_class = GcnInstFormatClass::Sopc,
        .mnemonic = "s_cmp_lt_u32",
        .operands = "",
        .decoded_operands = {},
    };

    const auto decoded = GcnInstDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 40u);
    EXPECT_EQ(decoded.mnemonic, "s_cmp_lt_u32");
    ASSERT_EQ(decoded.operands.size(), 2u);
    EXPECT_EQ(decoded.operands[0].text, "s1");
    EXPECT_EQ(decoded.operands[1].text, "s2");
  }
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

TEST(GcnInstDecoderTest, DecodesAdditionalVectorAndLdsInstructions) {
  {
    RawGcnInstruction raw{
        .pc = 0x1a10,
        .size_bytes = 4,
        .words = {EncodeVop2Word(/*opcode=*/1, /*vdst=*/2, /*src0=*/0x100u + 6u, /*vsrc1=*/7u)},
        .format_class = GcnInstFormatClass::Vop2,
        .mnemonic = "v_add_f32_e32",
        .operands = "",
        .decoded_operands = {},
    };

    const auto decoded = GcnInstDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 11u);
    EXPECT_EQ(decoded.mnemonic, "v_add_f32_e32");
    ASSERT_EQ(decoded.operands.size(), 3u);
    EXPECT_EQ(decoded.operands[0].text, "v2");
    EXPECT_EQ(decoded.operands[1].text, "v6");
    EXPECT_EQ(decoded.operands[2].text, "v7");
  }

  {
    RawGcnInstruction raw{
        .pc = 0x1a14,
        .size_bytes = 4,
        .words = {EncodeVop2Word(/*opcode=*/5, /*vdst=*/4, /*src0=*/0x100u + 1u, /*vsrc1=*/2u)},
        .format_class = GcnInstFormatClass::Vop2,
        .mnemonic = "v_mul_f32_e32",
        .operands = "",
        .decoded_operands = {},
    };

    const auto decoded = GcnInstDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.mnemonic, "v_mul_f32_e32");
    ASSERT_EQ(decoded.operands.size(), 3u);
    EXPECT_EQ(decoded.operands[0].text, "v4");
    EXPECT_EQ(decoded.operands[1].text, "v1");
    EXPECT_EQ(decoded.operands[2].text, "v2");
  }

  {
    RawGcnInstruction raw{
        .pc = 0x1a18,
        .size_bytes = 4,
        .words = {EncodeVopcWord(/*opcode=*/202, /*src0=*/0x100u + 2u, /*vsrc1=*/3u)},
        .format_class = GcnInstFormatClass::Vopc,
        .mnemonic = "v_cmp_eq_u32_e32",
        .operands = "",
        .decoded_operands = {},
    };

    const auto decoded = GcnInstDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 66u);
    EXPECT_EQ(decoded.mnemonic, "v_cmp_eq_u32_e32");
    ASSERT_EQ(decoded.operands.size(), 3u);
    EXPECT_EQ(decoded.operands[0].text, "vcc");
    EXPECT_EQ(decoded.operands[1].text, "v2");
    EXPECT_EQ(decoded.operands[2].text, "v3");
  }

  {
    RawGcnInstruction raw{
        .pc = 0x1a1c,
        .size_bytes = 4,
        .words = {EncodeVop1Word(/*opcode=*/5, /*vdst=*/1, /*src0=*/0x100u + 4u)},
        .format_class = GcnInstFormatClass::Vop1,
        .mnemonic = "v_cvt_f32_i32_e32",
        .operands = "",
        .decoded_operands = {},
    };

    const auto decoded = GcnInstDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 80u);
    EXPECT_EQ(decoded.mnemonic, "v_cvt_f32_i32_e32");
    ASSERT_EQ(decoded.operands.size(), 2u);
    EXPECT_EQ(decoded.operands[0].text, "v1");
    EXPECT_EQ(decoded.operands[1].text, "v4");
  }

  {
    RawGcnInstruction raw{
        .pc = 0x1a1e,
        .size_bytes = 4,
        .words = {EncodeVopcWord(/*opcode=*/196, /*src0=*/0x100u + 2u, /*vsrc1=*/3u)},
        .format_class = GcnInstFormatClass::Vopc,
        .mnemonic = "v_cmp_gt_i32_e32",
        .operands = "",
        .decoded_operands = {},
    };

    const auto decoded = GcnInstDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 8u);
    EXPECT_EQ(decoded.mnemonic, "v_cmp_gt_i32_e32");
    ASSERT_EQ(decoded.operands.size(), 3u);
    EXPECT_EQ(decoded.operands[0].text, "vcc");
    EXPECT_EQ(decoded.operands[1].text, "v2");
    EXPECT_EQ(decoded.operands[2].text, "v3");
  }

  {
    RawGcnInstruction raw{
        .pc = 0x1a40,
        .size_bytes = 8,
        .words = {EncodeDsWord0(/*opcode=*/13), EncodeDsWord1(/*addr=*/4, /*data0=*/5, /*data1=*/0, /*vdst=*/0)},
        .format_class = GcnInstFormatClass::Ds,
        .mnemonic = "ds_write_b32",
        .operands = "",
        .decoded_operands = {},
    };

    const auto decoded = GcnInstDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 30u);
    EXPECT_EQ(decoded.mnemonic, "ds_write_b32");
    ASSERT_EQ(decoded.operands.size(), 2u);
    EXPECT_EQ(decoded.operands[0].text, "v4");
    EXPECT_EQ(decoded.operands[1].text, "v5");
  }

  {
    RawGcnInstruction raw{
        .pc = 0x1a48,
        .size_bytes = 8,
        .words = {EncodeDsWord0(/*opcode=*/54), EncodeDsWord1(/*addr=*/4, /*data0=*/0, /*data1=*/0, /*vdst=*/6)},
        .format_class = GcnInstFormatClass::Ds,
        .mnemonic = "ds_read_b32",
        .operands = "",
        .decoded_operands = {},
    };

    const auto decoded = GcnInstDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 31u);
    EXPECT_EQ(decoded.mnemonic, "ds_read_b32");
    ASSERT_EQ(decoded.operands.size(), 2u);
    EXPECT_EQ(decoded.operands[0].text, "v6");
    EXPECT_EQ(decoded.operands[1].text, "v4");
  }

  {
    RawGcnInstruction raw{
        .pc = 0x1a4c,
        .size_bytes = 4,
        .words = {EncodeVopcWord(/*opcode=*/204, /*src0=*/0x100u + 4u, /*vsrc1=*/5u)},
        .format_class = GcnInstFormatClass::Vopc,
        .mnemonic = "v_cmp_gt_u32_e32",
        .operands = "",
        .decoded_operands = {},
    };

    const auto decoded = GcnInstDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 56u);
    EXPECT_EQ(decoded.mnemonic, "v_cmp_gt_u32_e32");
    ASSERT_EQ(decoded.operands.size(), 3u);
    EXPECT_EQ(decoded.operands[1].text, "v4");
    EXPECT_EQ(decoded.operands[2].text, "v5");
  }

  {
    RawGcnInstruction raw{
        .pc = 0x1a50,
        .size_bytes = 4,
        .words = {EncodeVopcWord(/*opcode=*/195, /*src0=*/0x100u + 6u, /*vsrc1=*/7u)},
        .format_class = GcnInstFormatClass::Vopc,
        .mnemonic = "v_cmp_le_i32_e32",
        .operands = "",
        .decoded_operands = {},
    };

    const auto decoded = GcnInstDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 75u);
    EXPECT_EQ(decoded.mnemonic, "v_cmp_le_i32_e32");
    ASSERT_EQ(decoded.operands.size(), 3u);
    EXPECT_EQ(decoded.operands[1].text, "v6");
    EXPECT_EQ(decoded.operands[2].text, "v7");
  }

  {
    RawGcnInstruction raw{
        .pc = 0x1a54,
        .size_bytes = 4,
        .words = {EncodeVopcWord(/*opcode=*/193, /*src0=*/0x100u + 8u, /*vsrc1=*/9u)},
        .format_class = GcnInstFormatClass::Vopc,
        .mnemonic = "v_cmp_lt_i32_e32",
        .operands = "",
        .decoded_operands = {},
    };

    const auto decoded = GcnInstDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 76u);
    EXPECT_EQ(decoded.mnemonic, "v_cmp_lt_i32_e32");
    ASSERT_EQ(decoded.operands.size(), 3u);
    EXPECT_EQ(decoded.operands[1].text, "v8");
    EXPECT_EQ(decoded.operands[2].text, "v9");
  }
}

TEST(GcnInstDecoderTest, FallsBackToGeneratedNameForReservedPlaceholderFamilies) {
  {
    RawGcnInstruction raw{
        .pc = 0x1b00,
        .size_bytes = 8,
        .words = {0xe0500000u, 0x00000000u},
        .format_class = GcnInstFormatClass::Mubuf,
        .mnemonic = "unknown",
        .operands = "",
        .decoded_operands = {},
    };
    const auto decoded = GcnInstDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.mnemonic, "buffer_load_dword");
  }

  {
    RawGcnInstruction raw{
        .pc = 0x1b08,
        .size_bytes = 8,
        .words = {0xe8000000u, 0x00000000u},
        .format_class = GcnInstFormatClass::Mtbuf,
        .mnemonic = "unknown",
        .operands = "",
        .decoded_operands = {},
    };
    const auto decoded = GcnInstDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.mnemonic, "tbuffer_load_format_x");
  }

  {
    RawGcnInstruction raw{
        .pc = 0x1b10,
        .size_bytes = 8,
        .words = {0xf0000000u, 0x00000000u},
        .format_class = GcnInstFormatClass::Mimg,
        .mnemonic = "unknown",
        .operands = "",
        .decoded_operands = {},
    };
    const auto decoded = GcnInstDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.mnemonic, "image_load");
  }

  {
    RawGcnInstruction raw{
        .pc = 0x1b18,
        .size_bytes = 8,
        .words = {0xf8000000u, 0x00000000u},
        .format_class = GcnInstFormatClass::Exp,
        .mnemonic = "unknown",
        .operands = "",
        .decoded_operands = {},
    };
    const auto decoded = GcnInstDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.mnemonic, "exp");
  }

  {
    RawGcnInstruction raw{
        .pc = 0x1b20,
        .size_bytes = 4,
        .words = {0xc8000000u},
        .format_class = GcnInstFormatClass::Vintrp,
        .mnemonic = "unknown",
        .operands = "",
        .decoded_operands = {},
    };
    const auto decoded = GcnInstDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.mnemonic, "v_interp_p1_f32");
  }
}

TEST(GcnInstDecoderTest, DecodesRepresentativeVop3B64ShiftInstruction) {
  RawGcnInstruction raw{
      .pc = 0x1a04,
      .size_bytes = 8,
      .words = {0xd28e0002u, 0x00020c01u},
      .format_class = GcnInstFormatClass::Vop3a,
      .mnemonic = "v_lshlrev_b64",
      .operands = "",
      .decoded_operands = {},
  };

  const auto decoded = GcnInstDecoder{}.Decode(raw);
  EXPECT_EQ(decoded.encoding_id, 17u);
  EXPECT_EQ(decoded.mnemonic, "v_lshlrev_b64");
  ASSERT_EQ(decoded.operands.size(), 3u);
  EXPECT_EQ(decoded.operands[0].text, "v[2:3]");
  EXPECT_EQ(decoded.operands[1].text, "s1");
  EXPECT_EQ(decoded.operands[2].text, "v[6:7]");
}

TEST(GcnInstDecoderTest, DecodesRepresentativeMadU64U32Instruction) {
  RawGcnInstruction raw{
      .pc = 0x1a08,
      .size_bytes = 8,
      .words = {0xd1e80402u, 0x04180a03u},
      .format_class = GcnInstFormatClass::Vop3a,
      .mnemonic = "v_mad_u64_u32",
      .operands = "",
      .decoded_operands = {},
  };

  const auto decoded = GcnInstDecoder{}.Decode(raw);
  EXPECT_EQ(decoded.encoding_id, 79u);
  EXPECT_EQ(decoded.mnemonic, "v_mad_u64_u32");
  ASSERT_EQ(decoded.operands.size(), 5u);
  EXPECT_EQ(decoded.operands[0].text, "v[2:3]");
  EXPECT_EQ(decoded.operands[1].text, "s[4:5]");
  EXPECT_EQ(decoded.operands[2].text, "s3");
  EXPECT_EQ(decoded.operands[3].text, "s5");
  EXPECT_EQ(decoded.operands[4].text, "v[6:7]");
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
