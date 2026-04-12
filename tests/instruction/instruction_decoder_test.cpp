#include <gtest/gtest.h>

#include "instruction/decode/encoded/instruction_decoder.h"
#include "instruction/decode/encoded/internal/encoded_gcn_db_lookup.h"

namespace gpu_model {
namespace {

uint32_t EncodeSoppWord(uint32_t opcode, int16_t simm16) {
  return 0xbf800000u | (opcode << 16u) | static_cast<uint16_t>(simm16);
}

uint32_t EncodeSop2Word(uint32_t opcode, uint32_t sdst, uint32_t ssrc0, uint32_t ssrc1) {
  return 0x80000000u | (opcode << 23u) | (sdst << 16u) | (ssrc1 << 8u) | ssrc0;
}

uint32_t EncodeSop1Word(uint32_t opcode, uint32_t sdst, uint32_t ssrc0) {
  return 0xbe800000u | (sdst << 16u) | (opcode << 8u) | ssrc0;
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

std::vector<uint32_t> EncodeVop3aWords(uint32_t opcode,
                                       uint32_t vdst,
                                       uint32_t src0,
                                       uint32_t src1,
                                       uint32_t src2,
                                       bool set_opcode_bit16 = false) {
  uint32_t low = 0xd0000000u | (opcode << 17u) | vdst;
  if (set_opcode_bit16) {
    low |= 0x00010000u;
  }
  const uint32_t high = src0 | (src1 << 9u) | (src2 << 18u);
  return {low, high};
}

std::vector<uint32_t> EncodeVop3bWords(uint32_t opcode,
                                       uint32_t vdst,
                                       uint32_t sdst,
                                       uint32_t src0,
                                       uint32_t src1,
                                       uint32_t src2) {
  const uint32_t low = 0xd0000000u | (opcode << 17u) | (sdst << 8u) | vdst;
  const uint32_t high = src0 | (src1 << 9u) | (src2 << 18u);
  return {low, high};
}

std::vector<uint32_t> EncodeVop3pWords(uint32_t opcode,
                                       uint32_t vdst,
                                       uint32_t src0,
                                       uint32_t src1,
                                       uint32_t src2) {
  const uint32_t low = 0xd3800000u | (opcode << 16u) | vdst;
  const uint32_t high = src0 | (src1 << 9u) | (src2 << 18u);
  return {low, high};
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

std::vector<uint32_t> EncodeGlobalFlatWords(uint32_t opcode,
                                            uint32_t addr,
                                            uint32_t data,
                                            uint32_t saddr,
                                            uint32_t vdst,
                                            uint32_t offset = 0) {
  const uint32_t low = 0xdc000000u | (opcode << 18u) | (0x2u << 14u) | (offset & 0x1fffu);
  const uint32_t high = addr | (data << 8u) | (saddr << 16u) | (vdst << 24u);
  return {low, high};
}

TEST(InstructionDecoderTest, DecodesRepresentativeScalarMemoryInstruction) {
  EncodedGcnInstruction raw{
      .pc = 0x1900,
      .size_bytes = 8,
      .words = {0xc0020002u, 0x0000002cu},
      .format_class = EncodedGcnInstFormatClass::Smrd,
      .mnemonic = "s_load_dword",
      .operands = "",
      .decoded_operands = {},
  };

  const auto decoded = InstructionDecoder{}.Decode(raw);
  EXPECT_EQ(decoded.encoding_id, 2u);
  EXPECT_EQ(decoded.mnemonic, "s_load_dword");
  EXPECT_EQ(decoded.format_class, EncodedGcnInstFormatClass::Smrd);
  ASSERT_EQ(decoded.operands.size(), 3u);
}

TEST(InstructionDecoderTest, DecodesRepresentativeBranchInstruction) {
  EncodedGcnInstruction raw{
      .pc = 0x192c,
      .size_bytes = 4,
      .words = {0xbf880019u},
      .format_class = EncodedGcnInstFormatClass::Sopp,
      .mnemonic = "s_cbranch_execz",
      .operands = "",
      .decoded_operands = {},
  };

  const auto decoded = InstructionDecoder{}.Decode(raw);
  EXPECT_EQ(decoded.encoding_id, 10u);
  EXPECT_EQ(decoded.mnemonic, "s_cbranch_execz");
  ASSERT_EQ(decoded.operands.size(), 1u);
  EXPECT_EQ(decoded.operands[0].text, "25");
  EXPECT_EQ(decoded.operands[0].kind, DecodedInstructionOperandKind::BranchTarget);
  EXPECT_TRUE(decoded.operands[0].info.has_immediate);
  EXPECT_EQ(decoded.operands[0].info.immediate, 25);
}

TEST(InstructionDecoderTest, DecodesNoOperandTerminationInstruction) {
  EncodedGcnInstruction raw{
      .pc = 0x1904,
      .size_bytes = 4,
      .words = {0xbf810000u},
      .format_class = EncodedGcnInstFormatClass::Sopp,
      .mnemonic = "s_endpgm",
      .operands = "",
      .decoded_operands = {},
  };

  const auto decoded = InstructionDecoder{}.Decode(raw);
  EXPECT_EQ(decoded.encoding_id, 1u);
  EXPECT_EQ(decoded.mnemonic, "s_endpgm");
  EXPECT_TRUE(decoded.operands.empty());
}

TEST(InstructionDecoderTest, DecodesRepresentativeWaitcntInstruction) {
  EncodedGcnInstruction raw{
      .pc = 0x1910,
      .size_bytes = 4,
      .words = {0xbf8cc07fu},
      .format_class = EncodedGcnInstFormatClass::Sopp,
      .mnemonic = "s_waitcnt",
      .operands = "",
      .decoded_operands = {},
  };

  const auto decoded = InstructionDecoder{}.Decode(raw);
  EXPECT_EQ(decoded.encoding_id, 12u);
  EXPECT_EQ(decoded.mnemonic, "s_waitcnt");
  ASSERT_EQ(decoded.operands.size(), 1u);
  EXPECT_EQ(decoded.operands[0].text, "lgkmcnt(0)");
  EXPECT_TRUE(decoded.operands[0].info.has_waitcnt);
  EXPECT_EQ(decoded.operands[0].info.wait_lgkmcnt, 0u);
  EXPECT_EQ(decoded.operands[0].info.wait_vmcnt, 15u);
  EXPECT_EQ(decoded.operands[0].info.wait_expcnt, 7u);
}

TEST(InstructionDecoderTest, DecodesRepresentativeVectorMoveInstruction) {
  EncodedGcnInstruction raw{
      .pc = 0x1950,
      .size_bytes = 4,
      .words = {0x7e060203u},
      .format_class = EncodedGcnInstFormatClass::Vop1,
      .mnemonic = "v_mov_b32_e32",
      .operands = "",
      .decoded_operands = {},
  };

  const auto decoded = InstructionDecoder{}.Decode(raw);
  EXPECT_EQ(decoded.encoding_id, 13u);
  EXPECT_EQ(decoded.mnemonic, "v_mov_b32_e32");
  ASSERT_EQ(decoded.operands.size(), 2u);
  EXPECT_EQ(decoded.operands[0].text, "v3");
}

TEST(InstructionDecoderTest, DecodesRepresentativeScalarMoveLiteralInstruction) {
  EncodedGcnInstruction raw{
      .pc = 0x1954,
      .size_bytes = 8,
      .words = {0xbe8000ffu, 0x0000002au},
      .format_class = EncodedGcnInstFormatClass::Sop1,
      .mnemonic = "s_mov_b32",
      .operands = "",
      .decoded_operands = {},
  };

  const auto decoded = InstructionDecoder{}.Decode(raw);
  EXPECT_EQ(decoded.encoding_id, 54u);
  EXPECT_EQ(decoded.mnemonic, "s_mov_b32");
  ASSERT_EQ(decoded.operands.size(), 2u);
  EXPECT_EQ(decoded.operands[0].text, "s0");
  EXPECT_EQ(decoded.operands[1].text, "0x2a");
}

TEST(InstructionDecoderTest, DecodesRepresentativeScalarMovkInstruction) {
  EncodedGcnInstruction raw{
      .pc = 0x1958,
      .size_bytes = 4,
      .words = {0xb000002au},
      .format_class = EncodedGcnInstFormatClass::Sopk,
      .mnemonic = "s_movk_i32",
      .operands = "",
      .decoded_operands = {},
  };

  const auto decoded = InstructionDecoder{}.Decode(raw);
  EXPECT_EQ(decoded.encoding_id, 78u);
  EXPECT_EQ(decoded.mnemonic, "s_movk_i32");
  ASSERT_EQ(decoded.operands.size(), 2u);
  EXPECT_EQ(decoded.operands[0].text, "s0");
  EXPECT_EQ(decoded.operands[1].text, "42");
}

TEST(InstructionDecoderTest, PrefersSizeAwareMnemonicLookupForLiteralExtendedInstructions) {
  {
    EncodedGcnInstruction raw{
        .pc = 0x1954,
        .size_bytes = 8,
        .words = {0xbe8000ffu, 0x0000002au},
        .format_class = EncodedGcnInstFormatClass::Sop1,
        .mnemonic = "s_mov_b32",
        .operands = "",
        .asm_op = "",
        .asm_text = "",
        .decoded_operands = {},
    };

    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 54u);
    EXPECT_EQ(decoded.mnemonic, "s_mov_b32");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x1aa4,
        .size_bytes = 4,
        .words = {EncodeVop2Word(/*opcode=*/59, /*vdst=*/4, /*src0=*/257, /*vsrc1=*/2)},
        .format_class = EncodedGcnInstFormatClass::Vop2,
        .mnemonic = "v_fmac_f32_e32",
        .operands = "",
        .asm_op = "",
        .asm_text = "",
        .decoded_operands = {},
    };

    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 65u);
    EXPECT_EQ(decoded.mnemonic, "v_fmac_f32_e32");
  }
}

TEST(InstructionDecoderTest, DecodesRepresentativeScalarShiftLeftB64Instruction) {
  EncodedGcnInstruction raw{
      .pc = 0x195c,
      .size_bytes = 4,
      .words = {0x8e848101u},
      .format_class = EncodedGcnInstFormatClass::Sop2,
      .mnemonic = "s_lshl_b64",
      .operands = "",
      .decoded_operands = {},
  };

  const auto decoded = InstructionDecoder{}.Decode(raw);
  EXPECT_EQ(decoded.encoding_id, 73u);
  EXPECT_EQ(decoded.mnemonic, "s_lshl_b64");
  ASSERT_EQ(decoded.operands.size(), 3u);
  EXPECT_EQ(decoded.operands[0].text, "s[4:5]");
  EXPECT_EQ(decoded.operands[1].text, "s[1:2]");
  EXPECT_EQ(decoded.operands[2].text, "1");
}

TEST(InstructionDecoderTest, DecodesScalarShiftLeftB64WithScalarShiftRegister) {
  EncodedGcnInstruction raw{
      .pc = 0x7b48,
      .size_bytes = 4,
      .words = {0x8e860381u},
      .format_class = EncodedGcnInstFormatClass::Sop2,
      .mnemonic = "s_lshl_b64",
      .operands = "",
      .decoded_operands = {},
  };

  const auto decoded = InstructionDecoder{}.Decode(raw);
  EXPECT_EQ(decoded.encoding_id, 73u);
  EXPECT_EQ(decoded.mnemonic, "s_lshl_b64");
  ASSERT_EQ(decoded.operands.size(), 3u);
  EXPECT_EQ(decoded.operands[0].text, "s[6:7]");
  EXPECT_EQ(decoded.operands[1].text, "1");
  EXPECT_EQ(decoded.operands[2].text, "s3");
}

TEST(InstructionDecoderTest, DecodesRepresentativeSop2ScalarAluInstructions) {
  {
    EncodedGcnInstruction raw{
        .pc = 0x1960,
        .size_bytes = 4,
        .words = {EncodeSop2Word(/*opcode=*/0, /*sdst=*/4, /*ssrc0=*/1, /*ssrc1=*/2)},
        .format_class = EncodedGcnInstFormatClass::Sop2,
        .mnemonic = "s_add_u32",
        .operands = "",
        .decoded_operands = {},
    };
    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 69u);
    EXPECT_EQ(decoded.mnemonic, "s_add_u32");
    ASSERT_EQ(decoded.operands.size(), 3u);
    EXPECT_EQ(decoded.operands[0].text, "s4");
    EXPECT_EQ(decoded.operands[1].text, "s1");
    EXPECT_EQ(decoded.operands[2].text, "s2");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x1964,
        .size_bytes = 4,
        .words = {EncodeSop2Word(/*opcode=*/4, /*sdst=*/5, /*ssrc0=*/3, /*ssrc1=*/6)},
        .format_class = EncodedGcnInstFormatClass::Sop2,
        .mnemonic = "s_addc_u32",
        .operands = "",
        .decoded_operands = {},
    };
    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 70u);
    EXPECT_EQ(decoded.mnemonic, "s_addc_u32");
    ASSERT_EQ(decoded.operands.size(), 3u);
    EXPECT_EQ(decoded.operands[0].text, "s5");
    EXPECT_EQ(decoded.operands[1].text, "s3");
    EXPECT_EQ(decoded.operands[2].text, "s6");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x1968,
        .size_bytes = 4,
        .words = {EncodeSop2Word(/*opcode=*/12, /*sdst=*/7, /*ssrc0=*/1, /*ssrc1=*/2)},
        .format_class = EncodedGcnInstFormatClass::Sop2,
        .mnemonic = "s_and_b32",
        .operands = "",
        .decoded_operands = {},
    };
    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 5u);
    EXPECT_EQ(decoded.mnemonic, "s_and_b32");
    ASSERT_EQ(decoded.operands.size(), 3u);
    EXPECT_EQ(decoded.operands[0].text, "s7");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x1970,
        .size_bytes = 8,
        .words = {EncodeSop2Word(/*opcode=*/12, /*sdst=*/0, /*ssrc0=*/255, /*ssrc1=*/12), 0xffffu},
        .format_class = EncodedGcnInstFormatClass::Sop2,
        .mnemonic = "s_and_b32",
        .operands = "",
        .decoded_operands = {},
    };
    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 20u);
    EXPECT_EQ(decoded.mnemonic, "s_and_b32");
    ASSERT_EQ(decoded.operands.size(), 3u);
    EXPECT_EQ(decoded.operands[0].text, "s0");
    EXPECT_EQ(decoded.operands[1].text, "0xffff");
    EXPECT_EQ(decoded.operands[2].text, "s12");
    ASSERT_TRUE(decoded.operands[1].info.has_immediate);
    EXPECT_EQ(decoded.operands[1].info.immediate, 0xffff);
    EXPECT_EQ(decoded.operands[2].info.reg_first, 12u);
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x196c,
        .size_bytes = 4,
        .words = {EncodeSop2Word(/*opcode=*/36, /*sdst=*/8, /*ssrc0=*/2, /*ssrc1=*/4)},
        .format_class = EncodedGcnInstFormatClass::Sop2,
        .mnemonic = "s_mul_i32",
        .operands = "",
        .decoded_operands = {},
    };
    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 6u);
    EXPECT_EQ(decoded.mnemonic, "s_mul_i32");
    ASSERT_EQ(decoded.operands.size(), 3u);
    EXPECT_EQ(decoded.operands[0].text, "s8");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x7b4c,
        .size_bytes = 4,
        .words = {0x84020802u},
        .format_class = EncodedGcnInstFormatClass::Sop2,
        .mnemonic = "s_max_i32",
        .operands = "",
        .decoded_operands = {},
    };
    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 125u);
    EXPECT_EQ(decoded.mnemonic, "s_max_i32");
    ASSERT_EQ(decoded.operands.size(), 3u);
    EXPECT_EQ(decoded.operands[0].text, "s2");
    EXPECT_EQ(decoded.operands[1].text, "s2");
    EXPECT_EQ(decoded.operands[2].text, "s8");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x1970,
        .size_bytes = 4,
        .words = {EncodeSop2Word(/*opcode=*/30, /*sdst=*/9, /*ssrc0=*/1, /*ssrc1=*/5)},
        .format_class = EncodedGcnInstFormatClass::Sop2,
        .mnemonic = "s_lshr_b32",
        .operands = "",
        .decoded_operands = {},
    };
    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 55u);
    EXPECT_EQ(decoded.mnemonic, "s_lshr_b32");
    ASSERT_EQ(decoded.operands.size(), 3u);
    EXPECT_EQ(decoded.operands[0].text, "s9");
  }
}

TEST(InstructionDecoderTest, DecodesRepresentativeScalarMemoryRangeInstructions) {
  {
    EncodedGcnInstruction raw{
        .pc = 0x1960,
        .size_bytes = 8,
        .words = {EncodeViSmrdWord(/*opcode=*/32, /*sdst_first=*/6, /*sbase_first=*/4), 0x10u},
        .format_class = EncodedGcnInstFormatClass::Smrd,
        .mnemonic = "s_load_dwordx2",
        .operands = "",
        .decoded_operands = {},
    };

    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 3u);
    EXPECT_EQ(decoded.mnemonic, "s_load_dwordx2");
    ASSERT_EQ(decoded.operands.size(), 3u);
    EXPECT_EQ(decoded.operands[0].text, "s[6:7]");
    EXPECT_EQ(decoded.operands[1].text, "s[4:5]");
    EXPECT_EQ(decoded.operands[2].text, "0x10");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x1968,
        .size_bytes = 8,
        .words = {EncodeViSmrdWord(/*opcode=*/64, /*sdst_first=*/0, /*sbase_first=*/4), 0x20u},
        .format_class = EncodedGcnInstFormatClass::Smrd,
        .mnemonic = "s_load_dwordx4",
        .operands = "",
        .decoded_operands = {},
    };

    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 4u);
    EXPECT_EQ(decoded.mnemonic, "s_load_dwordx4");
    ASSERT_EQ(decoded.operands.size(), 3u);
    EXPECT_EQ(decoded.operands[0].text, "s[0:3]");
    EXPECT_EQ(decoded.operands[1].text, "s[4:5]");
    EXPECT_EQ(decoded.operands[2].text, "0x20");
  }
}

TEST(InstructionDecoderTest, DecodesAdditionalScalarControlInstructions) {
  {
    EncodedGcnInstruction raw{
        .pc = 0x1978,
        .size_bytes = 4,
        .words = {EncodeSoppWord(/*opcode=*/2, /*simm16=*/5)},
        .format_class = EncodedGcnInstFormatClass::Sopp,
        .mnemonic = "s_branch",
        .operands = "",
        .decoded_operands = {},
    };

    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 27u);
    EXPECT_EQ(decoded.mnemonic, "s_branch");
    ASSERT_EQ(decoded.operands.size(), 1u);
    EXPECT_EQ(decoded.operands[0].text, "5");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x197c,
        .size_bytes = 4,
        .words = {EncodeSoppWord(/*opcode=*/5, /*simm16=*/3)},
        .format_class = EncodedGcnInstFormatClass::Sopp,
        .mnemonic = "s_cbranch_scc1",
        .operands = "",
        .decoded_operands = {},
    };

    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 22u);
    EXPECT_EQ(decoded.mnemonic, "s_cbranch_scc1");
    ASSERT_EQ(decoded.operands.size(), 1u);
    EXPECT_EQ(decoded.operands[0].text, "3");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x1980,
        .size_bytes = 4,
        .words = {EncodeSoppWord(/*opcode=*/10, /*simm16=*/0)},
        .format_class = EncodedGcnInstFormatClass::Sopp,
        .mnemonic = "s_barrier",
        .operands = "",
        .decoded_operands = {},
    };

    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 29u);
    EXPECT_EQ(decoded.mnemonic, "s_barrier");
    ASSERT_EQ(decoded.operands.size(), 1u);
    EXPECT_EQ(decoded.operands[0].text, "barrier");
  }
}

TEST(InstructionDecoderTest, DecodesAdditionalScalarCompareInstructions) {
  {
    EncodedGcnInstruction raw{
        .pc = 0x1984,
        .size_bytes = 4,
        .words = {0xbf060201u},
        .format_class = EncodedGcnInstFormatClass::Sopc,
        .mnemonic = "s_cmp_eq_u32",
        .operands = "",
        .decoded_operands = {},
    };

    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 24u);
    EXPECT_EQ(decoded.mnemonic, "s_cmp_eq_u32");
    ASSERT_EQ(decoded.operands.size(), 2u);
    EXPECT_EQ(decoded.operands[0].text, "s1");
    EXPECT_EQ(decoded.operands[1].text, "s2");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x1988,
        .size_bytes = 4,
        .words = {0xbf080201u},
        .format_class = EncodedGcnInstFormatClass::Sopc,
        .mnemonic = "s_cmp_gt_u32",
        .operands = "",
        .decoded_operands = {},
    };

    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 39u);
    EXPECT_EQ(decoded.mnemonic, "s_cmp_gt_u32");
    ASSERT_EQ(decoded.operands.size(), 2u);
    EXPECT_EQ(decoded.operands[0].text, "s1");
    EXPECT_EQ(decoded.operands[1].text, "s2");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x198c,
        .size_bytes = 4,
        .words = {0xbf0a0201u},
        .format_class = EncodedGcnInstFormatClass::Sopc,
        .mnemonic = "s_cmp_lt_u32",
        .operands = "",
        .decoded_operands = {},
    };

    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 40u);
    EXPECT_EQ(decoded.mnemonic, "s_cmp_lt_u32");
    ASSERT_EQ(decoded.operands.size(), 2u);
    EXPECT_EQ(decoded.operands[0].text, "s1");
    EXPECT_EQ(decoded.operands[1].text, "s2");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x7e5c,
        .size_bytes = 4,
        .words = {0xbf138000u},
        .format_class = EncodedGcnInstFormatClass::Sopc,
        .mnemonic = "s_cmp_lg_u64",
        .operands = "",
        .decoded_operands = {},
    };

    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 126u);
    EXPECT_EQ(decoded.mnemonic, "s_cmp_lg_u64");
    ASSERT_EQ(decoded.operands.size(), 2u);
    EXPECT_EQ(decoded.operands[0].text, "s[0:1]");
    EXPECT_EQ(decoded.operands[1].text, "0");
  }
}

TEST(InstructionDecoderTest, DecodesRepresentativeGlobalLoadInstruction) {
  EncodedGcnInstruction raw{
      .pc = 0x1968,
      .size_bytes = 8,
      .words = {0xdc508000u, 0x067f0004u},
      .format_class = EncodedGcnInstFormatClass::Flat,
      .mnemonic = "global_load_dword",
      .operands = "",
      .decoded_operands = {},
  };

  const auto decoded = InstructionDecoder{}.Decode(raw);
  EXPECT_EQ(decoded.encoding_id, 18u);
  EXPECT_EQ(decoded.mnemonic, "global_load_dword");
  ASSERT_EQ(decoded.operands.size(), 3u);
  EXPECT_EQ(decoded.operands[1].kind, DecodedInstructionOperandKind::VectorRegRange);
  EXPECT_EQ(decoded.operands[1].text, "v[4:5]");
  EXPECT_EQ(decoded.operands[1].info.reg_first, 4u);
  EXPECT_EQ(decoded.operands[1].info.reg_count, 2u);
  EXPECT_EQ(decoded.operands[0].text, "v6");
  EXPECT_EQ(decoded.operands[2].text, "off");
  EXPECT_TRUE(decoded.operands[2].info.has_immediate);
  EXPECT_EQ(decoded.operands[2].info.immediate, 0);
}

TEST(InstructionDecoderTest, DecodesRepresentativeGlobalStoreInstruction) {
  EncodedGcnInstruction raw{
      .pc = 0x1970,
      .size_bytes = 8,
      .words = {0xdc708000u, 0x047f0405u},
      .format_class = EncodedGcnInstFormatClass::Flat,
      .mnemonic = "global_store_dword",
      .operands = "",
      .decoded_operands = {},
  };

  const auto decoded = InstructionDecoder{}.Decode(raw);
  EXPECT_EQ(decoded.encoding_id, 19u);
  EXPECT_EQ(decoded.mnemonic, "global_store_dword");
  ASSERT_EQ(decoded.operands.size(), 3u);
  EXPECT_EQ(decoded.operands[0].kind, DecodedInstructionOperandKind::VectorRegRange);
  EXPECT_EQ(decoded.operands[0].text, "v[5:6]");
  EXPECT_EQ(decoded.operands[1].text, "v4");
  EXPECT_EQ(decoded.operands[2].text, "off");
}

TEST(InstructionDecoderTest, DecodesRepresentativeVop3aFmaInstruction) {
  EncodedGcnInstruction raw{
      .pc = 0x1a00,
      .size_bytes = 8,
      .words = {0xd1cb0002u, 0x04140503u},
      .format_class = EncodedGcnInstFormatClass::Vop3a,
      .mnemonic = "v_fma_f32",
      .operands = "",
      .decoded_operands = {},
  };

  const auto decoded = InstructionDecoder{}.Decode(raw);
  EXPECT_EQ(decoded.encoding_id, 25u);
  EXPECT_EQ(decoded.mnemonic, "v_fma_f32");
  ASSERT_EQ(decoded.operands.size(), 4u);
  EXPECT_EQ(decoded.operands[0].text, "v2");
  EXPECT_EQ(decoded.operands[1].text, "v3");
  EXPECT_EQ(decoded.operands[2].text, "s2");
  EXPECT_EQ(decoded.operands[3].text, "v5");
}

TEST(InstructionDecoderTest, DecodesAdditionalVectorAndLdsInstructions) {
  {
    EncodedGcnInstruction raw{
        .pc = 0x1a10,
        .size_bytes = 4,
        .words = {EncodeVop2Word(/*opcode=*/1, /*vdst=*/2, /*src0=*/0x100u + 6u, /*vsrc1=*/7u)},
        .format_class = EncodedGcnInstFormatClass::Vop2,
        .mnemonic = "v_add_f32_e32",
        .operands = "",
        .decoded_operands = {},
    };

    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 11u);
    EXPECT_EQ(decoded.mnemonic, "v_add_f32_e32");
    ASSERT_EQ(decoded.operands.size(), 3u);
    EXPECT_EQ(decoded.operands[0].text, "v2");
    EXPECT_EQ(decoded.operands[1].text, "v6");
    EXPECT_EQ(decoded.operands[2].text, "v7");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x1a14,
        .size_bytes = 4,
        .words = {EncodeVop2Word(/*opcode=*/5, /*vdst=*/4, /*src0=*/0x100u + 1u, /*vsrc1=*/2u)},
        .format_class = EncodedGcnInstFormatClass::Vop2,
        .mnemonic = "v_mul_f32_e32",
        .operands = "",
        .decoded_operands = {},
    };

    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.mnemonic, "v_mul_f32_e32");
    ASSERT_EQ(decoded.operands.size(), 3u);
    EXPECT_EQ(decoded.operands[0].text, "v4");
    EXPECT_EQ(decoded.operands[1].text, "v1");
    EXPECT_EQ(decoded.operands[2].text, "v2");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x1a18,
        .size_bytes = 4,
        .words = {EncodeVopcWord(/*opcode=*/202, /*src0=*/0x100u + 2u, /*vsrc1=*/3u)},
        .format_class = EncodedGcnInstFormatClass::Vopc,
        .mnemonic = "v_cmp_eq_u32_e32",
        .operands = "",
        .decoded_operands = {},
    };

    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 66u);
    EXPECT_EQ(decoded.mnemonic, "v_cmp_eq_u32_e32");
    ASSERT_EQ(decoded.operands.size(), 3u);
    EXPECT_EQ(decoded.operands[0].text, "vcc");
    EXPECT_EQ(decoded.operands[1].text, "v2");
    EXPECT_EQ(decoded.operands[2].text, "v3");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x1a1c,
        .size_bytes = 4,
        .words = {EncodeVop1Word(/*opcode=*/5, /*vdst=*/1, /*src0=*/0x100u + 4u)},
        .format_class = EncodedGcnInstFormatClass::Vop1,
        .mnemonic = "v_cvt_f32_i32_e32",
        .operands = "",
        .decoded_operands = {},
    };

    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 80u);
    EXPECT_EQ(decoded.mnemonic, "v_cvt_f32_i32_e32");
    ASSERT_EQ(decoded.operands.size(), 2u);
    EXPECT_EQ(decoded.operands[0].text, "v1");
    EXPECT_EQ(decoded.operands[1].text, "v4");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x1a1e,
        .size_bytes = 4,
        .words = {EncodeVopcWord(/*opcode=*/196, /*src0=*/0x100u + 2u, /*vsrc1=*/3u)},
        .format_class = EncodedGcnInstFormatClass::Vopc,
        .mnemonic = "v_cmp_gt_i32_e32",
        .operands = "",
        .decoded_operands = {},
    };

    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 8u);
    EXPECT_EQ(decoded.mnemonic, "v_cmp_gt_i32_e32");
    ASSERT_EQ(decoded.operands.size(), 3u);
    EXPECT_EQ(decoded.operands[0].text, "vcc");
    EXPECT_EQ(decoded.operands[1].text, "v2");
    EXPECT_EQ(decoded.operands[2].text, "v3");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x7b40,
        .size_bytes = 8,
        .words = {0xd2890008u, 0x00000700u},
        .format_class = EncodedGcnInstFormatClass::Vop3a,
        .mnemonic = "v_ldexp_f32",
        .operands = "",
        .asm_op = "v_readlane_b32",
        .asm_text = "v_readlane_b32 s8, v0, s3",
        .decoded_operands = {},
    };

    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 128u);
    EXPECT_EQ(decoded.mnemonic, "v_readlane_b32");
    EXPECT_EQ(decoded.asm_op, "v_readlane_b32");
    EXPECT_EQ(decoded.asm_text, "v_readlane_b32 s8, v0, s3");
    ASSERT_EQ(decoded.operands.size(), 3u);
    EXPECT_EQ(decoded.operands[0].text, "s8");
    EXPECT_EQ(decoded.operands[1].text, "v0");
    EXPECT_EQ(decoded.operands[2].text, "s3");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x7e74,
        .size_bytes = 8,
        .words = {0xd0ca0000u, 0x00020080u},
        .format_class = EncodedGcnInstFormatClass::Vop3a,
        .mnemonic = "unknown",
        .operands = "",
        .asm_op = "v_cmp_eq_u32_e64",
        .asm_text = "v_cmp_eq_u32_e64 s[0:1], 0, v0",
        .decoded_operands = {},
    };

    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 127u);
    EXPECT_EQ(decoded.mnemonic, "v_cmp_eq_u32_e64");
    EXPECT_EQ(decoded.asm_op, "v_cmp_eq_u32_e64");
    EXPECT_EQ(decoded.asm_text, "v_cmp_eq_u32_e64 s[0:1], 0, v0");
    ASSERT_EQ(decoded.operands.size(), 3u);
    EXPECT_EQ(decoded.operands[0].text, "s[0:1]");
    EXPECT_EQ(decoded.operands[1].text, "0");
    EXPECT_EQ(decoded.operands[2].text, "v0");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x1a40,
        .size_bytes = 8,
        .words = {EncodeDsWord0(/*opcode=*/13), EncodeDsWord1(/*addr=*/4, /*data0=*/5, /*data1=*/0, /*vdst=*/0)},
        .format_class = EncodedGcnInstFormatClass::Ds,
        .mnemonic = "ds_write_b32",
        .operands = "",
        .decoded_operands = {},
    };

    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 30u);
    EXPECT_EQ(decoded.mnemonic, "ds_write_b32");
    ASSERT_EQ(decoded.operands.size(), 2u);
    EXPECT_EQ(decoded.operands[0].text, "v4");
    EXPECT_EQ(decoded.operands[1].text, "v5");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x1a44,
        .size_bytes = 8,
        .words = {EncodeDsWord0(/*opcode=*/0), EncodeDsWord1(/*addr=*/4, /*data0=*/5, /*data1=*/0, /*vdst=*/6)},
        .format_class = EncodedGcnInstFormatClass::Ds,
        .mnemonic = "ds_add_u32",
        .operands = "",
        .decoded_operands = {},
    };

    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 96u);
    EXPECT_EQ(decoded.mnemonic, "ds_add_u32");
    ASSERT_EQ(decoded.operands.size(), 2u);
    EXPECT_EQ(decoded.operands[0].text, "v4");
    EXPECT_EQ(decoded.operands[1].text, "v5");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x1a48,
        .size_bytes = 8,
        .words = {EncodeDsWord0(/*opcode=*/54), EncodeDsWord1(/*addr=*/4, /*data0=*/0, /*data1=*/0, /*vdst=*/6)},
        .format_class = EncodedGcnInstFormatClass::Ds,
        .mnemonic = "ds_read_b32",
        .operands = "",
        .decoded_operands = {},
    };

    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 31u);
    EXPECT_EQ(decoded.mnemonic, "ds_read_b32");
    ASSERT_EQ(decoded.operands.size(), 2u);
    EXPECT_EQ(decoded.operands[0].text, "v6");
    EXPECT_EQ(decoded.operands[1].text, "v4");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x1a4c,
        .size_bytes = 4,
        .words = {EncodeVopcWord(/*opcode=*/204, /*src0=*/0x100u + 4u, /*vsrc1=*/5u)},
        .format_class = EncodedGcnInstFormatClass::Vopc,
        .mnemonic = "v_cmp_gt_u32_e32",
        .operands = "",
        .decoded_operands = {},
    };

    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 56u);
    EXPECT_EQ(decoded.mnemonic, "v_cmp_gt_u32_e32");
    ASSERT_EQ(decoded.operands.size(), 3u);
    EXPECT_EQ(decoded.operands[1].text, "v4");
    EXPECT_EQ(decoded.operands[2].text, "v5");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x1a4e,
        .size_bytes = 4,
        .words = {EncodeVopcWord(/*opcode=*/201, /*src0=*/0x100u + 10u, /*vsrc1=*/11u)},
        .format_class = EncodedGcnInstFormatClass::Vopc,
        .mnemonic = "v_cmp_lt_u32_e32",
        .operands = "",
        .decoded_operands = {},
    };

    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.mnemonic, "v_cmp_lt_u32_e32");
    ASSERT_EQ(decoded.operands.size(), 3u);
    EXPECT_EQ(decoded.operands[0].text, "vcc");
    EXPECT_EQ(decoded.operands[1].text, "v10");
    EXPECT_EQ(decoded.operands[2].text, "v11");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x1a50,
        .size_bytes = 4,
        .words = {EncodeVopcWord(/*opcode=*/195, /*src0=*/0x100u + 6u, /*vsrc1=*/7u)},
        .format_class = EncodedGcnInstFormatClass::Vopc,
        .mnemonic = "v_cmp_le_i32_e32",
        .operands = "",
        .decoded_operands = {},
    };

    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 75u);
    EXPECT_EQ(decoded.mnemonic, "v_cmp_le_i32_e32");
    ASSERT_EQ(decoded.operands.size(), 3u);
    EXPECT_EQ(decoded.operands[1].text, "v6");
    EXPECT_EQ(decoded.operands[2].text, "v7");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x1a54,
        .size_bytes = 4,
        .words = {EncodeVopcWord(/*opcode=*/193, /*src0=*/0x100u + 8u, /*vsrc1=*/9u)},
        .format_class = EncodedGcnInstFormatClass::Vopc,
        .mnemonic = "v_cmp_lt_i32_e32",
        .operands = "",
        .decoded_operands = {},
    };

    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 76u);
    EXPECT_EQ(decoded.mnemonic, "v_cmp_lt_i32_e32");
    ASSERT_EQ(decoded.operands.size(), 3u);
    EXPECT_EQ(decoded.operands[1].text, "v8");
    EXPECT_EQ(decoded.operands[2].text, "v9");
  }
}

TEST(InstructionDecoderTest, LeavesReservedPlaceholderFamiliesUnknownWhenNotInMatchTable) {
  {
    EncodedGcnInstruction raw{
        .pc = 0x1b00,
        .size_bytes = 8,
        .words = {0xe0500000u, 0x00000000u},
        .format_class = EncodedGcnInstFormatClass::Mubuf,
        .mnemonic = "unknown",
        .operands = "",
        .decoded_operands = {},
    };
    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.mnemonic, "unknown");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x1b08,
        .size_bytes = 8,
        .words = {0xe8000000u, 0x00000000u},
        .format_class = EncodedGcnInstFormatClass::Mtbuf,
        .mnemonic = "unknown",
        .operands = "",
        .decoded_operands = {},
    };
    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.mnemonic, "unknown");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x1b10,
        .size_bytes = 8,
        .words = {0xf0000000u, 0x00000000u},
        .format_class = EncodedGcnInstFormatClass::Mimg,
        .mnemonic = "unknown",
        .operands = "",
        .decoded_operands = {},
    };
    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.mnemonic, "unknown");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x1b18,
        .size_bytes = 8,
        .words = {0xf8000000u, 0x00000000u},
        .format_class = EncodedGcnInstFormatClass::Exp,
        .mnemonic = "unknown",
        .operands = "",
        .decoded_operands = {},
    };
    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.mnemonic, "unknown");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x1b20,
        .size_bytes = 4,
        .words = {0xc8000000u},
        .format_class = EncodedGcnInstFormatClass::Vintrp,
        .mnemonic = "unknown",
        .operands = "",
        .decoded_operands = {},
    };
    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.mnemonic, "unknown");
  }
}

TEST(InstructionDecoderTest, DecodesRepresentativeVop3B64ShiftInstruction) {
  EncodedGcnInstruction raw{
      .pc = 0x1a04,
      .size_bytes = 8,
      .words = {0xd28e0002u, 0x00020c01u},
      .format_class = EncodedGcnInstFormatClass::Vop3a,
      .mnemonic = "v_lshlrev_b64",
      .operands = "",
      .decoded_operands = {},
  };

  const auto decoded = InstructionDecoder{}.Decode(raw);
  EXPECT_EQ(decoded.encoding_id, 17u);
  EXPECT_EQ(decoded.mnemonic, "v_lshlrev_b64");
  ASSERT_EQ(decoded.operands.size(), 3u);
  EXPECT_EQ(decoded.operands[0].text, "v[2:3]");
  EXPECT_EQ(decoded.operands[1].text, "s1");
  EXPECT_EQ(decoded.operands[2].text, "v[6:7]");
}

TEST(InstructionDecoderTest, DecodesRepresentativeMadU64U32Instruction) {
  EncodedGcnInstruction raw{
      .pc = 0x1a08,
      .size_bytes = 8,
      .words = {0xd1e80402u, 0x04180a03u},
      .format_class = EncodedGcnInstFormatClass::Vop3a,
      .mnemonic = "v_mad_u64_u32",
      .operands = "",
      .decoded_operands = {},
  };

  const auto decoded = InstructionDecoder{}.Decode(raw);
  EXPECT_EQ(decoded.encoding_id, 79u);
  EXPECT_EQ(decoded.mnemonic, "v_mad_u64_u32");
  ASSERT_EQ(decoded.operands.size(), 5u);
  EXPECT_EQ(decoded.operands[0].text, "v[2:3]");
  EXPECT_EQ(decoded.operands[1].text, "s[4:5]");
  EXPECT_EQ(decoded.operands[2].text, "s3");
  EXPECT_EQ(decoded.operands[3].text, "s5");
  EXPECT_EQ(decoded.operands[4].text, "v[6:7]");
}

TEST(InstructionDecoderTest, DecodesRepresentativeCarryProducingVectorAddInstruction) {
  EncodedGcnInstruction raw{
      .pc = 0x1a20,
      .size_bytes = 4,
      .words = {0x32060401u},
      .format_class = EncodedGcnInstFormatClass::Vop2,
      .mnemonic = "v_add_co_u32_e32",
      .operands = "",
      .decoded_operands = {},
  };

  const auto decoded = InstructionDecoder{}.Decode(raw);
  EXPECT_EQ(decoded.encoding_id, 15u);
  EXPECT_EQ(decoded.mnemonic, "v_add_co_u32_e32");
  ASSERT_EQ(decoded.operands.size(), 4u);
  EXPECT_EQ(decoded.operands[0].text, "v3");
  EXPECT_EQ(decoded.operands[1].text, "vcc");
  EXPECT_EQ(decoded.operands[2].text, "s1");
  EXPECT_EQ(decoded.operands[3].text, "v2");
}

TEST(InstructionDecoderTest, DecodesRepresentativeCarryConsumingVectorAddInstruction) {
  EncodedGcnInstruction raw{
      .pc = 0x1a24,
      .size_bytes = 4,
      .words = {0x38060401u},
      .format_class = EncodedGcnInstFormatClass::Vop2,
      .mnemonic = "v_addc_co_u32_e32",
      .operands = "",
      .decoded_operands = {},
  };

  const auto decoded = InstructionDecoder{}.Decode(raw);
  EXPECT_EQ(decoded.encoding_id, 16u);
  EXPECT_EQ(decoded.mnemonic, "v_addc_co_u32_e32");
  ASSERT_EQ(decoded.operands.size(), 5u);
  EXPECT_EQ(decoded.operands[0].text, "v3");
  EXPECT_EQ(decoded.operands[1].text, "vcc");
  EXPECT_EQ(decoded.operands[2].text, "s1");
  EXPECT_EQ(decoded.operands[3].text, "v2");
  EXPECT_EQ(decoded.operands[4].text, "vcc");
}

TEST(InstructionDecoderTest, DecodesRepresentativeVop3CarryE64Instruction) {
  EncodedGcnInstruction raw{
      .pc = 0x1a30,
      .size_bytes = 8,
      .words = {0xd1180402u, 0x00000a03u},
      .format_class = EncodedGcnInstFormatClass::Vop3a,
      .mnemonic = "v_add_co_u32_e64",
      .operands = "",
      .decoded_operands = {},
  };

  const auto decoded = InstructionDecoder{}.Decode(raw);
  EXPECT_EQ(decoded.encoding_id, 35u);
  EXPECT_EQ(decoded.mnemonic, "v_add_co_u32_e64");
  ASSERT_EQ(decoded.operands.size(), 4u);
  EXPECT_EQ(decoded.operands[0].text, "v2");
  EXPECT_EQ(decoded.operands[1].text, "s[4:5]");
  EXPECT_EQ(decoded.operands[2].text, "s3");
  EXPECT_EQ(decoded.operands[3].text, "s5");
}

TEST(InstructionDecoderTest, DecodesRepresentativeVop3CndmaskE64Instruction) {
  EncodedGcnInstruction raw{
      .pc = 0x1a38,
      .size_bytes = 8,
      .words = {0xd1000001u, 0x00100002u},
      .format_class = EncodedGcnInstFormatClass::Vop3a,
      .mnemonic = "v_cndmask_b32_e64",
      .operands = "",
      .decoded_operands = {},
  };

  const auto decoded = InstructionDecoder{}.Decode(raw);
  EXPECT_EQ(decoded.encoding_id, 59u);
  EXPECT_EQ(decoded.mnemonic, "v_cndmask_b32_e64");
  ASSERT_EQ(decoded.operands.size(), 4u);
  EXPECT_EQ(decoded.operands[0].text, "v1");
  EXPECT_EQ(decoded.operands[1].text, "s2");
  EXPECT_EQ(decoded.operands[2].text, "s0");
  EXPECT_EQ(decoded.operands[3].text, "s[4:5]");
}

TEST(InstructionDecoderTest, DecodesRepresentativeSop1ScalarAluInstructions) {
  {
    EncodedGcnInstruction raw{
        .pc = 0x1a40,
        .size_bytes = 4,
        .words = {EncodeSop1Word(/*opcode=*/1, /*sdst=*/4, /*ssrc0=*/2)},
        .format_class = EncodedGcnInstFormatClass::Sop1,
        .mnemonic = "s_mov_b64",
        .operands = "",
        .decoded_operands = {},
    };

    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 71u);
    EXPECT_EQ(decoded.mnemonic, "s_mov_b64");
    ASSERT_EQ(decoded.operands.size(), 2u);
    EXPECT_EQ(decoded.operands[0].text, "s[4:5]");
    EXPECT_EQ(decoded.operands[1].text, "s[2:3]");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x1a44,
        .size_bytes = 4,
        .words = {EncodeSop1Word(/*opcode=*/13, /*sdst=*/7, /*ssrc0=*/10)},
        .format_class = EncodedGcnInstFormatClass::Sop1,
        .mnemonic = "s_bcnt1_i32_b64",
        .operands = "",
        .decoded_operands = {},
    };

    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 83u);
    EXPECT_EQ(decoded.mnemonic, "s_bcnt1_i32_b64");
    ASSERT_EQ(decoded.operands.size(), 2u);
    EXPECT_EQ(decoded.operands[0].text, "s7");
    EXPECT_EQ(decoded.operands[1].text, "s[10:11]");
  }
}

TEST(InstructionDecoderTest, DecodesRepresentativeVectorAluInstructionsAcrossFormats) {
  {
    EncodedGcnInstruction raw{
        .pc = 0x1a48,
        .size_bytes = 4,
        .words = {EncodeVop1Word(/*opcode=*/43, /*vdst=*/5, /*src0=*/257)},
        .format_class = EncodedGcnInstFormatClass::Vop1,
        .mnemonic = "v_not_b32_e32",
        .operands = "",
        .decoded_operands = {},
    };

    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 32u);
    EXPECT_EQ(decoded.mnemonic, "v_not_b32_e32");
    ASSERT_EQ(decoded.operands.size(), 2u);
    EXPECT_EQ(decoded.operands[0].text, "v5");
    EXPECT_EQ(decoded.operands[1].text, "v1");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x1a4c,
        .size_bytes = 4,
        .words = {EncodeVop1Word(/*opcode=*/30, /*vdst=*/6, /*src0=*/258)},
        .format_class = EncodedGcnInstFormatClass::Vop1,
        .mnemonic = "v_rndne_f32_e32",
        .operands = "",
        .decoded_operands = {},
    };

    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 50u);
    EXPECT_EQ(decoded.mnemonic, "v_rndne_f32_e32");
    ASSERT_EQ(decoded.operands.size(), 2u);
    EXPECT_EQ(decoded.operands[0].text, "v6");
    EXPECT_EQ(decoded.operands[1].text, "v2");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x1a50,
        .size_bytes = 4,
        .words = {EncodeVop2Word(/*opcode=*/17, /*vdst=*/7, /*src0=*/3, /*vsrc1=*/8)},
        .format_class = EncodedGcnInstFormatClass::Vop2,
        .mnemonic = "v_ashrrev_i32_e32",
        .operands = "",
        .decoded_operands = {},
    };

    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 14u);
    EXPECT_EQ(decoded.mnemonic, "v_ashrrev_i32_e32");
    ASSERT_EQ(decoded.operands.size(), 3u);
    EXPECT_EQ(decoded.operands[0].text, "v7");
    EXPECT_EQ(decoded.operands[1].text, "s3");
    EXPECT_EQ(decoded.operands[2].text, "v8");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x1a54,
        .size_bytes = 4,
        .words = {EncodeVop2Word(/*opcode=*/0, /*vdst=*/9, /*src0=*/260, /*vsrc1=*/10)},
        .format_class = EncodedGcnInstFormatClass::Vop2,
        .mnemonic = "v_cndmask_b32_e32",
        .operands = "",
        .decoded_operands = {},
    };

    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 48u);
    EXPECT_EQ(decoded.mnemonic, "v_cndmask_b32_e32");
    ASSERT_EQ(decoded.operands.size(), 3u);
    EXPECT_EQ(decoded.operands[0].text, "v9");
    EXPECT_EQ(decoded.operands[1].text, "v4");
    EXPECT_EQ(decoded.operands[2].text, "v10");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x1a58,
        .size_bytes = 8,
        .words = EncodeVop3aWords(/*opcode=*/324, /*vdst=*/11, /*src0=*/257, /*src1=*/2, /*src2=*/0),
        .format_class = EncodedGcnInstFormatClass::Vop3a,
        .mnemonic = "v_ldexp_f32",
        .operands = "",
        .decoded_operands = {},
    };

    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 63u);
    EXPECT_EQ(decoded.mnemonic, "v_ldexp_f32");
    ASSERT_EQ(decoded.operands.size(), 4u);
    EXPECT_EQ(decoded.operands[0].text, "v11");
    EXPECT_EQ(decoded.operands[1].text, "v1");
    EXPECT_EQ(decoded.operands[2].text, "s2");
    EXPECT_EQ(decoded.operands[3].text, "s0");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x1a60,
        .size_bytes = 8,
        .words = EncodeVop3aWords(
            /*opcode=*/326, /*vdst=*/12, /*src0=*/257, /*src1=*/258, /*src2=*/0, /*set_opcode_bit16=*/false),
        .format_class = EncodedGcnInstFormatClass::Vop3a,
        .mnemonic = "v_mbcnt_lo_u32_b32",
        .operands = "",
        .decoded_operands = {},
    };

    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 81u);
    EXPECT_EQ(decoded.mnemonic, "v_mbcnt_lo_u32_b32");
    ASSERT_EQ(decoded.operands.size(), 3u);
    EXPECT_EQ(decoded.operands[0].text, "v12");
    EXPECT_EQ(decoded.operands[1].text, "v1");
    EXPECT_EQ(decoded.operands[2].text, "v2");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x1a68,
        .size_bytes = 8,
        .words = EncodeVop3aWords(
            /*opcode=*/326, /*vdst=*/13, /*src0=*/257, /*src1=*/258, /*src2=*/0, /*set_opcode_bit16=*/true),
        .format_class = EncodedGcnInstFormatClass::Vop3a,
        .mnemonic = "v_mbcnt_hi_u32_b32",
        .operands = "",
        .decoded_operands = {},
    };

    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 82u);
    EXPECT_EQ(decoded.mnemonic, "v_mbcnt_hi_u32_b32");
    ASSERT_EQ(decoded.operands.size(), 3u);
    EXPECT_EQ(decoded.operands[0].text, "v13");
    EXPECT_EQ(decoded.operands[1].text, "v1");
    EXPECT_EQ(decoded.operands[2].text, "v2");
  }
}

TEST(InstructionDecoderTest, DecodesRemainingSupportedInstructionsAcrossFormats) {
  {
    EncodedGcnInstruction raw{
        .pc = 0x1a70,
        .size_bytes = 4,
        .words = {EncodeSop1Word(/*opcode=*/32, /*sdst=*/4, /*ssrc0=*/106)},
        .format_class = EncodedGcnInstFormatClass::Sop1,
        .mnemonic = "s_and_saveexec_b64",
        .operands = "",
        .decoded_operands = {},
    };
    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 9u);
    EXPECT_EQ(decoded.mnemonic, "s_and_saveexec_b64");
    ASSERT_EQ(decoded.operands.size(), 2u);
    EXPECT_EQ(decoded.operands[0].text, "s[4:5]");
    EXPECT_EQ(decoded.operands[1].text, "vcc");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x1a74,
        .size_bytes = 4,
        .words = {EncodeSoppWord(/*opcode=*/4, /*simm16=*/2)},
        .format_class = EncodedGcnInstFormatClass::Sopp,
        .mnemonic = "s_cbranch_scc0",
        .operands = "",
        .decoded_operands = {},
    };
    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 26u);
    EXPECT_EQ(decoded.mnemonic, "s_cbranch_scc0");
    ASSERT_EQ(decoded.operands.size(), 1u);
    EXPECT_EQ(decoded.operands[0].text, "2");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x1a78,
        .size_bytes = 4,
        .words = {EncodeSoppWord(/*opcode=*/6, /*simm16=*/4)},
        .format_class = EncodedGcnInstFormatClass::Sopp,
        .mnemonic = "s_cbranch_vccz",
        .operands = "",
        .decoded_operands = {},
    };
    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 43u);
    EXPECT_EQ(decoded.mnemonic, "s_cbranch_vccz");
    ASSERT_EQ(decoded.operands.size(), 1u);
    EXPECT_EQ(decoded.operands[0].text, "4");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x1a7c,
        .size_bytes = 4,
        .words = {EncodeSoppWord(/*opcode=*/9, /*simm16=*/1)},
        .format_class = EncodedGcnInstFormatClass::Sopp,
        .mnemonic = "s_cbranch_execnz",
        .operands = "",
        .decoded_operands = {},
    };
    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 74u);
    EXPECT_EQ(decoded.mnemonic, "s_cbranch_execnz");
    ASSERT_EQ(decoded.operands.size(), 1u);
    EXPECT_EQ(decoded.operands[0].text, "1");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x1a80,
        .size_bytes = 4,
        .words = {EncodeSoppWord(/*opcode=*/0, /*simm16=*/7)},
        .format_class = EncodedGcnInstFormatClass::Sopp,
        .mnemonic = "s_nop",
        .operands = "",
        .decoded_operands = {},
    };
    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 68u);
    EXPECT_EQ(decoded.mnemonic, "s_nop");
    EXPECT_TRUE(decoded.operands.empty());
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x1a84,
        .size_bytes = 4,
        .words = {0xbf040201u},
        .format_class = EncodedGcnInstFormatClass::Sopc,
        .mnemonic = "s_cmp_lt_i32",
        .operands = "",
        .decoded_operands = {},
    };
    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 21u);
    EXPECT_EQ(decoded.mnemonic, "s_cmp_lt_i32");
    ASSERT_EQ(decoded.operands.size(), 2u);
    EXPECT_EQ(decoded.operands[0].text, "s1");
    EXPECT_EQ(decoded.operands[1].text, "s2");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x1a88,
        .size_bytes = 4,
        .words = {EncodeSop2Word(/*opcode=*/2, /*sdst=*/5, /*ssrc0=*/1, /*ssrc1=*/2)},
        .format_class = EncodedGcnInstFormatClass::Sop2,
        .mnemonic = "s_add_i32",
        .operands = "",
        .decoded_operands = {},
    };
    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 23u);
    EXPECT_EQ(decoded.mnemonic, "s_add_i32");
    ASSERT_EQ(decoded.operands.size(), 3u);
    EXPECT_EQ(decoded.operands[0].text, "s5");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x1a8c,
        .size_bytes = 4,
        .words = {EncodeSop2Word(/*opcode=*/13, /*sdst=*/8, /*ssrc0=*/4, /*ssrc1=*/6)},
        .format_class = EncodedGcnInstFormatClass::Sop2,
        .mnemonic = "s_and_b64",
        .operands = "",
        .decoded_operands = {},
    };
    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 77u);
    EXPECT_EQ(decoded.mnemonic, "s_and_b64");
    ASSERT_EQ(decoded.operands.size(), 3u);
    EXPECT_EQ(decoded.operands[0].text, "s[8:9]");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x1a90,
        .size_bytes = 4,
        .words = {EncodeSop2Word(/*opcode=*/19, /*sdst=*/10, /*ssrc0=*/4, /*ssrc1=*/6)},
        .format_class = EncodedGcnInstFormatClass::Sop2,
        .mnemonic = "s_andn2_b64",
        .operands = "",
        .decoded_operands = {},
    };
    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 42u);
    EXPECT_EQ(decoded.mnemonic, "s_andn2_b64");
    ASSERT_EQ(decoded.operands.size(), 3u);
    EXPECT_EQ(decoded.operands[0].text, "s[10:11]");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x1a94,
        .size_bytes = 4,
        .words = {EncodeSop2Word(/*opcode=*/32, /*sdst=*/12, /*ssrc0=*/1, /*ssrc1=*/2)},
        .format_class = EncodedGcnInstFormatClass::Sop2,
        .mnemonic = "s_ashr_i32",
        .operands = "",
        .decoded_operands = {},
    };
    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 72u);
    EXPECT_EQ(decoded.mnemonic, "s_ashr_i32");
    ASSERT_EQ(decoded.operands.size(), 3u);
    EXPECT_EQ(decoded.operands[0].text, "s12");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x1a98,
        .size_bytes = 4,
        .words = {EncodeSop2Word(/*opcode=*/15, /*sdst=*/14, /*ssrc0=*/4, /*ssrc1=*/6)},
        .format_class = EncodedGcnInstFormatClass::Sop2,
        .mnemonic = "s_or_b64",
        .operands = "",
        .decoded_operands = {},
    };
    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 28u);
    EXPECT_EQ(decoded.mnemonic, "s_or_b64");
    ASSERT_EQ(decoded.operands.size(), 3u);
    EXPECT_EQ(decoded.operands[0].text, "s[14:15]");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x1a9a,
        .size_bytes = 4,
        .words = {EncodeSop2Word(/*opcode=*/17, /*sdst=*/18, /*ssrc0=*/4, /*ssrc1=*/6)},
        .format_class = EncodedGcnInstFormatClass::Sop2,
        .mnemonic = "s_xor_b64",
        .operands = "",
        .decoded_operands = {},
    };
    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.mnemonic, "s_xor_b64");
    ASSERT_EQ(decoded.operands.size(), 3u);
    EXPECT_EQ(decoded.operands[0].text, "s[18:19]");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x1a9c,
        .size_bytes = 4,
        .words = {EncodeSop2Word(/*opcode=*/11, /*sdst=*/16, /*ssrc0=*/4, /*ssrc1=*/6)},
        .format_class = EncodedGcnInstFormatClass::Sop2,
        .mnemonic = "s_cselect_b64",
        .operands = "",
        .decoded_operands = {},
    };
    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 41u);
    EXPECT_EQ(decoded.mnemonic, "s_cselect_b64");
    ASSERT_EQ(decoded.operands.size(), 3u);
    EXPECT_EQ(decoded.operands[0].text, "s[16:17]");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x1aa0,
        .size_bytes = 4,
        .words = {EncodeVop2Word(/*opcode=*/52, /*vdst=*/2, /*src0=*/257, /*vsrc1=*/3)},
        .format_class = EncodedGcnInstFormatClass::Vop2,
        .mnemonic = "v_add_u32_e32",
        .operands = "",
        .decoded_operands = {},
    };
    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 7u);
    EXPECT_EQ(decoded.mnemonic, "v_add_u32_e32");
    ASSERT_EQ(decoded.operands.size(), 3u);
    EXPECT_EQ(decoded.operands[0].text, "v2");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x1aa2,
        .size_bytes = 4,
        .words = {EncodeVop2Word(/*opcode=*/53, /*vdst=*/5, /*src0=*/257, /*vsrc1=*/6)},
        .format_class = EncodedGcnInstFormatClass::Vop2,
        .mnemonic = "v_sub_u32_e32",
        .operands = "",
        .decoded_operands = {},
    };
    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.mnemonic, "v_sub_u32_e32");
    ASSERT_EQ(decoded.operands.size(), 3u);
    EXPECT_EQ(decoded.operands[0].text, "v5");
    EXPECT_EQ(decoded.operands[1].text, "v1");
    EXPECT_EQ(decoded.operands[2].text, "v6");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x1aa4,
        .size_bytes = 4,
        .words = {EncodeVop2Word(/*opcode=*/59, /*vdst=*/4, /*src0=*/257, /*vsrc1=*/2)},
        .format_class = EncodedGcnInstFormatClass::Vop2,
        .mnemonic = "v_fmac_f32_e32",
        .operands = "",
        .decoded_operands = {},
    };
    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 65u);
    EXPECT_EQ(decoded.mnemonic, "v_fmac_f32_e32");
    ASSERT_EQ(decoded.operands.size(), 3u);
    EXPECT_EQ(decoded.operands[0].text, "v4");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x1aa8,
        .size_bytes = 4,
        .words = {EncodeVop1Word(/*opcode=*/8, /*vdst=*/5, /*src0=*/258)},
        .format_class = EncodedGcnInstFormatClass::Vop1,
        .mnemonic = "v_cvt_i32_f32_e32",
        .operands = "",
        .decoded_operands = {},
    };
    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 49u);
    EXPECT_EQ(decoded.mnemonic, "v_cvt_i32_f32_e32");
    ASSERT_EQ(decoded.operands.size(), 2u);
    EXPECT_EQ(decoded.operands[0].text, "v5");
    EXPECT_EQ(decoded.operands[1].text, "v2");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x1aac,
        .size_bytes = 4,
        .words = {EncodeVop1Word(/*opcode=*/32, /*vdst=*/6, /*src0=*/259)},
        .format_class = EncodedGcnInstFormatClass::Vop1,
        .mnemonic = "v_exp_f32_e32",
        .operands = "",
        .decoded_operands = {},
    };
    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 51u);
    EXPECT_EQ(decoded.mnemonic, "v_exp_f32_e32");
    ASSERT_EQ(decoded.operands.size(), 2u);
    EXPECT_EQ(decoded.operands[0].text, "v6");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x1ab0,
        .size_bytes = 4,
        .words = {EncodeVop1Word(/*opcode=*/34, /*vdst=*/7, /*src0=*/260)},
        .format_class = EncodedGcnInstFormatClass::Vop1,
        .mnemonic = "v_rcp_f32_e32",
        .operands = "",
        .decoded_operands = {},
    };
    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 52u);
    EXPECT_EQ(decoded.mnemonic, "v_rcp_f32_e32");
    ASSERT_EQ(decoded.operands.size(), 2u);
    EXPECT_EQ(decoded.operands[0].text, "v7");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x1ab4,
        .size_bytes = 4,
        .words = {EncodeVop2Word(/*opcode=*/2, /*vdst=*/8, /*src0=*/257, /*vsrc1=*/2)},
        .format_class = EncodedGcnInstFormatClass::Vop2,
        .mnemonic = "v_sub_f32_e32",
        .operands = "",
        .decoded_operands = {},
    };
    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 44u);
    EXPECT_EQ(decoded.mnemonic, "v_sub_f32_e32");
    ASSERT_EQ(decoded.operands.size(), 3u);
    EXPECT_EQ(decoded.operands[0].text, "v8");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x1ab8,
        .size_bytes = 4,
        .words = {EncodeVop2Word(/*opcode=*/11, /*vdst=*/9, /*src0=*/257, /*vsrc1=*/2)},
        .format_class = EncodedGcnInstFormatClass::Vop2,
        .mnemonic = "v_max_f32_e32",
        .operands = "",
        .decoded_operands = {},
    };
    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 46u);
    EXPECT_EQ(decoded.mnemonic, "v_max_f32_e32");
    ASSERT_EQ(decoded.operands.size(), 3u);
    EXPECT_EQ(decoded.operands[0].text, "v9");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x1abc,
        .size_bytes = 4,
        .words = {EncodeVop2Word(/*opcode=*/18, /*vdst=*/10, /*src0=*/3, /*vsrc1=*/4)},
        .format_class = EncodedGcnInstFormatClass::Vop2,
        .mnemonic = "v_lshlrev_b32_e32",
        .operands = "",
        .decoded_operands = {},
    };
    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 33u);
    EXPECT_EQ(decoded.mnemonic, "v_lshlrev_b32_e32");
    ASSERT_EQ(decoded.operands.size(), 3u);
    EXPECT_EQ(decoded.operands[1].text, "s3");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x1ac0,
        .size_bytes = 8,
        .words = EncodeVop3aWords(/*opcode=*/239, /*vdst=*/11, /*src0=*/257, /*src1=*/258, /*src2=*/259),
        .format_class = EncodedGcnInstFormatClass::Vop3a,
        .mnemonic = "v_div_fixup_f32",
        .operands = "",
        .decoded_operands = {},
    };
    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 60u);
    EXPECT_EQ(decoded.mnemonic, "v_div_fixup_f32");
    ASSERT_EQ(decoded.operands.size(), 4u);
    EXPECT_EQ(decoded.operands[0].text, "v11");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x1ac8,
        .size_bytes = 8,
        .words = EncodeVop3bWords(/*opcode=*/240, /*vdst=*/12, /*sdst=*/4, /*src0=*/257, /*src1=*/258, /*src2=*/259),
        .format_class = EncodedGcnInstFormatClass::Vop3a,
        .mnemonic = "v_div_scale_f32",
        .operands = "",
        .decoded_operands = {},
    };
    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 61u);
    EXPECT_EQ(decoded.mnemonic, "v_div_scale_f32");
    ASSERT_EQ(decoded.operands.size(), 4u);
    EXPECT_EQ(decoded.operands[0].text, "v12");
    EXPECT_EQ(decoded.operands[1].text, "v1");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x1ad0,
        .size_bytes = 8,
        .words = EncodeVop3aWords(/*opcode=*/241, /*vdst=*/13, /*src0=*/257, /*src1=*/258, /*src2=*/259),
        .format_class = EncodedGcnInstFormatClass::Vop3a,
        .mnemonic = "v_div_fmas_f32",
        .operands = "",
        .decoded_operands = {},
    };
    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 62u);
    EXPECT_EQ(decoded.mnemonic, "v_div_fmas_f32");
    ASSERT_EQ(decoded.operands.size(), 4u);
    EXPECT_EQ(decoded.operands[0].text, "v13");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x1ad8,
        .size_bytes = 8,
        .words = EncodeVop3aWords(/*opcode=*/254, /*vdst=*/14, /*src0=*/257, /*src1=*/258, /*src2=*/259),
        .format_class = EncodedGcnInstFormatClass::Vop3a,
        .mnemonic = "v_lshl_add_u32",
        .operands = "",
        .decoded_operands = {},
    };
    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 34u);
    EXPECT_EQ(decoded.mnemonic, "v_lshl_add_u32");
    ASSERT_EQ(decoded.operands.size(), 4u);
    EXPECT_EQ(decoded.operands[0].text, "v14");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x1ae0,
        .size_bytes = 8,
        .words = EncodeVop3bWords(/*opcode=*/142, /*vdst=*/15, /*sdst=*/4, /*src0=*/257, /*src1=*/258, /*src2=*/106),
        .format_class = EncodedGcnInstFormatClass::Vop3a,
        .mnemonic = "v_addc_co_u32_e64",
        .operands = "",
        .decoded_operands = {},
    };
    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 36u);
    EXPECT_EQ(decoded.mnemonic, "v_addc_co_u32_e64");
    ASSERT_EQ(decoded.operands.size(), 5u);
    EXPECT_EQ(decoded.operands[0].text, "v15");
    EXPECT_EQ(decoded.operands[1].text, "s[4:5]");
    EXPECT_EQ(decoded.operands[4].text, "s[106:107]");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x1ae8,
        .size_bytes = 8,
        .words = EncodeVop3aWords(/*opcode=*/98, /*vdst=*/0, /*src0=*/257, /*src1=*/258, /*src2=*/0),
        .format_class = EncodedGcnInstFormatClass::Vop3a,
        .mnemonic = "v_cmp_gt_i32_e64",
        .operands = "",
        .decoded_operands = {},
    };
    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 38u);
    EXPECT_EQ(decoded.mnemonic, "v_cmp_gt_i32_e64");
    ASSERT_EQ(decoded.operands.size(), 3u);
    EXPECT_EQ(decoded.operands[0].text, "s[0:1]");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x1aec,
        .size_bytes = 8,
        .words = EncodeVop3aWords(/*opcode=*/204, /*vdst=*/0, /*src0=*/257, /*src1=*/258, /*src2=*/0),
        .format_class = EncodedGcnInstFormatClass::Vop3a,
        .mnemonic = "v_cmp_gt_u32_e64",
        .operands = "",
        .decoded_operands = {},
    };
    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.mnemonic, "v_cmp_gt_u32_e64");
    ASSERT_EQ(decoded.operands.size(), 3u);
    EXPECT_EQ(decoded.operands[0].text, "s[0:1]");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x1af0,
        .size_bytes = 4,
        .words = {EncodeVopcWord(/*opcode=*/75, /*src0=*/257, /*vsrc1=*/2)},
        .format_class = EncodedGcnInstFormatClass::Vopc,
        .mnemonic = "v_cmp_ngt_f32_e32",
        .operands = "",
        .decoded_operands = {},
    };
    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 57u);
    EXPECT_EQ(decoded.mnemonic, "v_cmp_ngt_f32_e32");
    ASSERT_EQ(decoded.operands.size(), 3u);
    EXPECT_EQ(decoded.operands[0].text, "vcc");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x1af4,
        .size_bytes = 4,
        .words = {EncodeVopcWord(/*opcode=*/78, /*src0=*/257, /*vsrc1=*/2)},
        .format_class = EncodedGcnInstFormatClass::Vopc,
        .mnemonic = "v_cmp_nlt_f32_e32",
        .operands = "",
        .decoded_operands = {},
    };
    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 58u);
    EXPECT_EQ(decoded.mnemonic, "v_cmp_nlt_f32_e32");
    ASSERT_EQ(decoded.operands.size(), 3u);
    EXPECT_EQ(decoded.operands[0].text, "vcc");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x1af8,
        .size_bytes = 8,
        .words = EncodeVop3pWords(/*opcode=*/69, /*vdst=*/16, /*src0=*/257, /*src1=*/258, /*src2=*/259),
        .format_class = EncodedGcnInstFormatClass::Vop3p,
        .mnemonic = "v_mfma_f32_16x16x4f32",
        .operands = "",
        .decoded_operands = {},
    };
    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 67u);
    EXPECT_EQ(decoded.mnemonic, "v_mfma_f32_16x16x4f32");
    ASSERT_EQ(decoded.operands.size(), 4u);
    EXPECT_EQ(decoded.operands[0].text, "v[16:19]");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x1b00,
        .size_bytes = 8,
        .words = EncodeVop3pWords(/*opcode=*/73, /*vdst=*/20, /*src0=*/257, /*src1=*/258, /*src2=*/259),
        .format_class = EncodedGcnInstFormatClass::Vop3p,
        .mnemonic = "v_mfma_f32_16x16x4f16",
        .operands = "",
        .decoded_operands = {},
    };
    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 85u);
    EXPECT_EQ(decoded.mnemonic, "v_mfma_f32_16x16x4f16");
    ASSERT_EQ(decoded.operands.size(), 4u);
    EXPECT_EQ(decoded.operands[0].text, "v[20:23]");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x1b08,
        .size_bytes = 8,
        .words = EncodeVop3pWords(/*opcode=*/81, /*vdst=*/24, /*src0=*/257, /*src1=*/258, /*src2=*/259),
        .format_class = EncodedGcnInstFormatClass::Vop3p,
        .mnemonic = "v_mfma_i32_16x16x4i8",
        .operands = "",
        .decoded_operands = {},
    };
    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 86u);
    EXPECT_EQ(decoded.mnemonic, "v_mfma_i32_16x16x4i8");
    ASSERT_EQ(decoded.operands.size(), 4u);
    EXPECT_EQ(decoded.operands[0].text, "v[24:27]");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x1b10,
        .size_bytes = 8,
        .words = EncodeVop3pWords(/*opcode=*/105, /*vdst=*/28, /*src0=*/257, /*src1=*/258, /*src2=*/259),
        .format_class = EncodedGcnInstFormatClass::Vop3p,
        .mnemonic = "v_mfma_f32_16x16x2bf16",
        .operands = "",
        .decoded_operands = {},
    };
    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.encoding_id, 87u);
    EXPECT_EQ(decoded.mnemonic, "v_mfma_f32_16x16x2bf16");
    ASSERT_EQ(decoded.operands.size(), 4u);
    EXPECT_EQ(decoded.operands[0].text, "v[28:31]");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x1b18,
        .size_bytes = 8,
        .words = EncodeVop3pWords(/*opcode=*/68, /*vdst=*/32, /*src0=*/257, /*src1=*/258, /*src2=*/259),
        .format_class = EncodedGcnInstFormatClass::Vop3p,
        .mnemonic = "v_mfma_f32_32x32x2f32",
        .operands = "",
        .decoded_operands = {},
    };
    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.mnemonic, "v_mfma_f32_32x32x2f32");
    ASSERT_EQ(decoded.operands.size(), 4u);
    EXPECT_EQ(decoded.operands[0].text, "v[32:47]");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x1b20,
        .size_bytes = 8,
        .words = EncodeVop3pWords(/*opcode=*/85, /*vdst=*/48, /*src0=*/257, /*src1=*/258, /*src2=*/259),
        .format_class = EncodedGcnInstFormatClass::Vop3p,
        .mnemonic = "v_mfma_i32_16x16x16i8",
        .operands = "",
        .decoded_operands = {},
    };
    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.mnemonic, "v_mfma_i32_16x16x16i8");
    ASSERT_EQ(decoded.operands.size(), 4u);
    EXPECT_EQ(decoded.operands[0].text, "v[48:51]");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x1b28,
        .size_bytes = 8,
        .words = EncodeVop3pWords(/*opcode=*/88, /*vdst=*/6, /*src0=*/3, /*src1=*/0, /*src2=*/0),
        .format_class = EncodedGcnInstFormatClass::Vop3p,
        .mnemonic = "v_accvgpr_read_b32",
        .operands = "",
        .decoded_operands = {},
    };
    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.mnemonic, "v_accvgpr_read_b32");
    ASSERT_EQ(decoded.operands.size(), 2u);
    EXPECT_EQ(decoded.operands[0].text, "v6");
    EXPECT_EQ(decoded.operands[1].text, "a3");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x1b30,
        .size_bytes = 8,
        .words = EncodeVop3pWords(/*opcode=*/89, /*vdst=*/4, /*src0=*/257, /*src1=*/0, /*src2=*/0),
        .format_class = EncodedGcnInstFormatClass::Vop3p,
        .mnemonic = "v_accvgpr_write_b32",
        .operands = "",
        .decoded_operands = {},
    };
    const auto decoded = InstructionDecoder{}.Decode(raw);
    EXPECT_EQ(decoded.mnemonic, "v_accvgpr_write_b32");
    ASSERT_EQ(decoded.operands.size(), 2u);
    EXPECT_EQ(decoded.operands[0].text, "a4");
    EXPECT_EQ(decoded.operands[1].text, "v1");
  }

  {
    EncodedGcnInstruction raw{
        .pc = 0x1b38,
        .size_bytes = 8,
        .words = EncodeGlobalFlatWords(/*opcode=*/66, /*addr=*/1, /*data=*/2, /*saddr=*/2, /*vdst=*/0),
        .format_class = EncodedGcnInstFormatClass::Flat,
        .mnemonic = "global_atomic_add",
        .operands = "",
        .decoded_operands = {},
    };
    const auto decoded = InstructionDecoder{}.Decode(raw);
    const auto* expected = FindGeneratedGcnInstDefByMnemonic("global_atomic_add");
    ASSERT_NE(expected, nullptr);
    EXPECT_EQ(decoded.encoding_id, expected->id);
    EXPECT_EQ(decoded.mnemonic, "global_atomic_add");
    ASSERT_EQ(decoded.operands.size(), 3u);
    EXPECT_EQ(decoded.operands[0].text, "v1");
    EXPECT_EQ(decoded.operands[1].text, "v2");
    EXPECT_EQ(decoded.operands[2].text, "s[2:3]");
  }
}

}  // namespace
}  // namespace gpu_model
