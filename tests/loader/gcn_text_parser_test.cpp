#include <gtest/gtest.h>

#include "gpu_model/loader/gcn_text_parser.h"

namespace gpu_model {
namespace {

TEST(GcnTextParserTest, ParsesRegisterRangeSpecialRegisterAndOffOperands) {
  const auto range = GcnTextParser::ParseOperand("s[2:3]");
  ASSERT_EQ(range.kind, GcnTextOperandKind::RegisterRange);
  ASSERT_TRUE(range.reg_range.has_value());
  EXPECT_EQ(range.reg_range->prefix, 's');
  EXPECT_EQ(range.reg_range->first, 2u);
  EXPECT_EQ(range.reg_range->last, 3u);

  const auto special = GcnTextParser::ParseOperand("vcc");
  ASSERT_EQ(special.kind, GcnTextOperandKind::SpecialRegister);
  ASSERT_TRUE(special.special_reg.has_value());
  EXPECT_EQ(*special.special_reg, GcnSpecialRegister::Vcc);

  const auto off = GcnTextParser::ParseOperand("off");
  EXPECT_EQ(off.kind, GcnTextOperandKind::Off);
}

TEST(GcnTextParserTest, SplitsOperandsWithoutBreakingRegisterRanges) {
  const auto operands = GcnTextParser::SplitOperands("s[0:1], vcc, 0x10, off");
  ASSERT_EQ(operands.size(), 4u);
  EXPECT_EQ(operands[0], "s[0:1]");
  EXPECT_EQ(operands[1], "vcc");
  EXPECT_EQ(operands[2], "0x10");
  EXPECT_EQ(operands[3], "off");
}

TEST(GcnTextParserTest, ParsesInstructionLine) {
  const auto instruction =
      GcnTextParser::ParseInstruction("v_cmp_gt_i32_e32 vcc, s1, v0 // comment");
  EXPECT_EQ(instruction.mnemonic, "v_cmp_gt_i32_e32");
  ASSERT_EQ(instruction.operands.size(), 3u);
  EXPECT_EQ(instruction.operands[0].kind, GcnTextOperandKind::SpecialRegister);
  EXPECT_EQ(instruction.operands[1].kind, GcnTextOperandKind::Register);
  EXPECT_EQ(instruction.operands[2].kind, GcnTextOperandKind::Register);
}

}  // namespace
}  // namespace gpu_model
