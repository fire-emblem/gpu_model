#include <gtest/gtest.h>

#include "gpu_model/decode/gcn_inst_encoding_def.h"
#include "gpu_model/decode/gcn_inst_formatter.h"
#include "gpu_model/instruction/encoded/decoded_instruction.h"

namespace gpu_model {
namespace {

TEST(InstructionFormatterTest, ResolvesKnownEncodingDefinitions) {
  const auto* s_endpgm = FindGcnInstEncodingDef({0xbf810000u});
  ASSERT_NE(s_endpgm, nullptr);
  EXPECT_EQ(s_endpgm->mnemonic, "s_endpgm");

  const auto* s_load = FindGcnInstEncodingDef({0xc0020002u, 0x0000002cu});
  ASSERT_NE(s_load, nullptr);
  EXPECT_EQ(s_load->mnemonic, "s_load_dword");
}

TEST(InstructionFormatterTest, FormatsRawInstructionWithEncodingInfo) {
  EncodedGcnInstruction instruction{
      .pc = 0x1900,
      .size_bytes = 8,
      .words = {0xc0020002u, 0x0000002cu},
      .format_class = GcnInstFormatClass::Smrd,
      .encoding_id = 2,
      .mnemonic = "s_load_dword",
      .operands = "s0, s[4:5], 0x2c",
      .decoded_operands =
          {
              EncodedGcnOperand{.kind = EncodedGcnOperandKind::ScalarReg, .text = "s0", .info = {}},
              EncodedGcnOperand{.kind = EncodedGcnOperandKind::ScalarRegRange, .text = "s[4:5]", .info = {}},
              EncodedGcnOperand{.kind = EncodedGcnOperandKind::Immediate, .text = "0x2c", .info = {}},
          },
  };

  const std::string text = GcnInstFormatter{}.Format(instruction);
  EXPECT_NE(text.find("s_load_dword s0, s[4:5], 0x2c"), std::string::npos);
  EXPECT_NE(text.find("format=smrd"), std::string::npos);
}

TEST(InstructionFormatterTest, FormatsDecodedInstructionWithDecodedOperands) {
  DecodedInstruction instruction{
      .pc = 0x1968,
      .size_bytes = 8,
      .encoding_id = 18,
      .format_class = GcnInstFormatClass::Flat,
      .words = {0xdc508000u, 0x067f0004u},
      .mnemonic = "global_load_dword",
      .operands =
          {
              DecodedInstructionOperand{.kind = DecodedInstructionOperandKind::VectorReg, .text = "v6", .info = {}},
              DecodedInstructionOperand{.kind = DecodedInstructionOperandKind::VectorRegRange, .text = "v[4:5]", .info = {}},
              DecodedInstructionOperand{.kind = DecodedInstructionOperandKind::Immediate, .text = "off", .info = {}},
          },
  };

  const std::string text = GcnInstFormatter{}.Format(instruction);
  EXPECT_NE(text.find("global_load_dword v6, v[4:5], off"), std::string::npos);
  EXPECT_NE(text.find("format=flat"), std::string::npos);
}

}  // namespace
}  // namespace gpu_model
