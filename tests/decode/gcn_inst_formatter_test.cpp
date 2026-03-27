#include <gtest/gtest.h>

#include "gpu_model/decode/gcn_inst_encoding_def.h"
#include "gpu_model/decode/gcn_inst_formatter.h"

namespace gpu_model {
namespace {

TEST(GcnInstFormatterTest, ResolvesKnownEncodingDefinitions) {
  const auto* s_endpgm = FindGcnInstEncodingDef({0xbf810000u});
  ASSERT_NE(s_endpgm, nullptr);
  EXPECT_EQ(s_endpgm->mnemonic, "s_endpgm");

  const auto* s_load = FindGcnInstEncodingDef({0xc0020002u, 0x0000002cu});
  ASSERT_NE(s_load, nullptr);
  EXPECT_EQ(s_load->mnemonic, "s_load_dword");
}

TEST(GcnInstFormatterTest, FormatsRawInstructionWithEncodingInfo) {
  RawGcnInstruction instruction{
      .pc = 0x1900,
      .size_bytes = 8,
      .words = {0xc0020002u, 0x0000002cu},
      .format_class = GcnInstFormatClass::Smrd,
      .encoding_id = 2,
      .mnemonic = "s_load_dword",
      .operands = "s0, s[4:5], 0x2c",
      .decoded_operands =
          {
              RawGcnOperand{.kind = RawGcnOperandKind::ScalarReg, .text = "s0"},
              RawGcnOperand{.kind = RawGcnOperandKind::ScalarRegRange, .text = "s[4:5]"},
              RawGcnOperand{.kind = RawGcnOperandKind::Immediate, .text = "0x2c"},
          },
  };

  const std::string text = GcnInstFormatter{}.Format(instruction);
  EXPECT_NE(text.find("s_load_dword s0, s[4:5], 0x2c"), std::string::npos);
  EXPECT_NE(text.find("format=smrd"), std::string::npos);
}

}  // namespace
}  // namespace gpu_model
