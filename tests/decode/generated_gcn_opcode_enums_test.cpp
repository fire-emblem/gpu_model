#include <gtest/gtest.h>

#include "gpu_model/decode/generated_gcn_opcode_enums.h"

namespace gpu_model {
namespace {

TEST(GeneratedGcnOpcodeEnumsTest, ExposesOpTypeEncodingValues) {
  EXPECT_EQ(static_cast<uint16_t>(GcnOpTypeEncoding::SOP2), 0x2u);
  EXPECT_EQ(static_cast<uint16_t>(GcnOpTypeEncoding::SOP1), 0x17du);
  EXPECT_EQ(static_cast<uint16_t>(GcnOpTypeEncoding::VOP3A), 0x34u);
}

TEST(GeneratedGcnOpcodeEnumsTest, ResolvesNamesAndReverseNames) {
  EXPECT_EQ(ToString(GcnOpTypeEncoding::FLAT), "flat");
  const auto parsed = ParseGcnOpTypeEncoding("vopc");
  ASSERT_TRUE(parsed.has_value());
  EXPECT_EQ(*parsed, GcnOpTypeEncoding::VOPC);

  const auto* by_name = FindGcnOpcodeDescriptorByName("s_waitcnt");
  ASSERT_NE(by_name, nullptr);
  EXPECT_EQ(by_name->op_type, GcnOpTypeEncoding::SOPP);
  EXPECT_EQ(by_name->opcode, static_cast<uint16_t>(GcnSoppOpcode::S_WAITCNT));

  const auto* by_pair = FindGcnOpcodeDescriptor(
      GcnOpTypeEncoding::VOP3A, static_cast<uint16_t>(GcnVop3aOpcode::V_FMA_F32));
  ASSERT_NE(by_pair, nullptr);
  EXPECT_EQ(std::string_view(by_pair->name), "v_fma_f32");
}

}  // namespace
}  // namespace gpu_model
