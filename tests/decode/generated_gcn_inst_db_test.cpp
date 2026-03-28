#include <gtest/gtest.h>

#include <algorithm>
#include <string_view>

#include "gpu_model/decode/generated_gcn_inst_db.h"

namespace gpu_model {
namespace {

TEST(GeneratedGcnInstDbTest, ExposesFormatDefinitions) {
  const auto formats = GeneratedGcnFormatDefs();
  ASSERT_FALSE(formats.empty());
  EXPECT_EQ(formats.front().format_class, GcnInstFormatClass::Sop2);
  EXPECT_STREQ(formats.front().id, "sop2");
}

TEST(GeneratedGcnInstDbTest, ExposesProfileAndSemanticMetadata) {
  const auto profiles = GeneratedGcnProfileDefs();
  ASSERT_GE(profiles.size(), 2u);
  EXPECT_STREQ(profiles.front().id, "gfx6_gfx8");
  EXPECT_EQ(profiles.front().wave_size, 64u);

  const auto semantic_families = GeneratedGcnSemanticFamilyDefs();
  ASSERT_FALSE(semantic_families.empty());
  EXPECT_STREQ(semantic_families.front().id, "scalar_alu");
  EXPECT_STREQ(semantic_families.front().exec_domain, "compute");
}

TEST(GeneratedGcnInstDbTest, ExposesFlagsAndImplicitRegisters) {
  const auto insts = GeneratedGcnInstDefs();
  const auto it = std::find_if(insts.begin(), insts.end(), [](const auto& def) {
    return std::string_view(def.mnemonic) == "global_atomic_add";
  });
  ASSERT_NE(it, insts.end());
  EXPECT_STREQ(it->exec_domain, "memory");
  EXPECT_NE((it->flags & kGcnInstFlagIsMemory), 0u);
  EXPECT_NE((it->flags & kGcnInstFlagIsAtomic), 0u);
  ASSERT_GT(it->implicit_count, 0u);

  const auto implicits = GeneratedGcnImplicitRegRefs();
  ASSERT_LT(it->implicit_begin, implicits.size());
  EXPECT_STREQ(implicits[it->implicit_begin].name, "exec");
  EXPECT_FALSE(implicits[it->implicit_begin].is_write);
}

TEST(GeneratedGcnInstDbTest, ExposesOperandSchema) {
  const auto insts = GeneratedGcnInstDefs();
  const auto it = std::find_if(insts.begin(), insts.end(), [](const auto& def) {
    return std::string_view(def.mnemonic) == "s_load_dword";
  });
  ASSERT_NE(it, insts.end());
  ASSERT_EQ(it->operand_count, 3u);

  const auto operands = GeneratedGcnOperandSpecs();
  ASSERT_LT(it->operand_begin + 2, operands.size());
  EXPECT_STREQ(operands[it->operand_begin + 0].name, "sdst");
  EXPECT_STREQ(operands[it->operand_begin + 0].role, "def");
  EXPECT_STREQ(operands[it->operand_begin + 1].kind, "scalar_reg_range");
  EXPECT_STREQ(operands[it->operand_begin + 2].field, "offset");
}

TEST(GeneratedGcnInstDbTest, ExposesBatchOperandSchemasForSimpleFormats) {
  const auto insts = GeneratedGcnInstDefs();
  const auto sop2 = std::find_if(insts.begin(), insts.end(), [](const auto& def) {
    return std::string_view(def.mnemonic) == "s_add_i32";
  });
  ASSERT_NE(sop2, insts.end());
  EXPECT_EQ(sop2->operand_count, 3u);

  const auto vopc = std::find_if(insts.begin(), insts.end(), [](const auto& def) {
    return std::string_view(def.mnemonic) == "v_cmp_eq_u32_e32";
  });
  ASSERT_NE(vopc, insts.end());
  EXPECT_EQ(vopc->operand_count, 3u);

  const auto sop1 = std::find_if(insts.begin(), insts.end(), [](const auto& def) {
    return std::string_view(def.mnemonic) == "s_mov_b64";
  });
  ASSERT_NE(sop1, insts.end());
  EXPECT_EQ(sop1->operand_count, 2u);
}

TEST(GeneratedGcnInstDbTest, ExposesImportedEncodingDefinitions) {
  const auto defs = GeneratedGcnEncodingDefs();
  ASSERT_GE(defs.size(), 84u);
  EXPECT_EQ(defs[0].id, 1u);
  EXPECT_EQ(defs[0].format_class, GcnInstFormatClass::Sopp);
  EXPECT_EQ(defs[0].mnemonic, "s_endpgm");
}

TEST(GeneratedGcnInstDbTest, PreservesRepresentativeFlatInstruction) {
  const auto defs = GeneratedGcnEncodingDefs();
  const auto it = std::find_if(defs.begin(), defs.end(), [](const auto& def) {
    return std::string_view(def.mnemonic) == "global_load_dword";
  });
  ASSERT_NE(it, defs.end());
  EXPECT_EQ(it->format_class, GcnInstFormatClass::Flat);
  EXPECT_EQ(it->size_bytes, 8u);
}

}  // namespace
}  // namespace gpu_model
