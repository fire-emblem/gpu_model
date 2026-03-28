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
}

TEST(GeneratedGcnInstDbTest, ExposesFlagsAndImplicitRegisters) {
  const auto insts = GeneratedGcnInstDefs();
  const auto it = std::find_if(insts.begin(), insts.end(), [](const auto& def) {
    return std::string_view(def.mnemonic) == "global_atomic_add";
  });
  ASSERT_NE(it, insts.end());
  EXPECT_NE((it->flags & kGcnInstFlagIsMemory), 0u);
  EXPECT_NE((it->flags & kGcnInstFlagIsAtomic), 0u);
  ASSERT_GT(it->implicit_count, 0u);

  const auto implicits = GeneratedGcnImplicitRegRefs();
  ASSERT_LT(it->implicit_begin, implicits.size());
  EXPECT_STREQ(implicits[it->implicit_begin].name, "exec");
  EXPECT_FALSE(implicits[it->implicit_begin].is_write);
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
