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
