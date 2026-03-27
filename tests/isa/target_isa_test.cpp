#include <gtest/gtest.h>

#include "gpu_model/isa/target_isa.h"

namespace gpu_model {
namespace {

TEST(TargetIsaTest, ResolvesDefaultAndExplicitMetadata) {
  MetadataBlob default_metadata;
  EXPECT_EQ(ResolveTargetIsa(default_metadata), TargetIsa::CanonicalAsm);

  MetadataBlob gcn_metadata;
  SetTargetIsa(gcn_metadata, TargetIsa::GcnAsm);
  EXPECT_EQ(ResolveTargetIsa(gcn_metadata), TargetIsa::GcnAsm);
  EXPECT_EQ(gcn_metadata.values.at("target_isa"), "gcn_asm");
}

TEST(TargetIsaTest, ParsesTextNames) {
  EXPECT_EQ(ParseTargetIsa("canonical_asm"), TargetIsa::CanonicalAsm);
  EXPECT_EQ(ParseTargetIsa("gcn_asm"), TargetIsa::GcnAsm);
  EXPECT_THROW(ParseTargetIsa("unknown_isa"), std::invalid_argument);
}

}  // namespace
}  // namespace gpu_model
