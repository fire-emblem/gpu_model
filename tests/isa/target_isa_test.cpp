#include <gtest/gtest.h>

#include "gpu_model/isa/target_isa.h"

namespace gpu_model {
namespace {

TEST(TargetIsaTest, ResolvesDefaultAndExplicitMetadata) {
  MetadataBlob default_metadata;
  EXPECT_EQ(ResolveTargetIsa(default_metadata), TargetIsa::CanonicalAsm);

  MetadataBlob canonical_metadata;
  SetTargetIsa(canonical_metadata, TargetIsa::CanonicalAsm);
  EXPECT_EQ(ResolveTargetIsa(canonical_metadata), TargetIsa::CanonicalAsm);
  EXPECT_EQ(canonical_metadata.values.at("target_isa"), "canonical_asm");

  MetadataBlob legacy_gcn_metadata;
  legacy_gcn_metadata.values["target_isa"] = "gcn_asm";
  EXPECT_EQ(ResolveTargetIsa(legacy_gcn_metadata), TargetIsa::CanonicalAsm);

  MetadataBlob raw_metadata;
  SetTargetIsa(raw_metadata, TargetIsa::GcnRawAsm);
  EXPECT_EQ(ResolveTargetIsa(raw_metadata), TargetIsa::GcnRawAsm);
  EXPECT_EQ(raw_metadata.values.at("target_isa"), "gcn_raw_asm");
}

TEST(TargetIsaTest, ParsesTextNames) {
  EXPECT_EQ(ParseTargetIsa("canonical_asm"), TargetIsa::CanonicalAsm);
  EXPECT_EQ(ParseTargetIsa("gcn_asm"), TargetIsa::CanonicalAsm);
  EXPECT_EQ(ParseTargetIsa("gcn_raw_asm"), TargetIsa::GcnRawAsm);
  EXPECT_THROW(ParseTargetIsa("unknown_isa"), std::invalid_argument);
}

}  // namespace
}  // namespace gpu_model
