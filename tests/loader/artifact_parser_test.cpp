#include <gtest/gtest.h>

#include <cstring>
#include <algorithm>

#include "gpu_model/loader/artifact_parser.h"

namespace gpu_model {
namespace {

// Note: ParseSectionInfo and ParseKernelMetadataNotes depend on actual tool output
// format which varies. These tests focus on the stable API and data structures.

TEST(ArtifactParserTest, SelectKernelSymbolReturnsFirstFuncByName) {
  std::vector<SymbolInfo> syms = {
      {.name = "data", .type = "OBJECT", .value = 0, .size = 0},
      {.name = "kernel_a", .type = "FUNC", .value = 0x1000, .size = 100},
      {.name = "kernel_b", .type = "FUNC", .value = 0x2000, .size = 200},
  };

  const auto selected = ArtifactParser::SelectKernelSymbol(syms, std::nullopt);
  EXPECT_EQ(selected.name, "kernel_a");
}

TEST(ArtifactParserTest, SelectKernelSymbolReturnsNamedFunc) {
  std::vector<SymbolInfo> syms = {
      {.name = "kernel_a", .type = "FUNC", .value = 0x1000, .size = 100},
      {.name = "kernel_b", .type = "FUNC", .value = 0x2000, .size = 200},
  };

  const auto selected = ArtifactParser::SelectKernelSymbol(syms, "kernel_b");
  EXPECT_EQ(selected.name, "kernel_b");
}

TEST(ArtifactParserTest, SelectKernelSymbolThrowsIfNotFound) {
  std::vector<SymbolInfo> syms = {
      {.name = "kernel_a", .type = "FUNC", .value = 0x1000, .size = 100},
  };

  EXPECT_THROW(ArtifactParser::SelectKernelSymbol(syms, "nonexistent"), std::runtime_error);
}

TEST(ArtifactParserTest, SelectDescriptorSymbolReturnsObject) {
  std::vector<SymbolInfo> syms = {
      {.name = "my_kernel", .type = "FUNC", .value = 0x1000, .size = 100},
      {.name = "my_kernel.kd", .type = "OBJECT", .value = 0x2000, .size = 64},
  };

  const auto desc = ArtifactParser::SelectDescriptorSymbol(syms, "my_kernel.kd");
  EXPECT_EQ(desc.name, "my_kernel.kd");
  EXPECT_EQ(desc.type, "OBJECT");
}

TEST(ArtifactParserTest, ParseKernelDescriptorExtractsFields) {
  // Minimal 64-byte kernel descriptor
  std::vector<std::byte> bytes(64, std::byte{0});

  // Set some known fields
  uint32_t group_segment_size = 256;
  uint32_t private_segment_size = 128;
  uint32_t kernarg_size = 64;
  memcpy(bytes.data() + 0, &group_segment_size, sizeof(group_segment_size));
  memcpy(bytes.data() + 4, &private_segment_size, sizeof(private_segment_size));
  memcpy(bytes.data() + 8, &kernarg_size, sizeof(kernarg_size));

  const auto desc = ArtifactParser::ParseKernelDescriptor(bytes);
  EXPECT_EQ(desc.group_segment_fixed_size, 256);
  EXPECT_EQ(desc.private_segment_fixed_size, 128);
  EXPECT_EQ(desc.kernarg_size, 64);
}

TEST(ArtifactParserTest, ParseKernelDescriptorReturnsEmptyForSmallInput) {
  std::vector<std::byte> bytes(32, std::byte{0});
  const auto desc = ArtifactParser::ParseKernelDescriptor(bytes);
  EXPECT_EQ(desc.group_segment_fixed_size, 0);
  EXPECT_EQ(desc.private_segment_fixed_size, 0);
  EXPECT_EQ(desc.kernarg_size, 0);
}

}  // namespace
}  // namespace gpu_model
