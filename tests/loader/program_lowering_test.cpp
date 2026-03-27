#include <gtest/gtest.h>

#include <span>
#include <vector>

#include "gpu_model/runtime/host_runtime.h"
#include "gpu_model/isa/target_isa.h"
#include "gpu_model/loader/program_lowering.h"

namespace gpu_model {
namespace {

TEST(ProgramLoweringTest, LowersCanonicalAsmProgramImage) {
  MetadataBlob metadata;
  SetTargetIsa(metadata, TargetIsa::CanonicalAsm);
  ProgramImage image("canonical_exit", "s_endpgm\n", metadata);

  const auto kernel = ProgramLoweringRegistry::Lower(image);
  ASSERT_EQ(kernel.instructions().size(), 1u);
  EXPECT_EQ(kernel.instructions()[0].opcode, Opcode::BExit);
}

TEST(ProgramLoweringTest, LowersGcnAsmSubsetProgramImage) {
  MetadataBlob metadata;
  SetTargetIsa(metadata, TargetIsa::GcnAsm);
  ProgramImage image("gcn_subset_exit", "s_endpgm\n", metadata);

  const auto kernel = ProgramLoweringRegistry::Lower(image);
  ASSERT_EQ(kernel.instructions().size(), 1u);
  EXPECT_EQ(kernel.instructions()[0].opcode, Opcode::BExit);
}

TEST(ProgramLoweringTest, LowersRepresentativeGcnControlFlowSubsetAndExecutesIt) {
  MetadataBlob metadata;
  SetTargetIsa(metadata, TargetIsa::GcnAsm);
  metadata.values["arch"] = "c500";
  ProgramImage image(
      "gcn_subset_store",
      R"(
        .meta arch=c500
        s_load_kernarg s0, 0
        s_load_kernarg s1, 1
        v_get_global_id_x v0
        v_cmp_gt_i32_e32 vcc, s1, v0
        s_and_saveexec_b64 s[2:3], vcc
        s_cbranch_execz exit
        v_mov_b32_e32 v1, 42
        buffer_store_dword s0, v0, v1, 4
      exit:
        s_endpgm
      )",
      metadata);

  HostRuntime runtime;
  constexpr uint32_t n = 70;
  std::vector<uint32_t> out(128, 0xffffffffu);

  const uint64_t out_addr = runtime.memory().AllocateGlobal(out.size() * sizeof(uint32_t));
  runtime.memory().WriteGlobal(out_addr,
                               std::as_bytes(std::span<const uint32_t>(out.data(), out.size())));

  LaunchRequest request;
  request.program_image = &image;
  request.config.grid_dim_x = 2;
  request.config.block_dim_x = 64;
  request.args.PushU64(out_addr);
  request.args.PushU32(n);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  for (uint32_t i = 0; i < out.size(); ++i) {
    const uint32_t actual =
        runtime.memory().LoadGlobalValue<uint32_t>(out_addr + i * sizeof(uint32_t));
    if (i < n) {
      EXPECT_EQ(actual, 42u);
    } else {
      EXPECT_EQ(actual, 0xffffffffu);
    }
  }
}

}  // namespace
}  // namespace gpu_model
