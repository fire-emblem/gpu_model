#include <gtest/gtest.h>

#include <span>
#include <vector>

#include "gpu_model/isa/target_isa.h"
#include "gpu_model/loader/program_lowering.h"
#include "gpu_model/program/program_object.h"
#include "gpu_model/runtime/runtime_engine.h"

namespace gpu_model {
namespace {

TEST(ProgramLoweringTest, LowersCanonicalAsmProgramObject) {
  MetadataBlob metadata;
  SetTargetIsa(metadata, TargetIsa::CanonicalAsm);
  ProgramObject image("canonical_exit", "s_endpgm\n", metadata);

  const auto kernel = ProgramLoweringRegistry::Lower(image);
  ASSERT_EQ(kernel.instructions().size(), 1u);
  EXPECT_EQ(kernel.instructions()[0].opcode, Opcode::BExit);
}

TEST(ProgramLoweringTest, LowersGcnAsmSubsetProgramObject) {
  MetadataBlob metadata;
  SetTargetIsa(metadata, TargetIsa::GcnAsm);
  ProgramObject image("gcn_subset_exit", "s_endpgm\n", metadata);

  const auto kernel = ProgramLoweringRegistry::Lower(image);
  ASSERT_EQ(kernel.instructions().size(), 1u);
  EXPECT_EQ(kernel.instructions()[0].opcode, Opcode::BExit);
}

TEST(ProgramLoweringTest, LowersRepresentativeGcnControlFlowSubsetAndExecutesIt) {
  MetadataBlob metadata;
  SetTargetIsa(metadata, TargetIsa::GcnAsm);
  metadata.values["arch"] = "c500";
  ProgramObject image(
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

  RuntimeEngine runtime;
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

TEST(ProgramLoweringTest, LowersGcnGlobalAddressLoadAndStoreSubset) {
  MetadataBlob metadata;
  SetTargetIsa(metadata, TargetIsa::GcnAsm);
  metadata.values["arch"] = "c500";
  ProgramObject image(
      "gcn_global_addr_subset",
      R"(
        .meta arch=c500
        s_load_kernarg s0, 0
        s_load_kernarg s1, 1
        s_load_kernarg s2, 2
        s_mov_b32 s20, 1
        v_get_global_id_x v0
        v_cmp_gt_i32_e32 vcc, s20, v0
        s_and_saveexec_b64 s[4:5], vcc
        s_cbranch_execz exit
        v_mov_b32_e32 v2, s1
        v_mov_b32_e32 v3, s2
        global_load_dword v4, v[2:3], off
        buffer_store_dword s0, v0, v4, 4
        v_mov_b32_e32 v5, 123
        global_store_dword v[2:3], v5, off
      exit:
        s_endpgm
      )",
      metadata);

  RuntimeEngine runtime;
  const uint64_t out_addr = runtime.memory().AllocateGlobal(sizeof(uint32_t));
  const uint64_t target_addr = runtime.memory().AllocateGlobal(sizeof(uint32_t));
  runtime.memory().StoreGlobalValue<uint32_t>(out_addr, 0u);
  runtime.memory().StoreGlobalValue<uint32_t>(target_addr, 88u);

  LaunchRequest request;
  request.program_image = &image;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;
  request.args.PushU64(out_addr);
  request.args.PushU64(target_addr & 0xffffffffULL);
  request.args.PushU64(target_addr >> 32ULL);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  EXPECT_EQ(runtime.memory().LoadGlobalValue<uint32_t>(out_addr), 88u);
  EXPECT_EQ(runtime.memory().LoadGlobalValue<uint32_t>(target_addr), 123u);
}

}  // namespace
}  // namespace gpu_model
