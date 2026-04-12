#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "instruction/isa/opcode.h"
#include "program/loader/asm_parser.h"
#include "runtime/exec_engine.h"

namespace gpu_model {
namespace {

TEST(AsmParserTest, PreservesMetadataConstSegmentAndLabels) {
  ProgramObject image(
      "tiny_kernel",
      R"(
        .meta arch=mac500
        s_load_kernarg s0, 0
      exit:
        s_endpgm
      )",
      MetadataBlob{.values = {{"source", "asm"}}},
      ConstSegment{.bytes = {std::byte{0x01}, std::byte{0x02}, std::byte{0x03}}});

  const auto kernel = AsmParser{}.Parse(image);

  EXPECT_EQ(kernel.instructions().size(), 2u);
  EXPECT_EQ(kernel.ResolveLabel("exit"), 8u);
  EXPECT_EQ(kernel.metadata().values.at("arch"), "mac500");
  EXPECT_EQ(kernel.metadata().values.at("source"), "asm");
  EXPECT_EQ(kernel.const_segment().bytes.size(), 3u);
}

TEST(AsmParserTest, ResolvesLabelsInPcSpaceUsingInstructionEncodingSize) {
  ProgramObject image(
      "pc_size_asm",
      R"(
        .meta arch=mac500
        buffer_load_dword v1, s0, v0, 4
      exit:
        s_endpgm
      )");

  const auto kernel = AsmParser{}.Parse(image);

  ASSERT_EQ(kernel.instructions().size(), 2u);
  EXPECT_EQ(kernel.ResolveLabel("exit"), 8u);
  ASSERT_EQ(kernel.instructions_by_pc().size(), 2u);
  EXPECT_TRUE(kernel.ContainsPc(0u));
  EXPECT_TRUE(kernel.ContainsPc(8u));
  EXPECT_FALSE(kernel.ContainsPc(4u));
  EXPECT_EQ(kernel.entry_pc(), 0u);
  ASSERT_TRUE(kernel.NextPc(0u).has_value());
  EXPECT_EQ(*kernel.NextPc(0u), 8u);
}

TEST(AsmParserTest, LaunchesParsedVecAddKernelFunctionally) {
  ProgramObject image(
      "vecadd_asm",
      R"(
        .meta arch=mac500
        s_load_kernarg s0, 0
        s_load_kernarg s1, 1
        s_load_kernarg s2, 2
        s_load_kernarg s3, 3
        v_get_global_id_x v0
        v_cmp_lt_i32_cmask v0, s3
        s_saveexec_b64 s10
        s_and_exec_cmask_b64
        s_cbranch_execz exit
        buffer_load_dword v1, s0, v0, 4
        buffer_load_dword v2, s1, v0, 4
        v_add_i32 v3, v1, v2
        buffer_store_dword s2, v0, v3, 4
      exit:
        s_restoreexec_b64 s10
        s_endpgm
      )");

  const auto kernel = AsmParser{}.Parse(image);
  constexpr uint32_t n = 130;
  ExecEngine runtime;

  const uint64_t a_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  const uint64_t b_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  const uint64_t c_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  for (uint32_t i = 0; i < n; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(a_addr + i * sizeof(int32_t),
                                               static_cast<int32_t>(i));
    runtime.memory().StoreGlobalValue<int32_t>(b_addr + i * sizeof(int32_t),
                                               static_cast<int32_t>(10 + i));
    runtime.memory().StoreGlobalValue<int32_t>(c_addr + i * sizeof(int32_t), -1);
  }

  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 3;
  request.config.block_dim_x = 64;
  request.args.PushU64(a_addr);
  request.args.PushU64(b_addr);
  request.args.PushU64(c_addr);
  request.args.PushU32(n);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  for (uint32_t i = 0; i < n; ++i) {
    const int32_t actual =
        runtime.memory().LoadGlobalValue<int32_t>(c_addr + i * sizeof(int32_t));
    EXPECT_EQ(actual, static_cast<int32_t>(10 + 2 * static_cast<int32_t>(i)));
  }
}

TEST(AsmParserTest, ParsesWaitCntAssemblySyntax) {
  ProgramObject image(
      "waitcnt_asm",
      R"(
        .meta arch=mac500
        s_waitcnt vmcnt(0) & lgkmcnt(1)
        s_endpgm
      )");

  const auto kernel = AsmParser{}.Parse(image);
  ASSERT_EQ(kernel.instructions().size(), 2u);
  EXPECT_EQ(kernel.instructions()[0].opcode, Opcode::SWaitCnt);
  ASSERT_EQ(kernel.instructions()[0].operands.size(), 4u);
  EXPECT_EQ(kernel.instructions()[0].operands[0].immediate, 0u);
  EXPECT_EQ(kernel.instructions()[0].operands[1].immediate, 1u);
  EXPECT_EQ(kernel.instructions()[0].operands[2].immediate, 1u);
  EXPECT_EQ(kernel.instructions()[0].operands[3].immediate, 1u);
}

TEST(AsmParserTest, LaunchesParsedVectorFloatAddKernelFunctionally) {
  ProgramObject image(
      "vecadd_f32_asm",
      R"(
        .meta arch=mac500
        s_load_kernarg s0, 0
        s_load_kernarg s1, 1
        s_load_kernarg s2, 2
        s_load_kernarg s3, 3
        v_get_global_id_x v0
        v_cmp_lt_i32_cmask v0, s3
        s_saveexec_b64 s10
        s_and_exec_cmask_b64
        s_cbranch_execz exit
        buffer_load_dword v1, s0, v0, 4
        buffer_load_dword v2, s1, v0, 4
        v_add_f32 v3, v1, v2
        buffer_store_dword s2, v0, v3, 4
      exit:
        s_restoreexec_b64 s10
        s_endpgm
      )");

  const auto kernel = AsmParser{}.Parse(image);
  constexpr uint32_t n = 65;
  ExecEngine runtime;

  const uint64_t a_addr = runtime.memory().AllocateGlobal(n * sizeof(float));
  const uint64_t b_addr = runtime.memory().AllocateGlobal(n * sizeof(float));
  const uint64_t c_addr = runtime.memory().AllocateGlobal(n * sizeof(float));
  for (uint32_t i = 0; i < n; ++i) {
    runtime.memory().StoreGlobalValue<float>(a_addr + i * sizeof(float), 0.5f * static_cast<float>(i));
    runtime.memory().StoreGlobalValue<float>(b_addr + i * sizeof(float), 1.25f);
    runtime.memory().StoreGlobalValue<float>(c_addr + i * sizeof(float), -1.0f);
  }

  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 2;
  request.config.block_dim_x = 64;
  request.args.PushU64(a_addr);
  request.args.PushU64(b_addr);
  request.args.PushU64(c_addr);
  request.args.PushU32(n);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;

  for (uint32_t i = 0; i < n; ++i) {
    const float actual = runtime.memory().LoadGlobalValue<float>(c_addr + i * sizeof(float));
    EXPECT_FLOAT_EQ(actual, 0.5f * static_cast<float>(i) + 1.25f);
  }
}

TEST(AsmParserTest, LaunchesParsedGlobalAddressLoadAndStoreFunctionally) {
  ProgramObject image(
      "global_addr_asm",
      R"(
        .meta arch=mac500
        s_load_kernarg s0, 0
        s_load_kernarg s1, 1
        s_load_kernarg s2, 2
        s_mov_b32 s20, 1
        v_get_global_id_x v0
        v_cmp_lt_i32_cmask v0, s20
        s_saveexec_b64 s10
        s_and_exec_cmask_b64
        s_cbranch_execz load_exit
        v_mov_b32 v2, s1
        v_mov_b32 v3, s2
        global_load_dword_addr v4, v2, v3, 0
        buffer_store_dword s0, v0, v4, 4
      load_exit:
        s_restoreexec_b64 s10
        v_cmp_lt_i32_cmask v0, s20
        s_saveexec_b64 s11
        s_and_exec_cmask_b64
        s_cbranch_execz store_exit
        v_mov_b32 v5, 99
        global_store_dword_addr v2, v3, v5, 0
      store_exit:
        s_restoreexec_b64 s11
        s_endpgm
      )");

  const auto kernel = AsmParser{}.Parse(image);
  ExecEngine runtime;
  const uint64_t out_addr = runtime.memory().AllocateGlobal(sizeof(uint32_t));
  const uint64_t target_addr = runtime.memory().AllocateGlobal(sizeof(uint32_t));
  runtime.memory().StoreGlobalValue<uint32_t>(out_addr, 0u);
  runtime.memory().StoreGlobalValue<uint32_t>(target_addr, 77u);

  LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;
  request.args.PushU64(out_addr);
  request.args.PushU64(target_addr & 0xffffffffULL);
  request.args.PushU64(target_addr >> 32ULL);

  const auto result = runtime.Launch(request);
  ASSERT_TRUE(result.ok) << result.error_message;
  EXPECT_EQ(runtime.memory().LoadGlobalValue<uint32_t>(out_addr), 77u);
  EXPECT_EQ(runtime.memory().LoadGlobalValue<uint32_t>(target_addr), 99u);
}

}  // namespace
}  // namespace gpu_model
