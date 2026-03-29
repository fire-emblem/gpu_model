#include <gtest/gtest.h>

#include <bit>
#include <cstdint>
#include <string>
#include <vector>

#include "gpu_model/exec/raw_gcn_semantic_handler.h"

namespace gpu_model {
namespace {

uint32_t FloatBits(float value) {
  return std::bit_cast<uint32_t>(value);
}

class RawGcnSemanticExecuteTest : public ::testing::Test {
 protected:
  struct Harness {
    WaveState wave;
    uint64_t vcc = 0;
    std::vector<std::byte> kernarg;
    MemorySystem memory;
    ExecutionStats stats;
    std::vector<std::byte> shared_memory;
    uint64_t barrier_generation = 11;
    uint32_t barrier_arrivals = 0;
    RawGcnBlockContext block;
    RawGcnWaveContext context;

    Harness()
        : block{
              .shared_memory = shared_memory,
              .barrier_generation = barrier_generation,
              .barrier_arrivals = barrier_arrivals,
              .wave_count = 1,
          },
          context{
              .wave = wave,
              .vcc = vcc,
              .kernarg = kernarg,
              .kernarg_base = 0,
              .memory = memory,
              .stats = stats,
              .block = block,
          } {
      wave.thread_count = kWaveSize;
      wave.ResetInitialExec();
      shared_memory.resize(256, std::byte{0});
    }
  };

  static DecodedGcnOperand SReg(uint32_t index) {
    return DecodedGcnOperand{
        .kind = DecodedGcnOperandKind::ScalarReg,
        .text = "s" + std::to_string(index),
        .info =
            GcnOperandInfo{
                .reg_first = index,
                .reg_count = 1,
            },
    };
  }

  static DecodedGcnOperand SRange(uint32_t first, uint32_t count) {
    return DecodedGcnOperand{
        .kind = DecodedGcnOperandKind::ScalarRegRange,
        .text = count == 1 ? ("s" + std::to_string(first))
                           : ("s[" + std::to_string(first) + ":" + std::to_string(first + count - 1) + "]"),
        .info =
            GcnOperandInfo{
                .reg_first = first,
                .reg_count = count,
            },
    };
  }

  static DecodedGcnOperand VReg(uint32_t index) {
    return DecodedGcnOperand{
        .kind = DecodedGcnOperandKind::VectorReg,
        .text = "v" + std::to_string(index),
        .info =
            GcnOperandInfo{
                .reg_first = index,
                .reg_count = 1,
            },
    };
  }

  static DecodedGcnOperand Special(GcnSpecialReg reg) {
    return DecodedGcnOperand{
        .kind = DecodedGcnOperandKind::SpecialReg,
        .text = reg == GcnSpecialReg::Vcc ? "vcc" : "exec",
        .info =
            GcnOperandInfo{
                .special_reg = reg,
            },
    };
  }

  static DecodedGcnOperand Imm(int64_t value) {
    return DecodedGcnOperand{
        .kind = DecodedGcnOperandKind::Immediate,
        .text = std::to_string(value),
        .info =
            GcnOperandInfo{
                .immediate = value,
                .has_immediate = true,
            },
    };
  }

  static DecodedGcnOperand BranchTarget(int64_t value) {
    return DecodedGcnOperand{
        .kind = DecodedGcnOperandKind::BranchTarget,
        .text = std::to_string(value),
        .info =
            GcnOperandInfo{
                .immediate = value,
                .has_immediate = true,
            },
    };
  }

  static DecodedGcnInstruction Inst(std::string mnemonic,
                                    uint32_t encoding_id,
                                    std::initializer_list<DecodedGcnOperand> operands,
                                    uint64_t pc = 0x1000,
                                    uint32_t size_bytes = 4,
                                    std::vector<uint32_t> words = {}) {
    DecodedGcnInstruction instruction;
    instruction.pc = pc;
    instruction.size_bytes = size_bytes;
    instruction.encoding_id = encoding_id;
    instruction.mnemonic = std::move(mnemonic);
    instruction.words = std::move(words);
    instruction.operands.assign(operands.begin(), operands.end());
    return instruction;
  }
};

TEST_F(RawGcnSemanticExecuteTest, ExecutesScalarMemoryLoadsX2AndX4) {
  Harness harness;
  const uint64_t base = harness.memory.AllocateGlobal(64);
  harness.memory.StoreGlobalValue<uint32_t>(base + 0x10, 0x11111111u);
  harness.memory.StoreGlobalValue<uint32_t>(base + 0x14, 0x22222222u);
  harness.memory.StoreGlobalValue<uint32_t>(base + 0x20, 0xaaaa0001u);
  harness.memory.StoreGlobalValue<uint32_t>(base + 0x24, 0xbbbb0002u);
  harness.memory.StoreGlobalValue<uint32_t>(base + 0x28, 0xcccc0003u);
  harness.memory.StoreGlobalValue<uint32_t>(base + 0x2c, 0xdddd0004u);
  harness.wave.sgpr.Write(4, static_cast<uint32_t>(base));
  harness.wave.sgpr.Write(5, static_cast<uint32_t>(base >> 32u));

  auto load_x2 = Inst("s_load_dwordx2", 3, {SRange(6, 2), SRange(4, 2), Imm(0x10)}, 0x1100, 8);
  harness.wave.pc = load_x2.pc;
  RawGcnSemanticHandlerRegistry::Get(load_x2).Execute(load_x2, harness.context);
  EXPECT_EQ(harness.wave.sgpr.Read(6), 0x11111111u);
  EXPECT_EQ(harness.wave.sgpr.Read(7), 0x22222222u);
  EXPECT_EQ(harness.wave.pc, 0x1108u);

  auto load_x4 = Inst("s_load_dwordx4", 4, {SRange(8, 4), SRange(4, 2), Imm(0x20)}, 0x1200, 8);
  harness.wave.pc = load_x4.pc;
  RawGcnSemanticHandlerRegistry::Get(load_x4).Execute(load_x4, harness.context);
  EXPECT_EQ(harness.wave.sgpr.Read(8), 0xaaaa0001u);
  EXPECT_EQ(harness.wave.sgpr.Read(9), 0xbbbb0002u);
  EXPECT_EQ(harness.wave.sgpr.Read(10), 0xcccc0003u);
  EXPECT_EQ(harness.wave.sgpr.Read(11), 0xdddd0004u);
  EXPECT_EQ(harness.wave.pc, 0x1208u);
}

TEST_F(RawGcnSemanticExecuteTest, ExecutesBranchAndConditionalBranchOnScc) {
  {
    Harness harness;
    auto branch = Inst("s_branch", 27, {BranchTarget(5)}, 0x1300, 4);
    harness.wave.pc = branch.pc;
    RawGcnSemanticHandlerRegistry::Get(branch).Execute(branch, harness.context);
    EXPECT_EQ(harness.wave.pc, 0x1318u);
  }

  {
    Harness taken;
    taken.wave.SetScalarMaskBit0(true);
    auto cbranch = Inst("s_cbranch_scc1", 22, {BranchTarget(3)}, 0x1320, 4);
    taken.wave.pc = cbranch.pc;
    RawGcnSemanticHandlerRegistry::Get(cbranch).Execute(cbranch, taken.context);
    EXPECT_EQ(taken.wave.pc, 0x1330u);
  }

  {
    Harness not_taken;
    not_taken.wave.SetScalarMaskBit0(false);
    auto cbranch = Inst("s_cbranch_scc1", 22, {BranchTarget(3)}, 0x1320, 4);
    not_taken.wave.pc = cbranch.pc;
    RawGcnSemanticHandlerRegistry::Get(cbranch).Execute(cbranch, not_taken.context);
    EXPECT_EQ(not_taken.wave.pc, 0x1324u);
  }
}

TEST_F(RawGcnSemanticExecuteTest, ExecutesAdditionalBranchControlInstructions) {
  {
    Harness harness;
    harness.vcc = 0;
    auto cbranch = Inst("s_cbranch_vccz", 43, {BranchTarget(4)}, 0x1340, 4);
    harness.wave.pc = cbranch.pc;
    RawGcnSemanticHandlerRegistry::Get(cbranch).Execute(cbranch, harness.context);
    EXPECT_EQ(harness.wave.pc, 0x1354u);
  }

  {
    Harness harness;
    harness.vcc = 1;
    auto cbranch = Inst("s_cbranch_vccz", 43, {BranchTarget(4)}, 0x1340, 4);
    harness.wave.pc = cbranch.pc;
    RawGcnSemanticHandlerRegistry::Get(cbranch).Execute(cbranch, harness.context);
    EXPECT_EQ(harness.wave.pc, 0x1344u);
  }

  {
    Harness harness;
    harness.wave.exec.reset();
    auto cbranch = Inst("s_cbranch_execnz", 74, {BranchTarget(2)}, 0x1360, 4);
    harness.wave.pc = cbranch.pc;
    RawGcnSemanticHandlerRegistry::Get(cbranch).Execute(cbranch, harness.context);
    EXPECT_EQ(harness.wave.pc, 0x1364u);
  }

  {
    Harness harness;
    harness.wave.exec.set(0);
    auto cbranch = Inst("s_cbranch_execnz", 74, {BranchTarget(2)}, 0x1360, 4);
    harness.wave.pc = cbranch.pc;
    RawGcnSemanticHandlerRegistry::Get(cbranch).Execute(cbranch, harness.context);
    EXPECT_EQ(harness.wave.pc, 0x136cu);
  }

  {
    Harness harness;
    auto nop = Inst("s_nop", 68, {}, 0x1380, 4);
    harness.wave.pc = nop.pc;
    RawGcnSemanticHandlerRegistry::Get(nop).Execute(nop, harness.context);
    EXPECT_EQ(harness.wave.pc, 0x1384u);
    EXPECT_EQ(harness.wave.status, WaveStatus::Active);
  }
}

TEST_F(RawGcnSemanticExecuteTest, ExecutesBarrierAndMarksWaveWaiting) {
  Harness harness;
  auto barrier = Inst("s_barrier", 29, {}, 0x1400, 4);
  harness.wave.pc = barrier.pc;
  RawGcnSemanticHandlerRegistry::Get(barrier).Execute(barrier, harness.context);

  EXPECT_EQ(harness.stats.barriers, 1u);
  EXPECT_EQ(harness.barrier_arrivals, 1u);
  EXPECT_TRUE(harness.wave.waiting_at_barrier);
  EXPECT_EQ(harness.wave.status, WaveStatus::Stalled);
  EXPECT_EQ(harness.wave.barrier_generation, 11u);
}

TEST_F(RawGcnSemanticExecuteTest, ExecutesSharedWriteAndRead) {
  Harness harness;
  harness.wave.vgpr.Write(4, 0, 12u);
  harness.wave.vgpr.Write(5, 0, 0xdeadbeefu);
  auto write = Inst("ds_write_b32", 30, {VReg(4), VReg(5)}, 0x1500, 8);
  harness.wave.pc = write.pc;
  RawGcnSemanticHandlerRegistry::Get(write).Execute(write, harness.context);

  EXPECT_EQ(harness.stats.shared_stores, 1u);
  EXPECT_EQ(harness.wave.pc, 0x1508u);

  harness.wave.vgpr.Write(4, 0, 12u);
  auto read = Inst("ds_read_b32", 31, {VReg(6), VReg(4)}, 0x1510, 8);
  harness.wave.pc = read.pc;
  RawGcnSemanticHandlerRegistry::Get(read).Execute(read, harness.context);

  EXPECT_EQ(harness.stats.shared_loads, 1u);
  EXPECT_EQ(harness.wave.vgpr.Read(6, 0), 0xdeadbeefu);
  EXPECT_EQ(harness.wave.pc, 0x1518u);
}

TEST_F(RawGcnSemanticExecuteTest, ExecutesVectorFloatMathAndConvert) {
  Harness harness;
  harness.wave.vgpr.Write(1, 0, FloatBits(1.5f));
  harness.wave.vgpr.Write(2, 0, FloatBits(2.25f));
  auto add = Inst("v_add_f32_e32", 11, {VReg(3), VReg(1), VReg(2)}, 0x1600, 4);
  harness.wave.pc = add.pc;
  RawGcnSemanticHandlerRegistry::Get(add).Execute(add, harness.context);
  EXPECT_EQ(harness.wave.vgpr.Read(3, 0), FloatBits(3.75f));

  harness.wave.vgpr.Write(4, 0, FloatBits(3.0f));
  harness.wave.vgpr.Write(5, 0, FloatBits(4.0f));
  auto mul = Inst("v_mul_f32_e32", 64, {VReg(6), VReg(4), VReg(5)}, 0x1604, 4);
  harness.wave.pc = mul.pc;
  RawGcnSemanticHandlerRegistry::Get(mul).Execute(mul, harness.context);
  EXPECT_EQ(harness.wave.vgpr.Read(6, 0), FloatBits(12.0f));

  harness.wave.vgpr.Write(7, 0, static_cast<uint32_t>(-3));
  auto cvt = Inst("v_cvt_f32_i32_e32", 80, {VReg(8), VReg(7)}, 0x1608, 4);
  harness.wave.pc = cvt.pc;
  RawGcnSemanticHandlerRegistry::Get(cvt).Execute(cvt, harness.context);
  EXPECT_EQ(harness.wave.vgpr.Read(8, 0), FloatBits(-3.0f));
}

TEST_F(RawGcnSemanticExecuteTest, ExecutesVectorCompareAndWritesVcc) {
  Harness harness;
  harness.wave.vgpr.Write(1, 0, 7u);
  harness.wave.vgpr.Write(2, 0, 7u);
  harness.wave.vgpr.Write(1, 1, 7u);
  harness.wave.vgpr.Write(2, 1, 9u);

  auto compare = Inst("v_cmp_eq_u32_e32", 66, {Special(GcnSpecialReg::Vcc), VReg(1), VReg(2)}, 0x1700, 4);
  harness.wave.pc = compare.pc;
  RawGcnSemanticHandlerRegistry::Get(compare).Execute(compare, harness.context);

  EXPECT_EQ(harness.vcc & 0x3ull, 0x1ull);
  EXPECT_EQ(harness.wave.pc, 0x1704u);
}

TEST_F(RawGcnSemanticExecuteTest, ExecutesScalarCompareInstructionsAndWritesScc) {
  {
    Harness harness;
    harness.wave.sgpr.Write(1, 9u);
    harness.wave.sgpr.Write(2, 9u);
    auto compare = Inst("s_cmp_eq_u32", 24, {SReg(1), SReg(2)}, 0x1720, 4, {0xbf060201u});
    harness.wave.pc = compare.pc;
    RawGcnSemanticHandlerRegistry::Get(compare).Execute(compare, harness.context);
    EXPECT_TRUE(harness.wave.ScalarMaskBit0());
    EXPECT_EQ(harness.wave.pc, 0x1724u);
  }

  {
    Harness harness;
    harness.wave.sgpr.Write(1, 11u);
    harness.wave.sgpr.Write(2, 3u);
    auto compare = Inst("s_cmp_gt_u32", 39, {SReg(1), SReg(2)}, 0x1724, 4, {0xbf080201u});
    harness.wave.pc = compare.pc;
    RawGcnSemanticHandlerRegistry::Get(compare).Execute(compare, harness.context);
    EXPECT_TRUE(harness.wave.ScalarMaskBit0());
  }

  {
    Harness harness;
    harness.wave.sgpr.Write(1, 1u);
    harness.wave.sgpr.Write(2, 7u);
    auto compare = Inst("s_cmp_lt_u32", 40, {SReg(1), SReg(2)}, 0x1728, 4, {0xbf0a0201u});
    harness.wave.pc = compare.pc;
    RawGcnSemanticHandlerRegistry::Get(compare).Execute(compare, harness.context);
    EXPECT_TRUE(harness.wave.ScalarMaskBit0());
  }
}

TEST_F(RawGcnSemanticExecuteTest, ExecutesAdditionalVectorCompareInstructions) {
  {
    Harness harness;
    harness.wave.vgpr.Write(1, 0, 8u);
    harness.wave.vgpr.Write(2, 0, 6u);
    auto compare = Inst("v_cmp_gt_i32_e32", 8, {Special(GcnSpecialReg::Vcc), VReg(1), VReg(2)}, 0x1740, 4);
    harness.wave.pc = compare.pc;
    RawGcnSemanticHandlerRegistry::Get(compare).Execute(compare, harness.context);
    EXPECT_EQ(harness.vcc & 0x1ull, 0x1ull);
  }

  {
    Harness harness;
    harness.wave.vgpr.Write(3, 0, 12u);
    harness.wave.vgpr.Write(4, 0, 5u);
    auto compare =
        Inst("v_cmp_gt_u32_e32", 56, {Special(GcnSpecialReg::Vcc), VReg(3), VReg(4)}, 0x1744, 4);
    harness.wave.pc = compare.pc;
    RawGcnSemanticHandlerRegistry::Get(compare).Execute(compare, harness.context);
    EXPECT_EQ(harness.vcc & 0x1ull, 0x1ull);
  }

  {
    Harness harness;
    harness.wave.vgpr.Write(5, 0, static_cast<uint32_t>(-2));
    harness.wave.vgpr.Write(6, 0, static_cast<uint32_t>(-1));
    auto compare =
        Inst("v_cmp_le_i32_e32", 75, {Special(GcnSpecialReg::Vcc), VReg(5), VReg(6)}, 0x1748, 4);
    harness.wave.pc = compare.pc;
    RawGcnSemanticHandlerRegistry::Get(compare).Execute(compare, harness.context);
    EXPECT_EQ(harness.vcc & 0x1ull, 0x1ull);
  }

  {
    Harness harness;
    harness.wave.vgpr.Write(7, 0, static_cast<uint32_t>(-4));
    harness.wave.vgpr.Write(8, 0, static_cast<uint32_t>(2));
    auto compare =
        Inst("v_cmp_lt_i32_e32", 76, {Special(GcnSpecialReg::Vcc), VReg(7), VReg(8)}, 0x174c, 4);
    harness.wave.pc = compare.pc;
    RawGcnSemanticHandlerRegistry::Get(compare).Execute(compare, harness.context);
    EXPECT_EQ(harness.vcc & 0x1ull, 0x1ull);
  }
}

TEST_F(RawGcnSemanticExecuteTest, ExecutesGlobalAtomicAddAndReturnsOldValue) {
  Harness harness;
  const uint64_t base = harness.memory.AllocateGlobal(sizeof(uint32_t));
  harness.memory.StoreGlobalValue<uint32_t>(base, 10u);
  harness.wave.sgpr.Write(2, static_cast<uint32_t>(base));
  harness.wave.sgpr.Write(3, static_cast<uint32_t>(base >> 32u));
  harness.wave.vgpr.Write(1, 0, 5u);
  harness.wave.vgpr.Write(1, 1, 7u);

  auto atomic_add = Inst("global_atomic_add", 84, {VReg(0), VReg(1), SRange(2, 2)}, 0x1800, 8);
  harness.wave.pc = atomic_add.pc;
  RawGcnSemanticHandlerRegistry::Get(atomic_add).Execute(atomic_add, harness.context);

  EXPECT_EQ(harness.wave.vgpr.Read(0, 0), 10u);
  EXPECT_EQ(harness.wave.vgpr.Read(0, 1), 15u);
  EXPECT_EQ(harness.memory.LoadGlobalValue<uint32_t>(base), 22u);
  EXPECT_EQ(harness.stats.global_loads, 1u);
  EXPECT_EQ(harness.stats.global_stores, 1u);
  EXPECT_EQ(harness.wave.pc, 0x1808u);
}

}  // namespace
}  // namespace gpu_model
