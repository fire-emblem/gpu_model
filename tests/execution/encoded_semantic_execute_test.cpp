#include <gtest/gtest.h>

#include <bit>
#include <cstdint>
#include <string>
#include <vector>

#include "gpu_model/execution/encoded_semantic_handler.h"

namespace gpu_model {
namespace {

uint32_t FloatBits(float value) {
  return std::bit_cast<uint32_t>(value);
}

uint32_t EncodeSop2Word(uint32_t opcode, uint32_t sdst, uint32_t ssrc0, uint32_t ssrc1) {
  return 0x80000000u | (opcode << 23u) | (sdst << 16u) | (ssrc1 << 8u) | ssrc0;
}

uint32_t EncodeSop1Word(uint32_t opcode, uint32_t sdst, uint32_t ssrc0) {
  return 0xbe800000u | (sdst << 16u) | (opcode << 8u) | ssrc0;
}

class EncodedSemanticExecuteTest : public ::testing::Test {
 protected:
  struct Harness {
    WaveContext wave;
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
          context(wave, vcc, kernarg, 0, memory, stats, block) {
      wave.thread_count = kWaveSize;
      wave.ResetInitialExec();
      shared_memory.resize(256, std::byte{0});
    }
  };

  static DecodedInstructionOperand SReg(uint32_t index) {
    return DecodedInstructionOperand{
        .kind = DecodedInstructionOperandKind::ScalarReg,
        .text = "s" + std::to_string(index),
        .info =
            GcnOperandInfo{
                .reg_first = index,
                .reg_count = 1,
            },
    };
  }

  static DecodedInstructionOperand SRange(uint32_t first, uint32_t count) {
    return DecodedInstructionOperand{
        .kind = DecodedInstructionOperandKind::ScalarRegRange,
        .text = count == 1 ? ("s" + std::to_string(first))
                           : ("s[" + std::to_string(first) + ":" + std::to_string(first + count - 1) + "]"),
        .info =
            GcnOperandInfo{
                .reg_first = first,
                .reg_count = count,
            },
    };
  }

  static DecodedInstructionOperand VReg(uint32_t index) {
    return DecodedInstructionOperand{
        .kind = DecodedInstructionOperandKind::VectorReg,
        .text = "v" + std::to_string(index),
        .info =
            GcnOperandInfo{
                .reg_first = index,
                .reg_count = 1,
            },
    };
  }

  static DecodedInstructionOperand VRange(uint32_t first, uint32_t count) {
    return DecodedInstructionOperand{
        .kind = DecodedInstructionOperandKind::VectorRegRange,
        .text = count == 1 ? ("v" + std::to_string(first))
                           : ("v[" + std::to_string(first) + ":" + std::to_string(first + count - 1) + "]"),
        .info =
            GcnOperandInfo{
                .reg_first = first,
                .reg_count = count,
            },
    };
  }

  static DecodedInstructionOperand Special(GcnSpecialReg reg) {
    return DecodedInstructionOperand{
        .kind = DecodedInstructionOperandKind::SpecialReg,
        .text = reg == GcnSpecialReg::Vcc ? "vcc" : "exec",
        .info =
            GcnOperandInfo{
                .special_reg = reg,
            },
    };
  }

  static DecodedInstructionOperand Imm(int64_t value) {
    return DecodedInstructionOperand{
        .kind = DecodedInstructionOperandKind::Immediate,
        .text = std::to_string(value),
        .info =
            GcnOperandInfo{
                .immediate = value,
                .has_immediate = true,
            },
    };
  }

  static DecodedInstructionOperand BranchTarget(int64_t value) {
    return DecodedInstructionOperand{
        .kind = DecodedInstructionOperandKind::BranchTarget,
        .text = std::to_string(value),
        .info =
            GcnOperandInfo{
                .immediate = value,
                .has_immediate = true,
            },
    };
  }

  static DecodedInstruction Inst(std::string mnemonic,
                                    uint32_t encoding_id,
                                    std::initializer_list<DecodedInstructionOperand> operands,
                                    uint64_t pc = 0x1000,
                                    uint32_t size_bytes = 4,
                                    std::vector<uint32_t> words = {}) {
    DecodedInstruction instruction;
    instruction.pc = pc;
    instruction.size_bytes = size_bytes;
    instruction.encoding_id = encoding_id;
    instruction.mnemonic = std::move(mnemonic);
    instruction.words = std::move(words);
    instruction.operands.assign(operands.begin(), operands.end());
    return instruction;
  }
};

TEST_F(EncodedSemanticExecuteTest, ExecutesScalarMemoryLoadsX2AndX4) {
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

TEST_F(EncodedSemanticExecuteTest, ExecutesBranchAndConditionalBranchOnScc) {
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

TEST_F(EncodedSemanticExecuteTest, ExecutesAdditionalBranchControlInstructions) {
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

TEST_F(EncodedSemanticExecuteTest, ExecutesBarrierAndMarksWaveWaiting) {
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

TEST_F(EncodedSemanticExecuteTest, ExecutesSharedWriteAndRead) {
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

TEST_F(EncodedSemanticExecuteTest, ExecutesVectorFloatMathAndConvert) {
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

TEST_F(EncodedSemanticExecuteTest, ExecutesVectorCompareAndWritesVcc) {
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

TEST_F(EncodedSemanticExecuteTest, ExecutesScalarCompareInstructionsAndWritesScc) {
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

TEST_F(EncodedSemanticExecuteTest, ExecutesRepresentativeSop2ScalarAluInstructions) {
  {
    Harness harness;
    harness.wave.sgpr.Write(1, 7u);
    harness.wave.sgpr.Write(2, 9u);
    auto inst = Inst("s_add_u32", 69, {SReg(4), SReg(1), SReg(2)}, 0x1760, 4,
                     {EncodeSop2Word(/*opcode=*/0, /*sdst=*/4, /*ssrc0=*/1, /*ssrc1=*/2)});
    harness.wave.pc = inst.pc;
    RawGcnSemanticHandlerRegistry::Get(inst).Execute(inst, harness.context);
    EXPECT_EQ(harness.wave.sgpr.Read(4), 16u);
    EXPECT_FALSE(harness.wave.ScalarMaskBit0());
  }

  {
    Harness harness;
    harness.wave.sgpr.Write(1, 0xffffffffu);
    harness.wave.sgpr.Write(2, 0u);
    harness.wave.SetScalarMaskBit0(true);
    auto inst = Inst("s_addc_u32", 70, {SReg(5), SReg(1), SReg(2)}, 0x1764, 4,
                     {EncodeSop2Word(/*opcode=*/4, /*sdst=*/5, /*ssrc0=*/1, /*ssrc1=*/2)});
    harness.wave.pc = inst.pc;
    RawGcnSemanticHandlerRegistry::Get(inst).Execute(inst, harness.context);
    EXPECT_EQ(harness.wave.sgpr.Read(5), 0u);
    EXPECT_TRUE(harness.wave.ScalarMaskBit0());
  }

  {
    Harness harness;
    harness.wave.sgpr.Write(1, 0xf0f0ff00u);
    harness.wave.sgpr.Write(2, 0x0ff00ff0u);
    auto inst = Inst("s_and_b32", 5, {SReg(6), SReg(1), SReg(2)}, 0x1768, 4,
                     {EncodeSop2Word(/*opcode=*/12, /*sdst=*/6, /*ssrc0=*/1, /*ssrc1=*/2)});
    harness.wave.pc = inst.pc;
    RawGcnSemanticHandlerRegistry::Get(inst).Execute(inst, harness.context);
    EXPECT_EQ(harness.wave.sgpr.Read(6), 0x00f00f00u);
  }

  {
    Harness harness;
    harness.wave.sgpr.Write(1, 6u);
    harness.wave.sgpr.Write(2, 7u);
    auto inst = Inst("s_mul_i32", 6, {SReg(7), SReg(1), SReg(2)}, 0x176c, 4,
                     {EncodeSop2Word(/*opcode=*/36, /*sdst=*/7, /*ssrc0=*/1, /*ssrc1=*/2)});
    harness.wave.pc = inst.pc;
    RawGcnSemanticHandlerRegistry::Get(inst).Execute(inst, harness.context);
    EXPECT_EQ(harness.wave.sgpr.Read(7), 42u);
  }

  {
    Harness harness;
    harness.wave.sgpr.Write(1, 0x80u);
    harness.wave.sgpr.Write(2, 2u);
    auto inst = Inst("s_lshr_b32", 55, {SReg(8), SReg(1), SReg(2)}, 0x1770, 4,
                     {EncodeSop2Word(/*opcode=*/30, /*sdst=*/8, /*ssrc0=*/1, /*ssrc1=*/2)});
    harness.wave.pc = inst.pc;
    RawGcnSemanticHandlerRegistry::Get(inst).Execute(inst, harness.context);
    EXPECT_EQ(harness.wave.sgpr.Read(8), 0x20u);
  }

  {
    Harness harness;
    harness.wave.sgpr.Write(4, 0x0f0f0f0fu);
    harness.wave.sgpr.Write(5, 0xf0f0f0f0u);
    harness.wave.sgpr.Write(6, 0xffff0000u);
    harness.wave.sgpr.Write(7, 0x00ffff00u);
    auto inst = Inst("s_and_b64", 77, {SRange(10, 2), SRange(4, 2), SRange(6, 2)}, 0x1774, 4,
                     {EncodeSop2Word(/*opcode=*/13, /*sdst=*/10, /*ssrc0=*/4, /*ssrc1=*/6)});
    harness.wave.pc = inst.pc;
    RawGcnSemanticHandlerRegistry::Get(inst).Execute(inst, harness.context);
    EXPECT_EQ(harness.wave.sgpr.Read(10), 0x0f0f0000u);
    EXPECT_EQ(harness.wave.sgpr.Read(11), 0x00f0f000u);
  }

  {
    Harness harness;
    harness.wave.sgpr.Write(4, 0x0000ffffu);
    harness.wave.sgpr.Write(5, 0x00000000u);
    harness.wave.sgpr.Write(6, 0xffff0000u);
    harness.wave.sgpr.Write(7, 0xffffffffu);
    auto inst = Inst("s_or_b64", 28, {SRange(12, 2), SRange(4, 2), SRange(6, 2)}, 0x1778, 4,
                     {EncodeSop2Word(/*opcode=*/15, /*sdst=*/12, /*ssrc0=*/4, /*ssrc1=*/6)});
    harness.wave.pc = inst.pc;
    RawGcnSemanticHandlerRegistry::Get(inst).Execute(inst, harness.context);
    EXPECT_EQ(harness.wave.sgpr.Read(12), 0xffffffffu);
    EXPECT_EQ(harness.wave.sgpr.Read(13), 0xffffffffu);
  }

  {
    Harness harness;
    harness.wave.sgpr.Write(4, 0xffff0000u);
    harness.wave.sgpr.Write(5, 0x0f0f0f0fu);
    harness.wave.sgpr.Write(6, 0x00ff00ffu);
    harness.wave.sgpr.Write(7, 0x00ff00ffu);
    auto inst = Inst("s_andn2_b64", 42, {SRange(14, 2), SRange(4, 2), SRange(6, 2)}, 0x177c, 4,
                     {EncodeSop2Word(/*opcode=*/19, /*sdst=*/14, /*ssrc0=*/4, /*ssrc1=*/6)});
    harness.wave.pc = inst.pc;
    RawGcnSemanticHandlerRegistry::Get(inst).Execute(inst, harness.context);
    EXPECT_EQ(harness.wave.sgpr.Read(14), 0xff000000u);
    EXPECT_EQ(harness.wave.sgpr.Read(15), 0x0f000f00u);
  }

  {
    Harness harness;
    harness.wave.sgpr.Write(4, 0x11111111u);
    harness.wave.sgpr.Write(5, 0x22222222u);
    harness.wave.sgpr.Write(6, 0xaaaaaaaau);
    harness.wave.sgpr.Write(7, 0xbbbbbbbbu);
    harness.wave.SetScalarMaskBit0(true);
    auto inst = Inst("s_cselect_b64", 41, {SRange(16, 2), SRange(4, 2), SRange(6, 2)}, 0x1780, 4,
                     {EncodeSop2Word(/*opcode=*/11, /*sdst=*/16, /*ssrc0=*/4, /*ssrc1=*/6)});
    harness.wave.pc = inst.pc;
    RawGcnSemanticHandlerRegistry::Get(inst).Execute(inst, harness.context);
    EXPECT_EQ(harness.wave.sgpr.Read(16), 0x11111111u);
    EXPECT_EQ(harness.wave.sgpr.Read(17), 0x22222222u);
  }
}

TEST_F(EncodedSemanticExecuteTest, ExecutesAdditionalSop1ScalarAluInstructions) {
  {
    Harness harness;
    harness.wave.sgpr.Write(2, 0x89abcdefu);
    harness.wave.sgpr.Write(3, 0x01234567u);
    auto inst = Inst("s_mov_b64", 71, {SRange(8, 2), SRange(2, 2)}, 0x1784, 4,
                     {EncodeSop1Word(/*opcode=*/1, /*sdst=*/8, /*ssrc0=*/2)});
    harness.wave.pc = inst.pc;
    RawGcnSemanticHandlerRegistry::Get(inst).Execute(inst, harness.context);
    EXPECT_EQ(harness.wave.sgpr.Read(8), 0x89abcdefu);
    EXPECT_EQ(harness.wave.sgpr.Read(9), 0x01234567u);
    EXPECT_EQ(harness.wave.pc, 0x1788u);
  }

  {
    Harness harness;
    harness.wave.sgpr.Write(10, 0xf0f0ffffu);
    harness.wave.sgpr.Write(11, 0x0000000fu);
    auto inst = Inst("s_bcnt1_i32_b64", 83, {SReg(7), SRange(10, 2)}, 0x1788, 4,
                     {0xbe870d0au});
    harness.wave.pc = inst.pc;
    RawGcnSemanticHandlerRegistry::Get(inst).Execute(inst, harness.context);
    EXPECT_EQ(harness.wave.sgpr.Read(7), 28u);
    EXPECT_EQ(harness.wave.pc, 0x178cu);
  }
}

TEST_F(EncodedSemanticExecuteTest, ExecutesAdditionalVectorCompareInstructions) {
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

TEST_F(EncodedSemanticExecuteTest, ExecutesAdditionalVectorAluInstructionsAcrossFormats) {
  {
    Harness harness;
    harness.wave.sgpr.Write(1, 0x0f0f0000u);
    harness.wave.vgpr.Write(2, 0, 0x0000ff00u);
    auto inst = Inst("v_or_b32_e32", 0, {VReg(3), SReg(1), VReg(2)}, 0x178e, 4);
    harness.wave.pc = inst.pc;
    RawGcnSemanticHandlerRegistry::Get(inst).Execute(inst, harness.context);
    EXPECT_EQ(harness.wave.vgpr.Read(3, 0), 0x0f0fff00u);
  }

  {
    Harness harness;
    harness.wave.vgpr.Write(1, 0, 0x0f0f00ffu);
    auto inst = Inst("v_not_b32_e32", 32, {VReg(3), VReg(1)}, 0x1790, 4);
    harness.wave.pc = inst.pc;
    RawGcnSemanticHandlerRegistry::Get(inst).Execute(inst, harness.context);
    EXPECT_EQ(harness.wave.vgpr.Read(3, 0), 0xf0f0ff00u);
  }

  {
    Harness harness;
    harness.wave.vgpr.Write(2, 0, FloatBits(2.6f));
    auto inst = Inst("v_rndne_f32_e32", 50, {VReg(4), VReg(2)}, 0x1794, 4);
    harness.wave.pc = inst.pc;
    RawGcnSemanticHandlerRegistry::Get(inst).Execute(inst, harness.context);
    EXPECT_EQ(harness.wave.vgpr.Read(4, 0), FloatBits(3.0f));
  }

  {
    Harness harness;
    harness.wave.vgpr.Write(2, 0, FloatBits(3.0f));
    auto inst = Inst("v_exp_f32_e32", 51, {VReg(5), VReg(2)}, 0x1798, 4);
    harness.wave.pc = inst.pc;
    RawGcnSemanticHandlerRegistry::Get(inst).Execute(inst, harness.context);
    EXPECT_EQ(harness.wave.vgpr.Read(5, 0), FloatBits(8.0f));
  }

  {
    Harness harness;
    harness.wave.vgpr.Write(2, 0, FloatBits(4.0f));
    auto inst = Inst("v_rcp_f32_e32", 52, {VReg(6), VReg(2)}, 0x179c, 4);
    harness.wave.pc = inst.pc;
    RawGcnSemanticHandlerRegistry::Get(inst).Execute(inst, harness.context);
    EXPECT_EQ(harness.wave.vgpr.Read(6, 0), FloatBits(0.25f));
  }

  {
    Harness harness;
    harness.wave.vgpr.Write(8, 0, static_cast<uint32_t>(-32));
    auto inst = Inst("v_ashrrev_i32_e32", 14, {VReg(7), Imm(2), VReg(8)}, 0x17a0, 4);
    harness.wave.pc = inst.pc;
    RawGcnSemanticHandlerRegistry::Get(inst).Execute(inst, harness.context);
    EXPECT_EQ(harness.wave.vgpr.Read(7, 0), static_cast<uint32_t>(-8));
  }

  {
    Harness harness;
    harness.wave.vgpr.Write(1, 0, 0x0000000fu);
    harness.wave.vgpr.Write(2, 0, 0x000000f0u);
    harness.wave.vgpr.Write(3, 0, 0x00000f00u);
    auto inst = Inst("v_or3_b32", 0, {VReg(4), VReg(1), VReg(2), VReg(3)}, 0x17a2, 8);
    harness.wave.pc = inst.pc;
    RawGcnSemanticHandlerRegistry::Get(inst).Execute(inst, harness.context);
    EXPECT_EQ(harness.wave.vgpr.Read(4, 0), 0x00000fffu);
  }

  {
    Harness harness;
    harness.wave.vgpr.Write(1, 0, 10u);
    harness.wave.vgpr.Write(2, 0, 20u);
    harness.vcc = 0x1ull;
    auto inst = Inst("v_cndmask_b32_e32", 48, {VReg(9), VReg(1), VReg(2)}, 0x17a4, 4);
    harness.wave.pc = inst.pc;
    RawGcnSemanticHandlerRegistry::Get(inst).Execute(inst, harness.context);
    EXPECT_EQ(harness.wave.vgpr.Read(9, 0), 20u);
  }

  {
    Harness harness;
    harness.wave.vgpr.Write(1, 0, FloatBits(7.5f));
    harness.wave.vgpr.Write(2, 0, FloatBits(2.25f));
    auto inst = Inst("v_sub_f32_e32", 44, {VReg(10), VReg(1), VReg(2)}, 0x17a8, 4);
    harness.wave.pc = inst.pc;
    RawGcnSemanticHandlerRegistry::Get(inst).Execute(inst, harness.context);
    EXPECT_EQ(harness.wave.vgpr.Read(10, 0), FloatBits(5.25f));
  }

  {
    Harness harness;
    harness.wave.vgpr.Write(1, 0, FloatBits(1.5f));
    harness.wave.vgpr.Write(2, 0, 2u);
    auto inst = Inst("v_ldexp_f32", 63, {VReg(11), VReg(1), VReg(2)}, 0x17ac, 8);
    harness.wave.pc = inst.pc;
    RawGcnSemanticHandlerRegistry::Get(inst).Execute(inst, harness.context);
    EXPECT_EQ(harness.wave.vgpr.Read(11, 0), FloatBits(6.0f));
  }

  {
    Harness harness;
    harness.wave.vgpr.Write(1, 0, FloatBits(2.0f));
    harness.wave.vgpr.Write(2, 0, FloatBits(3.0f));
    harness.wave.vgpr.Write(3, 0, FloatBits(4.0f));
    auto inst = Inst("v_div_fmas_f32", 62, {VReg(12), VReg(1), VReg(2), VReg(3)}, 0x17b4, 8);
    harness.wave.pc = inst.pc;
    RawGcnSemanticHandlerRegistry::Get(inst).Execute(inst, harness.context);
    EXPECT_EQ(harness.wave.vgpr.Read(12, 0), FloatBits(10.0f));
  }

  {
    Harness harness;
    harness.wave.vgpr.Write(1, 0, FloatBits(0.0f));
    harness.wave.vgpr.Write(2, 0, FloatBits(2.0f));
    harness.wave.vgpr.Write(3, 0, FloatBits(6.0f));
    auto inst = Inst("v_div_fixup_f32", 60, {VReg(13), VReg(1), VReg(2), VReg(3)}, 0x17bc, 8);
    harness.wave.pc = inst.pc;
    RawGcnSemanticHandlerRegistry::Get(inst).Execute(inst, harness.context);
    EXPECT_EQ(harness.wave.vgpr.Read(13, 0), FloatBits(3.0f));
  }

  {
    Harness harness;
    harness.wave.vgpr.Write(1, 0, 3u);
    harness.wave.vgpr.Write(2, 0, 4u);
    harness.wave.vgpr.Write(3, 0, 5u);
    auto inst = Inst("v_lshl_add_u32", 34, {VReg(14), VReg(1), VReg(2), VReg(3)}, 0x17c4, 8);
    harness.wave.pc = inst.pc;
    RawGcnSemanticHandlerRegistry::Get(inst).Execute(inst, harness.context);
    EXPECT_EQ(harness.wave.vgpr.Read(14, 0), 53u);
  }

  {
    Harness harness;
    harness.wave.vgpr.Write(1, 4, 0b10111u);
    harness.wave.vgpr.Write(2, 4, 2u);
    auto inst = Inst("v_mbcnt_lo_u32_b32", 81, {VReg(15), VReg(1), VReg(2)}, 0x17cc, 8);
    harness.wave.pc = inst.pc;
    RawGcnSemanticHandlerRegistry::Get(inst).Execute(inst, harness.context);
    EXPECT_EQ(harness.wave.vgpr.Read(15, 4), 5u);
  }

  {
    Harness harness;
    harness.wave.vgpr.Write(1, 35, 0b1111u);
    harness.wave.vgpr.Write(2, 35, 1u);
    auto inst = Inst("v_mbcnt_hi_u32_b32", 82, {VReg(16), VReg(1), VReg(2)}, 0x17d4, 8);
    harness.wave.pc = inst.pc;
    RawGcnSemanticHandlerRegistry::Get(inst).Execute(inst, harness.context);
    EXPECT_EQ(harness.wave.vgpr.Read(16, 35), 4u);
  }
}

TEST_F(EncodedSemanticExecuteTest, ExecutesRemainingSupportedScalarAndVectorInstructions) {
  {
    Harness harness;
    harness.wave.sgpr.Write(1, 0x0000f0f0u);
    harness.wave.sgpr.Write(2, 0x0000000fu);
    auto inst = Inst("s_or_b32", 0, {SReg(6), SReg(1), SReg(2)}, 0x17da, 4);
    harness.wave.pc = inst.pc;
    RawGcnSemanticHandlerRegistry::Get(inst).Execute(inst, harness.context);
    EXPECT_EQ(harness.wave.sgpr.Read(6), 0x0000f0ffu);
  }

  {
    Harness harness;
    harness.wave.sgpr.Write(1, static_cast<uint32_t>(-7));
    harness.wave.sgpr.Write(2, 2u);
    auto inst = Inst("s_ashr_i32", 72, {SReg(6), SReg(1), SReg(2)}, 0x17dc, 4,
                     {EncodeSop2Word(/*opcode=*/32, /*sdst=*/6, /*ssrc0=*/1, /*ssrc1=*/2)});
    harness.wave.pc = inst.pc;
    RawGcnSemanticHandlerRegistry::Get(inst).Execute(inst, harness.context);
    EXPECT_EQ(harness.wave.sgpr.Read(6), static_cast<uint32_t>(-2));
  }

  {
    Harness harness;
    harness.wave.sgpr.Write(1, static_cast<uint32_t>(-5));
    harness.wave.sgpr.Write(2, 3u);
    auto inst = Inst("s_add_i32", 23, {SReg(5), SReg(1), SReg(2)}, 0x17e0, 4,
                     {EncodeSop2Word(/*opcode=*/2, /*sdst=*/5, /*ssrc0=*/1, /*ssrc1=*/2)});
    harness.wave.pc = inst.pc;
    RawGcnSemanticHandlerRegistry::Get(inst).Execute(inst, harness.context);
    EXPECT_EQ(harness.wave.sgpr.Read(5), static_cast<uint32_t>(-2));
  }

  {
    Harness harness;
    harness.wave.vgpr.Write(1, 0, 3u);
    auto inst = Inst("v_lshlrev_b32_e32", 33, {VReg(2), Imm(4), VReg(1)}, 0x17e4, 4);
    harness.wave.pc = inst.pc;
    RawGcnSemanticHandlerRegistry::Get(inst).Execute(inst, harness.context);
    EXPECT_EQ(harness.wave.vgpr.Read(2, 0), 48u);
  }

  {
    Harness harness;
    harness.wave.vgpr.Write(3, 0, FloatBits(3.75f));
    auto inst = Inst("v_cvt_i32_f32_e32", 49, {VReg(4), VReg(3)}, 0x17e8, 4);
    harness.wave.pc = inst.pc;
    RawGcnSemanticHandlerRegistry::Get(inst).Execute(inst, harness.context);
    EXPECT_EQ(harness.wave.vgpr.Read(4, 0), 3u);
  }

  {
    Harness harness;
    harness.wave.vgpr.Write(5, 0, FloatBits(1.0f));
    harness.wave.vgpr.Write(6, 0, FloatBits(2.0f));
    harness.wave.vgpr.Write(7, 0, FloatBits(4.0f));
    auto inst = Inst("v_fmac_f32_e32", 65, {VReg(7), VReg(5), VReg(6)}, 0x17ec, 4);
    harness.wave.pc = inst.pc;
    RawGcnSemanticHandlerRegistry::Get(inst).Execute(inst, harness.context);
    EXPECT_EQ(harness.wave.vgpr.Read(7, 0), FloatBits(6.0f));
  }

  {
    Harness harness;
    harness.wave.vgpr.Write(1, 0, 0xffffffffu);
    harness.wave.vgpr.Write(2, 0, 1u);
    harness.vcc = 0x1ull;
    auto inst = Inst("v_addc_co_u32_e64", 36,
                     {VReg(8), Special(GcnSpecialReg::Vcc), VReg(1), VReg(2), Special(GcnSpecialReg::Vcc)},
                     0x17f0, 8);
    harness.wave.pc = inst.pc;
    RawGcnSemanticHandlerRegistry::Get(inst).Execute(inst, harness.context);
    EXPECT_EQ(harness.wave.vgpr.Read(8, 0), 1u);
    EXPECT_EQ(harness.vcc & 0x1ull, 0x1ull);
  }

  {
    Harness harness;
    harness.wave.vgpr.Write(1, 0, FloatBits(9.0f));
    harness.wave.vgpr.Write(2, 0, FloatBits(3.0f));
    harness.wave.vgpr.Write(3, 0, FloatBits(0.0f));
    auto inst = Inst("v_div_scale_f32", 61, {VReg(9), VReg(1), VReg(2), VReg(3)}, 0x17f8, 8);
    harness.wave.pc = inst.pc;
    RawGcnSemanticHandlerRegistry::Get(inst).Execute(inst, harness.context);
    EXPECT_EQ(harness.wave.vgpr.Read(9, 0), FloatBits(9.0f));
  }

  {
    Harness harness;
    harness.wave.vgpr.Write(1, 0, static_cast<uint32_t>(7));
    harness.wave.vgpr.Write(2, 0, static_cast<uint32_t>(3));
    auto inst = Inst("v_cmp_gt_i32_e64", 38, {Special(GcnSpecialReg::Vcc), VReg(1), VReg(2)}, 0x1800, 8);
    harness.wave.pc = inst.pc;
    RawGcnSemanticHandlerRegistry::Get(inst).Execute(inst, harness.context);
    EXPECT_EQ(harness.vcc & 0x1ull, 0x1ull);
  }

  {
    Harness harness;
    harness.wave.vgpr.Write(1, 0, FloatBits(2.0f));
    harness.wave.vgpr.Write(2, 0, FloatBits(2.0f));
    auto inst =
        Inst("v_cmp_ngt_f32_e32", 57, {Special(GcnSpecialReg::Vcc), VReg(1), VReg(2)}, 0x1808, 4);
    harness.wave.pc = inst.pc;
    RawGcnSemanticHandlerRegistry::Get(inst).Execute(inst, harness.context);
    EXPECT_EQ(harness.vcc & 0x1ull, 0x1ull);
  }

  {
    Harness harness;
    harness.wave.vgpr.Write(1, 0, FloatBits(2.0f));
    harness.wave.vgpr.Write(2, 0, FloatBits(2.0f));
    auto inst =
        Inst("v_cmp_nlt_f32_e32", 58, {Special(GcnSpecialReg::Vcc), VReg(1), VReg(2)}, 0x180c, 4);
    harness.wave.pc = inst.pc;
    RawGcnSemanticHandlerRegistry::Get(inst).Execute(inst, harness.context);
    EXPECT_EQ(harness.vcc & 0x1ull, 0x1ull);
  }
}

TEST_F(EncodedSemanticExecuteTest, ExecutesRemainingExecCoverageInstructions) {
  {
    Harness harness;
    harness.wave.SetScalarMaskBit0(false);
    auto inst = Inst("s_cbranch_scc0", 26, {BranchTarget(3)}, 0x1810, 4);
    harness.wave.pc = inst.pc;
    RawGcnSemanticHandlerRegistry::Get(inst).Execute(inst, harness.context);
    EXPECT_EQ(harness.wave.pc, 0x1820u);
  }

  {
    Harness harness;
    harness.wave.sgpr.Write(1, static_cast<uint32_t>(-4));
    harness.wave.sgpr.Write(2, static_cast<uint32_t>(3));
    auto inst = Inst("s_cmp_lt_i32", 21, {SReg(1), SReg(2)}, 0x1814, 4, {0xbf040201u});
    harness.wave.pc = inst.pc;
    RawGcnSemanticHandlerRegistry::Get(inst).Execute(inst, harness.context);
    EXPECT_TRUE(harness.wave.ScalarMaskBit0());
  }

  {
    Harness harness;
    harness.wave.sgpr.Write(1, 0x00000003u);
    harness.wave.sgpr.Write(2, 0x00000000u);
    auto inst = Inst("s_lshl_b64", 73, {SRange(4, 2), SRange(1, 2), Imm(1)}, 0x1818, 4,
                     {0x8e848101u});
    harness.wave.pc = inst.pc;
    RawGcnSemanticHandlerRegistry::Get(inst).Execute(inst, harness.context);
    EXPECT_EQ(harness.wave.sgpr.Read(4), 0x6u);
    EXPECT_EQ(harness.wave.sgpr.Read(5), 0x0u);
  }

  {
    Harness harness;
    auto inst = Inst("s_mov_b32", 53, {SReg(6), Imm(42)}, 0x181c, 8, {0xbe8600ffu, 42u});
    harness.wave.pc = inst.pc;
    RawGcnSemanticHandlerRegistry::Get(inst).Execute(inst, harness.context);
    EXPECT_EQ(harness.wave.sgpr.Read(6), 42u);
  }

  {
    Harness harness;
    auto inst = Inst("s_movk_i32", 78, {SReg(7), Imm(-9)}, 0x1820, 4, {0xb007fff7u});
    harness.wave.pc = inst.pc;
    RawGcnSemanticHandlerRegistry::Get(inst).Execute(inst, harness.context);
    EXPECT_EQ(harness.wave.sgpr.Read(7), static_cast<uint32_t>(-9));
  }

  {
    Harness harness;
    harness.wave.vgpr.Write(1, 0, 11u);
    auto inst = Inst("v_mov_b32_e32", 13, {VReg(8), VReg(1)}, 0x1824, 4);
    harness.wave.pc = inst.pc;
    RawGcnSemanticHandlerRegistry::Get(inst).Execute(inst, harness.context);
    EXPECT_EQ(harness.wave.vgpr.Read(8, 0), 11u);
  }

  {
    Harness harness;
    harness.wave.vgpr.Write(1, 0, 0xffffffffu);
    harness.wave.vgpr.Write(2, 0, 1u);
    auto inst = Inst("v_add_co_u32_e32", 15,
                     {VReg(9), Special(GcnSpecialReg::Vcc), VReg(1), VReg(2)}, 0x1828, 4);
    harness.wave.pc = inst.pc;
    RawGcnSemanticHandlerRegistry::Get(inst).Execute(inst, harness.context);
    EXPECT_EQ(harness.wave.vgpr.Read(9, 0), 0u);
    EXPECT_EQ(harness.vcc & 0x1ull, 0x1ull);
  }

  {
    Harness harness;
    harness.wave.vgpr.Write(1, 0, 0xffffffffu);
    harness.wave.vgpr.Write(2, 0, 0u);
    harness.vcc = 0x1ull;
    auto inst = Inst("v_addc_co_u32_e32", 16,
                     {VReg(10), Special(GcnSpecialReg::Vcc), VReg(1), VReg(2), Special(GcnSpecialReg::Vcc)},
                     0x182c, 4);
    harness.wave.pc = inst.pc;
    RawGcnSemanticHandlerRegistry::Get(inst).Execute(inst, harness.context);
    EXPECT_EQ(harness.wave.vgpr.Read(10, 0), 0u);
    EXPECT_EQ(harness.vcc & 0x1ull, 0x1ull);
  }

  {
    Harness harness;
    harness.wave.vgpr.Write(1, 0, 0xffffffffu);
    harness.wave.vgpr.Write(2, 0, 1u);
    auto inst = Inst("v_add_co_u32_e64", 35,
                     {VReg(11), Special(GcnSpecialReg::Vcc), VReg(1), VReg(2)}, 0x1830, 8);
    harness.wave.pc = inst.pc;
    RawGcnSemanticHandlerRegistry::Get(inst).Execute(inst, harness.context);
    EXPECT_EQ(harness.wave.vgpr.Read(11, 0), 0u);
    EXPECT_EQ(harness.vcc & 0x1ull, 0x1ull);
  }

  {
    Harness harness;
    harness.wave.vgpr.Write(2, 0, 1u);
    harness.wave.vgpr.Write(3, 0, 0u);
    harness.wave.vgpr.Write(4, 0, 0u);
    harness.wave.vgpr.Write(5, 0, 0u);
    auto inst = Inst("v_lshlrev_b64", 17, {VRange(6, 2), Imm(1), VRange(2, 2)}, 0x1838, 8);
    harness.wave.pc = inst.pc;
    RawGcnSemanticHandlerRegistry::Get(inst).Execute(inst, harness.context);
    EXPECT_EQ(harness.wave.vgpr.Read(6, 0), 2u);
    EXPECT_EQ(harness.wave.vgpr.Read(7, 0), 0u);
  }

  {
    Harness harness;
    harness.wave.vgpr.Write(1, 0, 10u);
    harness.wave.vgpr.Write(2, 0, 20u);
    harness.vcc = 0x1ull;
    auto inst =
        Inst("v_cndmask_b32_e64", 59, {VReg(12), VReg(1), VReg(2), Special(GcnSpecialReg::Vcc)}, 0x1840, 8);
    harness.wave.pc = inst.pc;
    RawGcnSemanticHandlerRegistry::Get(inst).Execute(inst, harness.context);
    EXPECT_EQ(harness.wave.vgpr.Read(12, 0), 20u);
  }

  {
    Harness harness;
    harness.wave.vgpr.Write(1, 0, FloatBits(4.0f));
    harness.wave.vgpr.Write(2, 0, FloatBits(5.0f));
    harness.wave.vgpr.Write(3, 0, FloatBits(6.0f));
    auto inst = Inst("v_fma_f32", 25, {VReg(13), VReg(1), VReg(2), VReg(3)}, 0x1848, 8);
    harness.wave.pc = inst.pc;
    RawGcnSemanticHandlerRegistry::Get(inst).Execute(inst, harness.context);
    EXPECT_EQ(harness.wave.vgpr.Read(13, 0), FloatBits(26.0f));
  }

  {
    Harness harness;
    harness.wave.vgpr.Write(1, 0, FloatBits(8.0f));
    harness.wave.vgpr.Write(2, 0, FloatBits(3.0f));
    auto inst = Inst("v_max_f32_e32", 46, {VReg(14), VReg(1), VReg(2)}, 0x1850, 4);
    harness.wave.pc = inst.pc;
    RawGcnSemanticHandlerRegistry::Get(inst).Execute(inst, harness.context);
    EXPECT_EQ(harness.wave.vgpr.Read(14, 0), FloatBits(8.0f));
  }

  {
    Harness harness;
    harness.wave.vgpr.Write(1, 0, 7u);
    harness.wave.sgpr.Write(2, 3u);
    harness.wave.vgpr.Write(4, 0, 5u);
    harness.wave.vgpr.Write(5, 0, 0u);
    auto inst =
        Inst("v_mad_u64_u32", 79, {VRange(8, 2), SRange(10, 2), VReg(1), SReg(2), VRange(4, 2)}, 0x1854, 8);
    harness.wave.pc = inst.pc;
    RawGcnSemanticHandlerRegistry::Get(inst).Execute(inst, harness.context);
    EXPECT_EQ(harness.wave.vgpr.Read(8, 0), 26u);
    EXPECT_EQ(harness.wave.vgpr.Read(9, 0), 0u);
  }

  {
    Harness harness;
    const uint64_t base = harness.memory.AllocateGlobal(16);
    harness.wave.sgpr.Write(2, static_cast<uint32_t>(base));
    harness.wave.sgpr.Write(3, static_cast<uint32_t>(base >> 32u));
    harness.wave.vgpr.Write(1, 0, 4u);
    harness.wave.vgpr.Write(4, 0, 0xabcdef01u);
    auto inst = Inst("global_store_dword", 19, {VReg(1), SRange(2, 2), VReg(4)}, 0x185c, 8);
    harness.wave.pc = inst.pc;
    RawGcnSemanticHandlerRegistry::Get(inst).Execute(inst, harness.context);
    EXPECT_EQ(harness.memory.LoadGlobalValue<uint32_t>(base + 4), 0xabcdef01u);
    EXPECT_EQ(harness.stats.global_stores, 1u);
  }
}

TEST_F(EncodedSemanticExecuteTest, ExecutesTensorCoreMfmaVariants) {
  {
    Harness harness;
    harness.wave.vgpr.Write(1, 0, FloatBits(2.0f));
    harness.wave.vgpr.Write(2, 0, FloatBits(3.0f));
    harness.wave.vgpr.Write(3, 0, FloatBits(4.0f));
    auto inst = Inst("v_mfma_f32_16x16x4f32", 67, {VRange(8, 4), VReg(1), VReg(2), VReg(3)}, 0x1864, 8);
    harness.wave.pc = inst.pc;
    RawGcnSemanticHandlerRegistry::Get(inst).Execute(inst, harness.context);
    EXPECT_EQ(harness.wave.vgpr.Read(8, 0), FloatBits(28.0f));
    EXPECT_EQ(harness.wave.vgpr.Read(11, 0), FloatBits(28.0f));
    EXPECT_EQ(harness.wave.agpr.Read(8, 0), FloatBits(28.0f));
    EXPECT_EQ(harness.wave.agpr.Read(11, 0), FloatBits(28.0f));
  }

  {
    Harness harness;
    harness.wave.vgpr.Write(1, 0, 0x40003c00u);
    harness.wave.vgpr.Write(2, 0, 0x44004200u);
    harness.wave.vgpr.Write(3, 0, FloatBits(1.0f));
    auto inst = Inst("v_mfma_f32_16x16x4f16", 85, {VRange(12, 4), VReg(1), VReg(2), VReg(3)}, 0x186c, 8);
    harness.wave.pc = inst.pc;
    RawGcnSemanticHandlerRegistry::Get(inst).Execute(inst, harness.context);
    EXPECT_EQ(harness.wave.vgpr.Read(12, 0), FloatBits(12.0f));
    EXPECT_EQ(harness.wave.vgpr.Read(15, 0), FloatBits(12.0f));
  }

  {
    Harness harness;
    harness.wave.vgpr.Write(1, 0, 0x04030201u);
    harness.wave.vgpr.Write(2, 0, 0x01010101u);
    harness.wave.vgpr.Write(3, 0, 10u);
    auto inst = Inst("v_mfma_i32_16x16x4i8", 86, {VRange(16, 4), VReg(1), VReg(2), VReg(3)}, 0x1874, 8);
    harness.wave.pc = inst.pc;
    RawGcnSemanticHandlerRegistry::Get(inst).Execute(inst, harness.context);
    EXPECT_EQ(harness.wave.vgpr.Read(16, 0), 20u);
    EXPECT_EQ(harness.wave.vgpr.Read(19, 0), 20u);
  }

  {
    Harness harness;
    harness.wave.vgpr.Write(1, 0, 0x40003f80u);
    harness.wave.vgpr.Write(2, 0, 0x40404000u);
    harness.wave.vgpr.Write(3, 0, FloatBits(1.0f));
    auto inst = Inst("v_mfma_f32_16x16x2bf16", 87, {VRange(20, 4), VReg(1), VReg(2), VReg(3)}, 0x187c, 8);
    harness.wave.pc = inst.pc;
    RawGcnSemanticHandlerRegistry::Get(inst).Execute(inst, harness.context);
    EXPECT_EQ(harness.wave.vgpr.Read(20, 0), FloatBits(9.0f));
    EXPECT_EQ(harness.wave.vgpr.Read(23, 0), FloatBits(9.0f));
  }

  {
    Harness harness;
    harness.wave.vgpr.Write(1, 0, FloatBits(2.0f));
    harness.wave.vgpr.Write(2, 0, FloatBits(3.0f));
    harness.wave.vgpr.Write(3, 0, FloatBits(4.0f));
    auto inst = Inst("v_mfma_f32_32x32x2f32", 88, {VRange(24, 16), VReg(1), VReg(2), VReg(3)}, 0x1884, 8);
    harness.wave.pc = inst.pc;
    RawGcnSemanticHandlerRegistry::Get(inst).Execute(inst, harness.context);
    EXPECT_EQ(harness.wave.vgpr.Read(24, 0), FloatBits(16.0f));
    EXPECT_EQ(harness.wave.vgpr.Read(39, 0), FloatBits(16.0f));
  }

  {
    Harness harness;
    harness.wave.vgpr.Write(1, 0, 0x04030201u);
    harness.wave.vgpr.Write(2, 0, 0x01010101u);
    harness.wave.vgpr.Write(3, 0, 10u);
    auto inst = Inst("v_mfma_i32_16x16x16i8", 89, {VRange(40, 4), VReg(1), VReg(2), VReg(3)}, 0x188c, 8);
    harness.wave.pc = inst.pc;
    RawGcnSemanticHandlerRegistry::Get(inst).Execute(inst, harness.context);
    EXPECT_EQ(harness.wave.vgpr.Read(40, 0), 50u);
    EXPECT_EQ(harness.wave.vgpr.Read(43, 0), 50u);
  }

  {
    Harness harness;
    harness.wave.agpr.Write(3, 0, 0x12345678u);
    auto inst = Inst("v_accvgpr_read_b32", 90, {VReg(50), DecodedInstructionOperand{
                                                       .kind = DecodedInstructionOperandKind::AccumulatorReg,
                                                       .text = "a3",
                                                       .info = GcnOperandInfo{.reg_first = 3, .reg_count = 1},
                                                   }},
                     0x1894, 8);
    harness.wave.pc = inst.pc;
    RawGcnSemanticHandlerRegistry::Get(inst).Execute(inst, harness.context);
    EXPECT_EQ(harness.wave.vgpr.Read(50, 0), 0x12345678u);
  }

  {
    Harness harness;
    harness.wave.vgpr.Write(1, 0, FloatBits(2.0f));
    harness.wave.vgpr.Write(2, 0, FloatBits(3.0f));
    harness.wave.vgpr.Write(3, 0, FloatBits(4.0f));
    auto mfma = Inst("v_mfma_f32_16x16x4f32", 67, {VRange(8, 4), VReg(1), VReg(2), VReg(3)}, 0x18a0, 8);
    harness.wave.pc = mfma.pc;
    RawGcnSemanticHandlerRegistry::Get(mfma).Execute(mfma, harness.context);

    auto read = Inst("v_accvgpr_read_b32", 90, {VReg(60), DecodedInstructionOperand{
                                                         .kind = DecodedInstructionOperandKind::AccumulatorReg,
                                                         .text = "a8",
                                                         .info = GcnOperandInfo{.reg_first = 8, .reg_count = 1},
                                                     }},
                     0x18a8, 8);
    harness.wave.pc = read.pc;
    RawGcnSemanticHandlerRegistry::Get(read).Execute(read, harness.context);
    EXPECT_EQ(harness.wave.vgpr.Read(60, 0), FloatBits(28.0f));
  }

  {
    Harness harness;
    harness.wave.vgpr.Write(1, 0, 0x87654321u);
    auto inst = Inst("v_accvgpr_write_b32", 91,
                     {DecodedInstructionOperand{
                          .kind = DecodedInstructionOperandKind::AccumulatorReg,
                          .text = "a4",
                          .info = GcnOperandInfo{.reg_first = 4, .reg_count = 1},
                      },
                      VReg(1)},
                     0x189c, 8);
    harness.wave.pc = inst.pc;
    RawGcnSemanticHandlerRegistry::Get(inst).Execute(inst, harness.context);
    EXPECT_EQ(harness.wave.agpr.Read(4, 0), 0x87654321u);
  }
}

TEST_F(EncodedSemanticExecuteTest, ExecutesGlobalAtomicAddAndReturnsOldValue) {
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
