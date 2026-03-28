#include "gpu_model/exec/raw_gcn_semantic_handler.h"

#include <bit>
#include <bitset>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpu_model {

namespace {

uint32_t LaneCount(const RawGcnWaveContext& context) {
  return context.wave.thread_count < kWaveSize ? context.wave.thread_count : kWaveSize;
}

std::bitset<64> MaskFromU64(uint64_t value) {
  return std::bitset<64>(value);
}

uint32_t LoadU32(const std::vector<std::byte>& bytes, uint32_t offset) {
  uint32_t value = 0;
  std::memcpy(&value, bytes.data() + offset, sizeof(value));
  return value;
}

void StoreU32(std::vector<std::byte>& bytes, uint32_t offset, uint32_t value) {
  std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

uint64_t BranchTarget(uint64_t pc, int32_t simm16) {
  const int64_t target = static_cast<int64_t>(pc) + 4 + static_cast<int64_t>(simm16) * 4;
  return static_cast<uint64_t>(target);
}

float U32AsFloat(uint32_t bits) {
  return std::bit_cast<float>(bits);
}

uint32_t FloatAsU32(float value) {
  return std::bit_cast<uint32_t>(value);
}

bool DebugEnabled() {
  return std::getenv("GPU_MODEL_RAW_GCN_DEBUG") != nullptr;
}

void DebugLog(const char* fmt, ...) {
  if (!DebugEnabled()) {
    return;
  }
  va_list args;
  va_start(args, fmt);
  std::fputs("[gpu_model_raw_gcn] ", stderr);
  std::vfprintf(stderr, fmt, args);
  std::fputc('\n', stderr);
  va_end(args);
}

uint32_t RequireScalarIndex(const DecodedGcnOperand& operand) {
  if (operand.kind != DecodedGcnOperandKind::ScalarReg || operand.info.reg_count != 1) {
    throw std::invalid_argument("expected scalar register operand");
  }
  return operand.info.reg_first;
}

uint32_t RequireVectorIndex(const DecodedGcnOperand& operand) {
  if (operand.kind != DecodedGcnOperandKind::VectorReg || operand.info.reg_count != 1) {
    throw std::invalid_argument("expected vector register operand");
  }
  return operand.info.reg_first;
}

std::pair<uint32_t, uint32_t> RequireScalarRange(const DecodedGcnOperand& operand) {
  if (operand.kind != DecodedGcnOperandKind::ScalarRegRange || operand.info.reg_count == 0) {
    throw std::invalid_argument("expected scalar register range operand");
  }
  return {operand.info.reg_first, operand.info.reg_first + operand.info.reg_count - 1};
}

std::pair<uint32_t, uint32_t> RequireVectorRange(const DecodedGcnOperand& operand) {
  if (operand.kind != DecodedGcnOperandKind::VectorRegRange || operand.info.reg_count == 0) {
    throw std::invalid_argument("expected vector register range operand");
  }
  return {operand.info.reg_first, operand.info.reg_first + operand.info.reg_count - 1};
}

uint64_t ResolveScalarLike(const DecodedGcnOperand& operand, const RawGcnWaveContext& context) {
  if (operand.kind == DecodedGcnOperandKind::Immediate ||
      operand.kind == DecodedGcnOperandKind::BranchTarget) {
    if (!operand.info.has_immediate) {
      throw std::invalid_argument("immediate operand missing value");
    }
    return static_cast<uint64_t>(operand.info.immediate);
  }
  if (operand.kind == DecodedGcnOperandKind::ScalarReg) {
    return context.wave.sgpr.Read(RequireScalarIndex(operand));
  }
  if (operand.kind == DecodedGcnOperandKind::SpecialReg &&
      operand.info.special_reg == GcnSpecialReg::Vcc) {
    return context.vcc;
  }
  if (operand.kind == DecodedGcnOperandKind::SpecialReg &&
      operand.info.special_reg == GcnSpecialReg::Exec) {
    return context.wave.exec.to_ullong();
  }
  throw std::invalid_argument("unsupported scalar-like raw operand");
}

uint64_t ResolveScalarPair(const DecodedGcnOperand& operand, const RawGcnWaveContext& context) {
  if (operand.kind == DecodedGcnOperandKind::Immediate ||
      operand.kind == DecodedGcnOperandKind::BranchTarget) {
    if (!operand.info.has_immediate) {
      throw std::invalid_argument("scalar pair immediate missing value");
    }
    return static_cast<uint64_t>(operand.info.immediate);
  }
  if (operand.kind == DecodedGcnOperandKind::ScalarRegRange && operand.info.reg_count == 2) {
    const uint32_t first = operand.info.reg_first;
    return static_cast<uint64_t>(context.wave.sgpr.Read(first)) |
           (static_cast<uint64_t>(context.wave.sgpr.Read(first + 1)) << 32u);
  }
  if (operand.kind == DecodedGcnOperandKind::SpecialReg &&
      operand.info.special_reg == GcnSpecialReg::Vcc) {
    return context.vcc;
  }
  if (operand.kind == DecodedGcnOperandKind::SpecialReg &&
      operand.info.special_reg == GcnSpecialReg::Exec) {
    return context.wave.exec.to_ullong();
  }
  throw std::invalid_argument("unsupported scalar pair operand");
}

void StoreScalarPair(const DecodedGcnOperand& operand, RawGcnWaveContext& context, uint64_t value) {
  if (operand.kind == DecodedGcnOperandKind::ScalarRegRange && operand.info.reg_count == 2) {
    const uint32_t first = operand.info.reg_first;
    context.wave.sgpr.Write(first, static_cast<uint32_t>(value & 0xffffffffu));
    context.wave.sgpr.Write(first + 1, static_cast<uint32_t>(value >> 32u));
    return;
  }
  if (operand.kind == DecodedGcnOperandKind::SpecialReg &&
      operand.info.special_reg == GcnSpecialReg::Vcc) {
    context.vcc = value;
    return;
  }
  if (operand.kind == DecodedGcnOperandKind::SpecialReg &&
      operand.info.special_reg == GcnSpecialReg::Exec) {
    context.wave.exec = MaskFromU64(value);
    return;
  }
  throw std::invalid_argument("unsupported scalar pair destination");
}

uint64_t ResolveVectorLane(const DecodedGcnOperand& operand,
                           const RawGcnWaveContext& context,
                           uint32_t lane) {
  if (operand.kind == DecodedGcnOperandKind::Immediate) {
    if (!operand.info.has_immediate) {
      throw std::invalid_argument("immediate operand missing value");
    }
    return static_cast<uint64_t>(operand.info.immediate);
  }
  if (operand.kind == DecodedGcnOperandKind::ScalarReg) {
    return context.wave.sgpr.Read(RequireScalarIndex(operand));
  }
  if (operand.kind == DecodedGcnOperandKind::VectorReg) {
    return context.wave.vgpr.Read(RequireVectorIndex(operand), lane);
  }
  throw std::invalid_argument("unsupported vector-lane raw operand");
}

class ScalarMemoryHandler final : public IRawGcnSemanticHandler {
 public:
  void Execute(const DecodedGcnInstruction& instruction, RawGcnWaveContext& context) const override {
    if (instruction.mnemonic == "s_load_dword") {
      const uint32_t offset =
          static_cast<uint32_t>(ResolveScalarLike(instruction.operands.at(2), context));
      const uint32_t sdst = RequireScalarIndex(instruction.operands.at(0));
      context.wave.sgpr.Write(
          sdst, context.memory.LoadValue<uint32_t>(MemoryPoolKind::Kernarg, context.kernarg_base + offset));
    } else if (instruction.mnemonic == "s_load_dwordx2") {
      const uint32_t offset =
          static_cast<uint32_t>(ResolveScalarLike(instruction.operands.at(2), context));
      const auto [sdst, _] = RequireScalarRange(instruction.operands.at(0));
      const uint64_t value =
          context.memory.LoadValue<uint64_t>(MemoryPoolKind::Kernarg, context.kernarg_base + offset);
      context.wave.sgpr.Write(sdst, static_cast<uint32_t>(value & 0xffffffffu));
      context.wave.sgpr.Write(sdst + 1, static_cast<uint32_t>(value >> 32u));
    } else if (instruction.mnemonic == "s_load_dwordx4") {
      const uint32_t offset =
          static_cast<uint32_t>(ResolveScalarLike(instruction.operands.at(2), context));
      const auto [sdst, _] = RequireScalarRange(instruction.operands.at(0));
      context.wave.sgpr.Write(sdst + 0, context.memory.LoadValue<uint32_t>(
                                            MemoryPoolKind::Kernarg, context.kernarg_base + offset + 0));
      context.wave.sgpr.Write(sdst + 1, context.memory.LoadValue<uint32_t>(
                                            MemoryPoolKind::Kernarg, context.kernarg_base + offset + 4));
      context.wave.sgpr.Write(sdst + 2, context.memory.LoadValue<uint32_t>(
                                            MemoryPoolKind::Kernarg, context.kernarg_base + offset + 8));
      context.wave.sgpr.Write(sdst + 3, context.memory.LoadValue<uint32_t>(
                                            MemoryPoolKind::Kernarg, context.kernarg_base + offset + 12));
    } else {
      throw std::invalid_argument("unsupported scalar memory opcode: " + instruction.mnemonic);
    }
    context.wave.pc += instruction.size_bytes;
  }
};

class ScalarAluHandler final : public IRawGcnSemanticHandler {
 public:
  void Execute(const DecodedGcnInstruction& instruction, RawGcnWaveContext& context) const override {
    if (instruction.mnemonic == "s_cselect_b64") {
      const bool take_true = context.wave.ScalarMaskBit0();
      const uint64_t value = take_true ? ResolveScalarPair(instruction.operands.at(1), context)
                                       : ResolveScalarPair(instruction.operands.at(2), context);
      StoreScalarPair(instruction.operands.at(0), context, value);
    } else if (instruction.mnemonic == "s_andn2_b64") {
      const uint64_t lhs = ResolveScalarPair(instruction.operands.at(1), context);
      const uint64_t rhs = ResolveScalarPair(instruction.operands.at(2), context);
      StoreScalarPair(instruction.operands.at(0), context, lhs & ~rhs);
    } else if (instruction.mnemonic == "s_mov_b32") {
      const uint32_t sdst = RequireScalarIndex(instruction.operands.at(0));
      const uint32_t value =
          static_cast<uint32_t>(ResolveScalarLike(instruction.operands.at(1), context));
      context.wave.sgpr.Write(sdst, value);
    } else if (instruction.mnemonic == "s_or_b64") {
      const uint64_t lhs = ResolveScalarPair(instruction.operands.at(1), context);
      const uint64_t rhs = ResolveScalarPair(instruction.operands.at(2), context);
      StoreScalarPair(instruction.operands.at(0), context, lhs | rhs);
    } else if (instruction.mnemonic == "s_and_b32") {
      const uint32_t sdst = RequireScalarIndex(instruction.operands.at(0));
      const uint32_t lhs =
          static_cast<uint32_t>(ResolveScalarLike(instruction.operands.at(1), context));
      const uint32_t rhs =
          static_cast<uint32_t>(ResolveScalarLike(instruction.operands.at(2), context));
      context.wave.sgpr.Write(sdst, lhs & rhs);
    } else if (instruction.mnemonic == "s_mul_i32") {
      const uint32_t sdst = RequireScalarIndex(instruction.operands.at(0));
      const uint32_t lhs =
          static_cast<uint32_t>(ResolveScalarLike(instruction.operands.at(1), context));
      const uint32_t rhs =
          static_cast<uint32_t>(ResolveScalarLike(instruction.operands.at(2), context));
      context.wave.sgpr.Write(sdst, lhs * rhs);
    } else if (instruction.mnemonic == "s_add_i32") {
      const uint32_t sdst = RequireScalarIndex(instruction.operands.at(0));
      const uint32_t lhs =
          static_cast<uint32_t>(ResolveScalarLike(instruction.operands.at(1), context));
      const uint32_t rhs =
          static_cast<uint32_t>(ResolveScalarLike(instruction.operands.at(2), context));
      context.wave.sgpr.Write(sdst, lhs + rhs);
    } else if (instruction.mnemonic == "s_lshr_b32") {
      const uint32_t sdst = RequireScalarIndex(instruction.operands.at(0));
      const uint32_t lhs =
          static_cast<uint32_t>(ResolveScalarLike(instruction.operands.at(1), context));
      const uint32_t rhs =
          static_cast<uint32_t>(ResolveScalarLike(instruction.operands.at(2), context));
      context.wave.sgpr.Write(sdst, lhs >> (rhs & 31u));
    } else {
      throw std::invalid_argument("unsupported scalar alu opcode: " + instruction.mnemonic);
    }
    context.wave.pc += instruction.size_bytes;
  }
};

class ScalarCompareHandler final : public IRawGcnSemanticHandler {
 public:
  void Execute(const DecodedGcnInstruction& instruction, RawGcnWaveContext& context) const override {
    if (instruction.mnemonic == "s_cmp_lt_i32") {
      const int32_t lhs =
          static_cast<int32_t>(ResolveScalarLike(instruction.operands.at(0), context));
      const int32_t rhs =
          static_cast<int32_t>(ResolveScalarLike(instruction.operands.at(1), context));
      context.wave.SetScalarMaskBit0(lhs < rhs);
    } else if (instruction.mnemonic == "s_cmp_eq_u32") {
      const uint32_t lhs =
          static_cast<uint32_t>(ResolveScalarLike(instruction.operands.at(0), context));
      const uint32_t rhs =
          static_cast<uint32_t>(ResolveScalarLike(instruction.operands.at(1), context));
      context.wave.SetScalarMaskBit0(lhs == rhs);
    } else if (instruction.mnemonic == "s_cmp_gt_u32") {
      const uint32_t lhs =
          static_cast<uint32_t>(ResolveScalarLike(instruction.operands.at(0), context));
      const uint32_t rhs =
          static_cast<uint32_t>(ResolveScalarLike(instruction.operands.at(1), context));
      context.wave.SetScalarMaskBit0(lhs > rhs);
    } else if (instruction.mnemonic == "s_cmp_lt_u32") {
      const uint32_t lhs =
          static_cast<uint32_t>(ResolveScalarLike(instruction.operands.at(0), context));
      const uint32_t rhs =
          static_cast<uint32_t>(ResolveScalarLike(instruction.operands.at(1), context));
      context.wave.SetScalarMaskBit0(lhs < rhs);
    } else {
      throw std::invalid_argument("unsupported scalar compare opcode: " + instruction.mnemonic);
    }
    context.wave.pc += instruction.size_bytes;
  }
};

class VectorAluHandler final : public IRawGcnSemanticHandler {
 public:
  void Execute(const DecodedGcnInstruction& instruction, RawGcnWaveContext& context) const override {
    if (instruction.mnemonic == "v_not_b32_e32") {
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        const uint32_t src =
            static_cast<uint32_t>(ResolveVectorLane(instruction.operands.at(1), context, lane));
        context.wave.vgpr.Write(vdst, lane, ~src);
      }
    } else if (instruction.mnemonic == "v_add_u32_e32") {
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        const uint32_t lhs =
            static_cast<uint32_t>(ResolveVectorLane(instruction.operands.at(1), context, lane));
        const uint32_t rhs =
            static_cast<uint32_t>(ResolveVectorLane(instruction.operands.at(2), context, lane));
        context.wave.vgpr.Write(vdst, lane, lhs + rhs);
      }
    } else if (instruction.mnemonic == "v_ashrrev_i32_e32") {
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        const uint32_t imm =
            static_cast<uint32_t>(ResolveVectorLane(instruction.operands.at(1), context, lane));
        const int32_t rhs =
            static_cast<int32_t>(ResolveVectorLane(instruction.operands.at(2), context, lane));
        context.wave.vgpr.Write(vdst, lane, static_cast<uint32_t>(rhs >> (imm & 31u)));
      }
    } else if (instruction.mnemonic == "v_lshlrev_b64") {
      const auto [vdst, _] = RequireVectorRange(instruction.operands.at(0));
      const uint32_t shift =
          static_cast<uint32_t>(ResolveVectorLane(instruction.operands.at(1), context, 0));
      const uint32_t src_pair = instruction.operands.at(2).kind == DecodedGcnOperandKind::VectorRegRange
                                    ? RequireVectorRange(instruction.operands.at(2)).first
                                    : RequireVectorIndex(instruction.operands.at(2));
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        const uint64_t lo = static_cast<uint32_t>(context.wave.vgpr.Read(src_pair, lane));
        const uint64_t hi = static_cast<uint32_t>(context.wave.vgpr.Read(src_pair + 1, lane));
        const uint64_t value = ((hi << 32u) | lo) << (shift & 63u);
        context.wave.vgpr.Write(vdst, lane, static_cast<uint32_t>(value & 0xffffffffu));
        context.wave.vgpr.Write(vdst + 1, lane, static_cast<uint32_t>(value >> 32u));
      }
    } else if (instruction.mnemonic == "v_mov_b32_e32") {
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        context.wave.vgpr.Write(vdst, lane, static_cast<uint32_t>(
                                                ResolveVectorLane(instruction.operands.at(1), context, lane)));
      }
    } else if (instruction.mnemonic == "v_lshlrev_b32_e32") {
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        const uint32_t shift =
            static_cast<uint32_t>(ResolveVectorLane(instruction.operands.at(1), context, lane));
        const uint32_t src =
            static_cast<uint32_t>(ResolveVectorLane(instruction.operands.at(2), context, lane));
        context.wave.vgpr.Write(vdst, lane, src << (shift & 31u));
      }
    } else if (instruction.mnemonic == "v_lshl_add_u32") {
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        const uint32_t src0 =
            static_cast<uint32_t>(ResolveVectorLane(instruction.operands.at(1), context, lane));
        const uint32_t shift =
            static_cast<uint32_t>(ResolveVectorLane(instruction.operands.at(2), context, lane));
        const uint32_t src2 =
            static_cast<uint32_t>(ResolveVectorLane(instruction.operands.at(3), context, lane));
        context.wave.vgpr.Write(vdst, lane, (src0 << (shift & 31u)) + src2);
      }
    } else if (instruction.mnemonic == "v_add_co_u32_e32") {
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      context.vcc = 0;
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        const uint64_t lhs =
            static_cast<uint32_t>(ResolveVectorLane(instruction.operands.at(2), context, lane));
        const uint64_t rhs =
            static_cast<uint32_t>(ResolveVectorLane(instruction.operands.at(3), context, lane));
        const uint64_t sum = lhs + rhs;
        context.wave.vgpr.Write(vdst, lane, static_cast<uint32_t>(sum));
        if ((sum >> 32u) != 0) {
          context.vcc |= (1ull << lane);
        }
      }
    } else if (instruction.mnemonic == "v_addc_co_u32_e32") {
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      uint64_t next_vcc = 0;
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        const uint64_t carry_in = (context.vcc >> lane) & 1ull;
        const uint64_t lhs =
            static_cast<uint32_t>(ResolveVectorLane(instruction.operands.at(2), context, lane));
        const uint64_t rhs =
            static_cast<uint32_t>(ResolveVectorLane(instruction.operands.at(3), context, lane));
        const uint64_t sum = lhs + rhs + carry_in;
        context.wave.vgpr.Write(vdst, lane, static_cast<uint32_t>(sum));
        if ((sum >> 32u) != 0) {
          next_vcc |= (1ull << lane);
        }
      }
      context.vcc = next_vcc;
    } else if (instruction.mnemonic == "v_add_co_u32_e64") {
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      uint64_t carry_mask = 0;
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        const uint64_t lhs =
            static_cast<uint32_t>(ResolveVectorLane(instruction.operands.at(2), context, lane));
        const uint64_t rhs =
            static_cast<uint32_t>(ResolveVectorLane(instruction.operands.at(3), context, lane));
        const uint64_t sum = lhs + rhs;
        context.wave.vgpr.Write(vdst, lane, static_cast<uint32_t>(sum));
        if ((sum >> 32u) != 0) {
          carry_mask |= (1ull << lane);
        }
      }
      StoreScalarPair(instruction.operands.at(1), context, carry_mask);
    } else if (instruction.mnemonic == "v_addc_co_u32_e64") {
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      const uint64_t carry_in_mask = ResolveScalarPair(instruction.operands.at(4), context);
      uint64_t carry_out_mask = 0;
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        const uint64_t carry_in = (carry_in_mask >> lane) & 1ull;
        const uint64_t lhs =
            static_cast<uint32_t>(ResolveVectorLane(instruction.operands.at(2), context, lane));
        const uint64_t rhs =
            static_cast<uint32_t>(ResolveVectorLane(instruction.operands.at(3), context, lane));
        const uint64_t sum = lhs + rhs + carry_in;
        context.wave.vgpr.Write(vdst, lane, static_cast<uint32_t>(sum));
        if ((sum >> 32u) != 0) {
          carry_out_mask |= (1ull << lane);
        }
      }
      StoreScalarPair(instruction.operands.at(1), context, carry_out_mask);
    } else if (instruction.mnemonic == "v_add_f32_e32") {
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        const float lhs = U32AsFloat(static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(1), context, lane)));
        const float rhs = U32AsFloat(static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(2), context, lane)));
        context.wave.vgpr.Write(vdst, lane, FloatAsU32(lhs + rhs));
      }
    } else if (instruction.mnemonic == "v_sub_f32_e32") {
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        const float lhs = U32AsFloat(static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(1), context, lane)));
        const float rhs = U32AsFloat(static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(2), context, lane)));
        context.wave.vgpr.Write(vdst, lane, FloatAsU32(lhs - rhs));
      }
    } else if (instruction.mnemonic == "v_mul_f32_e32") {
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        const float lhs = U32AsFloat(static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(1), context, lane)));
        const float rhs = U32AsFloat(static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(2), context, lane)));
        context.wave.vgpr.Write(vdst, lane, FloatAsU32(lhs * rhs));
      }
    } else if (instruction.mnemonic == "v_max_f32_e32") {
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        const float lhs = U32AsFloat(static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(1), context, lane)));
        const float rhs = U32AsFloat(static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(2), context, lane)));
        context.wave.vgpr.Write(vdst, lane, FloatAsU32(std::fmax(lhs, rhs)));
      }
    } else if (instruction.mnemonic == "v_fmac_f32_e32") {
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        const float mul_lhs = U32AsFloat(static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(1), context, lane)));
        const float mul_rhs = U32AsFloat(static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(2), context, lane)));
        const float acc = U32AsFloat(static_cast<uint32_t>(context.wave.vgpr.Read(vdst, lane)));
        context.wave.vgpr.Write(vdst, lane, FloatAsU32(acc + mul_lhs * mul_rhs));
      }
    } else if (instruction.mnemonic == "v_fma_f32") {
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        const float src0 = U32AsFloat(static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(1), context, lane)));
        const float src1 = U32AsFloat(static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(2), context, lane)));
        const float src2 = U32AsFloat(static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(3), context, lane)));
        context.wave.vgpr.Write(vdst, lane, FloatAsU32(src0 * src1 + src2));
      }
    } else if (instruction.mnemonic == "v_mfma_f32_16x16x4f32") {
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        const float src0 = U32AsFloat(static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(1), context, lane)));
        const float src1 = U32AsFloat(static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(2), context, lane)));
        const float src2 = U32AsFloat(static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(3), context, lane)));
        const float value = src2 + src0 * src1 * 4.0f;
        for (uint32_t reg = 0; reg < 4; ++reg) {
          context.wave.vgpr.Write(vdst + reg, lane, FloatAsU32(value));
        }
      }
    } else if (instruction.mnemonic == "v_rndne_f32_e32") {
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        const float src = U32AsFloat(static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(1), context, lane)));
        context.wave.vgpr.Write(vdst, lane, FloatAsU32(std::nearbyint(src)));
      }
    } else if (instruction.mnemonic == "v_cvt_i32_f32_e32") {
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        const float src = U32AsFloat(static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(1), context, lane)));
        context.wave.vgpr.Write(vdst, lane, static_cast<uint32_t>(static_cast<int32_t>(src)));
      }
    } else if (instruction.mnemonic == "v_exp_f32_e32") {
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        const float src = U32AsFloat(static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(1), context, lane)));
        context.wave.vgpr.Write(vdst, lane, FloatAsU32(std::exp2(src)));
      }
    } else if (instruction.mnemonic == "v_rcp_f32_e32") {
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        const float src = U32AsFloat(static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(1), context, lane)));
        context.wave.vgpr.Write(vdst, lane, FloatAsU32(1.0f / src));
      }
    } else if (instruction.mnemonic == "v_ldexp_f32") {
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        const float value = U32AsFloat(static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(1), context, lane)));
        const int exp = static_cast<int>(ResolveVectorLane(instruction.operands.at(2), context, lane));
        context.wave.vgpr.Write(vdst, lane, FloatAsU32(std::ldexp(value, exp)));
      }
    } else if (instruction.mnemonic == "v_div_scale_f32") {
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        const uint32_t value =
            static_cast<uint32_t>(ResolveVectorLane(instruction.operands.at(1), context, lane));
        context.wave.vgpr.Write(vdst, lane, value);
      }
    } else if (instruction.mnemonic == "v_div_fmas_f32") {
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        const float src0 = U32AsFloat(static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(1), context, lane)));
        const float src1 = U32AsFloat(static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(2), context, lane)));
        const float src2 = U32AsFloat(static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(3), context, lane)));
        context.wave.vgpr.Write(vdst, lane, FloatAsU32(src0 * src1 + src2));
      }
    } else if (instruction.mnemonic == "v_div_fixup_f32") {
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        const float denom = U32AsFloat(static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(2), context, lane)));
        const float numer = U32AsFloat(static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(3), context, lane)));
        context.wave.vgpr.Write(vdst, lane, FloatAsU32(numer / denom));
      }
    } else if (instruction.mnemonic == "v_cndmask_b32_e32") {
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        const uint32_t false_value =
            static_cast<uint32_t>(ResolveVectorLane(instruction.operands.at(1), context, lane));
        const uint32_t true_value =
            static_cast<uint32_t>(ResolveVectorLane(instruction.operands.at(2), context, lane));
        const bool select_true = ((context.vcc >> lane) & 1ull) != 0;
        context.wave.vgpr.Write(vdst, lane, select_true ? true_value : false_value);
      }
    } else if (instruction.mnemonic == "v_cndmask_b32_e64") {
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      const uint64_t mask = ResolveScalarPair(instruction.operands.at(3), context);
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        const uint32_t false_value =
            static_cast<uint32_t>(ResolveVectorLane(instruction.operands.at(1), context, lane));
        const uint32_t true_value =
            static_cast<uint32_t>(ResolveVectorLane(instruction.operands.at(2), context, lane));
        const bool select_true = ((mask >> lane) & 1ull) != 0;
        context.wave.vgpr.Write(vdst, lane, select_true ? true_value : false_value);
      }
    } else {
      throw std::invalid_argument("unsupported vector alu opcode: " + instruction.mnemonic);
    }
    context.wave.pc += instruction.size_bytes;
  }
};

class VectorCompareHandler final : public IRawGcnSemanticHandler {
 public:
  void Execute(const DecodedGcnInstruction& instruction, RawGcnWaveContext& context) const override {
    uint64_t mask = 0;
    if (instruction.mnemonic == "v_cmp_gt_i32_e32" || instruction.mnemonic == "v_cmp_gt_i32_e64") {
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        const int32_t lhs =
            static_cast<int32_t>(ResolveVectorLane(instruction.operands.at(1), context, lane));
        const int32_t rhs =
            static_cast<int32_t>(ResolveVectorLane(instruction.operands.at(2), context, lane));
        if (lhs > rhs) {
          mask |= (1ull << lane);
        }
      }
    } else if (instruction.mnemonic == "v_cmp_gt_u32_e32") {
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        const uint32_t lhs =
            static_cast<uint32_t>(ResolveVectorLane(instruction.operands.at(1), context, lane));
        const uint32_t rhs =
            static_cast<uint32_t>(ResolveVectorLane(instruction.operands.at(2), context, lane));
        if (lhs > rhs) {
          mask |= (1ull << lane);
        }
      }
    } else if (instruction.mnemonic == "v_cmp_eq_u32_e32") {
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        const uint32_t lhs =
            static_cast<uint32_t>(ResolveVectorLane(instruction.operands.at(1), context, lane));
        const uint32_t rhs =
            static_cast<uint32_t>(ResolveVectorLane(instruction.operands.at(2), context, lane));
        if (lhs == rhs) {
          mask |= (1ull << lane);
        }
      }
    } else if (instruction.mnemonic == "v_cmp_ngt_f32_e32") {
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        const float lhs = U32AsFloat(static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(1), context, lane)));
        const float rhs = U32AsFloat(static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(2), context, lane)));
        if (!(lhs > rhs)) {
          mask |= (1ull << lane);
        }
      }
    } else if (instruction.mnemonic == "v_cmp_nlt_f32_e32") {
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        const float lhs = U32AsFloat(static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(1), context, lane)));
        const float rhs = U32AsFloat(static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(2), context, lane)));
        if (!(lhs < rhs)) {
          mask |= (1ull << lane);
        }
      }
    } else {
      throw std::invalid_argument("unsupported vector compare opcode: " + instruction.mnemonic);
    }
    context.vcc = mask;
    if (instruction.operands.at(0).kind == DecodedGcnOperandKind::ScalarRegRange ||
        instruction.operands.at(0).kind == DecodedGcnOperandKind::SpecialReg) {
      StoreScalarPair(instruction.operands.at(0), context, mask);
    }
    DebugLog("pc=0x%llx %s mask=0x%llx",
             static_cast<unsigned long long>(instruction.pc),
             instruction.mnemonic.c_str(),
             static_cast<unsigned long long>(mask));
    context.wave.pc += instruction.size_bytes;
  }
};

class MaskHandler final : public IRawGcnSemanticHandler {
 public:
  void Execute(const DecodedGcnInstruction& instruction, RawGcnWaveContext& context) const override {
    if (instruction.mnemonic != "s_and_saveexec_b64") {
      throw std::invalid_argument("unsupported mask opcode: " + instruction.mnemonic);
    }
    const auto [sdst, _] = RequireScalarRange(instruction.operands.at(0));
    const uint64_t exec_before = context.wave.exec.to_ullong();
    context.wave.sgpr.Write(sdst, static_cast<uint32_t>(exec_before & 0xffffffffu));
    context.wave.sgpr.Write(sdst + 1, static_cast<uint32_t>(exec_before >> 32u));
    const uint64_t mask = ResolveScalarPair(instruction.operands.at(1), context);
    context.wave.exec = context.wave.exec & MaskFromU64(mask);
    DebugLog("pc=0x%llx s_and_saveexec_b64 exec=0x%llx",
             static_cast<unsigned long long>(instruction.pc),
             static_cast<unsigned long long>(context.wave.exec.to_ullong()));
    context.wave.pc += instruction.size_bytes;
  }
};

class BranchHandler final : public IRawGcnSemanticHandler {
 public:
  void Execute(const DecodedGcnInstruction& instruction, RawGcnWaveContext& context) const override {
    if (instruction.mnemonic == "s_cbranch_execz") {
      if (context.wave.exec.none()) {
        context.wave.pc =
            BranchTarget(context.wave.pc, static_cast<int32_t>(instruction.operands.at(0).info.immediate));
      } else {
        context.wave.pc += instruction.size_bytes;
      }
      return;
    }
    if (instruction.mnemonic == "s_cbranch_scc1") {
      if (context.wave.ScalarMaskBit0()) {
        context.wave.pc =
            BranchTarget(context.wave.pc, static_cast<int32_t>(instruction.operands.at(0).info.immediate));
      } else {
        context.wave.pc += instruction.size_bytes;
      }
      return;
    }
    if (instruction.mnemonic == "s_cbranch_scc0") {
      if (!context.wave.ScalarMaskBit0()) {
        context.wave.pc =
            BranchTarget(context.wave.pc, static_cast<int32_t>(instruction.operands.at(0).info.immediate));
      } else {
        context.wave.pc += instruction.size_bytes;
      }
      return;
    }
    if (instruction.mnemonic == "s_cbranch_vccz") {
      if (context.vcc == 0) {
        context.wave.pc =
            BranchTarget(context.wave.pc, static_cast<int32_t>(instruction.operands.at(0).info.immediate));
      } else {
        context.wave.pc += instruction.size_bytes;
      }
      return;
    }
    if (instruction.mnemonic == "s_branch") {
      context.wave.pc =
          BranchTarget(context.wave.pc, static_cast<int32_t>(instruction.operands.at(0).info.immediate));
      return;
    }
    {
      throw std::invalid_argument("unsupported branch opcode: " + instruction.mnemonic);
    }
  }
};

class FlatMemoryHandler final : public IRawGcnSemanticHandler {
 public:
  void Execute(const DecodedGcnInstruction& instruction, RawGcnWaveContext& context) const override {
    if (instruction.mnemonic == "global_load_dword") {
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      const auto [addr, _] = RequireVectorRange(instruction.operands.at(1));
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        const uint64_t lo = static_cast<uint32_t>(context.wave.vgpr.Read(addr, lane));
        const uint64_t hi = static_cast<uint32_t>(context.wave.vgpr.Read(addr + 1, lane));
        const uint64_t address = (hi << 32u) | lo;
        if (!context.wave.exec.test(lane) &&
            !context.memory.HasGlobalRange(address, sizeof(uint32_t))) {
          continue;
        }
        context.wave.vgpr.Write(vdst, lane, context.memory.LoadGlobalValue<uint32_t>(address));
        if (lane == 0) {
          DebugLog("pc=0x%llx global_load addr=0x%llx -> v%u=0x%llx",
                   static_cast<unsigned long long>(instruction.pc),
                   static_cast<unsigned long long>(address), vdst,
                   static_cast<unsigned long long>(context.wave.vgpr.Read(vdst, lane)));
        }
      }
      ++context.stats.global_loads;
    } else if (instruction.mnemonic == "global_store_dword") {
      if (instruction.operands.at(0).kind == DecodedGcnOperandKind::VectorRegRange) {
        const auto [addr, _] = RequireVectorRange(instruction.operands.at(0));
        const uint32_t data = RequireVectorIndex(instruction.operands.at(1));
        for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
          const uint64_t lo = static_cast<uint32_t>(context.wave.vgpr.Read(addr, lane));
          const uint64_t hi = static_cast<uint32_t>(context.wave.vgpr.Read(addr + 1, lane));
          const uint64_t address = (hi << 32u) | lo;
          if (!context.wave.exec.test(lane) &&
              !context.memory.HasGlobalRange(address, sizeof(uint32_t))) {
            continue;
          }
          context.memory.StoreGlobalValue<uint32_t>(
              address, static_cast<uint32_t>(context.wave.vgpr.Read(data, lane)));
          if (lane == 0) {
            DebugLog("pc=0x%llx global_store addr=0x%llx value=0x%llx",
                     static_cast<unsigned long long>(instruction.pc),
                     static_cast<unsigned long long>(address),
                     static_cast<unsigned long long>(context.wave.vgpr.Read(data, lane)));
          }
        }
      } else {
        const uint32_t vaddr = RequireVectorIndex(instruction.operands.at(0));
        const auto [saddr, _] = RequireScalarRange(instruction.operands.at(1));
        const uint32_t data = RequireVectorIndex(instruction.operands.at(2));
        const uint64_t base = static_cast<uint64_t>(context.wave.sgpr.Read(saddr)) |
                              (static_cast<uint64_t>(context.wave.sgpr.Read(saddr + 1)) << 32u);
        for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
          const int32_t offset = static_cast<int32_t>(context.wave.vgpr.Read(vaddr, lane));
          const uint64_t address = base + static_cast<int64_t>(offset);
          if (!context.wave.exec.test(lane) &&
              !context.memory.HasGlobalRange(address, sizeof(uint32_t))) {
            continue;
          }
          context.memory.StoreGlobalValue<uint32_t>(
              address, static_cast<uint32_t>(context.wave.vgpr.Read(data, lane)));
          if (lane == 0) {
            DebugLog("pc=0x%llx global_store addr=0x%llx value=0x%llx",
                     static_cast<unsigned long long>(instruction.pc),
                     static_cast<unsigned long long>(address),
                     static_cast<unsigned long long>(context.wave.vgpr.Read(data, lane)));
          }
        }
      }
      ++context.stats.global_stores;
    } else {
      throw std::invalid_argument("unsupported flat memory opcode: " + instruction.mnemonic);
    }
    context.wave.pc += instruction.size_bytes;
  }
};

class SharedMemoryHandler final : public IRawGcnSemanticHandler {
 public:
  void Execute(const DecodedGcnInstruction& instruction, RawGcnWaveContext& context) const override {
    if (instruction.mnemonic == "ds_write_b32") {
      const uint32_t addr_vgpr = RequireVectorIndex(instruction.operands.at(0));
      const uint32_t data_vgpr = RequireVectorIndex(instruction.operands.at(1));
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        const uint32_t byte_offset =
            static_cast<uint32_t>(context.wave.vgpr.Read(addr_vgpr, lane));
        const uint32_t value =
            static_cast<uint32_t>(context.wave.vgpr.Read(data_vgpr, lane));
        StoreU32(context.block.shared_memory, byte_offset, value);
      }
      ++context.stats.shared_stores;
      context.wave.pc += instruction.size_bytes;
      return;
    }
    if (instruction.mnemonic == "ds_read_b32") {
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      const uint32_t addr_vgpr = RequireVectorIndex(instruction.operands.at(1));
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        const uint32_t byte_offset =
            static_cast<uint32_t>(context.wave.vgpr.Read(addr_vgpr, lane));
        context.wave.vgpr.Write(vdst, lane, LoadU32(context.block.shared_memory, byte_offset));
      }
      ++context.stats.shared_loads;
      context.wave.pc += instruction.size_bytes;
      return;
    }
    throw std::invalid_argument("unsupported shared memory opcode: " + instruction.mnemonic);
  }
};

class SpecialHandler final : public IRawGcnSemanticHandler {
 public:
  void Execute(const DecodedGcnInstruction& instruction, RawGcnWaveContext& context) const override {
    if (instruction.mnemonic == "s_barrier") {
      ++context.stats.barriers;
      context.wave.status = WaveStatus::Stalled;
      context.wave.waiting_at_barrier = true;
      context.wave.barrier_generation = context.block.barrier_generation;
      ++context.block.barrier_arrivals;
      return;
    }
    if (instruction.mnemonic == "s_nop") {
      context.wave.pc += instruction.size_bytes;
      return;
    }
    if (instruction.mnemonic == "s_waitcnt") {
      context.wave.pc += instruction.size_bytes;
      return;
    }
    if (instruction.mnemonic == "s_endpgm") {
      context.wave.status = WaveStatus::Exited;
      ++context.stats.wave_exits;
      return;
    }
    throw std::invalid_argument("unsupported special opcode: " + instruction.mnemonic);
  }
};

struct HandlerBinding {
  const char* mnemonic = nullptr;
  const IRawGcnSemanticHandler* handler = nullptr;
};

const std::vector<HandlerBinding>& HandlerBindings() {
  static const ScalarMemoryHandler kScalarMemoryHandler;
  static const ScalarAluHandler kScalarAluHandler;
  static const ScalarCompareHandler kScalarCompareHandler;
  static const VectorAluHandler kVectorAluHandler;
  static const VectorCompareHandler kVectorCompareHandler;
  static const MaskHandler kMaskHandler;
  static const BranchHandler kBranchHandler;
  static const FlatMemoryHandler kFlatMemoryHandler;
  static const SharedMemoryHandler kSharedMemoryHandler;
  static const SpecialHandler kSpecialHandler;
  static const std::vector<HandlerBinding> kBindings = {
      {.mnemonic = "s_load_dword", .handler = &kScalarMemoryHandler},
      {.mnemonic = "s_load_dwordx2", .handler = &kScalarMemoryHandler},
      {.mnemonic = "s_load_dwordx4", .handler = &kScalarMemoryHandler},
      {.mnemonic = "s_mov_b32", .handler = &kScalarAluHandler},
      {.mnemonic = "s_cselect_b64", .handler = &kScalarAluHandler},
      {.mnemonic = "s_andn2_b64", .handler = &kScalarAluHandler},
      {.mnemonic = "s_or_b64", .handler = &kScalarAluHandler},
      {.mnemonic = "s_and_b32", .handler = &kScalarAluHandler},
      {.mnemonic = "s_mul_i32", .handler = &kScalarAluHandler},
      {.mnemonic = "s_add_i32", .handler = &kScalarAluHandler},
      {.mnemonic = "s_lshr_b32", .handler = &kScalarAluHandler},
      {.mnemonic = "s_cmp_lt_i32", .handler = &kScalarCompareHandler},
      {.mnemonic = "s_cmp_eq_u32", .handler = &kScalarCompareHandler},
      {.mnemonic = "s_cmp_gt_u32", .handler = &kScalarCompareHandler},
      {.mnemonic = "s_cmp_lt_u32", .handler = &kScalarCompareHandler},
      {.mnemonic = "v_not_b32_e32", .handler = &kVectorAluHandler},
      {.mnemonic = "v_add_u32_e32", .handler = &kVectorAluHandler},
      {.mnemonic = "v_ashrrev_i32_e32", .handler = &kVectorAluHandler},
      {.mnemonic = "v_lshlrev_b32_e32", .handler = &kVectorAluHandler},
      {.mnemonic = "v_lshlrev_b64", .handler = &kVectorAluHandler},
      {.mnemonic = "v_lshl_add_u32", .handler = &kVectorAluHandler},
      {.mnemonic = "v_mov_b32_e32", .handler = &kVectorAluHandler},
      {.mnemonic = "v_add_co_u32_e32", .handler = &kVectorAluHandler},
      {.mnemonic = "v_addc_co_u32_e32", .handler = &kVectorAluHandler},
      {.mnemonic = "v_add_co_u32_e64", .handler = &kVectorAluHandler},
      {.mnemonic = "v_addc_co_u32_e64", .handler = &kVectorAluHandler},
      {.mnemonic = "v_add_f32_e32", .handler = &kVectorAluHandler},
      {.mnemonic = "v_sub_f32_e32", .handler = &kVectorAluHandler},
      {.mnemonic = "v_mul_f32_e32", .handler = &kVectorAluHandler},
      {.mnemonic = "v_max_f32_e32", .handler = &kVectorAluHandler},
      {.mnemonic = "v_fmac_f32_e32", .handler = &kVectorAluHandler},
      {.mnemonic = "v_fma_f32", .handler = &kVectorAluHandler},
      {.mnemonic = "v_mfma_f32_16x16x4f32", .handler = &kVectorAluHandler},
      {.mnemonic = "v_rndne_f32_e32", .handler = &kVectorAluHandler},
      {.mnemonic = "v_cvt_i32_f32_e32", .handler = &kVectorAluHandler},
      {.mnemonic = "v_exp_f32_e32", .handler = &kVectorAluHandler},
      {.mnemonic = "v_rcp_f32_e32", .handler = &kVectorAluHandler},
      {.mnemonic = "v_ldexp_f32", .handler = &kVectorAluHandler},
      {.mnemonic = "v_div_scale_f32", .handler = &kVectorAluHandler},
      {.mnemonic = "v_div_fmas_f32", .handler = &kVectorAluHandler},
      {.mnemonic = "v_div_fixup_f32", .handler = &kVectorAluHandler},
      {.mnemonic = "v_cndmask_b32_e32", .handler = &kVectorAluHandler},
      {.mnemonic = "v_cndmask_b32_e64", .handler = &kVectorAluHandler},
      {.mnemonic = "v_cmp_gt_i32_e32", .handler = &kVectorCompareHandler},
      {.mnemonic = "v_cmp_gt_i32_e64", .handler = &kVectorCompareHandler},
      {.mnemonic = "v_cmp_eq_u32_e32", .handler = &kVectorCompareHandler},
      {.mnemonic = "v_cmp_gt_u32_e32", .handler = &kVectorCompareHandler},
      {.mnemonic = "v_cmp_ngt_f32_e32", .handler = &kVectorCompareHandler},
      {.mnemonic = "v_cmp_nlt_f32_e32", .handler = &kVectorCompareHandler},
      {.mnemonic = "s_and_saveexec_b64", .handler = &kMaskHandler},
      {.mnemonic = "s_cbranch_execz", .handler = &kBranchHandler},
      {.mnemonic = "s_cbranch_scc1", .handler = &kBranchHandler},
      {.mnemonic = "s_cbranch_scc0", .handler = &kBranchHandler},
      {.mnemonic = "s_cbranch_vccz", .handler = &kBranchHandler},
      {.mnemonic = "s_branch", .handler = &kBranchHandler},
      {.mnemonic = "global_load_dword", .handler = &kFlatMemoryHandler},
      {.mnemonic = "global_store_dword", .handler = &kFlatMemoryHandler},
      {.mnemonic = "ds_read_b32", .handler = &kSharedMemoryHandler},
      {.mnemonic = "ds_write_b32", .handler = &kSharedMemoryHandler},
      {.mnemonic = "s_barrier", .handler = &kSpecialHandler},
      {.mnemonic = "s_nop", .handler = &kSpecialHandler},
      {.mnemonic = "s_waitcnt", .handler = &kSpecialHandler},
      {.mnemonic = "s_endpgm", .handler = &kSpecialHandler},
  };
  return kBindings;
}

}  // namespace

const IRawGcnSemanticHandler& RawGcnSemanticHandlerRegistry::Get(std::string_view mnemonic) {
  for (const auto& binding : HandlerBindings()) {
    if (binding.mnemonic == mnemonic) {
      return *binding.handler;
    }
  }
  throw std::invalid_argument("unsupported raw GCN opcode: " + std::string(mnemonic));
}

}  // namespace gpu_model
