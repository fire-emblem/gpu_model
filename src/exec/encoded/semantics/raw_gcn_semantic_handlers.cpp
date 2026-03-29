#include "gpu_model/exec/encoded/semantics/raw_gcn_semantic_handler.h"

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

#include "gpu_model/decode/gcn_inst_encoding_def.h"
#include "gpu_model/decode/gcn_inst_db_lookup.h"
#include "gpu_model/exec/execution_sync_ops.h"

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

float HalfToFloat(uint16_t bits) {
  const uint32_t sign = static_cast<uint32_t>(bits & 0x8000u) << 16u;
  const uint32_t exp = (bits >> 10u) & 0x1fu;
  const uint32_t frac = bits & 0x03ffu;

  if (exp == 0u) {
    if (frac == 0u) {
      return std::bit_cast<float>(sign);
    }
    float mantissa = static_cast<float>(frac) / 1024.0f;
    float value = std::ldexp(mantissa, -14);
    return (bits & 0x8000u) != 0 ? -value : value;
  }
  if (exp == 0x1fu) {
    const uint32_t out = sign | 0x7f800000u | (frac << 13u);
    return std::bit_cast<float>(out);
  }
  const uint32_t out = sign | ((exp + 112u) << 23u) | (frac << 13u);
  return std::bit_cast<float>(out);
}

float BFloat16ToFloat(uint16_t bits) {
  return std::bit_cast<float>(static_cast<uint32_t>(bits) << 16u);
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

uint32_t RequireAccumulatorIndex(const DecodedGcnOperand& operand) {
  if (operand.kind != DecodedGcnOperandKind::AccumulatorReg || operand.info.reg_count != 1) {
    throw std::invalid_argument("expected accumulator register operand");
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
  if (operand.kind == DecodedGcnOperandKind::ScalarReg) {
    const uint32_t first = operand.info.reg_first;
    return static_cast<uint64_t>(context.wave.sgpr.Read(first)) |
           (static_cast<uint64_t>(context.wave.sgpr.Read(first + 1)) << 32u);
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
  if (operand.kind == DecodedGcnOperandKind::ScalarReg) {
    const uint32_t first = operand.info.reg_first;
    context.wave.sgpr.Write(first, static_cast<uint32_t>(value & 0xffffffffu));
    context.wave.sgpr.Write(first + 1, static_cast<uint32_t>(value >> 32u));
    return;
  }
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

const GcnIsaOpcodeDescriptor& RequireCanonicalOpcode(const DecodedGcnInstruction& instruction) {
  if (const auto* descriptor = FindGcnFallbackOpcodeDescriptor(instruction.words); descriptor != nullptr) {
    return *descriptor;
  }
  throw std::invalid_argument("missing canonical opcode descriptor: " + instruction.mnemonic);
}

class ScalarMemoryHandler final : public IRawGcnSemanticHandler {
 public:
  void Execute(const DecodedGcnInstruction& instruction, RawGcnWaveContext& context) const override {
    if (instruction.mnemonic == "s_load_dword") {
      const uint32_t offset =
          static_cast<uint32_t>(ResolveScalarLike(instruction.operands.at(2), context));
      const uint32_t sdst = RequireScalarIndex(instruction.operands.at(0));
      const uint64_t base = ResolveScalarPair(instruction.operands.at(1), context);
      context.wave.sgpr.Write(
          sdst, context.memory.LoadGlobalValue<uint32_t>(base + offset));
    } else if (instruction.mnemonic == "s_load_dwordx2") {
      const uint32_t offset =
          static_cast<uint32_t>(ResolveScalarLike(instruction.operands.at(2), context));
      const auto [sdst, _] = RequireScalarRange(instruction.operands.at(0));
      const uint64_t base = ResolveScalarPair(instruction.operands.at(1), context);
      const uint64_t value = context.memory.LoadGlobalValue<uint64_t>(base + offset);
      context.wave.sgpr.Write(sdst, static_cast<uint32_t>(value & 0xffffffffu));
      context.wave.sgpr.Write(sdst + 1, static_cast<uint32_t>(value >> 32u));
    } else if (instruction.mnemonic == "s_load_dwordx4") {
      const uint32_t offset =
          static_cast<uint32_t>(ResolveScalarLike(instruction.operands.at(2), context));
      const auto [sdst, _] = RequireScalarRange(instruction.operands.at(0));
      const uint64_t base = ResolveScalarPair(instruction.operands.at(1), context);
      context.wave.sgpr.Write(sdst + 0, context.memory.LoadGlobalValue<uint32_t>(base + offset + 0));
      context.wave.sgpr.Write(sdst + 1, context.memory.LoadGlobalValue<uint32_t>(base + offset + 4));
      context.wave.sgpr.Write(sdst + 2, context.memory.LoadGlobalValue<uint32_t>(base + offset + 8));
      context.wave.sgpr.Write(sdst + 3, context.memory.LoadGlobalValue<uint32_t>(base + offset + 12));
    } else {
      throw std::invalid_argument("unsupported scalar memory opcode: " + instruction.mnemonic);
    }
    context.wave.pc += instruction.size_bytes;
  }
};

class ScalarAluHandler final : public IRawGcnSemanticHandler {
 public:
  void Execute(const DecodedGcnInstruction& instruction, RawGcnWaveContext& context) const override {
    const auto& descriptor = RequireCanonicalOpcode(instruction);
    if (descriptor.op_type == GcnIsaOpType::Sop2 &&
        descriptor.opcode == static_cast<uint16_t>(GcnIsaSop2Opcode::S_CSELECT_B64)) {
      const bool take_true = context.wave.ScalarMaskBit0();
      const uint64_t value = take_true ? ResolveScalarPair(instruction.operands.at(1), context)
                                       : ResolveScalarPair(instruction.operands.at(2), context);
      StoreScalarPair(instruction.operands.at(0), context, value);
    } else if (descriptor.op_type == GcnIsaOpType::Sop2 &&
               descriptor.opcode == static_cast<uint16_t>(GcnIsaSop2Opcode::S_ANDN2_B64)) {
      const uint64_t lhs = ResolveScalarPair(instruction.operands.at(1), context);
      const uint64_t rhs = ResolveScalarPair(instruction.operands.at(2), context);
      StoreScalarPair(instruction.operands.at(0), context, lhs & ~rhs);
    } else if (descriptor.op_type == GcnIsaOpType::Sop1 &&
               descriptor.opcode == static_cast<uint16_t>(GcnIsaSop1Opcode::S_MOV_B32)) {
      const uint32_t sdst = RequireScalarIndex(instruction.operands.at(0));
      const uint32_t value =
          static_cast<uint32_t>(ResolveScalarLike(instruction.operands.at(1), context));
      context.wave.sgpr.Write(sdst, value);
    } else if (descriptor.op_type == GcnIsaOpType::Sop1 &&
               descriptor.opcode == static_cast<uint16_t>(GcnIsaSop1Opcode::S_MOV_B64)) {
      const uint64_t value = ResolveScalarPair(instruction.operands.at(1), context);
      StoreScalarPair(instruction.operands.at(0), context, value);
    } else if (descriptor.op_type == GcnIsaOpType::Sop2 &&
               descriptor.opcode == static_cast<uint16_t>(GcnIsaSop2Opcode::S_OR_B64)) {
      const uint64_t lhs = ResolveScalarPair(instruction.operands.at(1), context);
      const uint64_t rhs = ResolveScalarPair(instruction.operands.at(2), context);
      const uint64_t value = lhs | rhs;
      StoreScalarPair(instruction.operands.at(0), context, value);
      if (instruction.operands.at(0).kind == DecodedGcnOperandKind::SpecialReg &&
          instruction.operands.at(0).info.special_reg == GcnSpecialReg::Exec) {
        DebugLog("pc=0x%llx s_or_b64 exec lhs=0x%llx rhs=0x%llx out=0x%llx",
                 static_cast<unsigned long long>(instruction.pc),
                 static_cast<unsigned long long>(lhs),
                 static_cast<unsigned long long>(rhs),
                 static_cast<unsigned long long>(value));
      }
    } else if (descriptor.op_type == GcnIsaOpType::Sop2 &&
               descriptor.opcode == static_cast<uint16_t>(GcnIsaSop2Opcode::S_AND_B64)) {
      const uint64_t lhs = ResolveScalarPair(instruction.operands.at(1), context);
      const uint64_t rhs = ResolveScalarPair(instruction.operands.at(2), context);
      StoreScalarPair(instruction.operands.at(0), context, lhs & rhs);
    } else if (descriptor.op_type == GcnIsaOpType::Sop2 &&
               descriptor.opcode == static_cast<uint16_t>(GcnIsaSop2Opcode::S_AND_B32)) {
      const uint32_t sdst = RequireScalarIndex(instruction.operands.at(0));
      const uint32_t lhs =
          static_cast<uint32_t>(ResolveScalarLike(instruction.operands.at(1), context));
      const uint32_t rhs =
          static_cast<uint32_t>(ResolveScalarLike(instruction.operands.at(2), context));
      context.wave.sgpr.Write(sdst, lhs & rhs);
    } else if (descriptor.op_type == GcnIsaOpType::Sop2 &&
               descriptor.opcode == static_cast<uint16_t>(GcnIsaSop2Opcode::S_MUL_I32)) {
      const uint32_t sdst = RequireScalarIndex(instruction.operands.at(0));
      const uint32_t lhs =
          static_cast<uint32_t>(ResolveScalarLike(instruction.operands.at(1), context));
      const uint32_t rhs =
          static_cast<uint32_t>(ResolveScalarLike(instruction.operands.at(2), context));
      context.wave.sgpr.Write(sdst, lhs * rhs);
    } else if (descriptor.op_type == GcnIsaOpType::Sop2 &&
               descriptor.opcode == static_cast<uint16_t>(GcnIsaSop2Opcode::S_ADD_I32)) {
      const uint32_t sdst = RequireScalarIndex(instruction.operands.at(0));
      const uint32_t lhs =
          static_cast<uint32_t>(ResolveScalarLike(instruction.operands.at(1), context));
      const uint32_t rhs =
          static_cast<uint32_t>(ResolveScalarLike(instruction.operands.at(2), context));
      context.wave.sgpr.Write(sdst, lhs + rhs);
    } else if (descriptor.op_type == GcnIsaOpType::Sop2 &&
               descriptor.opcode == static_cast<uint16_t>(GcnIsaSop2Opcode::S_ADD_U32)) {
      const uint32_t sdst = RequireScalarIndex(instruction.operands.at(0));
      const uint64_t lhs =
          static_cast<uint32_t>(ResolveScalarLike(instruction.operands.at(1), context));
      const uint64_t rhs =
          static_cast<uint32_t>(ResolveScalarLike(instruction.operands.at(2), context));
      const uint64_t sum = lhs + rhs;
      context.wave.sgpr.Write(sdst, static_cast<uint32_t>(sum));
      context.wave.SetScalarMaskBit0((sum >> 32u) != 0);
    } else if (descriptor.op_type == GcnIsaOpType::Sop2 &&
               descriptor.opcode == static_cast<uint16_t>(GcnIsaSop2Opcode::S_ADDC_U32)) {
      const uint32_t sdst = RequireScalarIndex(instruction.operands.at(0));
      const uint64_t lhs =
          static_cast<uint32_t>(ResolveScalarLike(instruction.operands.at(1), context));
      const uint64_t rhs =
          static_cast<uint32_t>(ResolveScalarLike(instruction.operands.at(2), context));
      const uint64_t carry_in = context.wave.ScalarMaskBit0() ? 1ull : 0ull;
      const uint64_t sum = lhs + rhs + carry_in;
      context.wave.sgpr.Write(sdst, static_cast<uint32_t>(sum));
      context.wave.SetScalarMaskBit0((sum >> 32u) != 0);
    } else if (descriptor.op_type == GcnIsaOpType::Sop2 &&
               descriptor.opcode == static_cast<uint16_t>(GcnIsaSop2Opcode::S_LSHR_B32)) {
      const uint32_t sdst = RequireScalarIndex(instruction.operands.at(0));
      const uint32_t lhs =
          static_cast<uint32_t>(ResolveScalarLike(instruction.operands.at(1), context));
      const uint32_t rhs =
          static_cast<uint32_t>(ResolveScalarLike(instruction.operands.at(2), context));
      context.wave.sgpr.Write(sdst, lhs >> (rhs & 31u));
    } else if (descriptor.op_type == GcnIsaOpType::Sop2 &&
               descriptor.opcode == static_cast<uint16_t>(GcnIsaSop2Opcode::S_ASHR_I32)) {
      const uint32_t sdst = RequireScalarIndex(instruction.operands.at(0));
      const int32_t lhs =
          static_cast<int32_t>(ResolveScalarLike(instruction.operands.at(1), context));
      const uint32_t rhs =
          static_cast<uint32_t>(ResolveScalarLike(instruction.operands.at(2), context));
      context.wave.sgpr.Write(sdst, static_cast<uint32_t>(lhs >> (rhs & 31u)));
    } else if (descriptor.op_type == GcnIsaOpType::Sop2 &&
               descriptor.opcode == static_cast<uint16_t>(GcnIsaSop2Opcode::S_LSHL_B64)) {
      const uint64_t lhs = ResolveScalarPair(instruction.operands.at(1), context);
      const uint32_t rhs =
          static_cast<uint32_t>(ResolveScalarLike(instruction.operands.at(2), context));
      StoreScalarPair(instruction.operands.at(0), context, lhs << (rhs & 63u));
    } else if (descriptor.op_type == GcnIsaOpType::Sopk &&
               descriptor.opcode == static_cast<uint16_t>(GcnIsaSopkOpcode::S_MOVK_I32)) {
      const uint32_t sdst = RequireScalarIndex(instruction.operands.at(0));
      const int32_t value =
          static_cast<int32_t>(ResolveScalarLike(instruction.operands.at(1), context));
      context.wave.sgpr.Write(sdst, static_cast<uint32_t>(value));
    } else if (descriptor.op_type == GcnIsaOpType::Sop1 &&
               descriptor.opcode == static_cast<uint16_t>(GcnIsaSop1Opcode::S_BCNT1_I32_B64)) {
      const uint32_t sdst = RequireScalarIndex(instruction.operands.at(0));
      const uint64_t value = ResolveScalarPair(instruction.operands.at(1), context);
      context.wave.sgpr.Write(sdst, static_cast<uint32_t>(std::popcount(value)));
    } else {
      throw std::invalid_argument("unsupported scalar alu opcode: " + instruction.mnemonic);
    }
    context.wave.pc += instruction.size_bytes;
  }
};

class ScalarCompareHandler final : public IRawGcnSemanticHandler {
 public:
  void Execute(const DecodedGcnInstruction& instruction, RawGcnWaveContext& context) const override {
    const auto& descriptor = RequireCanonicalOpcode(instruction);
    if (descriptor.op_type == GcnIsaOpType::Sopc &&
        descriptor.opcode == static_cast<uint16_t>(GcnIsaSopcOpcode::S_CMP_LT_I32)) {
      const int32_t lhs =
          static_cast<int32_t>(ResolveScalarLike(instruction.operands.at(0), context));
      const int32_t rhs =
          static_cast<int32_t>(ResolveScalarLike(instruction.operands.at(1), context));
      context.wave.SetScalarMaskBit0(lhs < rhs);
    } else if (descriptor.op_type == GcnIsaOpType::Sopc &&
               descriptor.opcode == static_cast<uint16_t>(GcnIsaSopcOpcode::S_CMP_EQ_U32)) {
      const uint32_t lhs =
          static_cast<uint32_t>(ResolveScalarLike(instruction.operands.at(0), context));
      const uint32_t rhs =
          static_cast<uint32_t>(ResolveScalarLike(instruction.operands.at(1), context));
      context.wave.SetScalarMaskBit0(lhs == rhs);
    } else if (descriptor.op_type == GcnIsaOpType::Sopc &&
               descriptor.opcode == static_cast<uint16_t>(GcnIsaSopcOpcode::S_CMP_GT_U32)) {
      const uint32_t lhs =
          static_cast<uint32_t>(ResolveScalarLike(instruction.operands.at(0), context));
      const uint32_t rhs =
          static_cast<uint32_t>(ResolveScalarLike(instruction.operands.at(1), context));
      context.wave.SetScalarMaskBit0(lhs > rhs);
    } else if (descriptor.op_type == GcnIsaOpType::Sopc &&
               descriptor.opcode == static_cast<uint16_t>(GcnIsaSopcOpcode::S_CMP_LT_U32)) {
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
        if (!context.wave.exec.test(lane)) {
          continue;
        }
        const uint32_t src =
            static_cast<uint32_t>(ResolveVectorLane(instruction.operands.at(1), context, lane));
        context.wave.vgpr.Write(vdst, lane, ~src);
      }
    } else if (instruction.mnemonic == "v_add_u32_e32") {
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        if (!context.wave.exec.test(lane)) {
          continue;
        }
        const uint32_t lhs =
            static_cast<uint32_t>(ResolveVectorLane(instruction.operands.at(1), context, lane));
        const uint32_t rhs =
            static_cast<uint32_t>(ResolveVectorLane(instruction.operands.at(2), context, lane));
        context.wave.vgpr.Write(vdst, lane, lhs + rhs);
      }
    } else if (instruction.mnemonic == "v_ashrrev_i32_e32") {
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        if (!context.wave.exec.test(lane)) {
          continue;
        }
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
        if (!context.wave.exec.test(lane)) {
          continue;
        }
        const uint64_t lo = static_cast<uint32_t>(context.wave.vgpr.Read(src_pair, lane));
        const uint64_t hi = static_cast<uint32_t>(context.wave.vgpr.Read(src_pair + 1, lane));
        const uint64_t value = ((hi << 32u) | lo) << (shift & 63u);
        context.wave.vgpr.Write(vdst, lane, static_cast<uint32_t>(value & 0xffffffffu));
        context.wave.vgpr.Write(vdst + 1, lane, static_cast<uint32_t>(value >> 32u));
      }
    } else if (instruction.mnemonic == "v_mov_b32_e32") {
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        if (!context.wave.exec.test(lane)) {
          continue;
        }
        context.wave.vgpr.Write(vdst, lane, static_cast<uint32_t>(
                                                ResolveVectorLane(instruction.operands.at(1), context, lane)));
      }
    } else if (instruction.mnemonic == "v_lshlrev_b32_e32") {
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        if (!context.wave.exec.test(lane)) {
          continue;
        }
        const uint32_t shift =
            static_cast<uint32_t>(ResolveVectorLane(instruction.operands.at(1), context, lane));
        const uint32_t src =
            static_cast<uint32_t>(ResolveVectorLane(instruction.operands.at(2), context, lane));
        context.wave.vgpr.Write(vdst, lane, src << (shift & 31u));
      }
    } else if (instruction.mnemonic == "v_lshl_add_u32") {
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        if (!context.wave.exec.test(lane)) {
          continue;
        }
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
        if (!context.wave.exec.test(lane)) {
          continue;
        }
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
        if (!context.wave.exec.test(lane)) {
          continue;
        }
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
        if (!context.wave.exec.test(lane)) {
          continue;
        }
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
        if (!context.wave.exec.test(lane)) {
          continue;
        }
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
        if (!context.wave.exec.test(lane)) {
          continue;
        }
        const float lhs = U32AsFloat(static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(1), context, lane)));
        const float rhs = U32AsFloat(static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(2), context, lane)));
        context.wave.vgpr.Write(vdst, lane, FloatAsU32(lhs + rhs));
      }
    } else if (instruction.mnemonic == "v_sub_f32_e32") {
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        if (!context.wave.exec.test(lane)) {
          continue;
        }
        const float lhs = U32AsFloat(static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(1), context, lane)));
        const float rhs = U32AsFloat(static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(2), context, lane)));
        context.wave.vgpr.Write(vdst, lane, FloatAsU32(lhs - rhs));
      }
    } else if (instruction.mnemonic == "v_mul_f32_e32") {
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        if (!context.wave.exec.test(lane)) {
          continue;
        }
        const float lhs = U32AsFloat(static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(1), context, lane)));
        const float rhs = U32AsFloat(static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(2), context, lane)));
        context.wave.vgpr.Write(vdst, lane, FloatAsU32(lhs * rhs));
      }
    } else if (instruction.mnemonic == "v_max_f32_e32") {
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        if (!context.wave.exec.test(lane)) {
          continue;
        }
        const float lhs = U32AsFloat(static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(1), context, lane)));
        const float rhs = U32AsFloat(static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(2), context, lane)));
        context.wave.vgpr.Write(vdst, lane, FloatAsU32(std::fmax(lhs, rhs)));
      }
    } else if (instruction.mnemonic == "v_fmac_f32_e32") {
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        if (!context.wave.exec.test(lane)) {
          continue;
        }
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
        if (!context.wave.exec.test(lane)) {
          continue;
        }
        const float src0 = U32AsFloat(static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(1), context, lane)));
        const float src1 = U32AsFloat(static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(2), context, lane)));
        const float src2 = U32AsFloat(static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(3), context, lane)));
        context.wave.vgpr.Write(vdst, lane, FloatAsU32(src0 * src1 + src2));
      }
    } else if (instruction.mnemonic == "v_mfma_f32_16x16x4f32") {
      const uint32_t vdst = instruction.operands.at(0).kind == DecodedGcnOperandKind::VectorRegRange
                                ? RequireVectorRange(instruction.operands.at(0)).first
                                : RequireVectorIndex(instruction.operands.at(0));
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        if (!context.wave.exec.test(lane)) {
          continue;
        }
        const float src0 = U32AsFloat(static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(1), context, lane)));
        const float src1 = U32AsFloat(static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(2), context, lane)));
        const float src2 = U32AsFloat(static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(3), context, lane)));
        const float value = src2 + src0 * src1 * 4.0f;
        for (uint32_t reg = 0; reg < 4; ++reg) {
          context.wave.vgpr.Write(vdst + reg, lane, FloatAsU32(value));
          context.wave.agpr.Write(vdst + reg, lane, FloatAsU32(value));
        }
      }
    } else if (instruction.mnemonic == "v_mfma_f32_32x32x2f32") {
      const uint32_t vdst = instruction.operands.at(0).kind == DecodedGcnOperandKind::VectorRegRange
                                ? RequireVectorRange(instruction.operands.at(0)).first
                                : RequireVectorIndex(instruction.operands.at(0));
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        if (!context.wave.exec.test(lane)) {
          continue;
        }
        const float src0 = U32AsFloat(static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(1), context, lane)));
        const float src1 = U32AsFloat(static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(2), context, lane)));
        const float src2 = U32AsFloat(static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(3), context, lane)));
        const float value = src2 + src0 * src1 * 2.0f;
        for (uint32_t reg = 0; reg < 16; ++reg) {
          context.wave.vgpr.Write(vdst + reg, lane, FloatAsU32(value));
          context.wave.agpr.Write(vdst + reg, lane, FloatAsU32(value));
        }
      }
    } else if (instruction.mnemonic == "v_mfma_f32_16x16x4f16") {
      const uint32_t vdst = instruction.operands.at(0).kind == DecodedGcnOperandKind::VectorRegRange
                                ? RequireVectorRange(instruction.operands.at(0)).first
                                : RequireVectorIndex(instruction.operands.at(0));
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        if (!context.wave.exec.test(lane)) {
          continue;
        }
        const uint32_t src0_bits =
            static_cast<uint32_t>(ResolveVectorLane(instruction.operands.at(1), context, lane));
        const uint32_t src1_bits =
            static_cast<uint32_t>(ResolveVectorLane(instruction.operands.at(2), context, lane));
        const float acc = U32AsFloat(static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(3), context, lane)));
        const float a0 = HalfToFloat(static_cast<uint16_t>(src0_bits & 0xffffu));
        const float a1 = HalfToFloat(static_cast<uint16_t>(src0_bits >> 16u));
        const float b0 = HalfToFloat(static_cast<uint16_t>(src1_bits & 0xffffu));
        const float b1 = HalfToFloat(static_cast<uint16_t>(src1_bits >> 16u));
        const float value = acc + a0 * b0 + a1 * b1;
        for (uint32_t reg = 0; reg < 4; ++reg) {
          context.wave.vgpr.Write(vdst + reg, lane, FloatAsU32(value));
          context.wave.agpr.Write(vdst + reg, lane, FloatAsU32(value));
        }
      }
    } else if (instruction.mnemonic == "v_mfma_i32_16x16x4i8") {
      const uint32_t vdst = instruction.operands.at(0).kind == DecodedGcnOperandKind::VectorRegRange
                                ? RequireVectorRange(instruction.operands.at(0)).first
                                : RequireVectorIndex(instruction.operands.at(0));
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        if (!context.wave.exec.test(lane)) {
          continue;
        }
        const uint32_t src0_bits =
            static_cast<uint32_t>(ResolveVectorLane(instruction.operands.at(1), context, lane));
        const uint32_t src1_bits =
            static_cast<uint32_t>(ResolveVectorLane(instruction.operands.at(2), context, lane));
        int32_t acc = static_cast<int32_t>(
            ResolveVectorLane(instruction.operands.at(3), context, lane));
        for (uint32_t i = 0; i < 4; ++i) {
          const int8_t a = static_cast<int8_t>((src0_bits >> (i * 8u)) & 0xffu);
          const int8_t b = static_cast<int8_t>((src1_bits >> (i * 8u)) & 0xffu);
          acc += static_cast<int32_t>(a) * static_cast<int32_t>(b);
        }
        for (uint32_t reg = 0; reg < 4; ++reg) {
          context.wave.vgpr.Write(vdst + reg, lane, static_cast<uint32_t>(acc));
          context.wave.agpr.Write(vdst + reg, lane, static_cast<uint32_t>(acc));
        }
      }
    } else if (instruction.mnemonic == "v_mfma_i32_16x16x16i8") {
      const uint32_t vdst = instruction.operands.at(0).kind == DecodedGcnOperandKind::VectorRegRange
                                ? RequireVectorRange(instruction.operands.at(0)).first
                                : RequireVectorIndex(instruction.operands.at(0));
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        if (!context.wave.exec.test(lane)) {
          continue;
        }
        const uint32_t src0_bits =
            static_cast<uint32_t>(ResolveVectorLane(instruction.operands.at(1), context, lane));
        const uint32_t src1_bits =
            static_cast<uint32_t>(ResolveVectorLane(instruction.operands.at(2), context, lane));
        int32_t acc = static_cast<int32_t>(
            ResolveVectorLane(instruction.operands.at(3), context, lane));
        for (uint32_t i = 0; i < 4; ++i) {
          const int8_t a = static_cast<int8_t>((src0_bits >> (i * 8u)) & 0xffu);
          const int8_t b = static_cast<int8_t>((src1_bits >> (i * 8u)) & 0xffu);
          acc += 4 * static_cast<int32_t>(a) * static_cast<int32_t>(b);
        }
        for (uint32_t reg = 0; reg < 4; ++reg) {
          context.wave.vgpr.Write(vdst + reg, lane, static_cast<uint32_t>(acc));
          context.wave.agpr.Write(vdst + reg, lane, static_cast<uint32_t>(acc));
        }
      }
    } else if (instruction.mnemonic == "v_mfma_f32_16x16x2bf16") {
      const uint32_t vdst = instruction.operands.at(0).kind == DecodedGcnOperandKind::VectorRegRange
                                ? RequireVectorRange(instruction.operands.at(0)).first
                                : RequireVectorIndex(instruction.operands.at(0));
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        if (!context.wave.exec.test(lane)) {
          continue;
        }
        const uint32_t src0_bits =
            static_cast<uint32_t>(ResolveVectorLane(instruction.operands.at(1), context, lane));
        const uint32_t src1_bits =
            static_cast<uint32_t>(ResolveVectorLane(instruction.operands.at(2), context, lane));
        const float acc = U32AsFloat(static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(3), context, lane)));
        const float a0 = BFloat16ToFloat(static_cast<uint16_t>(src0_bits & 0xffffu));
        const float a1 = BFloat16ToFloat(static_cast<uint16_t>(src0_bits >> 16u));
        const float b0 = BFloat16ToFloat(static_cast<uint16_t>(src1_bits & 0xffffu));
        const float b1 = BFloat16ToFloat(static_cast<uint16_t>(src1_bits >> 16u));
        const float value = acc + a0 * b0 + a1 * b1;
        for (uint32_t reg = 0; reg < 4; ++reg) {
          context.wave.vgpr.Write(vdst + reg, lane, FloatAsU32(value));
          context.wave.agpr.Write(vdst + reg, lane, FloatAsU32(value));
        }
      }
    } else if (instruction.mnemonic == "v_accvgpr_read_b32") {
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      const uint32_t asrc = RequireAccumulatorIndex(instruction.operands.at(1));
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        if (!context.wave.exec.test(lane)) {
          continue;
        }
        context.wave.vgpr.Write(vdst, lane, context.wave.agpr.Read(asrc, lane));
      }
    } else if (instruction.mnemonic == "v_accvgpr_write_b32") {
      const uint32_t adst = RequireAccumulatorIndex(instruction.operands.at(0));
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        if (!context.wave.exec.test(lane)) {
          continue;
        }
        const uint32_t value = static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(1), context, lane));
        context.wave.agpr.Write(adst, lane, value);
      }
    } else if (instruction.mnemonic == "v_rndne_f32_e32") {
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        if (!context.wave.exec.test(lane)) {
          continue;
        }
        const float src = U32AsFloat(static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(1), context, lane)));
        context.wave.vgpr.Write(vdst, lane, FloatAsU32(std::nearbyint(src)));
      }
    } else if (instruction.mnemonic == "v_cvt_i32_f32_e32") {
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        if (!context.wave.exec.test(lane)) {
          continue;
        }
        const float src = U32AsFloat(static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(1), context, lane)));
        context.wave.vgpr.Write(vdst, lane, static_cast<uint32_t>(static_cast<int32_t>(src)));
      }
    } else if (instruction.mnemonic == "v_cvt_f32_i32_e32") {
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        if (!context.wave.exec.test(lane)) {
          continue;
        }
        const int32_t src = static_cast<int32_t>(
            ResolveVectorLane(instruction.operands.at(1), context, lane));
        context.wave.vgpr.Write(vdst, lane, FloatAsU32(static_cast<float>(src)));
      }
    } else if (instruction.mnemonic == "v_mbcnt_lo_u32_b32") {
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        if (!context.wave.exec.test(lane)) {
          continue;
        }
        const uint32_t mask = static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(1), context, lane));
        const uint32_t carry = static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(2), context, lane));
        const uint32_t lane_mask =
            lane >= 32 ? 0xffffffffu : ((lane == 0) ? 0u : ((1u << lane) - 1u));
        context.wave.vgpr.Write(vdst, lane, carry + std::popcount(mask & lane_mask));
      }
    } else if (instruction.mnemonic == "v_mbcnt_hi_u32_b32") {
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        if (!context.wave.exec.test(lane)) {
          continue;
        }
        const uint32_t mask = static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(1), context, lane));
        const uint32_t carry = static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(2), context, lane));
        const uint32_t upper_lane = lane > 32 ? (lane - 32) : 0;
        const uint32_t lane_mask =
            upper_lane >= 32 ? 0xffffffffu : ((upper_lane == 0) ? 0u : ((1u << upper_lane) - 1u));
        context.wave.vgpr.Write(vdst, lane, carry + std::popcount(mask & lane_mask));
      }
    } else if (instruction.mnemonic == "v_exp_f32_e32") {
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        if (!context.wave.exec.test(lane)) {
          continue;
        }
        const float src = U32AsFloat(static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(1), context, lane)));
        context.wave.vgpr.Write(vdst, lane, FloatAsU32(std::exp2(src)));
      }
    } else if (instruction.mnemonic == "v_rcp_f32_e32") {
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        if (!context.wave.exec.test(lane)) {
          continue;
        }
        const float src = U32AsFloat(static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(1), context, lane)));
        context.wave.vgpr.Write(vdst, lane, FloatAsU32(1.0f / src));
      }
    } else if (instruction.mnemonic == "v_ldexp_f32") {
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        if (!context.wave.exec.test(lane)) {
          continue;
        }
        const float value = U32AsFloat(static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(1), context, lane)));
        const int exp = static_cast<int>(ResolveVectorLane(instruction.operands.at(2), context, lane));
        context.wave.vgpr.Write(vdst, lane, FloatAsU32(std::ldexp(value, exp)));
      }
    } else if (instruction.mnemonic == "v_div_scale_f32") {
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        if (!context.wave.exec.test(lane)) {
          continue;
        }
        const uint32_t value =
            static_cast<uint32_t>(ResolveVectorLane(instruction.operands.at(1), context, lane));
        context.wave.vgpr.Write(vdst, lane, value);
      }
    } else if (instruction.mnemonic == "v_div_fmas_f32") {
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        if (!context.wave.exec.test(lane)) {
          continue;
        }
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
        if (!context.wave.exec.test(lane)) {
          continue;
        }
        const float denom = U32AsFloat(static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(2), context, lane)));
        const float numer = U32AsFloat(static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(3), context, lane)));
        const float result = numer / denom;
        context.wave.vgpr.Write(vdst, lane, FloatAsU32(result));
        if (lane == 0) {
          DebugLog("pc=0x%llx v_div_fixup_f32 denom=%f numer=%f result=%f",
                   static_cast<unsigned long long>(instruction.pc), denom, numer, result);
        }
      }
    } else if (instruction.mnemonic == "v_cndmask_b32_e32") {
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        if (!context.wave.exec.test(lane)) {
          continue;
        }
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
        if (!context.wave.exec.test(lane)) {
          continue;
        }
        const uint32_t false_value =
            static_cast<uint32_t>(ResolveVectorLane(instruction.operands.at(1), context, lane));
        const uint32_t true_value =
            static_cast<uint32_t>(ResolveVectorLane(instruction.operands.at(2), context, lane));
        const bool select_true = ((mask >> lane) & 1ull) != 0;
        context.wave.vgpr.Write(vdst, lane, select_true ? true_value : false_value);
      }
    } else if (instruction.mnemonic == "v_mad_u64_u32") {
      const auto [vdst, _] = RequireVectorRange(instruction.operands.at(0));
      const uint32_t src0 = RequireVectorIndex(instruction.operands.at(2));
      const uint32_t src1 = RequireScalarIndex(instruction.operands.at(3));
      const uint32_t src2_pair = RequireVectorRange(instruction.operands.at(4)).first;
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        if (!context.wave.exec.test(lane)) {
          continue;
        }
        const uint64_t mul_lhs = static_cast<uint32_t>(context.wave.vgpr.Read(src0, lane));
        const uint64_t mul_rhs = static_cast<uint32_t>(context.wave.sgpr.Read(src1));
        const uint64_t acc_lo = static_cast<uint32_t>(context.wave.vgpr.Read(src2_pair, lane));
        const uint64_t acc_hi = static_cast<uint32_t>(context.wave.vgpr.Read(src2_pair + 1, lane));
        const uint64_t acc = (acc_hi << 32u) | acc_lo;
        const uint64_t value = acc + mul_lhs * mul_rhs;
        context.wave.vgpr.Write(vdst, lane, static_cast<uint32_t>(value & 0xffffffffu));
        context.wave.vgpr.Write(vdst + 1, lane, static_cast<uint32_t>(value >> 32u));
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
    switch (instruction.encoding_id) {
      case 8:
      case 38: {  // v_cmp_gt_i32
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        if (!context.wave.exec.test(lane)) {
          continue;
        }
        const int32_t lhs =
            static_cast<int32_t>(ResolveVectorLane(instruction.operands.at(1), context, lane));
        const int32_t rhs =
            static_cast<int32_t>(ResolveVectorLane(instruction.operands.at(2), context, lane));
        if (lhs > rhs) {
          mask |= (1ull << lane);
        }
      }
      break;
      }
      case 75: {  // v_cmp_le_i32_e32
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        if (!context.wave.exec.test(lane)) {
          continue;
        }
        const int32_t lhs =
            static_cast<int32_t>(ResolveVectorLane(instruction.operands.at(1), context, lane));
        const int32_t rhs =
            static_cast<int32_t>(ResolveVectorLane(instruction.operands.at(2), context, lane));
        if (lhs <= rhs) {
          mask |= (1ull << lane);
        }
      }
      break;
      }
      case 76: {  // v_cmp_lt_i32_e32
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        if (!context.wave.exec.test(lane)) {
          continue;
        }
        const int32_t lhs =
            static_cast<int32_t>(ResolveVectorLane(instruction.operands.at(1), context, lane));
        const int32_t rhs =
            static_cast<int32_t>(ResolveVectorLane(instruction.operands.at(2), context, lane));
        if (lhs < rhs) {
          mask |= (1ull << lane);
        }
      }
      break;
      }
      case 56: {  // v_cmp_gt_u32_e32
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        if (!context.wave.exec.test(lane)) {
          continue;
        }
        const uint32_t lhs =
            static_cast<uint32_t>(ResolveVectorLane(instruction.operands.at(1), context, lane));
        const uint32_t rhs =
            static_cast<uint32_t>(ResolveVectorLane(instruction.operands.at(2), context, lane));
        if (lhs > rhs) {
          mask |= (1ull << lane);
        }
      }
      break;
      }
      case 66: {  // v_cmp_eq_u32_e32
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        if (!context.wave.exec.test(lane)) {
          continue;
        }
        const uint32_t lhs =
            static_cast<uint32_t>(ResolveVectorLane(instruction.operands.at(1), context, lane));
        const uint32_t rhs =
            static_cast<uint32_t>(ResolveVectorLane(instruction.operands.at(2), context, lane));
        if (lhs == rhs) {
          mask |= (1ull << lane);
        }
      }
      break;
      }
      case 57: {  // v_cmp_ngt_f32_e32
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        if (!context.wave.exec.test(lane)) {
          continue;
        }
        const float lhs = U32AsFloat(static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(1), context, lane)));
        const float rhs = U32AsFloat(static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(2), context, lane)));
        if (!(lhs > rhs)) {
          mask |= (1ull << lane);
        }
      }
      break;
      }
      case 58: {  // v_cmp_nlt_f32_e32
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        if (!context.wave.exec.test(lane)) {
          continue;
        }
        const float lhs = U32AsFloat(static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(1), context, lane)));
        const float rhs = U32AsFloat(static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(2), context, lane)));
        if (!(lhs < rhs)) {
          mask |= (1ull << lane);
        }
      }
      break;
      }
      default:
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
    switch (instruction.encoding_id) {
      case 10: {  // s_cbranch_execz
      if (context.wave.exec.none()) {
        context.wave.pc =
            BranchTarget(context.wave.pc, static_cast<int32_t>(instruction.operands.at(0).info.immediate));
      } else {
        context.wave.pc += instruction.size_bytes;
      }
      return;
      }
      case 22: {  // s_cbranch_scc1
      if (context.wave.ScalarMaskBit0()) {
        context.wave.pc =
            BranchTarget(context.wave.pc, static_cast<int32_t>(instruction.operands.at(0).info.immediate));
      } else {
        context.wave.pc += instruction.size_bytes;
      }
      return;
      }
      case 26: {  // s_cbranch_scc0
      if (!context.wave.ScalarMaskBit0()) {
        context.wave.pc =
            BranchTarget(context.wave.pc, static_cast<int32_t>(instruction.operands.at(0).info.immediate));
      } else {
        context.wave.pc += instruction.size_bytes;
      }
      return;
      }
      case 43: {  // s_cbranch_vccz
      if (context.vcc == 0) {
        context.wave.pc =
            BranchTarget(context.wave.pc, static_cast<int32_t>(instruction.operands.at(0).info.immediate));
      } else {
        context.wave.pc += instruction.size_bytes;
      }
      return;
      }
      case 74: {  // s_cbranch_execnz
      if (context.wave.exec.any()) {
        context.wave.pc =
            BranchTarget(context.wave.pc, static_cast<int32_t>(instruction.operands.at(0).info.immediate));
      } else {
        context.wave.pc += instruction.size_bytes;
      }
      return;
      }
      case 27: {  // s_branch
      context.wave.pc =
          BranchTarget(context.wave.pc, static_cast<int32_t>(instruction.operands.at(0).info.immediate));
      return;
      }
      default:
        throw std::invalid_argument("unsupported branch opcode: " + instruction.mnemonic);
    }
  }
};

class FlatMemoryHandler final : public IRawGcnSemanticHandler {
 public:
  void Execute(const DecodedGcnInstruction& instruction, RawGcnWaveContext& context) const override {
    if (instruction.mnemonic == "global_load_dword") {
      const int64_t offset = instruction.operands.size() >= 3 && instruction.operands.back().info.has_immediate
                                 ? instruction.operands.back().info.immediate
                                 : 0;
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      const auto [addr, _] = RequireVectorRange(instruction.operands.at(1));
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        if (!context.wave.exec.test(lane)) {
          continue;
        }
        const uint64_t lo = static_cast<uint32_t>(context.wave.vgpr.Read(addr, lane));
        const uint64_t hi = static_cast<uint32_t>(context.wave.vgpr.Read(addr + 1, lane));
        const uint64_t address = static_cast<uint64_t>(static_cast<int64_t>((hi << 32u) | lo) + offset);
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
      const int64_t offset = instruction.operands.size() >= 4 && instruction.operands.back().info.has_immediate
                                 ? instruction.operands.back().info.immediate
                                 : 0;
      if (instruction.operands.at(0).kind == DecodedGcnOperandKind::VectorRegRange) {
        const auto [addr, _] = RequireVectorRange(instruction.operands.at(0));
        const uint32_t data = RequireVectorIndex(instruction.operands.at(1));
        for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
          if (!context.wave.exec.test(lane)) {
            continue;
          }
          const uint64_t lo = static_cast<uint32_t>(context.wave.vgpr.Read(addr, lane));
          const uint64_t hi = static_cast<uint32_t>(context.wave.vgpr.Read(addr + 1, lane));
          const uint64_t address = static_cast<uint64_t>(static_cast<int64_t>((hi << 32u) | lo) + offset);
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
          if (!context.wave.exec.test(lane)) {
            continue;
          }
          const int32_t offset = static_cast<int32_t>(context.wave.vgpr.Read(vaddr, lane));
          const uint64_t address = static_cast<uint64_t>(static_cast<int64_t>(base) + offset);
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
    } else if (instruction.mnemonic == "global_atomic_add") {
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      const uint32_t data = RequireVectorIndex(instruction.operands.at(1));
      const uint32_t saddr = RequireScalarRange(instruction.operands.at(2)).first;
      const uint64_t base = static_cast<uint64_t>(context.wave.sgpr.Read(saddr)) |
                            (static_cast<uint64_t>(context.wave.sgpr.Read(saddr + 1)) << 32u);
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        if (!context.wave.exec.test(lane)) {
          continue;
        }
        const uint32_t old_value = context.memory.LoadGlobalValue<uint32_t>(base);
        const uint32_t add_value = static_cast<uint32_t>(context.wave.vgpr.Read(data, lane));
        context.memory.StoreGlobalValue<uint32_t>(base, old_value + add_value);
        context.wave.vgpr.Write(vdst, lane, old_value);
      }
      ++context.stats.global_loads;
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
        if (!context.wave.exec.test(lane) ||
            static_cast<size_t>(byte_offset) + sizeof(uint32_t) > context.block.shared_memory.size()) {
          continue;
        }
        const uint32_t value =
            static_cast<uint32_t>(context.wave.vgpr.Read(data_vgpr, lane));
        StoreU32(context.block.shared_memory, byte_offset, value);
        if (lane == 0) {
          DebugLog("pc=0x%llx ds_write_b32 addr=0x%x value=0x%x",
                   static_cast<unsigned long long>(instruction.pc), byte_offset, value);
        }
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
        if (!context.wave.exec.test(lane) ||
            static_cast<size_t>(byte_offset) + sizeof(uint32_t) > context.block.shared_memory.size()) {
          continue;
        }
        const uint32_t value = LoadU32(context.block.shared_memory, byte_offset);
        context.wave.vgpr.Write(vdst, lane, value);
        if (lane == 0) {
          DebugLog("pc=0x%llx ds_read_b32 addr=0x%x value=0x%x",
                   static_cast<unsigned long long>(instruction.pc), byte_offset, value);
        }
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
    switch (instruction.encoding_id) {
      case 29: {  // s_barrier
      ++context.stats.barriers;
      execution_sync_ops::MarkWaveAtBarrier(context.wave,
                                            context.block.barrier_generation,
                                            context.block.barrier_arrivals,
                                            false);
      return;
      }
      case 68:  // s_nop
      context.wave.pc += instruction.size_bytes;
      return;
      case 12:  // s_waitcnt
      context.wave.pc += instruction.size_bytes;
      return;
      case 1:  // s_endpgm
      context.wave.status = WaveStatus::Exited;
      ++context.stats.wave_exits;
      return;
      default:
        throw std::invalid_argument("unsupported special opcode: " + instruction.mnemonic);
    }
  }
};

struct HandlerBinding {
  const char* mnemonic = nullptr;
  const IRawGcnSemanticHandler* handler = nullptr;
};

const IRawGcnSemanticHandler* HandlerForSemanticFamily(std::string_view semantic_family,
                                                       std::string_view mnemonic) {
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

  if (semantic_family == "scalar_memory") {
    return &kScalarMemoryHandler;
  }
  if (semantic_family == "scalar_alu") {
    return &kScalarAluHandler;
  }
  if (semantic_family == "scalar_compare") {
    return &kScalarCompareHandler;
  }
  if (semantic_family == "vector_alu") {
    return &kVectorAluHandler;
  }
  if (semantic_family == "vector_compare") {
    return &kVectorCompareHandler;
  }
  if (semantic_family == "vector_memory") {
    return &kFlatMemoryHandler;
  }
  if (semantic_family == "lds") {
    return &kSharedMemoryHandler;
  }
  if (semantic_family == "branch_or_sync") {
    if (mnemonic == "s_barrier" || mnemonic == "s_waitcnt" || mnemonic == "s_endpgm" ||
        mnemonic == "s_nop") {
      return &kSpecialHandler;
    }
    return &kBranchHandler;
  }
  return nullptr;
}

const std::vector<HandlerBinding>& HandlerBindings() {
  static const MaskHandler kMaskHandler;
  static const VectorCompareHandler kVectorCompareHandler;
  static const std::vector<HandlerBinding> kBindings = {
      {.mnemonic = "s_and_saveexec_b64", .handler = &kMaskHandler},
      {.mnemonic = "v_cmp_gt_i32_e64", .handler = &kVectorCompareHandler},
  };
  return kBindings;
}

}  // namespace

const IRawGcnSemanticHandler& RawGcnSemanticHandlerRegistry::Get(std::string_view mnemonic) {
  if (const auto* def = FindGeneratedGcnInstDefByMnemonic(mnemonic); def != nullptr) {
    if (const auto* handler = HandlerForSemanticFamily(def->semantic_family, def->mnemonic)) {
      return *handler;
    }
  }
  for (const auto& binding : HandlerBindings()) {
    if (binding.mnemonic == mnemonic) {
      return *binding.handler;
    }
  }
  throw std::invalid_argument("unsupported raw GCN opcode: " + std::string(mnemonic));
}

const IRawGcnSemanticHandler& RawGcnSemanticHandlerRegistry::Get(
    const DecodedGcnInstruction& instruction) {
  for (const auto& binding : HandlerBindings()) {
    if (binding.mnemonic == instruction.mnemonic) {
      return *binding.handler;
    }
  }
  if (instruction.encoding_id != 0) {
    if (const auto* def = FindGeneratedGcnInstDefById(instruction.encoding_id); def != nullptr) {
      if (const auto* handler =
              HandlerForSemanticFamily(def->semantic_family, instruction.mnemonic)) {
        return *handler;
      }
    }
  }
  return Get(std::string_view(instruction.mnemonic));
}

}  // namespace gpu_model
