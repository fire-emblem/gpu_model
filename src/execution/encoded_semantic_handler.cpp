#include "gpu_model/execution/encoded_semantic_handler.h"

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

#include "gpu_model/instruction/encoded/internal/encoded_gcn_encoding_def.h"
#include "gpu_model/instruction/encoded/internal/encoded_gcn_db_lookup.h"
#include "gpu_model/execution/sync_ops.h"
#include "gpu_model/execution/internal/tensor_op_utils.h"

namespace gpu_model {

namespace {

uint32_t LaneCount(const EncodedWaveContext& context) {
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
  return std::getenv("GPU_MODEL_ENCODED_EXEC_DEBUG") != nullptr;
}

void DebugLog(const char* fmt, ...) {
  if (!DebugEnabled()) {
    return;
  }
  va_list args;
  va_start(args, fmt);
  std::fputs("[gpu_model_encoded_exec] ", stderr);
  std::vfprintf(stderr, fmt, args);
  std::fputc('\n', stderr);
  va_end(args);
}

uint32_t RequireScalarIndex(const DecodedInstructionOperand& operand) {
  if (operand.kind != DecodedInstructionOperandKind::ScalarReg || operand.info.reg_count != 1) {
    throw std::invalid_argument("expected scalar register operand");
  }
  return operand.info.reg_first;
}

uint32_t RequireVectorIndex(const DecodedInstructionOperand& operand) {
  if (operand.kind != DecodedInstructionOperandKind::VectorReg || operand.info.reg_count != 1) {
    throw std::invalid_argument("expected vector register operand");
  }
  return operand.info.reg_first;
}

uint32_t RequireAccumulatorIndex(const DecodedInstructionOperand& operand) {
  if (operand.kind != DecodedInstructionOperandKind::AccumulatorReg || operand.info.reg_count != 1) {
    throw std::invalid_argument("expected accumulator register operand");
  }
  return operand.info.reg_first;
}

std::pair<uint32_t, uint32_t> RequireScalarRange(const DecodedInstructionOperand& operand) {
  if (operand.kind != DecodedInstructionOperandKind::ScalarRegRange || operand.info.reg_count == 0) {
    throw std::invalid_argument("expected scalar register range operand");
  }
  return {operand.info.reg_first, operand.info.reg_first + operand.info.reg_count - 1};
}

std::pair<uint32_t, uint32_t> RequireVectorRange(const DecodedInstructionOperand& operand) {
  if (operand.kind != DecodedInstructionOperandKind::VectorRegRange || operand.info.reg_count == 0) {
    throw std::invalid_argument("expected vector register range operand");
  }
  return {operand.info.reg_first, operand.info.reg_first + operand.info.reg_count - 1};
}

uint64_t ResolveScalarLike(const DecodedInstructionOperand& operand, const EncodedWaveContext& context) {
  if (operand.kind == DecodedInstructionOperandKind::Immediate ||
      operand.kind == DecodedInstructionOperandKind::BranchTarget) {
    if (!operand.info.has_immediate) {
      throw std::invalid_argument("immediate operand missing value");
    }
    return static_cast<uint64_t>(operand.info.immediate);
  }
  if (operand.kind == DecodedInstructionOperandKind::ScalarReg) {
    return context.wave.sgpr.Read(RequireScalarIndex(operand));
  }
  if (operand.kind == DecodedInstructionOperandKind::SpecialReg &&
      operand.info.special_reg == GcnSpecialReg::Vcc) {
    return context.vcc;
  }
  if (operand.kind == DecodedInstructionOperandKind::SpecialReg &&
      operand.info.special_reg == GcnSpecialReg::Exec) {
    return context.wave.exec.to_ullong();
  }
  throw std::invalid_argument("unsupported scalar-like raw operand");
}

uint64_t ResolveScalarPair(const DecodedInstructionOperand& operand, const EncodedWaveContext& context) {
  if (operand.kind == DecodedInstructionOperandKind::Immediate ||
      operand.kind == DecodedInstructionOperandKind::BranchTarget) {
    if (!operand.info.has_immediate) {
      throw std::invalid_argument("scalar pair immediate missing value");
    }
    return static_cast<uint64_t>(operand.info.immediate);
  }
  if (operand.kind == DecodedInstructionOperandKind::ScalarReg) {
    const uint32_t first = operand.info.reg_first;
    return static_cast<uint64_t>(context.wave.sgpr.Read(first)) |
           (static_cast<uint64_t>(context.wave.sgpr.Read(first + 1)) << 32u);
  }
  if (operand.kind == DecodedInstructionOperandKind::ScalarRegRange && operand.info.reg_count == 2) {
    const uint32_t first = operand.info.reg_first;
    return static_cast<uint64_t>(context.wave.sgpr.Read(first)) |
           (static_cast<uint64_t>(context.wave.sgpr.Read(first + 1)) << 32u);
  }
  if (operand.kind == DecodedInstructionOperandKind::SpecialReg &&
      operand.info.special_reg == GcnSpecialReg::Vcc) {
    return context.vcc;
  }
  if (operand.kind == DecodedInstructionOperandKind::SpecialReg &&
      operand.info.special_reg == GcnSpecialReg::Exec) {
    return context.wave.exec.to_ullong();
  }
  throw std::invalid_argument("unsupported scalar pair operand");
}

void StoreScalarPair(const DecodedInstructionOperand& operand, EncodedWaveContext& context, uint64_t value) {
  if (operand.kind == DecodedInstructionOperandKind::ScalarReg) {
    const uint32_t first = operand.info.reg_first;
    context.wave.sgpr.Write(first, static_cast<uint32_t>(value & 0xffffffffu));
    context.wave.sgpr.Write(first + 1, static_cast<uint32_t>(value >> 32u));
    return;
  }
  if (operand.kind == DecodedInstructionOperandKind::ScalarRegRange && operand.info.reg_count == 2) {
    const uint32_t first = operand.info.reg_first;
    context.wave.sgpr.Write(first, static_cast<uint32_t>(value & 0xffffffffu));
    context.wave.sgpr.Write(first + 1, static_cast<uint32_t>(value >> 32u));
    return;
  }
  if (operand.kind == DecodedInstructionOperandKind::SpecialReg &&
      operand.info.special_reg == GcnSpecialReg::Vcc) {
    context.vcc = value;
    return;
  }
  if (operand.kind == DecodedInstructionOperandKind::SpecialReg &&
      operand.info.special_reg == GcnSpecialReg::Exec) {
    context.wave.exec = MaskFromU64(value);
    return;
  }
  throw std::invalid_argument("unsupported scalar pair destination");
}

uint64_t ResolveVectorLane(const DecodedInstructionOperand& operand,
                           const EncodedWaveContext& context,
                           uint32_t lane) {
  if (operand.kind == DecodedInstructionOperandKind::Immediate) {
    if (!operand.info.has_immediate) {
      throw std::invalid_argument("immediate operand missing value");
    }
    return static_cast<uint64_t>(operand.info.immediate);
  }
  if (operand.kind == DecodedInstructionOperandKind::ScalarReg) {
    return context.wave.sgpr.Read(RequireScalarIndex(operand));
  }
  if (operand.kind == DecodedInstructionOperandKind::ScalarRegRange) {
    return ResolveScalarPair(operand, context);
  }
  if (operand.kind == DecodedInstructionOperandKind::SpecialReg) {
    switch (operand.info.special_reg) {
      case GcnSpecialReg::Vcc:
        return ((context.vcc >> lane) & 1ull) != 0 ? 1ull : 0ull;
      case GcnSpecialReg::Exec:
        return context.wave.exec.test(lane) ? 1ull : 0ull;
      default:
        break;
    }
  }
  if (operand.kind == DecodedInstructionOperandKind::VectorReg) {
    return context.wave.vgpr.Read(RequireVectorIndex(operand), lane);
  }
  throw std::invalid_argument("unsupported vector-lane raw operand kind=" +
                              std::to_string(static_cast<int>(operand.kind)) +
                              " text=" + operand.text);
}

const GcnIsaOpcodeDescriptor& RequireCanonicalOpcode(const DecodedInstruction& instruction) {
  if (const auto* descriptor = FindEncodedGcnFallbackOpcodeDescriptor(instruction.words); descriptor != nullptr) {
    return *descriptor;
  }
  if (const auto* descriptor = FindGcnIsaOpcodeDescriptorByName(instruction.mnemonic);
      descriptor != nullptr) {
    return *descriptor;
  }
  throw std::invalid_argument("missing canonical opcode descriptor: " + instruction.mnemonic);
}

class ScalarMemoryHandler final : public IEncodedSemanticHandler {
 public:
  void Execute(const DecodedInstruction& instruction, EncodedWaveContext& context) const override {
    MemoryRequest request;
    request.space = MemorySpace::Constant;
    request.kind = AccessKind::Load;
    request.exec_snapshot.set(0);
    if (instruction.mnemonic == "s_load_dword") {
      const uint32_t offset =
          static_cast<uint32_t>(ResolveScalarLike(instruction.operands.at(2), context));
      const uint32_t sdst = RequireScalarIndex(instruction.operands.at(0));
      const uint64_t base = ResolveScalarPair(instruction.operands.at(1), context);
      request.lanes[0] = LaneAccess{.active = true, .addr = base + offset, .bytes = 4, .value = 0};
      request.dst = RegRef{.file = RegisterFile::Scalar, .index = sdst};
      context.wave.sgpr.Write(
          sdst, context.memory.LoadGlobalValue<uint32_t>(base + offset));
    } else if (instruction.mnemonic == "s_load_dwordx2") {
      const uint32_t offset =
          static_cast<uint32_t>(ResolveScalarLike(instruction.operands.at(2), context));
      const auto [sdst, _] = RequireScalarRange(instruction.operands.at(0));
      const uint64_t base = ResolveScalarPair(instruction.operands.at(1), context);
      request.lanes[0] = LaneAccess{.active = true, .addr = base + offset, .bytes = 8, .value = 0};
      request.dst = RegRef{.file = RegisterFile::Scalar, .index = sdst};
      const uint64_t value = context.memory.LoadGlobalValue<uint64_t>(base + offset);
      context.wave.sgpr.Write(sdst, static_cast<uint32_t>(value & 0xffffffffu));
      context.wave.sgpr.Write(sdst + 1, static_cast<uint32_t>(value >> 32u));
    } else if (instruction.mnemonic == "s_load_dwordx4") {
      const uint32_t offset =
          static_cast<uint32_t>(ResolveScalarLike(instruction.operands.at(2), context));
      const auto [sdst, _] = RequireScalarRange(instruction.operands.at(0));
      const uint64_t base = ResolveScalarPair(instruction.operands.at(1), context);
      request.lanes[0] = LaneAccess{.active = true, .addr = base + offset, .bytes = 16, .value = 0};
      request.dst = RegRef{.file = RegisterFile::Scalar, .index = sdst};
      context.wave.sgpr.Write(sdst + 0, context.memory.LoadGlobalValue<uint32_t>(base + offset + 0));
      context.wave.sgpr.Write(sdst + 1, context.memory.LoadGlobalValue<uint32_t>(base + offset + 4));
      context.wave.sgpr.Write(sdst + 2, context.memory.LoadGlobalValue<uint32_t>(base + offset + 8));
      context.wave.sgpr.Write(sdst + 3, context.memory.LoadGlobalValue<uint32_t>(base + offset + 12));
    } else {
      throw std::invalid_argument("unsupported scalar memory opcode: " + instruction.mnemonic);
    }
    if (context.captured_memory_request != nullptr) {
      *context.captured_memory_request = request;
    }
    context.wave.pc += instruction.size_bytes;
  }
};

class ScalarAluHandler final : public IEncodedSemanticHandler {
 public:
  void Execute(const DecodedInstruction& instruction, EncodedWaveContext& context) const override {
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
    } else if (descriptor.op_type == GcnIsaOpType::Sop1 &&
               descriptor.opcode == static_cast<uint16_t>(GcnIsaSop1Opcode::S_ABS_I32)) {
      const uint32_t sdst = RequireScalarIndex(instruction.operands.at(0));
      const int32_t value =
          static_cast<int32_t>(ResolveScalarLike(instruction.operands.at(1), context));
      context.wave.sgpr.Write(sdst, static_cast<uint32_t>(value < 0 ? -value : value));
    } else if (descriptor.op_type == GcnIsaOpType::Sop2 &&
               descriptor.opcode == static_cast<uint16_t>(GcnIsaSop2Opcode::S_SUB_I32)) {
      const uint32_t sdst = RequireScalarIndex(instruction.operands.at(0));
      const uint32_t lhs =
          static_cast<uint32_t>(ResolveScalarLike(instruction.operands.at(1), context));
      const uint32_t rhs =
          static_cast<uint32_t>(ResolveScalarLike(instruction.operands.at(2), context));
      context.wave.sgpr.Write(sdst, lhs - rhs);
    } else if (descriptor.op_type == GcnIsaOpType::Sop1 &&
               descriptor.opcode == static_cast<uint16_t>(GcnIsaSop1Opcode::S_SEXT_I32_I16)) {
      const uint32_t sdst = RequireScalarIndex(instruction.operands.at(0));
      const int16_t value = static_cast<int16_t>(
          static_cast<uint32_t>(ResolveScalarLike(instruction.operands.at(1), context)) & 0xffffu);
      context.wave.sgpr.Write(sdst, static_cast<uint32_t>(static_cast<int32_t>(value)));
    } else if (descriptor.op_type == GcnIsaOpType::Sop2 &&
               descriptor.opcode == static_cast<uint16_t>(GcnIsaSop2Opcode::S_OR_B64)) {
      const uint64_t lhs = ResolveScalarPair(instruction.operands.at(1), context);
      const uint64_t rhs = ResolveScalarPair(instruction.operands.at(2), context);
      const uint64_t value = lhs | rhs;
      StoreScalarPair(instruction.operands.at(0), context, value);
      if (instruction.operands.at(0).kind == DecodedInstructionOperandKind::SpecialReg &&
          instruction.operands.at(0).info.special_reg == GcnSpecialReg::Exec) {
        DebugLog("pc=0x%llx s_or_b64 exec lhs=0x%llx rhs=0x%llx out=0x%llx",
                 static_cast<unsigned long long>(instruction.pc),
                 static_cast<unsigned long long>(lhs),
                 static_cast<unsigned long long>(rhs),
                 static_cast<unsigned long long>(value));
      }
    } else if (descriptor.op_type == GcnIsaOpType::Sop2 &&
               descriptor.opcode == static_cast<uint16_t>(GcnIsaSop2Opcode::S_OR_B32)) {
      const uint32_t sdst = RequireScalarIndex(instruction.operands.at(0));
      const uint32_t lhs =
          static_cast<uint32_t>(ResolveScalarLike(instruction.operands.at(1), context));
      const uint32_t rhs =
          static_cast<uint32_t>(ResolveScalarLike(instruction.operands.at(2), context));
      context.wave.sgpr.Write(sdst, lhs | rhs);
    } else if (descriptor.op_type == GcnIsaOpType::Sop2 &&
               descriptor.opcode == static_cast<uint16_t>(GcnIsaSop2Opcode::S_XOR_B64)) {
      const uint64_t lhs = ResolveScalarPair(instruction.operands.at(1), context);
      const uint64_t rhs = ResolveScalarPair(instruction.operands.at(2), context);
      StoreScalarPair(instruction.operands.at(0), context, lhs ^ rhs);
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

class ScalarCompareHandler final : public IEncodedSemanticHandler {
 public:
  void Execute(const DecodedInstruction& instruction, EncodedWaveContext& context) const override {
    const auto& descriptor = RequireCanonicalOpcode(instruction);
    if (descriptor.op_type == GcnIsaOpType::Sopc &&
        descriptor.opcode == static_cast<uint16_t>(GcnIsaSopcOpcode::S_CMP_LT_I32)) {
      const int32_t lhs =
          static_cast<int32_t>(ResolveScalarLike(instruction.operands.at(0), context));
      const int32_t rhs =
          static_cast<int32_t>(ResolveScalarLike(instruction.operands.at(1), context));
      context.wave.SetScalarMaskBit0(lhs < rhs);
    } else if (descriptor.op_type == GcnIsaOpType::Sopc &&
               descriptor.opcode == static_cast<uint16_t>(GcnIsaSopcOpcode::S_CMP_GT_I32)) {
      const int32_t lhs =
          static_cast<int32_t>(ResolveScalarLike(instruction.operands.at(0), context));
      const int32_t rhs =
          static_cast<int32_t>(ResolveScalarLike(instruction.operands.at(1), context));
      context.wave.SetScalarMaskBit0(lhs > rhs);
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

class VectorAluHandler final : public IEncodedSemanticHandler {
 public:
  void Execute(const DecodedInstruction& instruction, EncodedWaveContext& context) const override {
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
    } else if (instruction.mnemonic == "v_sub_u32_e32") {
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        if (!context.wave.exec.test(lane)) {
          continue;
        }
        const uint32_t lhs =
            static_cast<uint32_t>(ResolveVectorLane(instruction.operands.at(1), context, lane));
        const uint32_t rhs =
            static_cast<uint32_t>(ResolveVectorLane(instruction.operands.at(2), context, lane));
        context.wave.vgpr.Write(vdst, lane, lhs - rhs);
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
      const uint32_t src_pair = instruction.operands.at(2).kind == DecodedInstructionOperandKind::VectorRegRange
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
    } else if (instruction.mnemonic == "v_cvt_f32_u32_e32") {
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        if (!context.wave.exec.test(lane)) {
          continue;
        }
        const uint32_t value =
            static_cast<uint32_t>(ResolveVectorLane(instruction.operands.at(1), context, lane));
        context.wave.vgpr.Write(vdst, lane, FloatAsU32(static_cast<float>(value)));
      }
    } else if (instruction.mnemonic == "v_cvt_u32_f32_e32") {
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        if (!context.wave.exec.test(lane)) {
          continue;
        }
        const float value = U32AsFloat(static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(1), context, lane)));
        context.wave.vgpr.Write(vdst, lane, static_cast<uint32_t>(value));
      }
    } else if (instruction.mnemonic == "v_rcp_iflag_f32_e32") {
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        if (!context.wave.exec.test(lane)) {
          continue;
        }
        const float value = U32AsFloat(static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(1), context, lane)));
        context.wave.vgpr.Write(vdst, lane, FloatAsU32(1.0f / value));
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
    } else if (instruction.mnemonic == "v_and_b32_e32") {
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        if (!context.wave.exec.test(lane)) {
          continue;
        }
        const uint32_t lhs =
            static_cast<uint32_t>(ResolveVectorLane(instruction.operands.at(1), context, lane));
        const uint32_t rhs =
            static_cast<uint32_t>(ResolveVectorLane(instruction.operands.at(2), context, lane));
        context.wave.vgpr.Write(vdst, lane, lhs & rhs);
      }
    } else if (instruction.mnemonic == "v_xor_b32_e32") {
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        if (!context.wave.exec.test(lane)) {
          continue;
        }
        const uint32_t lhs =
            static_cast<uint32_t>(ResolveVectorLane(instruction.operands.at(1), context, lane));
        const uint32_t rhs =
            static_cast<uint32_t>(ResolveVectorLane(instruction.operands.at(2), context, lane));
        context.wave.vgpr.Write(vdst, lane, lhs ^ rhs);
      }
    } else if (instruction.mnemonic == "v_subrev_u32_e32") {
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        if (!context.wave.exec.test(lane)) {
          continue;
        }
        const uint32_t lhs =
            static_cast<uint32_t>(ResolveVectorLane(instruction.operands.at(1), context, lane));
        const uint32_t rhs =
            static_cast<uint32_t>(ResolveVectorLane(instruction.operands.at(2), context, lane));
        context.wave.vgpr.Write(vdst, lane, rhs - lhs);
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
    } else if (instruction.mnemonic == "v_max_i32_e32") {
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        if (!context.wave.exec.test(lane)) {
          continue;
        }
        const int32_t lhs =
            static_cast<int32_t>(ResolveVectorLane(instruction.operands.at(1), context, lane));
        const int32_t rhs =
            static_cast<int32_t>(ResolveVectorLane(instruction.operands.at(2), context, lane));
        context.wave.vgpr.Write(vdst, lane, static_cast<uint32_t>(std::max(lhs, rhs)));
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
    } else if (instruction.mnemonic == "v_or_b32_e32") {
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        if (!context.wave.exec.test(lane)) {
          continue;
        }
        const uint32_t lhs = static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(1), context, lane));
        const uint32_t rhs = static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(2), context, lane));
        context.wave.vgpr.Write(vdst, lane, lhs | rhs);
      }
    } else if (instruction.mnemonic == "v_or3_b32") {
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        if (!context.wave.exec.test(lane)) {
          continue;
        }
        const uint32_t src0 = static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(1), context, lane));
        const uint32_t src1 = static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(2), context, lane));
        const uint32_t src2 = static_cast<uint32_t>(
            ResolveVectorLane(instruction.operands.at(3), context, lane));
        context.wave.vgpr.Write(vdst, lane, src0 | src1 | src2);
      }
    } else if (instruction.mnemonic == "v_mfma_f32_16x16x4f32") {
      const uint32_t vdst = instruction.operands.at(0).kind == DecodedInstructionOperandKind::VectorRegRange
                                ? RequireVectorRange(instruction.operands.at(0)).first
                                : RequireVectorIndex(instruction.operands.at(0));
      const auto storage_policy = DefaultTensorResultStoragePolicy();
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
        WriteTensorResultRange(
            context.wave, vdst, 4, lane, FloatAsU32(value), storage_policy);
      }
    } else if (instruction.mnemonic == "v_mfma_f32_32x32x2f32") {
      const uint32_t vdst = instruction.operands.at(0).kind == DecodedInstructionOperandKind::VectorRegRange
                                ? RequireVectorRange(instruction.operands.at(0)).first
                                : RequireVectorIndex(instruction.operands.at(0));
      const auto storage_policy = DefaultTensorResultStoragePolicy();
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
        WriteTensorResultRange(
            context.wave, vdst, 16, lane, FloatAsU32(value), storage_policy);
      }
    } else if (instruction.mnemonic == "v_mfma_f32_16x16x4f16") {
      const uint32_t vdst = instruction.operands.at(0).kind == DecodedInstructionOperandKind::VectorRegRange
                                ? RequireVectorRange(instruction.operands.at(0)).first
                                : RequireVectorIndex(instruction.operands.at(0));
      const auto storage_policy = DefaultTensorResultStoragePolicy();
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
        WriteTensorResultRange(
            context.wave, vdst, 4, lane, FloatAsU32(value), storage_policy);
      }
    } else if (instruction.mnemonic == "v_mfma_i32_16x16x4i8") {
      const uint32_t vdst = instruction.operands.at(0).kind == DecodedInstructionOperandKind::VectorRegRange
                                ? RequireVectorRange(instruction.operands.at(0)).first
                                : RequireVectorIndex(instruction.operands.at(0));
      const auto storage_policy = DefaultTensorResultStoragePolicy();
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
        WriteTensorResultRange(
            context.wave, vdst, 4, lane, static_cast<uint32_t>(acc), storage_policy);
      }
    } else if (instruction.mnemonic == "v_mfma_i32_16x16x16i8") {
      const uint32_t vdst = instruction.operands.at(0).kind == DecodedInstructionOperandKind::VectorRegRange
                                ? RequireVectorRange(instruction.operands.at(0)).first
                                : RequireVectorIndex(instruction.operands.at(0));
      const auto storage_policy = DefaultTensorResultStoragePolicy();
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
        WriteTensorResultRange(
            context.wave, vdst, 4, lane, static_cast<uint32_t>(acc), storage_policy);
      }
    } else if (instruction.mnemonic == "v_mfma_f32_16x16x2bf16") {
      const uint32_t vdst = instruction.operands.at(0).kind == DecodedInstructionOperandKind::VectorRegRange
                                ? RequireVectorRange(instruction.operands.at(0)).first
                                : RequireVectorIndex(instruction.operands.at(0));
      const auto storage_policy = DefaultTensorResultStoragePolicy();
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
        WriteTensorResultRange(
            context.wave, vdst, 4, lane, FloatAsU32(value), storage_policy);
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
    } else if (instruction.mnemonic == "v_mul_lo_i32") {
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        if (!context.wave.exec.test(lane)) {
          continue;
        }
        const int32_t lhs =
            static_cast<int32_t>(ResolveVectorLane(instruction.operands.at(1), context, lane));
        const int32_t rhs =
            static_cast<int32_t>(ResolveVectorLane(instruction.operands.at(2), context, lane));
        const int64_t product = static_cast<int64_t>(lhs) * static_cast<int64_t>(rhs);
        context.wave.vgpr.Write(vdst, lane, static_cast<uint32_t>(product & 0xffffffffu));
      }
    } else if (instruction.mnemonic == "v_mul_hi_u32") {
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        if (!context.wave.exec.test(lane)) {
          continue;
        }
        const uint64_t lhs =
            static_cast<uint32_t>(ResolveVectorLane(instruction.operands.at(1), context, lane));
        const uint64_t rhs =
            static_cast<uint32_t>(ResolveVectorLane(instruction.operands.at(2), context, lane));
        context.wave.vgpr.Write(vdst, lane, static_cast<uint32_t>((lhs * rhs) >> 32u));
      }
    } else {
      throw std::invalid_argument("unsupported vector alu opcode: " + instruction.mnemonic);
    }
    context.wave.pc += instruction.size_bytes;
  }
};

class VectorCompareHandler final : public IEncodedSemanticHandler {
 public:
  void Execute(const DecodedInstruction& instruction, EncodedWaveContext& context) const override {
    uint64_t mask = 0;
    if (instruction.mnemonic == "v_cmp_lt_u32_e32") {
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        if (!context.wave.exec.test(lane)) {
          continue;
        }
        const uint32_t lhs =
            static_cast<uint32_t>(ResolveVectorLane(instruction.operands.at(1), context, lane));
        const uint32_t rhs =
            static_cast<uint32_t>(ResolveVectorLane(instruction.operands.at(2), context, lane));
        if (lhs < rhs) {
          mask |= (1ull << lane);
        }
      }
      context.vcc = mask;
      context.wave.pc += instruction.size_bytes;
      return;
    }
    if (instruction.mnemonic == "v_cmp_gt_u32_e64") {
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
      if (instruction.operands.at(0).kind == DecodedInstructionOperandKind::ScalarRegRange ||
          instruction.operands.at(0).kind == DecodedInstructionOperandKind::SpecialReg) {
        StoreScalarPair(instruction.operands.at(0), context, mask);
      }
      context.wave.pc += instruction.size_bytes;
      return;
    }
    if (instruction.mnemonic == "v_cmp_lt_u32_e64") {
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        if (!context.wave.exec.test(lane)) {
          continue;
        }
        const uint32_t lhs =
            static_cast<uint32_t>(ResolveVectorLane(instruction.operands.at(1), context, lane));
        const uint32_t rhs =
            static_cast<uint32_t>(ResolveVectorLane(instruction.operands.at(2), context, lane));
        if (lhs < rhs) {
          mask |= (1ull << lane);
        }
      }
      if (instruction.operands.at(0).kind == DecodedInstructionOperandKind::ScalarRegRange ||
          instruction.operands.at(0).kind == DecodedInstructionOperandKind::SpecialReg) {
        StoreScalarPair(instruction.operands.at(0), context, mask);
      }
      context.wave.pc += instruction.size_bytes;
      return;
    }
    if (instruction.mnemonic == "v_cmp_le_u32_e32") {
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        if (!context.wave.exec.test(lane)) {
          continue;
        }
        const uint32_t lhs =
            static_cast<uint32_t>(ResolveVectorLane(instruction.operands.at(1), context, lane));
        const uint32_t rhs =
            static_cast<uint32_t>(ResolveVectorLane(instruction.operands.at(2), context, lane));
        if (lhs <= rhs) {
          mask |= (1ull << lane);
        }
      }
      context.vcc = mask;
      context.wave.pc += instruction.size_bytes;
      return;
    }
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
      case 9:    // v_cmp_lt_u32_e32
      case 56:
      case 204: {  // v_cmp_gt_u32_e32 / v_cmp_lt_u32_e32 / v_cmp_gt_u32_e64
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        if (!context.wave.exec.test(lane)) {
          continue;
        }
        const uint32_t lhs =
            static_cast<uint32_t>(ResolveVectorLane(instruction.operands.at(1), context, lane));
        const uint32_t rhs =
            static_cast<uint32_t>(ResolveVectorLane(instruction.operands.at(2), context, lane));
        const bool is_lt_u32 =
            instruction.mnemonic == "v_cmp_lt_u32_e32" || instruction.encoding_id == 9;
        const bool match = is_lt_u32 ? (lhs < rhs) : (lhs > rhs);
        if (match) {
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
      case 203: {  // v_cmp_le_u32_e32
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        if (!context.wave.exec.test(lane)) {
          continue;
        }
        const uint32_t lhs =
            static_cast<uint32_t>(ResolveVectorLane(instruction.operands.at(1), context, lane));
        const uint32_t rhs =
            static_cast<uint32_t>(ResolveVectorLane(instruction.operands.at(2), context, lane));
        if (lhs <= rhs) {
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
    if (instruction.operands.at(0).kind == DecodedInstructionOperandKind::SpecialReg &&
        instruction.operands.at(0).info.special_reg == GcnSpecialReg::Vcc) {
      context.vcc = mask;
    }
    if (instruction.operands.at(0).kind == DecodedInstructionOperandKind::ScalarRegRange ||
        instruction.operands.at(0).kind == DecodedInstructionOperandKind::SpecialReg) {
      StoreScalarPair(instruction.operands.at(0), context, mask);
    }
    DebugLog("pc=0x%llx %s mask=0x%llx",
             static_cast<unsigned long long>(instruction.pc),
             instruction.mnemonic.c_str(),
             static_cast<unsigned long long>(mask));
    context.wave.pc += instruction.size_bytes;
  }
};

class MaskHandler final : public IEncodedSemanticHandler {
 public:
  void Execute(const DecodedInstruction& instruction, EncodedWaveContext& context) const override {
    if (instruction.mnemonic != "s_and_saveexec_b64" &&
        instruction.mnemonic != "s_andn2_saveexec_b64") {
      throw std::invalid_argument("unsupported mask opcode: " + instruction.mnemonic);
    }
    const auto [sdst, _] = RequireScalarRange(instruction.operands.at(0));
    const uint64_t exec_before = context.wave.exec.to_ullong();
    const uint64_t mask = ResolveScalarPair(instruction.operands.at(1), context);
    context.wave.sgpr.Write(sdst, static_cast<uint32_t>(exec_before & 0xffffffffu));
    context.wave.sgpr.Write(sdst + 1, static_cast<uint32_t>(exec_before >> 32u));
    if (instruction.mnemonic == "s_and_saveexec_b64") {
      context.wave.exec = context.wave.exec & MaskFromU64(mask);
    } else {
      context.wave.exec = MaskFromU64(mask & ~exec_before);
    }
    DebugLog("pc=0x%llx %s exec=0x%llx",
             static_cast<unsigned long long>(instruction.pc),
             instruction.mnemonic.c_str(),
             static_cast<unsigned long long>(context.wave.exec.to_ullong()));
    context.wave.pc += instruction.size_bytes;
  }
};

class BranchHandler final : public IEncodedSemanticHandler {
 public:
  void Execute(const DecodedInstruction& instruction, EncodedWaveContext& context) const override {
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

class FlatMemoryHandler final : public IEncodedSemanticHandler {
 public:
  void Execute(const DecodedInstruction& instruction, EncodedWaveContext& context) const override {
    MemoryRequest request;
    request.space = MemorySpace::Global;
    if (instruction.mnemonic == "global_load_dword") {
      request.kind = AccessKind::Load;
      const int64_t offset = instruction.operands.size() >= 3 && instruction.operands.back().info.has_immediate
                                 ? instruction.operands.back().info.immediate
                                 : 0;
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      const auto [addr, _] = RequireVectorRange(instruction.operands.at(1));
      request.dst = RegRef{.file = RegisterFile::Vector, .index = vdst};
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        if (!context.wave.exec.test(lane)) {
          continue;
        }
        const uint64_t lo = static_cast<uint32_t>(context.wave.vgpr.Read(addr, lane));
        const uint64_t hi = static_cast<uint32_t>(context.wave.vgpr.Read(addr + 1, lane));
        const uint64_t address = static_cast<uint64_t>(static_cast<int64_t>((hi << 32u) | lo) + offset);
        request.exec_snapshot.set(lane);
        request.lanes[lane] = LaneAccess{.active = true, .addr = address, .bytes = 4, .value = 0};
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
      request.kind = AccessKind::Store;
      const int64_t offset = instruction.operands.size() >= 4 && instruction.operands.back().info.has_immediate
                                 ? instruction.operands.back().info.immediate
                                 : 0;
      if (instruction.operands.at(0).kind == DecodedInstructionOperandKind::VectorRegRange) {
        const auto [addr, _] = RequireVectorRange(instruction.operands.at(0));
        const uint32_t data = RequireVectorIndex(instruction.operands.at(1));
        for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
          if (!context.wave.exec.test(lane)) {
            continue;
          }
          const uint64_t lo = static_cast<uint32_t>(context.wave.vgpr.Read(addr, lane));
          const uint64_t hi = static_cast<uint32_t>(context.wave.vgpr.Read(addr + 1, lane));
          const uint64_t address = static_cast<uint64_t>(static_cast<int64_t>((hi << 32u) | lo) + offset);
          request.exec_snapshot.set(lane);
          request.lanes[lane] = LaneAccess{
              .active = true,
              .addr = address,
              .bytes = 4,
              .value = static_cast<uint32_t>(context.wave.vgpr.Read(data, lane)),
          };
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
          request.exec_snapshot.set(lane);
          request.lanes[lane] = LaneAccess{
              .active = true,
              .addr = address,
              .bytes = 4,
              .value = static_cast<uint32_t>(context.wave.vgpr.Read(data, lane)),
          };
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
      request.kind = AccessKind::Atomic;
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      const uint32_t data = RequireVectorIndex(instruction.operands.at(1));
      const uint32_t saddr = RequireScalarRange(instruction.operands.at(2)).first;
      const uint64_t base = static_cast<uint64_t>(context.wave.sgpr.Read(saddr)) |
                            (static_cast<uint64_t>(context.wave.sgpr.Read(saddr + 1)) << 32u);
      request.dst = RegRef{.file = RegisterFile::Vector, .index = vdst};
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        if (!context.wave.exec.test(lane)) {
          continue;
        }
        request.exec_snapshot.set(lane);
        request.lanes[lane] = LaneAccess{
            .active = true,
            .addr = base,
            .bytes = 4,
            .value = static_cast<uint32_t>(context.wave.vgpr.Read(data, lane)),
        };
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
    if (context.captured_memory_request != nullptr) {
      *context.captured_memory_request = request;
    }
    context.wave.pc += instruction.size_bytes;
  }
};

class SharedMemoryHandler final : public IEncodedSemanticHandler {
 public:
  void Execute(const DecodedInstruction& instruction, EncodedWaveContext& context) const override {
    MemoryRequest request;
    request.space = MemorySpace::Shared;
    if (instruction.mnemonic == "ds_write_b32") {
      request.kind = AccessKind::Store;
      const uint32_t addr_vgpr = RequireVectorIndex(instruction.operands.at(0));
      const uint32_t data_vgpr = RequireVectorIndex(instruction.operands.at(1));
      const uint32_t offset = instruction.operands.size() >= 3 && instruction.operands.back().info.has_immediate
                                  ? static_cast<uint32_t>(instruction.operands.back().info.immediate)
                                  : 0u;
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        uint32_t byte_offset = static_cast<uint32_t>(context.wave.vgpr.Read(addr_vgpr, lane));
        if (!context.wave.exec.test(lane) || context.block.shared_memory.empty()) {
          continue;
        }
        byte_offset += offset;
        byte_offset %= static_cast<uint32_t>(context.block.shared_memory.size());
        if (static_cast<size_t>(byte_offset) + sizeof(uint32_t) > context.block.shared_memory.size()) {
          continue;
        }
        const uint32_t value =
            static_cast<uint32_t>(context.wave.vgpr.Read(data_vgpr, lane));
        request.exec_snapshot.set(lane);
        request.lanes[lane] = LaneAccess{
            .active = true,
            .addr = byte_offset,
            .bytes = 4,
            .value = value,
        };
        StoreU32(context.block.shared_memory, byte_offset, value);
        if (lane == 0) {
          DebugLog("pc=0x%llx ds_write_b32 addr=0x%x value=0x%x",
                   static_cast<unsigned long long>(instruction.pc), byte_offset, value);
        }
      }
      ++context.stats.shared_stores;
      if (context.captured_memory_request != nullptr) {
        *context.captured_memory_request = request;
      }
      context.wave.pc += instruction.size_bytes;
      return;
    }
    if (instruction.mnemonic == "ds_read_b32") {
      request.kind = AccessKind::Load;
      const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
      const uint32_t addr_vgpr = RequireVectorIndex(instruction.operands.at(1));
      const uint32_t offset = instruction.operands.size() >= 3 && instruction.operands.back().info.has_immediate
                                  ? static_cast<uint32_t>(instruction.operands.back().info.immediate)
                                  : 0u;
      request.dst = RegRef{.file = RegisterFile::Vector, .index = vdst};
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        uint32_t byte_offset = static_cast<uint32_t>(context.wave.vgpr.Read(addr_vgpr, lane));
        if (!context.wave.exec.test(lane) || context.block.shared_memory.empty()) {
          continue;
        }
        byte_offset += offset;
        byte_offset %= static_cast<uint32_t>(context.block.shared_memory.size());
        if (static_cast<size_t>(byte_offset) + sizeof(uint32_t) > context.block.shared_memory.size()) {
          continue;
        }
        request.exec_snapshot.set(lane);
        request.lanes[lane] = LaneAccess{.active = true, .addr = byte_offset, .bytes = 4, .value = 0};
        const uint32_t value = LoadU32(context.block.shared_memory, byte_offset);
        context.wave.vgpr.Write(vdst, lane, value);
        if (lane == 0) {
          DebugLog("pc=0x%llx ds_read_b32 addr=0x%x value=0x%x",
                   static_cast<unsigned long long>(instruction.pc), byte_offset, value);
        }
      }
      ++context.stats.shared_loads;
      if (context.captured_memory_request != nullptr) {
        *context.captured_memory_request = request;
      }
      context.wave.pc += instruction.size_bytes;
      return;
    }
    if (instruction.mnemonic == "ds_read2_b32") {
      const auto [vdst, _] = RequireVectorRange(instruction.operands.at(0));
      const uint32_t addr_vgpr = RequireVectorIndex(instruction.operands.at(1));
      const uint32_t offset0 =
          static_cast<uint32_t>(ResolveScalarLike(instruction.operands.at(2), context));
      const uint32_t offset1 =
          static_cast<uint32_t>(ResolveScalarLike(instruction.operands.at(3), context));
      for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
        const uint32_t base_byte_offset =
            static_cast<uint32_t>(context.wave.vgpr.Read(addr_vgpr, lane));
        if (!context.wave.exec.test(lane)) {
          continue;
        }
        const uint32_t byte_offset0 = base_byte_offset + offset0 * sizeof(uint32_t);
        const uint32_t byte_offset1 = base_byte_offset + offset1 * sizeof(uint32_t);
        if (static_cast<size_t>(byte_offset0) + sizeof(uint32_t) > context.block.shared_memory.size() ||
            static_cast<size_t>(byte_offset1) + sizeof(uint32_t) > context.block.shared_memory.size()) {
          continue;
        }
        context.wave.vgpr.Write(vdst, lane, LoadU32(context.block.shared_memory, byte_offset0));
        context.wave.vgpr.Write(vdst + 1, lane, LoadU32(context.block.shared_memory, byte_offset1));
      }
      ++context.stats.shared_loads;
      context.wave.pc += instruction.size_bytes;
      return;
    }
    throw std::invalid_argument("unsupported shared memory opcode: " + instruction.mnemonic);
  }
};

class SpecialHandler final : public IEncodedSemanticHandler {
 public:
  void Execute(const DecodedInstruction& instruction, EncodedWaveContext& context) const override {
    switch (instruction.encoding_id) {
      case 29: {  // s_barrier
      ++context.stats.barriers;
      sync_ops::MarkWaveAtBarrier(context.wave,
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
  const IEncodedSemanticHandler* handler = nullptr;
};

const IEncodedSemanticHandler* HandlerForSemanticFamily(std::string_view semantic_family,
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
  static const ScalarAluHandler kScalarAluHandler;
  static const ScalarCompareHandler kScalarCompareHandler;
  static const SharedMemoryHandler kSharedMemoryHandler;
  static const VectorAluHandler kVectorAluHandler;
  static const VectorCompareHandler kVectorCompareHandler;
  static const std::vector<HandlerBinding> kBindings = {
      {.mnemonic = "s_and_saveexec_b64", .handler = &kMaskHandler},
      {.mnemonic = "s_andn2_saveexec_b64", .handler = &kMaskHandler},
      {.mnemonic = "s_abs_i32", .handler = &kScalarAluHandler},
      {.mnemonic = "s_sub_i32", .handler = &kScalarAluHandler},
      {.mnemonic = "s_or_b32", .handler = &kScalarAluHandler},
      {.mnemonic = "s_xor_b64", .handler = &kScalarAluHandler},
      {.mnemonic = "s_cmp_gt_i32", .handler = &kScalarCompareHandler},
      {.mnemonic = "ds_read2_b32", .handler = &kSharedMemoryHandler},
      {.mnemonic = "v_and_b32_e32", .handler = &kVectorAluHandler},
      {.mnemonic = "v_xor_b32_e32", .handler = &kVectorAluHandler},
      {.mnemonic = "v_subrev_u32_e32", .handler = &kVectorAluHandler},
      {.mnemonic = "v_cvt_f32_u32_e32", .handler = &kVectorAluHandler},
      {.mnemonic = "v_cvt_u32_f32_e32", .handler = &kVectorAluHandler},
      {.mnemonic = "v_rcp_iflag_f32_e32", .handler = &kVectorAluHandler},
      {.mnemonic = "v_or_b32_e32", .handler = &kVectorAluHandler},
      {.mnemonic = "v_max_i32_e32", .handler = &kVectorAluHandler},
      {.mnemonic = "v_sub_u32_e32", .handler = &kVectorAluHandler},
      {.mnemonic = "v_mul_lo_i32", .handler = &kVectorAluHandler},
      {.mnemonic = "v_mul_hi_u32", .handler = &kVectorAluHandler},
      {.mnemonic = "v_or3_b32", .handler = &kVectorAluHandler},
      {.mnemonic = "v_cmp_lt_u32_e32", .handler = &kVectorCompareHandler},
      {.mnemonic = "v_cmp_le_u32_e32", .handler = &kVectorCompareHandler},
      {.mnemonic = "v_cmp_gt_i32_e64", .handler = &kVectorCompareHandler},
      {.mnemonic = "v_cmp_lt_u32_e64", .handler = &kVectorCompareHandler},
      {.mnemonic = "v_cmp_gt_u32_e64", .handler = &kVectorCompareHandler},
  };
  return kBindings;
}

}  // namespace

const IEncodedSemanticHandler& EncodedSemanticHandlerRegistry::Get(std::string_view mnemonic) {
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

const IEncodedSemanticHandler& EncodedSemanticHandlerRegistry::Get(
    const DecodedInstruction& instruction) {
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
