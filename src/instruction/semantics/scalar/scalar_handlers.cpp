#include "instruction/semantics/internal/handler_support.h"

namespace gpu_model {
namespace semantics {

using handler_support::BaseHandler;
using handler_support::HandlerRegistry;
using handler_support::RequireCanonicalOpcode;
using handler_support::ReverseBits32;
using handler_support::ResolveScalarLike;
using handler_support::StoreScalarPair;
using handler_support::ThrowUnsupportedInstruction;

// ============================================================================
// ScalarMemoryHandler — handles s_load_dword, s_load_dwordx2, s_load_dwordx4
// ============================================================================

class ScalarMemoryHandler final : public BaseHandler {
 protected:
  void ExecuteImpl(const DecodedInstruction& instruction, EncodedWaveContext& context) const override {
    MemoryRequest request;
    request.space = MemorySpace::Constant;
    request.kind = AccessKind::Load;
    request.exec_snapshot.set(0);
    if (instruction.mnemonic == "s_load_dword") {
      const uint32_t offset =
          static_cast<uint32_t>(ResolveScalarLike(instruction.operands.at(2), context));
      const uint32_t sdst = RequireScalarIndex(instruction.operands.at(0));
      const uint64_t base = ResolveScalarPair(instruction.operands.at(1), context.wave, context.vcc);
      const uint32_t value = context.memory.LoadGlobalValue<uint32_t>(base + offset);
      request.lanes[0] = LaneAccess{
          .active = true,
          .addr = base + offset,
          .bytes = 4,
          .value = value,
          .has_read_value = true,
          .read_value = value,
      };
      request.dst = RegRef{.file = RegisterFile::Scalar, .index = sdst};
      context.wave.sgpr.Write(sdst, value);
    } else if (instruction.mnemonic == "s_load_dwordx2") {
      const uint32_t offset =
          static_cast<uint32_t>(ResolveScalarLike(instruction.operands.at(2), context));
      const auto [sdst, _] = RequireScalarRange(instruction.operands.at(0));
      const uint64_t base = ResolveScalarPair(instruction.operands.at(1), context.wave, context.vcc);
      const uint64_t value = context.memory.LoadGlobalValue<uint64_t>(base + offset);
      request.lanes[0] = LaneAccess{
          .active = true,
          .addr = base + offset,
          .bytes = 8,
          .value = value,
          .has_read_value = true,
          .read_value = value,
      };
      request.dst = RegRef{.file = RegisterFile::Scalar, .index = sdst};
      context.wave.sgpr.Write(sdst, static_cast<uint32_t>(value & 0xffffffffu));
      context.wave.sgpr.Write(sdst + 1, static_cast<uint32_t>(value >> 32u));
    } else if (instruction.mnemonic == "s_load_dwordx4") {
      const uint32_t offset =
          static_cast<uint32_t>(ResolveScalarLike(instruction.operands.at(2), context));
      const auto [sdst, _] = RequireScalarRange(instruction.operands.at(0));
      const uint64_t base = ResolveScalarPair(instruction.operands.at(1), context.wave, context.vcc);
      request.lanes[0] = LaneAccess{.active = true, .addr = base + offset, .bytes = 16, .value = 0};
      request.dst = RegRef{.file = RegisterFile::Scalar, .index = sdst};
      context.wave.sgpr.Write(sdst + 0, context.memory.LoadGlobalValue<uint32_t>(base + offset + 0));
      context.wave.sgpr.Write(sdst + 1, context.memory.LoadGlobalValue<uint32_t>(base + offset + 4));
      context.wave.sgpr.Write(sdst + 2, context.memory.LoadGlobalValue<uint32_t>(base + offset + 8));
      context.wave.sgpr.Write(sdst + 3, context.memory.LoadGlobalValue<uint32_t>(base + offset + 12));
    } else {
      ThrowUnsupportedInstruction("unsupported scalar memory opcode: ", instruction);
    }
    if (context.captured_memory_request != nullptr) {
      *context.captured_memory_request = request;
    }
  }
};

// ============================================================================
// ScalarAluHandler — handles scalar ALU operations
// ============================================================================

class ScalarAluHandler final : public BaseHandler {
 protected:
  void ExecuteImpl(const DecodedInstruction& instruction, EncodedWaveContext& context) const override {
    const auto& descriptor = RequireCanonicalOpcode(instruction);
    if (descriptor.op_type == GcnIsaOpType::Sop2 &&
        descriptor.opcode == static_cast<uint16_t>(GcnIsaSop2Opcode::S_CSELECT_B64)) {
      const bool take_true = context.wave.ScalarMaskBit0();
      const uint64_t value = take_true ? ResolveScalarPair(instruction.operands.at(1), context.wave, context.vcc)
                                       : ResolveScalarPair(instruction.operands.at(2), context.wave, context.vcc);
      StoreScalarPair(instruction.operands.at(0), context, value);
    } else if (descriptor.op_type == GcnIsaOpType::Sop2 &&
               descriptor.opcode == static_cast<uint16_t>(GcnIsaSop2Opcode::S_ANDN2_B64)) {
      const uint64_t lhs = ResolveScalarPair(instruction.operands.at(1), context.wave, context.vcc);
      const uint64_t rhs = ResolveScalarPair(instruction.operands.at(2), context.wave, context.vcc);
      StoreScalarPair(instruction.operands.at(0), context, lhs & ~rhs);
    } else if (descriptor.op_type == GcnIsaOpType::Sop1 &&
               descriptor.opcode == static_cast<uint16_t>(GcnIsaSop1Opcode::S_MOV_B32)) {
      const uint32_t sdst = RequireScalarIndex(instruction.operands.at(0));
      const uint32_t value =
          static_cast<uint32_t>(ResolveScalarLike(instruction.operands.at(1), context));
      context.wave.sgpr.Write(sdst, value);
    } else if (descriptor.op_type == GcnIsaOpType::Sop1 &&
               descriptor.opcode == static_cast<uint16_t>(GcnIsaSop1Opcode::S_MOV_B64)) {
      const uint64_t value = ResolveScalarPair(instruction.operands.at(1), context.wave, context.vcc);
      StoreScalarPair(instruction.operands.at(0), context, value);
    } else if (descriptor.op_type == GcnIsaOpType::Sop1 &&
               descriptor.opcode == static_cast<uint16_t>(GcnIsaSop1Opcode::S_ABS_I32)) {
      const uint32_t sdst = RequireScalarIndex(instruction.operands.at(0));
      const int32_t value =
          static_cast<int32_t>(ResolveScalarLike(instruction.operands.at(1), context));
      context.wave.sgpr.Write(sdst, static_cast<uint32_t>(value < 0 ? -value : value));
    } else if (descriptor.op_type == GcnIsaOpType::Sop1 &&
               descriptor.opcode == static_cast<uint16_t>(GcnIsaSop1Opcode::S_BREV_B32)) {
      const uint32_t sdst = RequireScalarIndex(instruction.operands.at(0));
      const uint32_t value =
          static_cast<uint32_t>(ResolveScalarLike(instruction.operands.at(1), context));
      context.wave.sgpr.Write(sdst, ReverseBits32(value));
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
    } else if (descriptor.op_type == GcnIsaOpType::Sop1 &&
               descriptor.opcode == static_cast<uint16_t>(GcnIsaSop1Opcode::S_FF1_I32_B64)) {
      const uint32_t sdst = RequireScalarIndex(instruction.operands.at(0));
      const uint64_t value = ResolveScalarPair(instruction.operands.at(1), context.wave, context.vcc);
      const int32_t result =
          value == 0 ? -1 : static_cast<int32_t>(std::countr_zero(value));
      context.wave.sgpr.Write(sdst, static_cast<uint32_t>(result));
    } else if (descriptor.op_type == GcnIsaOpType::Sop2 &&
               descriptor.opcode == static_cast<uint16_t>(GcnIsaSop2Opcode::S_OR_B64)) {
      const uint64_t lhs = ResolveScalarPair(instruction.operands.at(1), context.wave, context.vcc);
      const uint64_t rhs = ResolveScalarPair(instruction.operands.at(2), context.wave, context.vcc);
      const uint64_t value = lhs | rhs;
      StoreScalarPair(instruction.operands.at(0), context, value);
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
      const uint64_t lhs = ResolveScalarPair(instruction.operands.at(1), context.wave, context.vcc);
      const uint64_t rhs = ResolveScalarPair(instruction.operands.at(2), context.wave, context.vcc);
      StoreScalarPair(instruction.operands.at(0), context, lhs ^ rhs);
    } else if (descriptor.op_type == GcnIsaOpType::Sop2 &&
               descriptor.opcode == static_cast<uint16_t>(GcnIsaSop2Opcode::S_AND_B64)) {
      const uint64_t lhs = ResolveScalarPair(instruction.operands.at(1), context.wave, context.vcc);
      const uint64_t rhs = ResolveScalarPair(instruction.operands.at(2), context.wave, context.vcc);
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
               descriptor.opcode == static_cast<uint16_t>(GcnIsaSop2Opcode::S_XOR_B32)) {
      const uint32_t sdst = RequireScalarIndex(instruction.operands.at(0));
      const uint32_t lhs =
          static_cast<uint32_t>(ResolveScalarLike(instruction.operands.at(1), context));
      const uint32_t rhs =
          static_cast<uint32_t>(ResolveScalarLike(instruction.operands.at(2), context));
      context.wave.sgpr.Write(sdst, lhs ^ rhs);
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
               descriptor.opcode == static_cast<uint16_t>(GcnIsaSop2Opcode::S_MAX_I32)) {
      const uint32_t sdst = RequireScalarIndex(instruction.operands.at(0));
      const int32_t lhs =
          static_cast<int32_t>(ResolveScalarLike(instruction.operands.at(1), context));
      const int32_t rhs =
          static_cast<int32_t>(ResolveScalarLike(instruction.operands.at(2), context));
      context.wave.sgpr.Write(sdst, static_cast<uint32_t>(std::max(lhs, rhs)));
    } else if (descriptor.op_type == GcnIsaOpType::Sop2 &&
               descriptor.opcode == static_cast<uint16_t>(GcnIsaSop2Opcode::S_MIN_I32)) {
      const uint32_t sdst = RequireScalarIndex(instruction.operands.at(0));
      const int32_t lhs =
          static_cast<int32_t>(ResolveScalarLike(instruction.operands.at(1), context));
      const int32_t rhs =
          static_cast<int32_t>(ResolveScalarLike(instruction.operands.at(2), context));
      context.wave.sgpr.Write(sdst, static_cast<uint32_t>(std::min(lhs, rhs)));
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
      const uint64_t lhs = ResolveScalarPair(instruction.operands.at(1), context.wave, context.vcc);
      const uint32_t rhs =
          static_cast<uint32_t>(ResolveScalarLike(instruction.operands.at(2), context));
      StoreScalarPair(instruction.operands.at(0), context, lhs << (rhs & 63u));
    } else if (descriptor.op_type == GcnIsaOpType::Sop2 &&
               descriptor.opcode == static_cast<uint16_t>(GcnIsaSop2Opcode::S_LSHL_B32)) {
      const uint32_t sdst = RequireScalarIndex(instruction.operands.at(0));
      const uint32_t lhs =
          static_cast<uint32_t>(ResolveScalarLike(instruction.operands.at(1), context));
      const uint32_t rhs =
          static_cast<uint32_t>(ResolveScalarLike(instruction.operands.at(2), context));
      context.wave.sgpr.Write(sdst, lhs << (rhs & 31u));
    } else if (descriptor.op_type == GcnIsaOpType::Sopk &&
               descriptor.opcode == static_cast<uint16_t>(GcnIsaSopkOpcode::S_MOVK_I32)) {
      const uint32_t sdst = RequireScalarIndex(instruction.operands.at(0));
      const int32_t value =
          static_cast<int32_t>(ResolveScalarLike(instruction.operands.at(1), context));
      context.wave.sgpr.Write(sdst, static_cast<uint32_t>(value));
    } else if (descriptor.op_type == GcnIsaOpType::Sopk &&
               descriptor.opcode == static_cast<uint16_t>(GcnIsaSopkOpcode::S_ADDK_I32)) {
      const uint32_t sdst = RequireScalarIndex(instruction.operands.at(0));
      const int32_t src = static_cast<int32_t>(context.wave.sgpr.Read(sdst));
      const int32_t imm = static_cast<int32_t>(ResolveScalarLike(instruction.operands.at(1), context));
      context.wave.sgpr.Write(sdst, static_cast<uint32_t>(src + imm));
    } else if (descriptor.op_type == GcnIsaOpType::Sopk &&
               descriptor.opcode == static_cast<uint16_t>(GcnIsaSopkOpcode::S_MULK_I32)) {
      const uint32_t sdst = RequireScalarIndex(instruction.operands.at(0));
      const int32_t src = static_cast<int32_t>(context.wave.sgpr.Read(sdst));
      const int32_t imm = static_cast<int32_t>(ResolveScalarLike(instruction.operands.at(1), context));
      context.wave.sgpr.Write(sdst, static_cast<uint32_t>(src * imm));
    } else if (descriptor.op_type == GcnIsaOpType::Sop1 &&
               descriptor.opcode == static_cast<uint16_t>(GcnIsaSop1Opcode::S_BCNT1_I32_B64)) {
      const uint32_t sdst = RequireScalarIndex(instruction.operands.at(0));
      const uint64_t value = ResolveScalarPair(instruction.operands.at(1), context.wave, context.vcc);
      context.wave.sgpr.Write(sdst, static_cast<uint32_t>(std::popcount(value)));
    } else {
      ThrowUnsupportedInstruction("unsupported scalar alu opcode: ", instruction);
    }
  }
};

// ============================================================================
// ScalarCompareHandler — handles scalar comparison operations
// ============================================================================

class ScalarCompareHandler final : public BaseHandler {
 protected:
  void ExecuteImpl(const DecodedInstruction& instruction, EncodedWaveContext& context) const override {
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
               descriptor.opcode == static_cast<uint16_t>(GcnIsaSopcOpcode::S_CMP_LG_U32)) {
      const uint32_t lhs =
          static_cast<uint32_t>(ResolveScalarLike(instruction.operands.at(0), context));
      const uint32_t rhs =
          static_cast<uint32_t>(ResolveScalarLike(instruction.operands.at(1), context));
      context.wave.SetScalarMaskBit0(lhs != rhs);
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
    } else if (descriptor.op_type == GcnIsaOpType::Sopc &&
               descriptor.opcode == static_cast<uint16_t>(GcnIsaSopcOpcode::S_CMP_LG_U64)) {
      const uint64_t lhs = ResolveScalarPair(instruction.operands.at(0), context.wave, context.vcc);
      const uint64_t rhs = ResolveScalarPair(instruction.operands.at(1), context.wave, context.vcc);
      context.wave.SetScalarMaskBit0(lhs != rhs);
    } else {
      ThrowUnsupportedInstruction("unsupported scalar compare opcode: ", instruction);
    }
  }
};

// ============================================================================
// MaskHandler — handles exec mask operations (s_and_saveexec_b64, s_andn2_saveexec_b64)
// ============================================================================

class MaskHandler final : public BaseHandler {
 protected:
  void ExecuteImpl(const DecodedInstruction& instruction, EncodedWaveContext& context) const override {
    if (instruction.mnemonic != "s_and_saveexec_b64" &&
        instruction.mnemonic != "s_andn2_saveexec_b64") {
      ThrowUnsupportedInstruction("unsupported mask opcode: ", instruction);
    }
    const auto [sdst, _] = RequireScalarRange(instruction.operands.at(0));
    const uint64_t exec_before = context.wave.exec.to_ullong();
    const uint64_t mask = ResolveScalarPair(instruction.operands.at(1), context.wave, context.vcc);
    context.wave.sgpr.Write(sdst, static_cast<uint32_t>(exec_before & 0xffffffffu));
    context.wave.sgpr.Write(sdst + 1, static_cast<uint32_t>(exec_before >> 32u));
    if (instruction.mnemonic == "s_and_saveexec_b64") {
      context.wave.exec = context.wave.exec & MaskFromU64(mask);
    } else {
      context.wave.exec = MaskFromU64(mask & ~exec_before);
    }
  }
};

// ============================================================================
// Static instances
// ============================================================================

static const ScalarMemoryHandler kScalarMemoryHandler;
static const ScalarAluHandler kScalarAluHandler;
static const ScalarCompareHandler kScalarCompareHandler;
static const MaskHandler kMaskHandler;

// ============================================================================
// Accessors
// ============================================================================

const IEncodedSemanticHandler& GetScalarMemoryHandler() { return kScalarMemoryHandler; }
const IEncodedSemanticHandler& GetScalarAluHandler() { return kScalarAluHandler; }
const IEncodedSemanticHandler& GetScalarCompareHandler() { return kScalarCompareHandler; }
const IEncodedSemanticHandler& GetMaskHandler() { return kMaskHandler; }

// ============================================================================
// Self-registration for mnemonic-based lookup
// ============================================================================

struct ScalarHandlerRegistrar {
  ScalarHandlerRegistrar() {
    auto& registry = HandlerRegistry::MutableInstance();
    registry.Register("s_and_saveexec_b64", &kMaskHandler);
    registry.Register("s_andn2_saveexec_b64", &kMaskHandler);
    registry.Register("s_abs_i32", &kScalarAluHandler);
    registry.Register("s_brev_b32", &kScalarAluHandler);
    registry.Register("s_ff1_i32_b64", &kScalarAluHandler);
    registry.Register("s_max_i32", &kScalarAluHandler);
    registry.Register("s_min_i32", &kScalarAluHandler);
    registry.Register("s_sub_i32", &kScalarAluHandler);
    registry.Register("s_lshl_b32", &kScalarAluHandler);
    registry.Register("s_or_b32", &kScalarAluHandler);
    registry.Register("s_xor_b32", &kScalarAluHandler);
    registry.Register("s_xor_b64", &kScalarAluHandler);
    registry.Register("s_movk_i32", &kScalarAluHandler);
    registry.Register("s_cmp_gt_i32", &kScalarCompareHandler);
    registry.Register("s_cmp_lg_u32", &kScalarCompareHandler);
    registry.Register("s_cmp_lg_u64", &kScalarCompareHandler);
  }
};
static ScalarHandlerRegistrar s_scalar_registrar;

}  // namespace semantics
}  // namespace gpu_model
