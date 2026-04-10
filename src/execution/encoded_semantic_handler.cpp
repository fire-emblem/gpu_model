#include "gpu_model/execution/encoded_semantic_handler.h"

#include <bit>
#include <bitset>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "gpu_model/util/logging.h"
#include "gpu_model/instruction/encoded/internal/encoded_gcn_encoding_def.h"
#include "gpu_model/instruction/encoded/internal/encoded_gcn_db_lookup.h"
#include "gpu_model/execution/sync_ops.h"
#include "gpu_model/execution/internal/tensor_op_utils.h"
#include "gpu_model/execution/internal/float_utils.h"
#include "gpu_model/execution/internal/encoded_handler_utils.h"

namespace gpu_model {

namespace {

// Note: LaneCount, MaskFromU64, LoadU32, StoreU32, DebugEnabled/EncodedDebugLog,
// RequireScalarIndex, RequireVectorIndex, RequireAccumulatorIndex,
// RequireScalarRange, ResolveScalarPair are now provided by
// gpu_model/execution/internal/encoded_handler_utils.h

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
  if (const auto* match = FindEncodedGcnMatchRecord(instruction.words); match != nullptr &&
      match->opcode_descriptor != nullptr) {
    return *match->opcode_descriptor;
  }
  if (const auto* descriptor = FindGcnIsaOpcodeDescriptorByName(instruction.mnemonic);
      descriptor != nullptr) {
    return *descriptor;
  }
  throw std::invalid_argument("missing canonical opcode descriptor: " + instruction.mnemonic);
}

// Base handler class with trace support and unified PC update
// This eliminates the repetitive "context.wave.pc += instruction.size_bytes;" pattern
// and provides centralized trace callback invocation.
class BaseHandler : public IEncodedSemanticHandler {
 public:
  void Execute(const DecodedInstruction& instruction, EncodedWaveContext& context) const override {
    // Debug log for instruction execution
    EncodedDebugLog("Execute: pc=0x%llx mnemonic=%s",
                    static_cast<unsigned long long>(instruction.pc),
                    instruction.mnemonic.c_str());

    // Emit trace callback for instruction start
    if (context.on_execute) {
      context.on_execute(instruction, context, "start");
    }

    // Execute the actual instruction logic
    ExecuteImpl(instruction, context);

    // Emit trace callback for instruction end
    if (context.on_execute) {
      context.on_execute(instruction, context, "end");
    }

    // Unified PC advancement - all handlers need this
    context.wave.pc += instruction.size_bytes;
  }

 protected:
  virtual void ExecuteImpl(const DecodedInstruction& instruction,
                           EncodedWaveContext& context) const = 0;
};

// Vector lane handler base class - provides lane iteration with exec mask check
// Eliminates the repetitive "for (uint32_t lane = 0; lane < LaneCount(context); ++lane)"
// and "if (!context.wave.exec.test(lane)) { continue; }" patterns.
template <typename Impl>
class VectorLaneHandler : public BaseHandler {
 protected:
  void ExecuteImpl(const DecodedInstruction& instruction,
                   EncodedWaveContext& context) const override {
    ForEachActiveLane(context, [&](uint32_t lane) {
      static_cast<const Impl*>(this)->ExecuteLane(instruction, context, lane);
    });
  }

 private:
  void ForEachActiveLane(EncodedWaveContext& context, auto&& fn) const {
    for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
      if (context.wave.exec.test(lane)) {
        fn(lane);
      }
    }
  }
};

// ============================================================================
// Individual Vector ALU Handlers using VectorLaneHandler CRTP
// These replace the monolithic VectorAluHandler for better SRP and OCP.
// ============================================================================

// ============================================================================
// Generic Handler Templates - reduce code duplication for common patterns
// ============================================================================

// Binary integer operation: dst = lhs OP rhs
template <auto Op>
class BinaryU32Handler final : public VectorLaneHandler<BinaryU32Handler<Op>> {
 public:
  void ExecuteLane(const DecodedInstruction& instruction,
                   EncodedWaveContext& context, uint32_t lane) const {
    const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
    const uint32_t lhs = static_cast<uint32_t>(
        ResolveVectorLane(instruction.operands.at(1), context, lane));
    const uint32_t rhs = static_cast<uint32_t>(
        ResolveVectorLane(instruction.operands.at(2), context, lane));
    context.wave.vgpr.Write(vdst, lane, Op(lhs, rhs));
  }
};

// Binary float operation: dst = float(lhs) OP float(rhs)
template <auto Op>
class BinaryF32Handler final : public VectorLaneHandler<BinaryF32Handler<Op>> {
 public:
  void ExecuteLane(const DecodedInstruction& instruction,
                   EncodedWaveContext& context, uint32_t lane) const {
    const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
    const float lhs = U32AsFloat(static_cast<uint32_t>(
        ResolveVectorLane(instruction.operands.at(1), context, lane)));
    const float rhs = U32AsFloat(static_cast<uint32_t>(
        ResolveVectorLane(instruction.operands.at(2), context, lane)));
    context.wave.vgpr.Write(vdst, lane, FloatAsU32(Op(lhs, rhs)));
  }
};

// Unary integer operation: dst = OP(src)
template <auto Op>
class UnaryU32Handler final : public VectorLaneHandler<UnaryU32Handler<Op>> {
 public:
  void ExecuteLane(const DecodedInstruction& instruction,
                   EncodedWaveContext& context, uint32_t lane) const {
    const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
    const uint32_t src = static_cast<uint32_t>(
        ResolveVectorLane(instruction.operands.at(1), context, lane));
    context.wave.vgpr.Write(vdst, lane, Op(src));
  }
};

// Operation functors for template instantiation
constexpr auto OpAddU32 = [](uint32_t a, uint32_t b) { return a + b; };
constexpr auto OpSubU32 = [](uint32_t a, uint32_t b) { return a - b; };
constexpr auto OpAndU32 = [](uint32_t a, uint32_t b) { return a & b; };
constexpr auto OpOrU32  = [](uint32_t a, uint32_t b) { return a | b; };
constexpr auto OpXorU32 = [](uint32_t a, uint32_t b) { return a ^ b; };
constexpr auto OpAddF32 = [](float a, float b) { return a + b; };
constexpr auto OpSubF32 = [](float a, float b) { return a - b; };
constexpr auto OpMulF32 = [](float a, float b) { return a * b; };
constexpr auto OpNotU32 = [](uint32_t a) { return ~a; };
constexpr auto OpIdentity = [](uint32_t a) { return a; };

// Type aliases for concrete handlers
using VAddU32Handler = BinaryU32Handler<OpAddU32>;
using VSubU32Handler = BinaryU32Handler<OpSubU32>;
using VAndB32Handler = BinaryU32Handler<OpAndU32>;
using VOrB32Handler  = BinaryU32Handler<OpOrU32>;
using VXorB32Handler = BinaryU32Handler<OpXorU32>;
using VAddF32Handler = BinaryF32Handler<OpAddF32>;
using VSubF32Handler = BinaryF32Handler<OpSubF32>;
using VMulF32Handler = BinaryF32Handler<OpMulF32>;
using VNotB32Handler = UnaryU32Handler<OpNotU32>;
using VMovB32Handler = UnaryU32Handler<OpIdentity>;

// ============================================================================
// Specialized Handlers (not suitable for templating)
// ============================================================================

// v_lshlrev_b32_e32: dst = src1 << (src0 & 31) (shift left logical)
class VLshlrevB32Handler final : public VectorLaneHandler<VLshlrevB32Handler> {
 public:
  void ExecuteLane(const DecodedInstruction& instruction,
                   EncodedWaveContext& context, uint32_t lane) const {
    const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
    const uint32_t shift = static_cast<uint32_t>(
        ResolveVectorLane(instruction.operands.at(1), context, lane));
    const uint32_t src = static_cast<uint32_t>(
        ResolveVectorLane(instruction.operands.at(2), context, lane));
    context.wave.vgpr.Write(vdst, lane, src << (shift & 31u));
  }
};

// v_cvt_f32_u32_e32: dst = float(src) (unsigned to float)
class VCvtF32U32Handler final : public VectorLaneHandler<VCvtF32U32Handler> {
 public:
  void ExecuteLane(const DecodedInstruction& instruction,
                   EncodedWaveContext& context, uint32_t lane) const {
    const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
    const uint32_t value = static_cast<uint32_t>(
        ResolveVectorLane(instruction.operands.at(1), context, lane));
    context.wave.vgpr.Write(vdst, lane, FloatAsU32(static_cast<float>(value)));
  }
};

// v_cvt_u32_f32_e32: dst = uint32_t(src) (float to unsigned)
class VCvtU32F32Handler final : public VectorLaneHandler<VCvtU32F32Handler> {
 public:
  void ExecuteLane(const DecodedInstruction& instruction,
                   EncodedWaveContext& context, uint32_t lane) const {
    const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
    const float value = U32AsFloat(static_cast<uint32_t>(
        ResolveVectorLane(instruction.operands.at(1), context, lane)));
    context.wave.vgpr.Write(vdst, lane, static_cast<uint32_t>(value));
  }
};

// v_max_i32_e32: dst = max(src0, src1) (signed 32-bit)
class VMaxI32Handler final : public VectorLaneHandler<VMaxI32Handler> {
 public:
  void ExecuteLane(const DecodedInstruction& instruction,
                   EncodedWaveContext& context, uint32_t lane) const {
    const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
    const int32_t lhs = static_cast<int32_t>(
        ResolveVectorLane(instruction.operands.at(1), context, lane));
    const int32_t rhs = static_cast<int32_t>(
        ResolveVectorLane(instruction.operands.at(2), context, lane));
    context.wave.vgpr.Write(vdst, lane, static_cast<uint32_t>(std::max(lhs, rhs)));
  }
};

// v_max_f32_e32: dst = max(src0, src1) (float)
class VMaxF32Handler final : public VectorLaneHandler<VMaxF32Handler> {
 public:
  void ExecuteLane(const DecodedInstruction& instruction,
                   EncodedWaveContext& context, uint32_t lane) const {
    const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
    const float lhs = U32AsFloat(static_cast<uint32_t>(
        ResolveVectorLane(instruction.operands.at(1), context, lane)));
    const float rhs = U32AsFloat(static_cast<uint32_t>(
        ResolveVectorLane(instruction.operands.at(2), context, lane)));
    context.wave.vgpr.Write(vdst, lane, FloatAsU32(std::fmax(lhs, rhs)));
  }
};

// v_fmac_f32_e32: dst = dst + src0 * src1 (fused multiply-accumulate)
class VFmacF32Handler final : public VectorLaneHandler<VFmacF32Handler> {
 public:
  void ExecuteLane(const DecodedInstruction& instruction,
                   EncodedWaveContext& context, uint32_t lane) const {
    const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
    const float mul_lhs = U32AsFloat(static_cast<uint32_t>(
        ResolveVectorLane(instruction.operands.at(1), context, lane)));
    const float mul_rhs = U32AsFloat(static_cast<uint32_t>(
        ResolveVectorLane(instruction.operands.at(2), context, lane)));
    const float acc = U32AsFloat(static_cast<uint32_t>(context.wave.vgpr.Read(vdst, lane)));
    context.wave.vgpr.Write(vdst, lane, FloatAsU32(acc + mul_lhs * mul_rhs));
  }
};

// v_fma_f32: dst = src0 * src1 + src2 (fused multiply-add)
class VFmaF32Handler final : public VectorLaneHandler<VFmaF32Handler> {
 public:
  void ExecuteLane(const DecodedInstruction& instruction,
                   EncodedWaveContext& context, uint32_t lane) const {
    const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
    const float src0 = U32AsFloat(static_cast<uint32_t>(
        ResolveVectorLane(instruction.operands.at(1), context, lane)));
    const float src1 = U32AsFloat(static_cast<uint32_t>(
        ResolveVectorLane(instruction.operands.at(2), context, lane)));
    const float src2 = U32AsFloat(static_cast<uint32_t>(
        ResolveVectorLane(instruction.operands.at(3), context, lane)));
    context.wave.vgpr.Write(vdst, lane, FloatAsU32(src0 * src1 + src2));
  }
};

// v_rcp_f32_e32: dst = 1.0 / src (reciprocal)
class VRcpF32Handler final : public VectorLaneHandler<VRcpF32Handler> {
 public:
  void ExecuteLane(const DecodedInstruction& instruction,
                   EncodedWaveContext& context, uint32_t lane) const {
    const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
    const float src = U32AsFloat(static_cast<uint32_t>(
        ResolveVectorLane(instruction.operands.at(1), context, lane)));
    context.wave.vgpr.Write(vdst, lane, FloatAsU32(1.0f / src));
  }
};

// v_exp_f32_e32: dst = 2^src (exponential base 2)
class VExpF32Handler final : public VectorLaneHandler<VExpF32Handler> {
 public:
  void ExecuteLane(const DecodedInstruction& instruction,
                   EncodedWaveContext& context, uint32_t lane) const {
    const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
    const float src = U32AsFloat(static_cast<uint32_t>(
        ResolveVectorLane(instruction.operands.at(1), context, lane)));
    context.wave.vgpr.Write(vdst, lane, FloatAsU32(std::exp2(src)));
  }
};

// v_rndne_f32_e32: dst = round_to_nearest(src)
class VRndneF32Handler final : public VectorLaneHandler<VRndneF32Handler> {
 public:
  void ExecuteLane(const DecodedInstruction& instruction,
                   EncodedWaveContext& context, uint32_t lane) const {
    const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
    const float src = U32AsFloat(static_cast<uint32_t>(
        ResolveVectorLane(instruction.operands.at(1), context, lane)));
    context.wave.vgpr.Write(vdst, lane, FloatAsU32(std::nearbyint(src)));
  }
};

// v_cvt_i32_f32_e32: dst = int32_t(src) (float to signed)
class VCvtI32F32Handler final : public VectorLaneHandler<VCvtI32F32Handler> {
 public:
  void ExecuteLane(const DecodedInstruction& instruction,
                   EncodedWaveContext& context, uint32_t lane) const {
    const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
    const float src = U32AsFloat(static_cast<uint32_t>(
        ResolveVectorLane(instruction.operands.at(1), context, lane)));
    context.wave.vgpr.Write(vdst, lane, static_cast<uint32_t>(static_cast<int32_t>(src)));
  }
};

// v_cvt_f32_i32_e32: dst = float(src) (signed to float)
class VCvtF32I32Handler final : public VectorLaneHandler<VCvtF32I32Handler> {
 public:
  void ExecuteLane(const DecodedInstruction& instruction,
                   EncodedWaveContext& context, uint32_t lane) const {
    const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
    const int32_t src = static_cast<int32_t>(
        ResolveVectorLane(instruction.operands.at(1), context, lane));
    context.wave.vgpr.Write(vdst, lane, FloatAsU32(static_cast<float>(src)));
  }
};

// v_subrev_u32_e32: dst = src1 - src0 (reverse subtract)
class VSubrevU32Handler final : public VectorLaneHandler<VSubrevU32Handler> {
 public:
  void ExecuteLane(const DecodedInstruction& instruction,
                   EncodedWaveContext& context, uint32_t lane) const {
    const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
    const uint32_t lhs = static_cast<uint32_t>(
        ResolveVectorLane(instruction.operands.at(1), context, lane));
    const uint32_t rhs = static_cast<uint32_t>(
        ResolveVectorLane(instruction.operands.at(2), context, lane));
    context.wave.vgpr.Write(vdst, lane, rhs - lhs);
  }
};

// v_or3_b32: dst = src0 | src1 | src2 (triple OR)
class VOr3B32Handler final : public VectorLaneHandler<VOr3B32Handler> {
 public:
  void ExecuteLane(const DecodedInstruction& instruction,
                   EncodedWaveContext& context, uint32_t lane) const {
    const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
    const uint32_t src0 = static_cast<uint32_t>(
        ResolveVectorLane(instruction.operands.at(1), context, lane));
    const uint32_t src1 = static_cast<uint32_t>(
        ResolveVectorLane(instruction.operands.at(2), context, lane));
    const uint32_t src2 = static_cast<uint32_t>(
        ResolveVectorLane(instruction.operands.at(3), context, lane));
    context.wave.vgpr.Write(vdst, lane, src0 | src1 | src2);
  }
};

// v_add3_u32: dst = src0 + src1 + src2 (triple add)
class VAdd3U32Handler final : public VectorLaneHandler<VAdd3U32Handler> {
 public:
  void ExecuteLane(const DecodedInstruction& instruction,
                   EncodedWaveContext& context, uint32_t lane) const {
    const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
    const uint32_t src0 = static_cast<uint32_t>(
        ResolveVectorLane(instruction.operands.at(1), context, lane));
    const uint32_t src1 = static_cast<uint32_t>(
        ResolveVectorLane(instruction.operands.at(2), context, lane));
    const uint32_t src2 = static_cast<uint32_t>(
        ResolveVectorLane(instruction.operands.at(3), context, lane));
    context.wave.vgpr.Write(vdst, lane, src0 + src1 + src2);
  }
};

// v_mad_u32_u24: dst = (src0[23:0] * src1[23:0]) + src2 (24-bit multiply-add)
class VMadU32U24Handler final : public VectorLaneHandler<VMadU32U24Handler> {
 public:
  void ExecuteLane(const DecodedInstruction& instruction,
                   EncodedWaveContext& context, uint32_t lane) const {
    const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
    const uint32_t src0 = static_cast<uint32_t>(
        ResolveVectorLane(instruction.operands.at(1), context, lane)) & 0x00ffffffu;
    const uint32_t src1 = static_cast<uint32_t>(
        ResolveVectorLane(instruction.operands.at(2), context, lane)) & 0x00ffffffu;
    const uint32_t src2 = static_cast<uint32_t>(
        ResolveVectorLane(instruction.operands.at(3), context, lane));
    context.wave.vgpr.Write(vdst, lane, src0 * src1 + src2);
  }
};

// v_mul_lo_i32: dst = (src0 * src1)[31:0] (low 32 bits of signed multiply)
class VMulLoI32Handler final : public VectorLaneHandler<VMulLoI32Handler> {
 public:
  void ExecuteLane(const DecodedInstruction& instruction,
                   EncodedWaveContext& context, uint32_t lane) const {
    const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
    const int32_t lhs = static_cast<int32_t>(
        ResolveVectorLane(instruction.operands.at(1), context, lane));
    const int32_t rhs = static_cast<int32_t>(
        ResolveVectorLane(instruction.operands.at(2), context, lane));
    const int64_t product = static_cast<int64_t>(lhs) * static_cast<int64_t>(rhs);
    context.wave.vgpr.Write(vdst, lane, static_cast<uint32_t>(product & 0xffffffffu));
  }
};

// v_mul_hi_u32: dst = (src0 * src1)[63:32] (high 32 bits of unsigned multiply)
class VMulHiU32Handler final : public VectorLaneHandler<VMulHiU32Handler> {
 public:
  void ExecuteLane(const DecodedInstruction& instruction,
                   EncodedWaveContext& context, uint32_t lane) const {
    const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
    const uint64_t lhs = static_cast<uint32_t>(
        ResolveVectorLane(instruction.operands.at(1), context, lane));
    const uint64_t rhs = static_cast<uint32_t>(
        ResolveVectorLane(instruction.operands.at(2), context, lane));
    context.wave.vgpr.Write(vdst, lane, static_cast<uint32_t>((lhs * rhs) >> 32u));
  }
};

// v_rcp_iflag_f32_e32: dst = 1.0 / src (integer flag reciprocal)
class VRcpIflagF32Handler final : public VectorLaneHandler<VRcpIflagF32Handler> {
 public:
  void ExecuteLane(const DecodedInstruction& instruction,
                   EncodedWaveContext& context, uint32_t lane) const {
    const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
    const float value = U32AsFloat(static_cast<uint32_t>(
        ResolveVectorLane(instruction.operands.at(1), context, lane)));
    context.wave.vgpr.Write(vdst, lane, FloatAsU32(1.0f / value));
  }
};

// v_ashrrev_i32_e32: dst = src1 >> (src0 & 31) (arithmetic shift right)
class VAshrrevI32Handler final : public VectorLaneHandler<VAshrrevI32Handler> {
 public:
  void ExecuteLane(const DecodedInstruction& instruction,
                   EncodedWaveContext& context, uint32_t lane) const {
    const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
    const uint32_t imm = static_cast<uint32_t>(
        ResolveVectorLane(instruction.operands.at(1), context, lane));
    const int32_t rhs = static_cast<int32_t>(
        ResolveVectorLane(instruction.operands.at(2), context, lane));
    context.wave.vgpr.Write(vdst, lane, static_cast<uint32_t>(rhs >> (imm & 31u)));
  }
};

// v_lshl_add_u32: dst = (src0 << (src1 & 31)) + src2
class VLshlAddU32Handler final : public VectorLaneHandler<VLshlAddU32Handler> {
 public:
  void ExecuteLane(const DecodedInstruction& instruction,
                   EncodedWaveContext& context, uint32_t lane) const {
    const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
    const uint32_t src0 = static_cast<uint32_t>(
        ResolveVectorLane(instruction.operands.at(1), context, lane));
    const uint32_t shift = static_cast<uint32_t>(
        ResolveVectorLane(instruction.operands.at(2), context, lane));
    const uint32_t src2 = static_cast<uint32_t>(
        ResolveVectorLane(instruction.operands.at(3), context, lane));
    context.wave.vgpr.Write(vdst, lane, (src0 << (shift & 31u)) + src2);
  }
};

// v_ldexp_f32: dst = src0 * 2^src1
class VLdexpF32Handler final : public VectorLaneHandler<VLdexpF32Handler> {
 public:
  void ExecuteLane(const DecodedInstruction& instruction,
                   EncodedWaveContext& context, uint32_t lane) const {
    const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
    const float value = U32AsFloat(static_cast<uint32_t>(
        ResolveVectorLane(instruction.operands.at(1), context, lane)));
    const int exp = static_cast<int>(
        ResolveVectorLane(instruction.operands.at(2), context, lane));
    context.wave.vgpr.Write(vdst, lane, FloatAsU32(std::ldexp(value, exp)));
  }
};

// v_div_scale_f32: dst = src1 (pass-through for division scaling)
class VDivScaleF32Handler final : public VectorLaneHandler<VDivScaleF32Handler> {
 public:
  void ExecuteLane(const DecodedInstruction& instruction,
                   EncodedWaveContext& context, uint32_t lane) const {
    const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
    const uint32_t value = static_cast<uint32_t>(
        ResolveVectorLane(instruction.operands.at(1), context, lane));
    context.wave.vgpr.Write(vdst, lane, value);
  }
};

// v_div_fmas_f32: dst = src0 * src1 + src2 (fused multiply-add for division)
class VDivFmasF32Handler final : public VectorLaneHandler<VDivFmasF32Handler> {
 public:
  void ExecuteLane(const DecodedInstruction& instruction,
                   EncodedWaveContext& context, uint32_t lane) const {
    const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
    const float src0 = U32AsFloat(static_cast<uint32_t>(
        ResolveVectorLane(instruction.operands.at(1), context, lane)));
    const float src1 = U32AsFloat(static_cast<uint32_t>(
        ResolveVectorLane(instruction.operands.at(2), context, lane)));
    const float src2 = U32AsFloat(static_cast<uint32_t>(
        ResolveVectorLane(instruction.operands.at(3), context, lane)));
    context.wave.vgpr.Write(vdst, lane, FloatAsU32(src0 * src1 + src2));
  }
};

// v_div_fixup_f32: dst = src2 / src1 (division fixup)
class VDivFixupF32Handler final : public VectorLaneHandler<VDivFixupF32Handler> {
 public:
  void ExecuteLane(const DecodedInstruction& instruction,
                   EncodedWaveContext& context, uint32_t lane) const {
    const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
    const float denom = U32AsFloat(static_cast<uint32_t>(
        ResolveVectorLane(instruction.operands.at(2), context, lane)));
    const float numer = U32AsFloat(static_cast<uint32_t>(
        ResolveVectorLane(instruction.operands.at(3), context, lane)));
    context.wave.vgpr.Write(vdst, lane, FloatAsU32(numer / denom));
  }
};

// v_mbcnt_lo_u32_b32: dst = carry + popcount(mask & ((1 << lane) - 1))
class VMbcntLoU32Handler final : public VectorLaneHandler<VMbcntLoU32Handler> {
 public:
  void ExecuteLane(const DecodedInstruction& instruction,
                   EncodedWaveContext& context, uint32_t lane) const {
    const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
    const uint32_t mask = static_cast<uint32_t>(
        ResolveVectorLane(instruction.operands.at(1), context, lane));
    const uint32_t carry = static_cast<uint32_t>(
        ResolveVectorLane(instruction.operands.at(2), context, lane));
    const uint32_t lane_mask =
        lane >= 32 ? 0xffffffffu : ((lane == 0) ? 0u : ((1u << lane) - 1u));
    context.wave.vgpr.Write(vdst, lane, carry + std::popcount(mask & lane_mask));
  }
};

// v_mbcnt_hi_u32_b32: dst = carry + popcount(mask & ((1 << (lane-32)) - 1))
class VMbcntHiU32Handler final : public VectorLaneHandler<VMbcntHiU32Handler> {
 public:
  void ExecuteLane(const DecodedInstruction& instruction,
                   EncodedWaveContext& context, uint32_t lane) const {
    const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
    const uint32_t mask = static_cast<uint32_t>(
        ResolveVectorLane(instruction.operands.at(1), context, lane));
    const uint32_t carry = static_cast<uint32_t>(
        ResolveVectorLane(instruction.operands.at(2), context, lane));
    const uint32_t upper_lane = lane > 32 ? (lane - 32) : 0;
    const uint32_t lane_mask =
        upper_lane >= 32 ? 0xffffffffu : ((upper_lane == 0) ? 0u : ((1u << upper_lane) - 1u));
    context.wave.vgpr.Write(vdst, lane, carry + std::popcount(mask & lane_mask));
  }
};

// v_accvgpr_read_b32: dst = agpr[src]
class VAccvgprReadHandler final : public VectorLaneHandler<VAccvgprReadHandler> {
 public:
  void ExecuteLane(const DecodedInstruction& instruction,
                   EncodedWaveContext& context, uint32_t lane) const {
    const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
    const uint32_t asrc = RequireAccumulatorIndex(instruction.operands.at(1));
    context.wave.vgpr.Write(vdst, lane, context.wave.agpr.Read(asrc, lane));
  }
};

// v_accvgpr_write_b32: agpr[dst] = src
class VAccvgprWriteHandler final : public VectorLaneHandler<VAccvgprWriteHandler> {
 public:
  void ExecuteLane(const DecodedInstruction& instruction,
                   EncodedWaveContext& context, uint32_t lane) const {
    const uint32_t adst = RequireAccumulatorIndex(instruction.operands.at(0));
    const uint32_t value = static_cast<uint32_t>(
        ResolveVectorLane(instruction.operands.at(1), context, lane));
    context.wave.agpr.Write(adst, lane, value);
  }
};

// v_cndmask_b32_e32: dst = vcc[lane] ? src1 : src0
class VCndmaskB32E32Handler final : public VectorLaneHandler<VCndmaskB32E32Handler> {
 public:
  void ExecuteLane(const DecodedInstruction& instruction,
                   EncodedWaveContext& context, uint32_t lane) const {
    const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
    const uint32_t false_value = static_cast<uint32_t>(
        ResolveVectorLane(instruction.operands.at(1), context, lane));
    const uint32_t true_value = static_cast<uint32_t>(
        ResolveVectorLane(instruction.operands.at(2), context, lane));
    const bool select_true = ((context.vcc >> lane) & 1ull) != 0;
    context.wave.vgpr.Write(vdst, lane, select_true ? true_value : false_value);
  }
};

// v_cndmask_b32_e64: dst = mask[lane] ? src1 : src0
class VCndmaskB32E64Handler final : public VectorLaneHandler<VCndmaskB32E64Handler> {
 public:
  void ExecuteLane(const DecodedInstruction& instruction,
                   EncodedWaveContext& context, uint32_t lane) const {
    const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
    const uint64_t mask = ResolveScalarPair(instruction.operands.at(3), context);
    const uint32_t false_value = static_cast<uint32_t>(
        ResolveVectorLane(instruction.operands.at(1), context, lane));
    const uint32_t true_value = static_cast<uint32_t>(
        ResolveVectorLane(instruction.operands.at(2), context, lane));
    const bool select_true = ((mask >> lane) & 1ull) != 0;
    context.wave.vgpr.Write(vdst, lane, select_true ? true_value : false_value);
  }
};

// v_pk_mov_b32: no-op placeholder for pack-move
class VPkMovB32Handler final : public BaseHandler {
 protected:
  void ExecuteImpl(const DecodedInstruction& /*instruction*/,
                   EncodedWaveContext& /*context*/) const override {
    // Until VOP3P pack-move operand decoding is modeled explicitly, treat the
    // instruction as a supported no-op for decode/binding purposes.
  }
};

// v_add_co_u32_e32: dst = src0 + src1 with carry output to VCC
// Complex: modifies VCC
class VAddCoU32E32Handler final : public BaseHandler {
 protected:
  void ExecuteImpl(const DecodedInstruction& instruction,
                   EncodedWaveContext& context) const override {
    const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
    context.vcc = 0;
    for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
      if (!context.wave.exec.test(lane)) continue;
      const uint64_t lhs = static_cast<uint32_t>(
          ResolveVectorLane(instruction.operands.at(2), context, lane));
      const uint64_t rhs = static_cast<uint32_t>(
          ResolveVectorLane(instruction.operands.at(3), context, lane));
      const uint64_t sum = lhs + rhs;
      context.wave.vgpr.Write(vdst, lane, static_cast<uint32_t>(sum));
      if ((sum >> 32u) != 0) {
        context.vcc |= (1ull << lane);
      }
    }
  }
};

// v_addc_co_u32_e32: dst = src0 + src1 + carry_in with carry output to VCC
// Complex: reads and writes VCC
class VAddcCoU32E32Handler final : public BaseHandler {
 protected:
  void ExecuteImpl(const DecodedInstruction& instruction,
                   EncodedWaveContext& context) const override {
    const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
    uint64_t next_vcc = 0;
    for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
      if (!context.wave.exec.test(lane)) continue;
      const uint64_t carry_in = (context.vcc >> lane) & 1ull;
      const uint64_t lhs = static_cast<uint32_t>(
          ResolveVectorLane(instruction.operands.at(2), context, lane));
      const uint64_t rhs = static_cast<uint32_t>(
          ResolveVectorLane(instruction.operands.at(3), context, lane));
      const uint64_t sum = lhs + rhs + carry_in;
      context.wave.vgpr.Write(vdst, lane, static_cast<uint32_t>(sum));
      if ((sum >> 32u) != 0) {
        next_vcc |= (1ull << lane);
      }
    }
    context.vcc = next_vcc;
  }
};

// v_add_co_u32_e64: dst = src0 + src1 with carry output to sdst
// Complex: carry output to arbitrary scalar register
class VAddCoU32E64Handler final : public BaseHandler {
 protected:
  void ExecuteImpl(const DecodedInstruction& instruction,
                   EncodedWaveContext& context) const override {
    const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
    uint64_t carry_mask = 0;
    for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
      if (!context.wave.exec.test(lane)) continue;
      const uint64_t lhs = static_cast<uint32_t>(
          ResolveVectorLane(instruction.operands.at(2), context, lane));
      const uint64_t rhs = static_cast<uint32_t>(
          ResolveVectorLane(instruction.operands.at(3), context, lane));
      const uint64_t sum = lhs + rhs;
      context.wave.vgpr.Write(vdst, lane, static_cast<uint32_t>(sum));
      if ((sum >> 32u) != 0) {
        carry_mask |= (1ull << lane);
      }
    }
    StoreScalarPair(instruction.operands.at(1), context, carry_mask);
  }
};

// v_addc_co_u32_e64: dst = src0 + src1 + carry_in with carry output to sdst
// Complex: carry from arbitrary scalar register
class VAddcCoU32E64Handler final : public BaseHandler {
 protected:
  void ExecuteImpl(const DecodedInstruction& instruction,
                   EncodedWaveContext& context) const override {
    const uint32_t vdst = RequireVectorIndex(instruction.operands.at(0));
    const uint64_t carry_in_mask = ResolveScalarPair(instruction.operands.at(4), context);
    uint64_t carry_out_mask = 0;
    for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
      if (!context.wave.exec.test(lane)) continue;
      const uint64_t carry_in = (carry_in_mask >> lane) & 1ull;
      const uint64_t lhs = static_cast<uint32_t>(
          ResolveVectorLane(instruction.operands.at(2), context, lane));
      const uint64_t rhs = static_cast<uint32_t>(
          ResolveVectorLane(instruction.operands.at(3), context, lane));
      const uint64_t sum = lhs + rhs + carry_in;
      context.wave.vgpr.Write(vdst, lane, static_cast<uint32_t>(sum));
      if ((sum >> 32u) != 0) {
        carry_out_mask |= (1ull << lane);
      }
    }
    StoreScalarPair(instruction.operands.at(1), context, carry_out_mask);
  }
};

// v_lshlrev_b64: dst[63:0] = src[63:0] << (shift & 63)
// Complex: 64-bit result writes to two VGPRs
class VLshlrevB64Handler final : public BaseHandler {
 protected:
  void ExecuteImpl(const DecodedInstruction& instruction,
                   EncodedWaveContext& context) const override {
    const auto [vdst, _] = RequireVectorRange(instruction.operands.at(0));
    const uint32_t shift = static_cast<uint32_t>(
        ResolveVectorLane(instruction.operands.at(1), context, 0));
    const uint32_t src_pair = instruction.operands.at(2).kind == DecodedInstructionOperandKind::VectorRegRange
                                  ? RequireVectorRange(instruction.operands.at(2)).first
                                  : RequireVectorIndex(instruction.operands.at(2));
    for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
      if (!context.wave.exec.test(lane)) continue;
      const uint64_t lo = static_cast<uint32_t>(context.wave.vgpr.Read(src_pair, lane));
      const uint64_t hi = static_cast<uint32_t>(context.wave.vgpr.Read(src_pair + 1, lane));
      const uint64_t value = ((hi << 32u) | lo) << (shift & 63u);
      context.wave.vgpr.Write(vdst, lane, static_cast<uint32_t>(value & 0xffffffffu));
      context.wave.vgpr.Write(vdst + 1, lane, static_cast<uint32_t>(value >> 32u));
    }
  }
};

// v_mad_u64_u32: dst[63:0] = acc[63:0] + (src0 * src1)
// Complex: 64-bit multiply-add with 64-bit result
class VMadU64U32Handler final : public BaseHandler {
 protected:
  void ExecuteImpl(const DecodedInstruction& instruction,
                   EncodedWaveContext& context) const override {
    const auto [vdst, _] = RequireVectorRange(instruction.operands.at(0));
    for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
      if (!context.wave.exec.test(lane)) continue;
      const uint64_t mul_lhs = ResolveVectorLane(instruction.operands.at(2), context, lane);
      const uint64_t mul_rhs = ResolveVectorLane(instruction.operands.at(3), context, lane);
      const auto src2_kind = instruction.operands.at(4).kind;
      const uint32_t src2_first = src2_kind == DecodedInstructionOperandKind::VectorRegRange
                                      ? RequireVectorRange(instruction.operands.at(4)).first
                                      : RequireVectorIndex(instruction.operands.at(4));
      const uint64_t acc_lo = static_cast<uint32_t>(
          src2_kind == DecodedInstructionOperandKind::VectorRegRange
              ? context.wave.vgpr.Read(src2_first, lane)
              : ResolveVectorLane(instruction.operands.at(4), context, lane));
      const uint64_t acc_hi = static_cast<uint32_t>(
          src2_kind == DecodedInstructionOperandKind::VectorRegRange
              ? context.wave.vgpr.Read(src2_first + 1, lane)
              : 0u);
      const uint64_t acc = (acc_hi << 32u) | acc_lo;
      const uint64_t value = acc + mul_lhs * mul_rhs;
      context.wave.vgpr.Write(vdst, lane, static_cast<uint32_t>(value & 0xffffffffu));
      context.wave.vgpr.Write(vdst + 1, lane, static_cast<uint32_t>(value >> 32u));
    }
  }
};

// MFMA handlers - use WriteTensorResultRange for tensor result storage
// v_mfma_f32_16x16x4f32: dst = acc + src0 * src1 * 4
class VMfmaF32_16x16x4f32Handler final : public BaseHandler {
 protected:
  void ExecuteImpl(const DecodedInstruction& instruction,
                   EncodedWaveContext& context) const override {
    const uint32_t vdst = instruction.operands.at(0).kind == DecodedInstructionOperandKind::VectorRegRange
                              ? RequireVectorRange(instruction.operands.at(0)).first
                              : RequireVectorIndex(instruction.operands.at(0));
    const auto storage_policy = DefaultTensorResultStoragePolicy();
    for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
      if (!context.wave.exec.test(lane)) continue;
      const float src0 = U32AsFloat(static_cast<uint32_t>(
          ResolveVectorLane(instruction.operands.at(1), context, lane)));
      const float src1 = U32AsFloat(static_cast<uint32_t>(
          ResolveVectorLane(instruction.operands.at(2), context, lane)));
      const float src2 = U32AsFloat(static_cast<uint32_t>(
          ResolveVectorLane(instruction.operands.at(3), context, lane)));
      const float value = src2 + src0 * src1 * 4.0f;
      WriteTensorResultRange(context.wave, vdst, 4, lane, FloatAsU32(value), storage_policy);
    }
  }
};

// v_mfma_f32_32x32x2f32: dst = acc + src0 * src1 * 2
class VMfmaF32_32x32x2f32Handler final : public BaseHandler {
 protected:
  void ExecuteImpl(const DecodedInstruction& instruction,
                   EncodedWaveContext& context) const override {
    const uint32_t vdst = instruction.operands.at(0).kind == DecodedInstructionOperandKind::VectorRegRange
                              ? RequireVectorRange(instruction.operands.at(0)).first
                              : RequireVectorIndex(instruction.operands.at(0));
    const auto storage_policy = DefaultTensorResultStoragePolicy();
    for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
      if (!context.wave.exec.test(lane)) continue;
      const float src0 = U32AsFloat(static_cast<uint32_t>(
          ResolveVectorLane(instruction.operands.at(1), context, lane)));
      const float src1 = U32AsFloat(static_cast<uint32_t>(
          ResolveVectorLane(instruction.operands.at(2), context, lane)));
      const float src2 = U32AsFloat(static_cast<uint32_t>(
          ResolveVectorLane(instruction.operands.at(3), context, lane)));
      const float value = src2 + src0 * src1 * 2.0f;
      WriteTensorResultRange(context.wave, vdst, 16, lane, FloatAsU32(value), storage_policy);
    }
  }
};

// v_mfma_f32_16x16x4f16: dst = acc + fp16_dot(src0, src1)
class VMfmaF32_16x16x4f16Handler final : public BaseHandler {
 protected:
  void ExecuteImpl(const DecodedInstruction& instruction,
                   EncodedWaveContext& context) const override {
    const uint32_t vdst = instruction.operands.at(0).kind == DecodedInstructionOperandKind::VectorRegRange
                              ? RequireVectorRange(instruction.operands.at(0)).first
                              : RequireVectorIndex(instruction.operands.at(0));
    const auto storage_policy = DefaultTensorResultStoragePolicy();
    for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
      if (!context.wave.exec.test(lane)) continue;
      const uint32_t src0_bits = static_cast<uint32_t>(
          ResolveVectorLane(instruction.operands.at(1), context, lane));
      const uint32_t src1_bits = static_cast<uint32_t>(
          ResolveVectorLane(instruction.operands.at(2), context, lane));
      const float acc = U32AsFloat(static_cast<uint32_t>(
          ResolveVectorLane(instruction.operands.at(3), context, lane)));
      const float a0 = HalfToFloat(static_cast<uint16_t>(src0_bits & 0xffffu));
      const float a1 = HalfToFloat(static_cast<uint16_t>(src0_bits >> 16u));
      const float b0 = HalfToFloat(static_cast<uint16_t>(src1_bits & 0xffffu));
      const float b1 = HalfToFloat(static_cast<uint16_t>(src1_bits >> 16u));
      const float value = acc + a0 * b0 + a1 * b1;
      WriteTensorResultRange(context.wave, vdst, 4, lane, FloatAsU32(value), storage_policy);
    }
  }
};

// v_mfma_i32_16x16x4i8: dst = acc + i8_dot(src0, src1)
class VMfmaI32_16x16x4i8Handler final : public BaseHandler {
 protected:
  void ExecuteImpl(const DecodedInstruction& instruction,
                   EncodedWaveContext& context) const override {
    const uint32_t vdst = instruction.operands.at(0).kind == DecodedInstructionOperandKind::VectorRegRange
                              ? RequireVectorRange(instruction.operands.at(0)).first
                              : RequireVectorIndex(instruction.operands.at(0));
    const auto storage_policy = DefaultTensorResultStoragePolicy();
    for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
      if (!context.wave.exec.test(lane)) continue;
      const uint32_t src0_bits = static_cast<uint32_t>(
          ResolveVectorLane(instruction.operands.at(1), context, lane));
      const uint32_t src1_bits = static_cast<uint32_t>(
          ResolveVectorLane(instruction.operands.at(2), context, lane));
      int32_t acc = static_cast<int32_t>(
          ResolveVectorLane(instruction.operands.at(3), context, lane));
      for (uint32_t i = 0; i < 4; ++i) {
        const int8_t a = static_cast<int8_t>((src0_bits >> (i * 8u)) & 0xffu);
        const int8_t b = static_cast<int8_t>((src1_bits >> (i * 8u)) & 0xffu);
        acc += static_cast<int32_t>(a) * static_cast<int32_t>(b);
      }
      WriteTensorResultRange(context.wave, vdst, 4, lane, static_cast<uint32_t>(acc), storage_policy);
    }
  }
};

// v_mfma_i32_16x16x16i8: dst = acc + 4 * i8_dot(src0, src1)
class VMfmaI32_16x16x16i8Handler final : public BaseHandler {
 protected:
  void ExecuteImpl(const DecodedInstruction& instruction,
                   EncodedWaveContext& context) const override {
    const uint32_t vdst = instruction.operands.at(0).kind == DecodedInstructionOperandKind::VectorRegRange
                              ? RequireVectorRange(instruction.operands.at(0)).first
                              : RequireVectorIndex(instruction.operands.at(0));
    const auto storage_policy = DefaultTensorResultStoragePolicy();
    for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
      if (!context.wave.exec.test(lane)) continue;
      const uint32_t src0_bits = static_cast<uint32_t>(
          ResolveVectorLane(instruction.operands.at(1), context, lane));
      const uint32_t src1_bits = static_cast<uint32_t>(
          ResolveVectorLane(instruction.operands.at(2), context, lane));
      int32_t acc = static_cast<int32_t>(
          ResolveVectorLane(instruction.operands.at(3), context, lane));
      for (uint32_t i = 0; i < 4; ++i) {
        const int8_t a = static_cast<int8_t>((src0_bits >> (i * 8u)) & 0xffu);
        const int8_t b = static_cast<int8_t>((src1_bits >> (i * 8u)) & 0xffu);
        acc += 4 * static_cast<int32_t>(a) * static_cast<int32_t>(b);
      }
      WriteTensorResultRange(context.wave, vdst, 4, lane, static_cast<uint32_t>(acc), storage_policy);
    }
  }
};

// v_mfma_f32_16x16x2bf16: dst = acc + bf16_dot(src0, src1)
class VMfmaF32_16x16x2bf16Handler final : public BaseHandler {
 protected:
  void ExecuteImpl(const DecodedInstruction& instruction,
                   EncodedWaveContext& context) const override {
    const uint32_t vdst = instruction.operands.at(0).kind == DecodedInstructionOperandKind::VectorRegRange
                              ? RequireVectorRange(instruction.operands.at(0)).first
                              : RequireVectorIndex(instruction.operands.at(0));
    const auto storage_policy = DefaultTensorResultStoragePolicy();
    for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
      if (!context.wave.exec.test(lane)) continue;
      const uint32_t src0_bits = static_cast<uint32_t>(
          ResolveVectorLane(instruction.operands.at(1), context, lane));
      const uint32_t src1_bits = static_cast<uint32_t>(
          ResolveVectorLane(instruction.operands.at(2), context, lane));
      const float acc = U32AsFloat(static_cast<uint32_t>(
          ResolveVectorLane(instruction.operands.at(3), context, lane)));
      const float a0 = BFloat16ToFloat(static_cast<uint16_t>(src0_bits & 0xffffu));
      const float a1 = BFloat16ToFloat(static_cast<uint16_t>(src0_bits >> 16u));
      const float b0 = BFloat16ToFloat(static_cast<uint16_t>(src1_bits & 0xffffu));
      const float b1 = BFloat16ToFloat(static_cast<uint16_t>(src1_bits >> 16u));
      const float value = acc + a0 * b0 + a1 * b1;
      WriteTensorResultRange(context.wave, vdst, 4, lane, FloatAsU32(value), storage_policy);
    }
  }
};

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
  }
};

class ScalarAluHandler final : public BaseHandler {
 protected:
  void ExecuteImpl(const DecodedInstruction& instruction, EncodedWaveContext& context) const override {
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
        EncodedDebugLog("pc=0x%llx s_or_b64 exec lhs=0x%llx rhs=0x%llx out=0x%llx",
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
    } else if (descriptor.op_type == GcnIsaOpType::Sop1 &&
               descriptor.opcode == static_cast<uint16_t>(GcnIsaSop1Opcode::S_BCNT1_I32_B64)) {
      const uint32_t sdst = RequireScalarIndex(instruction.operands.at(0));
      const uint64_t value = ResolveScalarPair(instruction.operands.at(1), context);
      context.wave.sgpr.Write(sdst, static_cast<uint32_t>(std::popcount(value)));
    } else {
      throw std::invalid_argument("unsupported scalar alu opcode: " + instruction.mnemonic);
    }
  }
};

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
  }
};

class VectorCompareHandler final : public BaseHandler {
 protected:
  void ExecuteImpl(const DecodedInstruction& instruction, EncodedWaveContext& context) const override {
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
    EncodedDebugLog("pc=0x%llx %s mask=0x%llx",
             static_cast<unsigned long long>(instruction.pc),
             instruction.mnemonic.c_str(),
             static_cast<unsigned long long>(mask));
  }
};

class MaskHandler final : public BaseHandler {
 protected:
  void ExecuteImpl(const DecodedInstruction& instruction, EncodedWaveContext& context) const override {
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
    EncodedDebugLog("pc=0x%llx %s exec=0x%llx",
             static_cast<unsigned long long>(instruction.pc),
             instruction.mnemonic.c_str(),
             static_cast<unsigned long long>(context.wave.exec.to_ullong()));
  }
};

class BranchHandler final : public IEncodedSemanticHandler {
 public:
  void Execute(const DecodedInstruction& instruction, EncodedWaveContext& context) const override {
    // Emit trace callback for instruction start
    if (context.on_execute) {
      context.on_execute(instruction, context, "start");
    }

    switch (instruction.encoding_id) {
      case 10: {  // s_cbranch_execz
      if (context.wave.exec.none()) {
        context.wave.pc =
            BranchTarget(context.wave.pc, static_cast<int32_t>(instruction.operands.at(0).info.immediate));
      } else {
        context.wave.pc += instruction.size_bytes;
      }
      break;
      }
      case 22: {  // s_cbranch_scc1
      if (context.wave.ScalarMaskBit0()) {
        context.wave.pc =
            BranchTarget(context.wave.pc, static_cast<int32_t>(instruction.operands.at(0).info.immediate));
      } else {
        context.wave.pc += instruction.size_bytes;
      }
      break;
      }
      case 26: {  // s_cbranch_scc0
      if (!context.wave.ScalarMaskBit0()) {
        context.wave.pc =
            BranchTarget(context.wave.pc, static_cast<int32_t>(instruction.operands.at(0).info.immediate));
      } else {
        context.wave.pc += instruction.size_bytes;
      }
      break;
      }
      case 43: {  // s_cbranch_vccz
      if (context.vcc == 0) {
        context.wave.pc =
            BranchTarget(context.wave.pc, static_cast<int32_t>(instruction.operands.at(0).info.immediate));
      } else {
        context.wave.pc += instruction.size_bytes;
      }
      break;
      }
      case 74: {  // s_cbranch_execnz
      if (context.wave.exec.any()) {
        context.wave.pc =
            BranchTarget(context.wave.pc, static_cast<int32_t>(instruction.operands.at(0).info.immediate));
      } else {
        context.wave.pc += instruction.size_bytes;
      }
      break;
      }
      case 27: {  // s_branch
      context.wave.pc =
          BranchTarget(context.wave.pc, static_cast<int32_t>(instruction.operands.at(0).info.immediate));
      break;
      }
      default:
        throw std::invalid_argument("unsupported branch opcode: " + instruction.mnemonic);
    }

    // Emit trace callback for instruction end
    if (context.on_execute) {
      context.on_execute(instruction, context, "end");
    }
  }
};

class FlatMemoryHandler final : public BaseHandler {
 protected:
  void ExecuteImpl(const DecodedInstruction& instruction, EncodedWaveContext& context) const override {
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
          EncodedDebugLog("pc=0x%llx global_load addr=0x%llx -> v%u=0x%llx",
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
            EncodedDebugLog("pc=0x%llx global_store addr=0x%llx value=0x%llx",
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
            EncodedDebugLog("pc=0x%llx global_store addr=0x%llx value=0x%llx",
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
  }
};

class SharedMemoryHandler final : public BaseHandler {
 protected:
  void ExecuteImpl(const DecodedInstruction& instruction, EncodedWaveContext& context) const override {
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
          EncodedDebugLog("pc=0x%llx ds_write_b32 addr=0x%x value=0x%x",
                   static_cast<unsigned long long>(instruction.pc), byte_offset, value);
        }
      }
      ++context.stats.shared_stores;
      if (context.captured_memory_request != nullptr) {
        *context.captured_memory_request = request;
      }
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
          EncodedDebugLog("pc=0x%llx ds_read_b32 addr=0x%x value=0x%x",
                   static_cast<unsigned long long>(instruction.pc), byte_offset, value);
        }
      }
      ++context.stats.shared_loads;
      if (context.captured_memory_request != nullptr) {
        *context.captured_memory_request = request;
      }
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
      return;
    }
    throw std::invalid_argument("unsupported shared memory opcode: " + instruction.mnemonic);
  }
};

class SpecialHandler final : public IEncodedSemanticHandler {
 public:
  void Execute(const DecodedInstruction& instruction, EncodedWaveContext& context) const override {
    // Emit trace callback for instruction start
    if (context.on_execute) {
      context.on_execute(instruction, context, "start");
    }

    switch (instruction.encoding_id) {
      case 29: {  // s_barrier
        ++context.stats.barriers;
        sync_ops::MarkWaveAtBarrier(context.wave,
                                    context.block.barrier_generation,
                                    context.block.barrier_arrivals,
                                    false);
        // Do NOT advance PC here - ResumeBarrierReleasedWave handles it when barrier releases
        break;
      }
      case 68:  // s_nop
        context.wave.pc += instruction.size_bytes;
        break;
      case 12:  // s_waitcnt
        context.wave.pc += instruction.size_bytes;
        break;
      case 1:  // s_endpgm - do NOT advance PC, execution terminates
        context.wave.status = WaveStatus::Exited;
        ++context.stats.wave_exits;
        break;
      default:
        throw std::invalid_argument("unsupported special opcode: " + instruction.mnemonic);
    }

    // Emit trace callback for instruction end
    if (context.on_execute) {
      context.on_execute(instruction, context, "end");
    }
  }
};

// Unified handler registry for O(1) lookup.
// This replaces the O(n) linear search in HandlerBindings().
// Supports incremental migration from large monolithic handlers to individual handlers.
class HandlerRegistry {
 public:
  static const HandlerRegistry& Instance() {
    static const HandlerRegistry kInstance;
    return kInstance;
  }

  const IEncodedSemanticHandler* Find(std::string_view mnemonic) const {
    auto it = map_.find(mnemonic);
    if (it != map_.end()) {
      return it->second;
    }
    return nullptr;
  }

  void Register(std::string_view mnemonic, const IEncodedSemanticHandler* handler) {
    map_[mnemonic] = handler;
  }

 private:
  HandlerRegistry() = default;
  std::unordered_map<std::string_view, const IEncodedSemanticHandler*> map_;
};

struct HandlerBinding {
  const char* mnemonic = nullptr;
  const IEncodedSemanticHandler* handler = nullptr;
};

// Static instances of individual vector ALU handlers
// These are registered in the unified HandlerRegistry for O(1) lookup
namespace {
static const VAddU32Handler kVAddU32Handler;
static const VSubU32Handler kVSubU32Handler;
static const VMovB32Handler kVMovB32Handler;
static const VNotB32Handler kVNotB32Handler;
static const VAndB32Handler kVAndB32Handler;
static const VOrB32Handler kVOrB32Handler;
static const VXorB32Handler kVXorB32Handler;
static const VAddF32Handler kVAddF32Handler;
static const VSubF32Handler kVSubF32Handler;
static const VMulF32Handler kVMulF32Handler;
static const VLshlrevB32Handler kVLshlrevB32Handler;
static const VCvtF32U32Handler kVCvtF32U32Handler;
static const VCvtU32F32Handler kVCvtU32F32Handler;
static const VMaxI32Handler kVMaxI32Handler;
static const VMaxF32Handler kVMaxF32Handler;
static const VFmacF32Handler kVFmacF32Handler;
static const VFmaF32Handler kVFmaF32Handler;
static const VRcpF32Handler kVRcpF32Handler;
static const VExpF32Handler kVExpF32Handler;
static const VRndneF32Handler kVRndneF32Handler;
static const VCvtI32F32Handler kVCvtI32F32Handler;
static const VCvtF32I32Handler kVCvtF32I32Handler;
static const VSubrevU32Handler kVSubrevU32Handler;
static const VOr3B32Handler kVOr3B32Handler;
static const VAdd3U32Handler kVAdd3U32Handler;
static const VMadU32U24Handler kVMadU32U24Handler;
static const VMulLoI32Handler kVMulLoI32Handler;
static const VMulHiU32Handler kVMulHiU32Handler;
static const VRcpIflagF32Handler kVRcpIflagF32Handler;
static const VAshrrevI32Handler kVAshrrevI32Handler;
static const VLshlAddU32Handler kVLshlAddU32Handler;
static const VLdexpF32Handler kVLdexpF32Handler;
static const VDivScaleF32Handler kVDivScaleF32Handler;
static const VDivFmasF32Handler kVDivFmasF32Handler;
static const VDivFixupF32Handler kVDivFixupF32Handler;
static const VMbcntLoU32Handler kVMbcntLoU32Handler;
static const VMbcntHiU32Handler kVMbcntHiU32Handler;
static const VAccvgprReadHandler kVAccvgprReadHandler;
static const VAccvgprWriteHandler kVAccvgprWriteHandler;
static const VCndmaskB32E32Handler kVCndmaskB32E32Handler;
static const VCndmaskB32E64Handler kVCndmaskB32E64Handler;
static const VPkMovB32Handler kVPkMovB32Handler;
static const VAddCoU32E32Handler kVAddCoU32E32Handler;
static const VAddcCoU32E32Handler kVAddcCoU32E32Handler;
static const VAddCoU32E64Handler kVAddCoU32E64Handler;
static const VAddcCoU32E64Handler kVAddcCoU32E64Handler;
static const VLshlrevB64Handler kVLshlrevB64Handler;
static const VMadU64U32Handler kVMadU64U32Handler;
static const VMfmaF32_16x16x4f32Handler kVMfmaF32_16x16x4f32Handler;
static const VMfmaF32_32x32x2f32Handler kVMfmaF32_32x32x2f32Handler;
static const VMfmaF32_16x16x4f16Handler kVMfmaF32_16x16x4f16Handler;
static const VMfmaI32_16x16x4i8Handler kVMfmaI32_16x16x4i8Handler;
static const VMfmaI32_16x16x16i8Handler kVMfmaI32_16x16x16i8Handler;
static const VMfmaF32_16x16x2bf16Handler kVMfmaF32_16x16x2bf16Handler;

// Register all individual handlers in the unified registry
void RegisterVectorAluHandlers() {
  auto& registry = const_cast<HandlerRegistry&>(HandlerRegistry::Instance());
  registry.Register("v_add_u32_e32", &kVAddU32Handler);
  registry.Register("v_sub_u32_e32", &kVSubU32Handler);
  registry.Register("v_mov_b32_e32", &kVMovB32Handler);
  registry.Register("v_not_b32_e32", &kVNotB32Handler);
  registry.Register("v_and_b32_e32", &kVAndB32Handler);
  registry.Register("v_or_b32_e32", &kVOrB32Handler);
  registry.Register("v_xor_b32_e32", &kVXorB32Handler);
  registry.Register("v_add_f32_e32", &kVAddF32Handler);
  registry.Register("v_sub_f32_e32", &kVSubF32Handler);
  registry.Register("v_mul_f32_e32", &kVMulF32Handler);
  registry.Register("v_lshlrev_b32_e32", &kVLshlrevB32Handler);
  registry.Register("v_cvt_f32_u32_e32", &kVCvtF32U32Handler);
  registry.Register("v_cvt_u32_f32_e32", &kVCvtU32F32Handler);
  registry.Register("v_max_i32_e32", &kVMaxI32Handler);
  registry.Register("v_max_f32_e32", &kVMaxF32Handler);
  registry.Register("v_fmac_f32_e32", &kVFmacF32Handler);
  registry.Register("v_fma_f32", &kVFmaF32Handler);
  registry.Register("v_rcp_f32_e32", &kVRcpF32Handler);
  registry.Register("v_exp_f32_e32", &kVExpF32Handler);
  registry.Register("v_rndne_f32_e32", &kVRndneF32Handler);
  registry.Register("v_cvt_i32_f32_e32", &kVCvtI32F32Handler);
  registry.Register("v_cvt_f32_i32_e32", &kVCvtF32I32Handler);
  registry.Register("v_subrev_u32_e32", &kVSubrevU32Handler);
  registry.Register("v_or3_b32", &kVOr3B32Handler);
  registry.Register("v_add3_u32", &kVAdd3U32Handler);
  registry.Register("v_mad_u32_u24", &kVMadU32U24Handler);
  registry.Register("v_mul_lo_i32", &kVMulLoI32Handler);
  registry.Register("v_mul_hi_u32", &kVMulHiU32Handler);
  registry.Register("v_rcp_iflag_f32_e32", &kVRcpIflagF32Handler);
  registry.Register("v_ashrrev_i32_e32", &kVAshrrevI32Handler);
  registry.Register("v_lshl_add_u32", &kVLshlAddU32Handler);
  registry.Register("v_ldexp_f32", &kVLdexpF32Handler);
  registry.Register("v_div_scale_f32", &kVDivScaleF32Handler);
  registry.Register("v_div_fmas_f32", &kVDivFmasF32Handler);
  registry.Register("v_div_fixup_f32", &kVDivFixupF32Handler);
  registry.Register("v_mbcnt_lo_u32_b32", &kVMbcntLoU32Handler);
  registry.Register("v_mbcnt_hi_u32_b32", &kVMbcntHiU32Handler);
  registry.Register("v_accvgpr_read_b32", &kVAccvgprReadHandler);
  registry.Register("v_accvgpr_write_b32", &kVAccvgprWriteHandler);
  registry.Register("v_cndmask_b32_e32", &kVCndmaskB32E32Handler);
  registry.Register("v_cndmask_b32_e64", &kVCndmaskB32E64Handler);
  registry.Register("v_pk_mov_b32", &kVPkMovB32Handler);
  registry.Register("v_add_co_u32_e32", &kVAddCoU32E32Handler);
  registry.Register("v_addc_co_u32_e32", &kVAddcCoU32E32Handler);
  registry.Register("v_add_co_u32_e64", &kVAddCoU32E64Handler);
  registry.Register("v_addc_co_u32_e64", &kVAddcCoU32E64Handler);
  registry.Register("v_lshlrev_b64", &kVLshlrevB64Handler);
  registry.Register("v_mad_u64_u32", &kVMadU64U32Handler);
  registry.Register("v_mfma_f32_16x16x4f32", &kVMfmaF32_16x16x4f32Handler);
  registry.Register("v_mfma_f32_32x32x2f32", &kVMfmaF32_32x32x2f32Handler);
  registry.Register("v_mfma_f32_16x16x4f16", &kVMfmaF32_16x16x4f16Handler);
  registry.Register("v_mfma_i32_16x16x4i8", &kVMfmaI32_16x16x4i8Handler);
  registry.Register("v_mfma_i32_16x16x16i8", &kVMfmaI32_16x16x16i8Handler);
  registry.Register("v_mfma_f32_16x16x2bf16", &kVMfmaF32_16x16x2bf16Handler);
}

// Auto-registration on first use
struct HandlerRegistrar {
  HandlerRegistrar() {
    RegisterVectorAluHandlers();
  }
};

static const HandlerRegistrar kHandlerRegistrar;
}  // namespace

const IEncodedSemanticHandler* HandlerForSemanticFamily(std::string_view semantic_family,
                                                       std::string_view mnemonic) {
  static const ScalarMemoryHandler kScalarMemoryHandler;
  static const ScalarAluHandler kScalarAluHandler;
  static const ScalarCompareHandler kScalarCompareHandler;
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
  // vector_alu handlers are now in HandlerRegistry for O(1) lookup
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
  static const VectorCompareHandler kVectorCompareHandler;
  // Note: Vector ALU handlers are now in HandlerRegistry for O(1) lookup
  static const std::vector<HandlerBinding> kBindings = {
      {.mnemonic = "s_and_saveexec_b64", .handler = &kMaskHandler},
      {.mnemonic = "s_andn2_saveexec_b64", .handler = &kMaskHandler},
      {.mnemonic = "s_abs_i32", .handler = &kScalarAluHandler},
      {.mnemonic = "s_sub_i32", .handler = &kScalarAluHandler},
      {.mnemonic = "s_lshl_b32", .handler = &kScalarAluHandler},
      {.mnemonic = "s_or_b32", .handler = &kScalarAluHandler},
      {.mnemonic = "s_xor_b64", .handler = &kScalarAluHandler},
      {.mnemonic = "s_movk_i32", .handler = &kScalarAluHandler},
      {.mnemonic = "s_cmp_gt_i32", .handler = &kScalarCompareHandler},
      {.mnemonic = "s_cmp_lg_u32", .handler = &kScalarCompareHandler},
      {.mnemonic = "ds_read2_b32", .handler = &kSharedMemoryHandler},
      {.mnemonic = "ds_read_b32", .handler = &kSharedMemoryHandler},
      {.mnemonic = "ds_write_b32", .handler = &kSharedMemoryHandler},
      // Vector ALU handlers moved to HandlerRegistry
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
  // O(1) lookup via unified registry
  if (const auto* handler = HandlerRegistry::Instance().Find(mnemonic)) {
    return *handler;
  }
  // Fallback: semantic family lookup (for backward compatibility during migration)
  if (const auto* def = FindGeneratedGcnInstDefByMnemonic(mnemonic); def != nullptr) {
    if (const auto* handler = HandlerForSemanticFamily(def->semantic_family, def->mnemonic)) {
      return *handler;
    }
  }
  // Fallback: linear search in HandlerBindings() (deprecated, will be removed)
  for (const auto& binding : HandlerBindings()) {
    if (binding.mnemonic == mnemonic) {
      return *binding.handler;
    }
  }
  EncodedDebugLog("EncodedSemanticHandlerRegistry::Get mnemonic=%.*s not found",
           static_cast<int>(mnemonic.size()), mnemonic.data());
  throw std::invalid_argument("unsupported raw GCN opcode: " + std::string(mnemonic));
}

const IEncodedSemanticHandler& EncodedSemanticHandlerRegistry::Get(
    const DecodedInstruction& instruction) {
  // O(1) lookup via unified registry
  if (const auto* handler = HandlerRegistry::Instance().Find(instruction.mnemonic)) {
    return *handler;
  }
  // Fallback: linear search in HandlerBindings() (deprecated)
  for (const auto& binding : HandlerBindings()) {
    if (binding.mnemonic == instruction.mnemonic) {
      return *binding.handler;
    }
  }
  // Fallback: semantic family lookup
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
