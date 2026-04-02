#include "gpu_model/instruction/encoded/internal/encoded_instruction_binding.h"

#include <bitset>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include "gpu_model/instruction/encoded/internal/encoded_gcn_encoding_def.h"
#include "gpu_model/instruction/encoded/internal/encoded_instruction_descriptor.h"
#include "gpu_model/execution/encoded_semantic_handler.h"

namespace gpu_model {

namespace {

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

class UnsupportedInstructionHandler final : public IEncodedSemanticHandler {
 public:
  void Execute(const DecodedInstruction& instruction, EncodedWaveContext&) const override {
    throw std::invalid_argument("unsupported instantiated raw GCN opcode: " + instruction.mnemonic);
  }
};

std::bitset<64> MaskFromU64(uint64_t value) {
  return std::bitset<64>(value);
}

uint64_t BranchTarget(uint64_t pc, int32_t simm16) {
  const int64_t target = static_cast<int64_t>(pc) + 4 + static_cast<int64_t>(simm16) * 4;
  return static_cast<uint64_t>(target);
}

std::pair<uint32_t, uint32_t> RequireScalarRange(const DecodedInstructionOperand& operand) {
  if (operand.kind != DecodedInstructionOperandKind::ScalarRegRange || operand.info.reg_count == 0) {
    throw std::invalid_argument("expected scalar register range operand");
  }
  return {operand.info.reg_first, operand.info.reg_first + operand.info.reg_count - 1};
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

class EncodedInstructionObject : public InstructionObject {
 public:
  using InstructionObject::InstructionObject;

  virtual void Execute(EncodedWaveContext& context) const {
    InstructionObject::Execute(context);
  }
};

class SmrdInstructionBase : public EncodedInstructionObject {
 public:
  using EncodedInstructionObject::EncodedInstructionObject;
  std::string_view op_type_name() const override { return "smrd"; }
};

class Sop1InstructionBase : public EncodedInstructionObject {
 public:
  using EncodedInstructionObject::EncodedInstructionObject;
  std::string_view op_type_name() const override { return "sop1"; }
};

class Sop2InstructionBase : public EncodedInstructionObject {
 public:
  using EncodedInstructionObject::EncodedInstructionObject;
  std::string_view op_type_name() const override { return "sop2"; }
};

class SopkInstructionBase : public EncodedInstructionObject {
 public:
  using EncodedInstructionObject::EncodedInstructionObject;
  std::string_view op_type_name() const override { return "sopk"; }
};

class SopcInstructionBase : public EncodedInstructionObject {
 public:
  using EncodedInstructionObject::EncodedInstructionObject;
  std::string_view op_type_name() const override { return "sopc"; }
};

class SoppInstructionBase : public EncodedInstructionObject {
 public:
  using EncodedInstructionObject::EncodedInstructionObject;
  std::string_view op_type_name() const override { return "sopp"; }
};

class Vop1InstructionBase : public EncodedInstructionObject {
 public:
  using EncodedInstructionObject::EncodedInstructionObject;
  std::string_view op_type_name() const override { return "vop1"; }
};

class Vop2InstructionBase : public EncodedInstructionObject {
 public:
  using EncodedInstructionObject::EncodedInstructionObject;
  std::string_view op_type_name() const override { return "vop2"; }
};

class Vop3aInstructionBase : public EncodedInstructionObject {
 public:
  using EncodedInstructionObject::EncodedInstructionObject;
  std::string_view op_type_name() const override { return "vop3a"; }
};

class Vop3bInstructionBase : public EncodedInstructionObject {
 public:
  using EncodedInstructionObject::EncodedInstructionObject;
  std::string_view op_type_name() const override { return "vop3b"; }
};

class Vop3pInstructionBase : public EncodedInstructionObject {
 public:
  using EncodedInstructionObject::EncodedInstructionObject;
  std::string_view op_type_name() const override { return "vop3p"; }
};

class VopcInstructionBase : public EncodedInstructionObject {
 public:
  using EncodedInstructionObject::EncodedInstructionObject;
  std::string_view op_type_name() const override { return "vopc"; }
};

class FlatInstructionBase : public EncodedInstructionObject {
 public:
  using EncodedInstructionObject::EncodedInstructionObject;
  std::string_view op_type_name() const override { return "flat"; }
};

class DsInstructionBase : public EncodedInstructionObject {
 public:
  using EncodedInstructionObject::EncodedInstructionObject;
  std::string_view op_type_name() const override { return "ds"; }
};

class PlaceholderInstructionBase : public EncodedInstructionObject {
 public:
  PlaceholderInstructionBase(DecodedInstruction instruction,
                             const IEncodedSemanticHandler& handler,
                             std::string_view op_type_name,
                             std::string_view class_name)
      : EncodedInstructionObject(std::move(instruction), handler),
        op_type_name_(op_type_name),
        class_name_(class_name) {}

  std::string_view op_type_name() const override { return op_type_name_; }
  std::string_view class_name() const override { return class_name_; }

 private:
  std::string_view op_type_name_;
  std::string_view class_name_;
};

class SAndSaveexecB64Instruction final : public Sop1InstructionBase {
 public:
  using Sop1InstructionBase::Sop1InstructionBase;

  std::string_view class_name() const override { return "s_and_saveexec_b64"; }

  void Execute(EncodedWaveContext& context) const override {
    const auto& instruction = decoded();
    const auto [sdst, _] = RequireScalarRange(instruction.operands.at(0));
    const uint64_t exec_before = context.wave.exec.to_ullong();
    const uint64_t mask = ResolveScalarPair(instruction.operands.at(1), context);
    context.wave.sgpr.Write(sdst, static_cast<uint32_t>(exec_before & 0xffffffffu));
    context.wave.sgpr.Write(sdst + 1, static_cast<uint32_t>(exec_before >> 32u));
    context.wave.exec = context.wave.exec & MaskFromU64(mask);
    DebugLog("pc=0x%llx s_and_saveexec_b64 before=0x%llx mask=0x%llx after=0x%llx",
             static_cast<unsigned long long>(instruction.pc),
             static_cast<unsigned long long>(exec_before),
             static_cast<unsigned long long>(mask),
             static_cast<unsigned long long>(context.wave.exec.to_ullong()));
    context.wave.pc += instruction.size_bytes;
  }
};

class SAndn2SaveexecB64Instruction final : public Sop1InstructionBase {
 public:
  using Sop1InstructionBase::Sop1InstructionBase;

  std::string_view class_name() const override { return "s_andn2_saveexec_b64"; }

  void Execute(EncodedWaveContext& context) const override {
    const auto& instruction = decoded();
    const auto [sdst, _] = RequireScalarRange(instruction.operands.at(0));
    const uint64_t exec_before = context.wave.exec.to_ullong();
    const uint64_t mask = ResolveScalarPair(instruction.operands.at(1), context);
    context.wave.sgpr.Write(sdst, static_cast<uint32_t>(exec_before & 0xffffffffu));
    context.wave.sgpr.Write(sdst + 1, static_cast<uint32_t>(exec_before >> 32u));
    context.wave.exec = MaskFromU64(mask & ~exec_before);
    context.wave.pc += instruction.size_bytes;
  }
};

class SNopInstruction final : public SoppInstructionBase {
 public:
  using SoppInstructionBase::SoppInstructionBase;

  std::string_view class_name() const override { return "s_nop"; }

  void Execute(EncodedWaveContext& context) const override {
    context.wave.pc += decoded().size_bytes;
  }
};

class SWaitcntInstruction final : public SoppInstructionBase {
 public:
  using SoppInstructionBase::SoppInstructionBase;

  std::string_view class_name() const override { return "s_waitcnt"; }

  void Execute(EncodedWaveContext& context) const override {
    context.wave.pc += decoded().size_bytes;
  }
};

class SEndpgmInstruction final : public SoppInstructionBase {
 public:
  using SoppInstructionBase::SoppInstructionBase;

  std::string_view class_name() const override { return "s_endpgm"; }

  void Execute(EncodedWaveContext& context) const override {
    context.wave.status = WaveStatus::Exited;
    ++context.stats.wave_exits;
  }
};

class SBarrierInstruction final : public SoppInstructionBase {
 public:
  using SoppInstructionBase::SoppInstructionBase;

  std::string_view class_name() const override { return "s_barrier"; }

  void Execute(EncodedWaveContext& context) const override {
    ++context.stats.barriers;
    context.wave.status = WaveStatus::Stalled;
    context.wave.waiting_at_barrier = true;
    context.wave.barrier_generation = context.block.barrier_generation;
    ++context.block.barrier_arrivals;
  }
};

class BranchInstructionBase : public SoppInstructionBase {
 public:
  using SoppInstructionBase::SoppInstructionBase;

 protected:
  int32_t branch_offset() const {
    const auto& operand = decoded().operands.at(0);
    if (!operand.info.has_immediate) {
      throw std::invalid_argument("branch instruction missing immediate");
    }
    return static_cast<int32_t>(operand.info.immediate);
  }

  void BranchOrAdvance(EncodedWaveContext& context, bool take_branch) const {
    if (take_branch) {
      context.wave.pc = BranchTarget(context.wave.pc, branch_offset());
      return;
    }
    context.wave.pc += decoded().size_bytes;
  }
};

class SBranchInstruction final : public BranchInstructionBase {
 public:
  using BranchInstructionBase::BranchInstructionBase;

  std::string_view class_name() const override { return "s_branch"; }

  void Execute(EncodedWaveContext& context) const override {
    BranchOrAdvance(context, true);
  }
};

class SCbranchScc0Instruction final : public BranchInstructionBase {
 public:
  using BranchInstructionBase::BranchInstructionBase;

  std::string_view class_name() const override { return "s_cbranch_scc0"; }

  void Execute(EncodedWaveContext& context) const override {
    BranchOrAdvance(context, !context.wave.ScalarMaskBit0());
  }
};

class SCbranchScc1Instruction final : public BranchInstructionBase {
 public:
  using BranchInstructionBase::BranchInstructionBase;

  std::string_view class_name() const override { return "s_cbranch_scc1"; }

  void Execute(EncodedWaveContext& context) const override {
    BranchOrAdvance(context, context.wave.ScalarMaskBit0());
  }
};

class SCbranchVcczInstruction final : public BranchInstructionBase {
 public:
  using BranchInstructionBase::BranchInstructionBase;

  std::string_view class_name() const override { return "s_cbranch_vccz"; }

  void Execute(EncodedWaveContext& context) const override {
    BranchOrAdvance(context, context.vcc == 0);
  }
};

class SCbranchExeczInstruction final : public BranchInstructionBase {
 public:
  using BranchInstructionBase::BranchInstructionBase;

  std::string_view class_name() const override { return "s_cbranch_execz"; }

  void Execute(EncodedWaveContext& context) const override {
    BranchOrAdvance(context, context.wave.exec.none());
  }
};

class SCbranchExecnzInstruction final : public BranchInstructionBase {
 public:
  using BranchInstructionBase::BranchInstructionBase;

  std::string_view class_name() const override { return "s_cbranch_execnz"; }

  void Execute(EncodedWaveContext& context) const override {
    BranchOrAdvance(context, context.wave.exec.any());
  }
};

#define DEFINE_RAW_GCN_OPCODE_CLASS(ClassName, BaseClass, TextName) \
  class ClassName final : public BaseClass {                        \
   public:                                                          \
    using BaseClass::BaseClass;                                     \
    std::string_view class_name() const override { return TextName; } \
  }

DEFINE_RAW_GCN_OPCODE_CLASS(SLoadDwordInstruction, SmrdInstructionBase, "s_load_dword");
DEFINE_RAW_GCN_OPCODE_CLASS(SLoadDwordx2Instruction, SmrdInstructionBase, "s_load_dwordx2");
DEFINE_RAW_GCN_OPCODE_CLASS(SLoadDwordx4Instruction, SmrdInstructionBase, "s_load_dwordx4");

DEFINE_RAW_GCN_OPCODE_CLASS(SMovB32Instruction, Sop1InstructionBase, "s_mov_b32");
DEFINE_RAW_GCN_OPCODE_CLASS(SMovB64Instruction, Sop1InstructionBase, "s_mov_b64");
DEFINE_RAW_GCN_OPCODE_CLASS(SAbsI32Instruction, Sop1InstructionBase, "s_abs_i32");
DEFINE_RAW_GCN_OPCODE_CLASS(SSextI32I16Instruction, Sop1InstructionBase, "s_sext_i32_i16");
DEFINE_RAW_GCN_OPCODE_CLASS(SBcnt1I32B64Instruction, Sop1InstructionBase, "s_bcnt1_i32_b64");
DEFINE_RAW_GCN_OPCODE_CLASS(SMovkI32Instruction, SopkInstructionBase, "s_movk_i32");
DEFINE_RAW_GCN_OPCODE_CLASS(SCselectB64Instruction, Sop2InstructionBase, "s_cselect_b64");
DEFINE_RAW_GCN_OPCODE_CLASS(SAndn2B64Instruction, Sop2InstructionBase, "s_andn2_b64");
DEFINE_RAW_GCN_OPCODE_CLASS(SOrB64Instruction, Sop2InstructionBase, "s_or_b64");
DEFINE_RAW_GCN_OPCODE_CLASS(SXorB64Instruction, Sop2InstructionBase, "s_xor_b64");
DEFINE_RAW_GCN_OPCODE_CLASS(SOrB32Instruction, Sop2InstructionBase, "s_or_b32");
DEFINE_RAW_GCN_OPCODE_CLASS(SAndB64Instruction, Sop2InstructionBase, "s_and_b64");
DEFINE_RAW_GCN_OPCODE_CLASS(SAndB32Instruction, Sop2InstructionBase, "s_and_b32");
DEFINE_RAW_GCN_OPCODE_CLASS(SMulI32Instruction, Sop2InstructionBase, "s_mul_i32");
DEFINE_RAW_GCN_OPCODE_CLASS(SSubI32Instruction, Sop2InstructionBase, "s_sub_i32");
DEFINE_RAW_GCN_OPCODE_CLASS(SAddI32Instruction, Sop2InstructionBase, "s_add_i32");
DEFINE_RAW_GCN_OPCODE_CLASS(SAddU32Instruction, Sop2InstructionBase, "s_add_u32");
DEFINE_RAW_GCN_OPCODE_CLASS(SAddcU32Instruction, Sop2InstructionBase, "s_addc_u32");
DEFINE_RAW_GCN_OPCODE_CLASS(SLshrB32Instruction, Sop2InstructionBase, "s_lshr_b32");
DEFINE_RAW_GCN_OPCODE_CLASS(SAshrI32Instruction, Sop2InstructionBase, "s_ashr_i32");
DEFINE_RAW_GCN_OPCODE_CLASS(SLshlB64Instruction, Sop2InstructionBase, "s_lshl_b64");

DEFINE_RAW_GCN_OPCODE_CLASS(SCmpLtI32Instruction, SopcInstructionBase, "s_cmp_lt_i32");
DEFINE_RAW_GCN_OPCODE_CLASS(SCmpGtI32Instruction, SopcInstructionBase, "s_cmp_gt_i32");
DEFINE_RAW_GCN_OPCODE_CLASS(SCmpEqU32Instruction, SopcInstructionBase, "s_cmp_eq_u32");
DEFINE_RAW_GCN_OPCODE_CLASS(SCmpGtU32Instruction, SopcInstructionBase, "s_cmp_gt_u32");
DEFINE_RAW_GCN_OPCODE_CLASS(SCmpLtU32Instruction, SopcInstructionBase, "s_cmp_lt_u32");

DEFINE_RAW_GCN_OPCODE_CLASS(VNotB32Instruction, Vop1InstructionBase, "v_not_b32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VMovB32Instruction, Vop1InstructionBase, "v_mov_b32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VRndneF32Instruction, Vop1InstructionBase, "v_rndne_f32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VCvtI32F32Instruction, Vop1InstructionBase, "v_cvt_i32_f32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VCvtF32I32Instruction, Vop1InstructionBase, "v_cvt_f32_i32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VCvtF32U32Instruction, Vop1InstructionBase, "v_cvt_f32_u32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VCvtU32F32Instruction, Vop1InstructionBase, "v_cvt_u32_f32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VExpF32Instruction, Vop1InstructionBase, "v_exp_f32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VRcpF32Instruction, Vop1InstructionBase, "v_rcp_f32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VRcpIflagF32Instruction, Vop1InstructionBase, "v_rcp_iflag_f32_e32");

DEFINE_RAW_GCN_OPCODE_CLASS(VAddU32Instruction, Vop2InstructionBase, "v_add_u32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VSubU32Instruction, Vop2InstructionBase, "v_sub_u32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VAshrrevI32Instruction, Vop2InstructionBase, "v_ashrrev_i32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VLshlrevB32Instruction, Vop2InstructionBase, "v_lshlrev_b32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VAndB32Instruction, Vop2InstructionBase, "v_and_b32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VOrB32Instruction, Vop2InstructionBase, "v_or_b32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VXorB32Instruction, Vop2InstructionBase, "v_xor_b32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VSubrevU32Instruction, Vop2InstructionBase, "v_subrev_u32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VAddCoU32E32Instruction, Vop2InstructionBase, "v_add_co_u32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VAddcCoU32E32Instruction, Vop2InstructionBase, "v_addc_co_u32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VAddF32Instruction, Vop2InstructionBase, "v_add_f32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VSubF32Instruction, Vop2InstructionBase, "v_sub_f32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VMulF32Instruction, Vop2InstructionBase, "v_mul_f32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VMaxF32Instruction, Vop2InstructionBase, "v_max_f32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VMaxI32Instruction, Vop2InstructionBase, "v_max_i32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VFmacF32Instruction, Vop2InstructionBase, "v_fmac_f32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VCndmaskB32E32Instruction, Vop2InstructionBase, "v_cndmask_b32_e32");

DEFINE_RAW_GCN_OPCODE_CLASS(VLshlrevB64Instruction, Vop3aInstructionBase, "v_lshlrev_b64");
DEFINE_RAW_GCN_OPCODE_CLASS(VOr3B32Instruction, Vop3aInstructionBase, "v_or3_b32");
DEFINE_RAW_GCN_OPCODE_CLASS(VLshlAddU32Instruction, Vop3aInstructionBase, "v_lshl_add_u32");
DEFINE_RAW_GCN_OPCODE_CLASS(VFmaF32Instruction, Vop3aInstructionBase, "v_fma_f32");
DEFINE_RAW_GCN_OPCODE_CLASS(VMbcntLoInstruction, Vop3aInstructionBase, "v_mbcnt_lo_u32_b32");
DEFINE_RAW_GCN_OPCODE_CLASS(VMbcntHiInstruction, Vop3aInstructionBase, "v_mbcnt_hi_u32_b32");
DEFINE_RAW_GCN_OPCODE_CLASS(VLdexpF32Instruction, Vop3aInstructionBase, "v_ldexp_f32");
DEFINE_RAW_GCN_OPCODE_CLASS(VDivFmasF32Instruction, Vop3aInstructionBase, "v_div_fmas_f32");
DEFINE_RAW_GCN_OPCODE_CLASS(VDivFixupF32Instruction, Vop3aInstructionBase, "v_div_fixup_f32");
DEFINE_RAW_GCN_OPCODE_CLASS(VCndmaskB32E64Instruction, Vop3aInstructionBase, "v_cndmask_b32_e64");
DEFINE_RAW_GCN_OPCODE_CLASS(VCmpLtU32E64Instruction, Vop3aInstructionBase, "v_cmp_lt_u32_e64");
DEFINE_RAW_GCN_OPCODE_CLASS(VCmpGtI32E64Instruction, Vop3aInstructionBase, "v_cmp_gt_i32_e64");
DEFINE_RAW_GCN_OPCODE_CLASS(VCmpGtU32E64Instruction, Vop3aInstructionBase, "v_cmp_gt_u32_e64");
DEFINE_RAW_GCN_OPCODE_CLASS(VMulLoI32Instruction, Vop3aInstructionBase, "v_mul_lo_i32");
DEFINE_RAW_GCN_OPCODE_CLASS(VMulHiU32Instruction, Vop3aInstructionBase, "v_mul_hi_u32");

DEFINE_RAW_GCN_OPCODE_CLASS(VAddCoU32E64Instruction, Vop3bInstructionBase, "v_add_co_u32_e64");
DEFINE_RAW_GCN_OPCODE_CLASS(VAddcCoU32E64Instruction, Vop3bInstructionBase, "v_addc_co_u32_e64");
DEFINE_RAW_GCN_OPCODE_CLASS(VDivScaleF32Instruction, Vop3bInstructionBase, "v_div_scale_f32");
DEFINE_RAW_GCN_OPCODE_CLASS(VMadU64U32Instruction, Vop3bInstructionBase, "v_mad_u64_u32");

DEFINE_RAW_GCN_OPCODE_CLASS(VMfmaF32Instruction, Vop3pInstructionBase, "v_mfma_f32_16x16x4f32");
DEFINE_RAW_GCN_OPCODE_CLASS(VMfmaF16Instruction, Vop3pInstructionBase, "v_mfma_f32_16x16x4f16");
DEFINE_RAW_GCN_OPCODE_CLASS(VMfmaI8Instruction, Vop3pInstructionBase, "v_mfma_i32_16x16x4i8");
DEFINE_RAW_GCN_OPCODE_CLASS(VMfmaBf16Instruction, Vop3pInstructionBase, "v_mfma_f32_16x16x2bf16");
DEFINE_RAW_GCN_OPCODE_CLASS(VMfmaF32WideInstruction, Vop3pInstructionBase, "v_mfma_f32_32x32x2f32");
DEFINE_RAW_GCN_OPCODE_CLASS(VMfmaI8WideInstruction, Vop3pInstructionBase, "v_mfma_i32_16x16x16i8");
DEFINE_RAW_GCN_OPCODE_CLASS(VAccvgprReadB32Instruction, Vop3pInstructionBase, "v_accvgpr_read_b32");
DEFINE_RAW_GCN_OPCODE_CLASS(VAccvgprWriteB32Instruction, Vop3pInstructionBase, "v_accvgpr_write_b32");

DEFINE_RAW_GCN_OPCODE_CLASS(VCmpGtI32Instruction, VopcInstructionBase, "v_cmp_gt_i32");
DEFINE_RAW_GCN_OPCODE_CLASS(VCmpLeI32Instruction, VopcInstructionBase, "v_cmp_le_i32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VCmpLtI32Instruction, VopcInstructionBase, "v_cmp_lt_i32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VCmpLtU32Instruction, VopcInstructionBase, "v_cmp_lt_u32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VCmpLeU32Instruction, VopcInstructionBase, "v_cmp_le_u32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VCmpGtU32Instruction, VopcInstructionBase, "v_cmp_gt_u32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VCmpEqU32Instruction, VopcInstructionBase, "v_cmp_eq_u32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VCmpNgtF32Instruction, VopcInstructionBase, "v_cmp_ngt_f32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VCmpNltF32Instruction, VopcInstructionBase, "v_cmp_nlt_f32_e32");

DEFINE_RAW_GCN_OPCODE_CLASS(GlobalLoadDwordInstruction, FlatInstructionBase, "global_load_dword");
DEFINE_RAW_GCN_OPCODE_CLASS(GlobalStoreDwordInstruction, FlatInstructionBase, "global_store_dword");
DEFINE_RAW_GCN_OPCODE_CLASS(GlobalAtomicAddInstruction, FlatInstructionBase, "global_atomic_add");

DEFINE_RAW_GCN_OPCODE_CLASS(DsWriteB32Instruction, DsInstructionBase, "ds_write_b32");
DEFINE_RAW_GCN_OPCODE_CLASS(DsReadB32Instruction, DsInstructionBase, "ds_read_b32");
DEFINE_RAW_GCN_OPCODE_CLASS(DsRead2B32Instruction, DsInstructionBase, "ds_read2_b32");

#undef DEFINE_RAW_GCN_OPCODE_CLASS

InstructionObjectPtr MakePlaceholderInstruction(DecodedInstruction instruction,
                                                      std::string_view op_type_name,
                                                      std::string_view class_name) {
  static const UnsupportedInstructionHandler kUnsupportedHandler;
  return std::make_unique<PlaceholderInstructionBase>(std::move(instruction), kUnsupportedHandler,
                                                      op_type_name, class_name);
}

InstructionObjectPtr CreateScalarMemoryInstruction(const GcnIsaOpcodeDescriptor& descriptor,
                                                         DecodedInstruction instruction) {
  if (descriptor.op_type == GcnIsaOpType::Smrd || descriptor.op_type == GcnIsaOpType::Smem) {
    switch (descriptor.opcode) {
      case static_cast<uint16_t>(GcnIsaSmrdOpcode::S_LOAD_DWORD):
        return std::make_unique<SLoadDwordInstruction>(
            std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
      case static_cast<uint16_t>(GcnIsaSmrdOpcode::S_LOAD_DWORDX2):
        return std::make_unique<SLoadDwordx2Instruction>(
            std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
      case static_cast<uint16_t>(GcnIsaSmrdOpcode::S_LOAD_DWORDX4):
        return std::make_unique<SLoadDwordx4Instruction>(
            std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
      default:
        break;
    }
  }
  return MakePlaceholderInstruction(std::move(instruction), "scalar_memory",
                                    "scalar_memory_placeholder");
}

InstructionObjectPtr CreateScalarInstruction(const GcnIsaOpcodeDescriptor& descriptor,
                                                   DecodedInstruction instruction) {
  switch (descriptor.op_type) {
    case GcnIsaOpType::Sop1:
      switch (descriptor.opcode) {
        case static_cast<uint16_t>(GcnIsaSop1Opcode::S_MOV_B32):
          return std::make_unique<SMovB32Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaSop1Opcode::S_MOV_B64):
          return std::make_unique<SMovB64Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaSop1Opcode::S_ABS_I32):
          return std::make_unique<SAbsI32Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaSop1Opcode::S_SEXT_I32_I16):
          return std::make_unique<SSextI32I16Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaSop1Opcode::S_BCNT1_I32_B64):
          return std::make_unique<SBcnt1I32B64Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaSop1Opcode::S_AND_SAVEEXEC_B64):
          return std::make_unique<SAndSaveexecB64Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaSop1Opcode::S_ANDN2_SAVEEXEC_B64):
          return std::make_unique<SAndn2SaveexecB64Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        default:
          break;
      }
      return MakePlaceholderInstruction(std::move(instruction), "sop1", "sop1_placeholder");
    case GcnIsaOpType::Sop2:
      switch (descriptor.opcode) {
        case static_cast<uint16_t>(GcnIsaSop2Opcode::S_ADD_U32):
          return std::make_unique<SAddU32Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaSop2Opcode::S_SUB_I32):
          return std::make_unique<SSubI32Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaSop2Opcode::S_ADD_I32):
          return std::make_unique<SAddI32Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaSop2Opcode::S_ADDC_U32):
          return std::make_unique<SAddcU32Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaSop2Opcode::S_CSELECT_B64):
          return std::make_unique<SCselectB64Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaSop2Opcode::S_AND_B32):
          return std::make_unique<SAndB32Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaSop2Opcode::S_AND_B64):
          return std::make_unique<SAndB64Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaSop2Opcode::S_OR_B64):
          return std::make_unique<SOrB64Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaSop2Opcode::S_XOR_B64):
          return std::make_unique<SXorB64Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaSop2Opcode::S_OR_B32):
          return std::make_unique<SOrB32Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaSop2Opcode::S_ANDN2_B64):
          return std::make_unique<SAndn2B64Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaSop2Opcode::S_LSHL_B64):
          return std::make_unique<SLshlB64Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaSop2Opcode::S_LSHR_B32):
          return std::make_unique<SLshrB32Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaSop2Opcode::S_ASHR_I32):
          return std::make_unique<SAshrI32Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaSop2Opcode::S_MUL_I32):
          return std::make_unique<SMulI32Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        default:
          break;
      }
      return MakePlaceholderInstruction(std::move(instruction), "sop2", "sop2_placeholder");
    case GcnIsaOpType::Sopk:
      if (descriptor.opcode == static_cast<uint16_t>(GcnIsaSopkOpcode::S_MOVK_I32)) {
        return std::make_unique<SMovkI32Instruction>(
            std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
      }
      return MakePlaceholderInstruction(std::move(instruction), "sopk", "sopk_placeholder");
    case GcnIsaOpType::Sopc:
      switch (descriptor.opcode) {
        case static_cast<uint16_t>(GcnIsaSopcOpcode::S_CMP_GT_I32):
          return std::make_unique<SCmpGtI32Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaSopcOpcode::S_CMP_LT_I32):
          return std::make_unique<SCmpLtI32Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaSopcOpcode::S_CMP_EQ_U32):
          return std::make_unique<SCmpEqU32Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaSopcOpcode::S_CMP_GT_U32):
          return std::make_unique<SCmpGtU32Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaSopcOpcode::S_CMP_LT_U32):
          return std::make_unique<SCmpLtU32Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        default:
          break;
      }
      return MakePlaceholderInstruction(std::move(instruction), "sopc", "sopc_placeholder");
    case GcnIsaOpType::Sopp:
      switch (descriptor.opcode) {
        case static_cast<uint16_t>(GcnIsaSoppOpcode::S_NOP):
          return std::make_unique<SNopInstruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaSoppOpcode::S_ENDPGM):
          return std::make_unique<SEndpgmInstruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaSoppOpcode::S_BRANCH):
          return std::make_unique<SBranchInstruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaSoppOpcode::S_CBRANCH_SCC0):
          return std::make_unique<SCbranchScc0Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaSoppOpcode::S_CBRANCH_SCC1):
          return std::make_unique<SCbranchScc1Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaSoppOpcode::S_CBRANCH_VCCZ):
          return std::make_unique<SCbranchVcczInstruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaSoppOpcode::S_CBRANCH_EXECZ):
          return std::make_unique<SCbranchExeczInstruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaSoppOpcode::S_CBRANCH_EXECNZ):
          return std::make_unique<SCbranchExecnzInstruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaSoppOpcode::S_BARRIER):
          return std::make_unique<SBarrierInstruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaSoppOpcode::S_WAITCNT):
          return std::make_unique<SWaitcntInstruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        default:
          break;
      }
      return MakePlaceholderInstruction(std::move(instruction), "sopp", "sopp_placeholder");
    default:
      break;
  }
  return MakePlaceholderInstruction(std::move(instruction), "scalar", "scalar_placeholder");
}

InstructionObjectPtr CreateVectorInstruction(const GcnIsaOpcodeDescriptor& descriptor,
                                                   DecodedInstruction instruction) {
  if (instruction.mnemonic == "v_max_i32_e32") {
    return std::make_unique<VMaxI32Instruction>(
        std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
  }
  if (instruction.mnemonic == "v_cmp_le_u32_e32") {
    return std::make_unique<VCmpLeU32Instruction>(
        std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
  }
  switch (descriptor.op_type) {
    case GcnIsaOpType::Vop1:
      switch (descriptor.opcode) {
        case static_cast<uint16_t>(GcnIsaVop1Opcode::V_MOV_B32_E32):
          return std::make_unique<VMovB32Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaVop1Opcode::V_CVT_F32_I32_E32):
          return std::make_unique<VCvtF32I32Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaVop1Opcode::V_CVT_F32_U32_E32):
          return std::make_unique<VCvtF32U32Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaVop1Opcode::V_CVT_I32_F32_E32):
          return std::make_unique<VCvtI32F32Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaVop1Opcode::V_CVT_U32_F32_E32):
          return std::make_unique<VCvtU32F32Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaVop1Opcode::V_RNDNE_F32_E32):
          return std::make_unique<VRndneF32Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaVop1Opcode::V_EXP_F32_E32):
          return std::make_unique<VExpF32Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaVop1Opcode::V_RCP_F32_E32):
          return std::make_unique<VRcpF32Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaVop1Opcode::V_RCP_IFLAG_F32_E32):
          return std::make_unique<VRcpIflagF32Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaVop1Opcode::V_NOT_B32_E32):
          return std::make_unique<VNotB32Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        default:
          break;
      }
      return MakePlaceholderInstruction(std::move(instruction), "vop1", "vop1_placeholder");
    case GcnIsaOpType::Vop2:
      switch (descriptor.opcode) {
        case static_cast<uint16_t>(GcnIsaVop2Opcode::V_CNDMASK_B32_E32):
          return std::make_unique<VCndmaskB32E32Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaVop2Opcode::V_ADD_F32_E32):
          return std::make_unique<VAddF32Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaVop2Opcode::V_SUB_F32_E32):
          return std::make_unique<VSubF32Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaVop2Opcode::V_MUL_F32_E32):
          return std::make_unique<VMulF32Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaVop2Opcode::V_MAX_F32_E32):
          return std::make_unique<VMaxF32Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaVop2Opcode::V_MAX_I32_E32):
          return std::make_unique<VMaxI32Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaVop2Opcode::V_ASHRREV_I32_E32):
          return std::make_unique<VAshrrevI32Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaVop2Opcode::V_LSHLREV_B32_E32):
          return std::make_unique<VLshlrevB32Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaVop2Opcode::V_AND_B32_E32):
          return std::make_unique<VAndB32Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaVop2Opcode::V_OR_B32_E32):
          return std::make_unique<VOrB32Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaVop2Opcode::V_XOR_B32_E32):
          return std::make_unique<VXorB32Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaVop2Opcode::V_SUBREV_U32_E32):
          return std::make_unique<VSubrevU32Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaVop2Opcode::V_ADD_CO_U32_E32):
          return std::make_unique<VAddCoU32E32Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaVop2Opcode::V_ADDC_CO_U32_E32):
          return std::make_unique<VAddcCoU32E32Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaVop2Opcode::V_ADD_U32_E32):
          return std::make_unique<VAddU32Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaVop2Opcode::V_SUB_U32_E32):
          return std::make_unique<VSubU32Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaVop2Opcode::V_FMAC_F32_E32):
          return std::make_unique<VFmacF32Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        default:
          break;
      }
      return MakePlaceholderInstruction(std::move(instruction), "vop2", "vop2_placeholder");
    case GcnIsaOpType::Vop3a:
      switch (descriptor.opcode) {
        case static_cast<uint16_t>(GcnIsaVop3aOpcode::V_CNDMASK_B32_E64):
          return std::make_unique<VCndmaskB32E64Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaVop3aOpcode::V_CMP_LT_U32_E64):
          return std::make_unique<VCmpLtU32E64Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaVop3aOpcode::V_CMP_GT_I32_E64):
          return std::make_unique<VCmpGtI32E64Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaVop3aOpcode::V_CMP_GT_U32_E64):
          return std::make_unique<VCmpGtU32E64Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaVop3aOpcode::V_MUL_LO_I32):
          return std::make_unique<VMulLoI32Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaVop3aOpcode::V_MUL_HI_U32):
          return std::make_unique<VMulHiU32Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaVop3aOpcode::V_FMA_F32):
          return std::make_unique<VFmaF32Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaVop3aOpcode::V_DIV_FIXUP_F32):
          return std::make_unique<VDivFixupF32Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaVop3aOpcode::V_DIV_FMAS_F32):
          return std::make_unique<VDivFmasF32Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaVop3aOpcode::V_LSHL_ADD_U32):
          return std::make_unique<VLshlAddU32Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaVop3aOpcode::V_LDEXP_F32):
          return std::make_unique<VLdexpF32Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaVop3aOpcode::V_MBCNT_LO_U32_B32):
          return std::make_unique<VMbcntLoInstruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaVop3aOpcode::V_MBCNT_HI_U32_B32):
          return std::make_unique<VMbcntHiInstruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaVop3aOpcode::V_LSHLREV_B64):
          return std::make_unique<VLshlrevB64Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaVop3aOpcode::V_OR3_B32):
          return std::make_unique<VOr3B32Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        default:
          break;
      }
      return MakePlaceholderInstruction(std::move(instruction), "vop3a", "vop3a_placeholder");
    case GcnIsaOpType::Vop3b:
      switch (descriptor.opcode) {
        case static_cast<uint16_t>(GcnIsaVop3bOpcode::V_ADD_CO_U32_E64):
          return std::make_unique<VAddCoU32E64Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaVop3bOpcode::V_ADDC_CO_U32_E64):
          return std::make_unique<VAddcCoU32E64Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaVop3bOpcode::V_DIV_SCALE_F32):
          return std::make_unique<VDivScaleF32Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaVop3bOpcode::V_MAD_U64_U32):
          return std::make_unique<VMadU64U32Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        default:
          break;
      }
      return MakePlaceholderInstruction(std::move(instruction), "vop3b", "vop3b_placeholder");
    case GcnIsaOpType::Vop3p:
      switch (descriptor.opcode) {
        case static_cast<uint16_t>(GcnIsaVop3pOpcode::V_MFMA_F32_16X16X4F32):
          return std::make_unique<VMfmaF32Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaVop3pOpcode::V_MFMA_F32_16X16X4F16):
          return std::make_unique<VMfmaF16Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaVop3pOpcode::V_MFMA_I32_16X16X4I8):
          return std::make_unique<VMfmaI8Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaVop3pOpcode::V_MFMA_F32_16X16X2BF16):
          return std::make_unique<VMfmaBf16Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaVop3pOpcode::V_MFMA_F32_32X32X2F32):
          return std::make_unique<VMfmaF32WideInstruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaVop3pOpcode::V_MFMA_I32_16X16X16I8):
          return std::make_unique<VMfmaI8WideInstruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaVop3pOpcode::V_ACCVGPR_READ_B32):
          return std::make_unique<VAccvgprReadB32Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaVop3pOpcode::V_ACCVGPR_WRITE_B32):
          return std::make_unique<VAccvgprWriteB32Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        default:
          break;
      }
      return MakePlaceholderInstruction(std::move(instruction), "vop3p", "vop3p_placeholder");
    case GcnIsaOpType::Vopc:
      switch (descriptor.opcode) {
        case static_cast<uint16_t>(GcnIsaVopcOpcode::V_CMP_LT_U32_E32):
          return std::make_unique<VCmpLtU32Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaVopcOpcode::V_CMP_LE_U32_E32):
          return std::make_unique<VCmpLeU32Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaVopcOpcode::V_CMP_NGT_F32_E32):
          return std::make_unique<VCmpNgtF32Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaVopcOpcode::V_CMP_NLT_F32_E32):
          return std::make_unique<VCmpNltF32Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaVopcOpcode::V_CMP_LT_I32_E32):
          return std::make_unique<VCmpLtI32Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaVopcOpcode::V_CMP_LE_I32_E32):
          return std::make_unique<VCmpLeI32Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaVopcOpcode::V_CMP_GT_I32_E32):
          return std::make_unique<VCmpGtI32Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaVopcOpcode::V_CMP_EQ_U32_E32):
          return std::make_unique<VCmpEqU32Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        case static_cast<uint16_t>(GcnIsaVopcOpcode::V_CMP_GT_U32_E32):
          return std::make_unique<VCmpGtU32Instruction>(
              std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
        default:
          break;
      }
      return MakePlaceholderInstruction(std::move(instruction), "vopc", "vopc_placeholder");
    default:
      break;
  }
  return MakePlaceholderInstruction(std::move(instruction), "vector", "vector_placeholder");
}

InstructionObjectPtr CreateMemoryInstruction(const GcnIsaOpcodeDescriptor& descriptor,
                                                   DecodedInstruction instruction) {
  if (descriptor.op_type == GcnIsaOpType::Flat) {
    if (descriptor.opname == std::string_view("global_load_dword") ||
        descriptor.opname == std::string_view("flat_load_dword")) {
      return std::make_unique<GlobalLoadDwordInstruction>(
          std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
    }
    if (descriptor.opname == std::string_view("global_store_dword") ||
        descriptor.opname == std::string_view("flat_store_dword")) {
      return std::make_unique<GlobalStoreDwordInstruction>(
          std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
    }
    if (descriptor.opname == std::string_view("global_atomic_add") ||
        descriptor.opname == std::string_view("flat_atomic_add")) {
      return std::make_unique<GlobalAtomicAddInstruction>(
          std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
    }
    return MakePlaceholderInstruction(std::move(instruction), "flat", "flat_placeholder");
  }
  if (descriptor.op_type == GcnIsaOpType::Ds) {
    switch (descriptor.opcode) {
      case static_cast<uint16_t>(GcnIsaDsOpcode::DS_WRITE_B32):
        return std::make_unique<DsWriteB32Instruction>(
            std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
      case static_cast<uint16_t>(GcnIsaDsOpcode::DS_READ_B32):
        return std::make_unique<DsReadB32Instruction>(
            std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
      case static_cast<uint16_t>(GcnIsaDsOpcode::DS_READ2_B32):
        return std::make_unique<DsRead2B32Instruction>(
            std::move(instruction), EncodedSemanticHandlerRegistry::Get(instruction));
      default:
        break;
    }
    return MakePlaceholderInstruction(std::move(instruction), "ds", "ds_placeholder");
  }
  if (descriptor.op_type == GcnIsaOpType::Mubuf) {
    return MakePlaceholderInstruction(std::move(instruction), "mubuf", "mubuf_placeholder");
  }
  if (descriptor.op_type == GcnIsaOpType::Mtbuf) {
    return MakePlaceholderInstruction(std::move(instruction), "mtbuf", "mtbuf_placeholder");
  }
  if (descriptor.op_type == GcnIsaOpType::Mimg) {
    return MakePlaceholderInstruction(std::move(instruction), "mimg", "mimg_placeholder");
  }
  if (descriptor.op_type == GcnIsaOpType::Vintrp) {
    return MakePlaceholderInstruction(std::move(instruction), "vintrp", "vintrp_placeholder");
  }
  if (descriptor.op_type == GcnIsaOpType::Exp) {
    return MakePlaceholderInstruction(std::move(instruction), "exp", "exp_placeholder");
  }
  return MakePlaceholderInstruction(std::move(instruction), "memory", "memory_placeholder");
}

}  // namespace

InstructionObjectPtr BindEncodedInstructionObject(DecodedInstruction instruction) {
  const EncodedInstructionDescriptor descriptor = DescribeEncodedInstruction(instruction);
  if (!descriptor.known()) {
    return MakePlaceholderInstruction(std::move(instruction), descriptor.placeholder_op_type_name,
                                      descriptor.placeholder_class_name);
  }

  switch (descriptor.category) {
    case EncodedInstructionCategory::ScalarMemory:
      return CreateScalarMemoryInstruction(*descriptor.opcode_descriptor, std::move(instruction));
    case EncodedInstructionCategory::Scalar:
      return CreateScalarInstruction(*descriptor.opcode_descriptor, std::move(instruction));
    case EncodedInstructionCategory::Vector:
      return CreateVectorInstruction(*descriptor.opcode_descriptor, std::move(instruction));
    case EncodedInstructionCategory::Memory:
      return CreateMemoryInstruction(*descriptor.opcode_descriptor, std::move(instruction));
    case EncodedInstructionCategory::Unknown:
      break;
  }
  return MakePlaceholderInstruction(std::move(instruction), descriptor.placeholder_op_type_name,
                                    descriptor.placeholder_class_name);
}

}  // namespace gpu_model
