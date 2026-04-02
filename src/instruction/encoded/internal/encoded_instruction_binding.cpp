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

using InstructionFactoryFn = InstructionObjectPtr (*)(DecodedInstruction);

template <typename T>
InstructionObjectPtr MakeInstruction(DecodedInstruction instruction) {
  const auto& handler = EncodedSemanticHandlerRegistry::Get(instruction);
  return std::make_unique<T>(std::move(instruction), handler);
}

struct InstructionFactoryEntry {
  std::string_view mnemonic;
  InstructionFactoryFn factory;
};

constexpr InstructionFactoryEntry kInstructionFactories[] = {
    {"s_load_dword", &MakeInstruction<SLoadDwordInstruction>},
    {"s_load_dwordx2", &MakeInstruction<SLoadDwordx2Instruction>},
    {"s_load_dwordx4", &MakeInstruction<SLoadDwordx4Instruction>},
    {"s_mov_b32", &MakeInstruction<SMovB32Instruction>},
    {"s_mov_b64", &MakeInstruction<SMovB64Instruction>},
    {"s_abs_i32", &MakeInstruction<SAbsI32Instruction>},
    {"s_sext_i32_i16", &MakeInstruction<SSextI32I16Instruction>},
    {"s_bcnt1_i32_b64", &MakeInstruction<SBcnt1I32B64Instruction>},
    {"s_movk_i32", &MakeInstruction<SMovkI32Instruction>},
    {"s_and_saveexec_b64", &MakeInstruction<SAndSaveexecB64Instruction>},
    {"s_andn2_saveexec_b64", &MakeInstruction<SAndn2SaveexecB64Instruction>},
    {"s_cselect_b64", &MakeInstruction<SCselectB64Instruction>},
    {"s_andn2_b64", &MakeInstruction<SAndn2B64Instruction>},
    {"s_or_b64", &MakeInstruction<SOrB64Instruction>},
    {"s_xor_b64", &MakeInstruction<SXorB64Instruction>},
    {"s_or_b32", &MakeInstruction<SOrB32Instruction>},
    {"s_and_b64", &MakeInstruction<SAndB64Instruction>},
    {"s_and_b32", &MakeInstruction<SAndB32Instruction>},
    {"s_mul_i32", &MakeInstruction<SMulI32Instruction>},
    {"s_sub_i32", &MakeInstruction<SSubI32Instruction>},
    {"s_add_i32", &MakeInstruction<SAddI32Instruction>},
    {"s_add_u32", &MakeInstruction<SAddU32Instruction>},
    {"s_addc_u32", &MakeInstruction<SAddcU32Instruction>},
    {"s_lshr_b32", &MakeInstruction<SLshrB32Instruction>},
    {"s_ashr_i32", &MakeInstruction<SAshrI32Instruction>},
    {"s_lshl_b64", &MakeInstruction<SLshlB64Instruction>},
    {"s_cmp_lt_i32", &MakeInstruction<SCmpLtI32Instruction>},
    {"s_cmp_gt_i32", &MakeInstruction<SCmpGtI32Instruction>},
    {"s_cmp_eq_u32", &MakeInstruction<SCmpEqU32Instruction>},
    {"s_cmp_gt_u32", &MakeInstruction<SCmpGtU32Instruction>},
    {"s_cmp_lt_u32", &MakeInstruction<SCmpLtU32Instruction>},
    {"s_nop", &MakeInstruction<SNopInstruction>},
    {"s_endpgm", &MakeInstruction<SEndpgmInstruction>},
    {"s_branch", &MakeInstruction<SBranchInstruction>},
    {"s_cbranch_scc0", &MakeInstruction<SCbranchScc0Instruction>},
    {"s_cbranch_scc1", &MakeInstruction<SCbranchScc1Instruction>},
    {"s_cbranch_vccz", &MakeInstruction<SCbranchVcczInstruction>},
    {"s_cbranch_execz", &MakeInstruction<SCbranchExeczInstruction>},
    {"s_cbranch_execnz", &MakeInstruction<SCbranchExecnzInstruction>},
    {"s_barrier", &MakeInstruction<SBarrierInstruction>},
    {"s_waitcnt", &MakeInstruction<SWaitcntInstruction>},
    {"v_not_b32_e32", &MakeInstruction<VNotB32Instruction>},
    {"v_mov_b32_e32", &MakeInstruction<VMovB32Instruction>},
    {"v_rndne_f32_e32", &MakeInstruction<VRndneF32Instruction>},
    {"v_cvt_i32_f32_e32", &MakeInstruction<VCvtI32F32Instruction>},
    {"v_cvt_f32_i32_e32", &MakeInstruction<VCvtF32I32Instruction>},
    {"v_cvt_f32_u32_e32", &MakeInstruction<VCvtF32U32Instruction>},
    {"v_cvt_u32_f32_e32", &MakeInstruction<VCvtU32F32Instruction>},
    {"v_exp_f32_e32", &MakeInstruction<VExpF32Instruction>},
    {"v_rcp_f32_e32", &MakeInstruction<VRcpF32Instruction>},
    {"v_rcp_iflag_f32_e32", &MakeInstruction<VRcpIflagF32Instruction>},
    {"v_add_u32_e32", &MakeInstruction<VAddU32Instruction>},
    {"v_sub_u32_e32", &MakeInstruction<VSubU32Instruction>},
    {"v_ashrrev_i32_e32", &MakeInstruction<VAshrrevI32Instruction>},
    {"v_lshlrev_b32_e32", &MakeInstruction<VLshlrevB32Instruction>},
    {"v_and_b32_e32", &MakeInstruction<VAndB32Instruction>},
    {"v_or_b32_e32", &MakeInstruction<VOrB32Instruction>},
    {"v_xor_b32_e32", &MakeInstruction<VXorB32Instruction>},
    {"v_subrev_u32_e32", &MakeInstruction<VSubrevU32Instruction>},
    {"v_add_co_u32_e32", &MakeInstruction<VAddCoU32E32Instruction>},
    {"v_addc_co_u32_e32", &MakeInstruction<VAddcCoU32E32Instruction>},
    {"v_add_f32_e32", &MakeInstruction<VAddF32Instruction>},
    {"v_sub_f32_e32", &MakeInstruction<VSubF32Instruction>},
    {"v_mul_f32_e32", &MakeInstruction<VMulF32Instruction>},
    {"v_max_f32_e32", &MakeInstruction<VMaxF32Instruction>},
    {"v_max_i32_e32", &MakeInstruction<VMaxI32Instruction>},
    {"v_fmac_f32_e32", &MakeInstruction<VFmacF32Instruction>},
    {"v_cndmask_b32_e32", &MakeInstruction<VCndmaskB32E32Instruction>},
    {"v_lshlrev_b64", &MakeInstruction<VLshlrevB64Instruction>},
    {"v_or3_b32", &MakeInstruction<VOr3B32Instruction>},
    {"v_lshl_add_u32", &MakeInstruction<VLshlAddU32Instruction>},
    {"v_fma_f32", &MakeInstruction<VFmaF32Instruction>},
    {"v_mbcnt_lo_u32_b32", &MakeInstruction<VMbcntLoInstruction>},
    {"v_mbcnt_hi_u32_b32", &MakeInstruction<VMbcntHiInstruction>},
    {"v_ldexp_f32", &MakeInstruction<VLdexpF32Instruction>},
    {"v_div_fmas_f32", &MakeInstruction<VDivFmasF32Instruction>},
    {"v_div_fixup_f32", &MakeInstruction<VDivFixupF32Instruction>},
    {"v_cndmask_b32_e64", &MakeInstruction<VCndmaskB32E64Instruction>},
    {"v_cmp_lt_u32_e64", &MakeInstruction<VCmpLtU32E64Instruction>},
    {"v_cmp_gt_i32_e64", &MakeInstruction<VCmpGtI32E64Instruction>},
    {"v_cmp_gt_u32_e64", &MakeInstruction<VCmpGtU32E64Instruction>},
    {"v_mul_lo_i32", &MakeInstruction<VMulLoI32Instruction>},
    {"v_mul_hi_u32", &MakeInstruction<VMulHiU32Instruction>},
    {"v_add_co_u32_e64", &MakeInstruction<VAddCoU32E64Instruction>},
    {"v_addc_co_u32_e64", &MakeInstruction<VAddcCoU32E64Instruction>},
    {"v_div_scale_f32", &MakeInstruction<VDivScaleF32Instruction>},
    {"v_mad_u64_u32", &MakeInstruction<VMadU64U32Instruction>},
    {"v_mfma_f32_16x16x4f32", &MakeInstruction<VMfmaF32Instruction>},
    {"v_mfma_f32_16x16x4f16", &MakeInstruction<VMfmaF16Instruction>},
    {"v_mfma_i32_16x16x4i8", &MakeInstruction<VMfmaI8Instruction>},
    {"v_mfma_f32_16x16x2bf16", &MakeInstruction<VMfmaBf16Instruction>},
    {"v_mfma_f32_32x32x2f32", &MakeInstruction<VMfmaF32WideInstruction>},
    {"v_mfma_i32_16x16x16i8", &MakeInstruction<VMfmaI8WideInstruction>},
    {"v_accvgpr_read_b32", &MakeInstruction<VAccvgprReadB32Instruction>},
    {"v_accvgpr_write_b32", &MakeInstruction<VAccvgprWriteB32Instruction>},
    {"v_cmp_gt_i32_e32", &MakeInstruction<VCmpGtI32Instruction>},
    {"v_cmp_le_i32_e32", &MakeInstruction<VCmpLeI32Instruction>},
    {"v_cmp_lt_i32_e32", &MakeInstruction<VCmpLtI32Instruction>},
    {"v_cmp_lt_u32_e32", &MakeInstruction<VCmpLtU32Instruction>},
    {"v_cmp_le_u32_e32", &MakeInstruction<VCmpLeU32Instruction>},
    {"v_cmp_gt_u32_e32", &MakeInstruction<VCmpGtU32Instruction>},
    {"v_cmp_eq_u32_e32", &MakeInstruction<VCmpEqU32Instruction>},
    {"v_cmp_ngt_f32_e32", &MakeInstruction<VCmpNgtF32Instruction>},
    {"v_cmp_nlt_f32_e32", &MakeInstruction<VCmpNltF32Instruction>},
    {"global_load_dword", &MakeInstruction<GlobalLoadDwordInstruction>},
    {"global_store_dword", &MakeInstruction<GlobalStoreDwordInstruction>},
    {"global_atomic_add", &MakeInstruction<GlobalAtomicAddInstruction>},
    {"ds_write_b32", &MakeInstruction<DsWriteB32Instruction>},
    {"ds_read_b32", &MakeInstruction<DsReadB32Instruction>},
    {"ds_read2_b32", &MakeInstruction<DsRead2B32Instruction>},
};

const InstructionFactoryEntry* FindInstructionFactory(std::string_view mnemonic) {
  for (const auto& entry : kInstructionFactories) {
    if (entry.mnemonic == mnemonic) {
      return &entry;
    }
  }
  return nullptr;
}

}  // namespace

InstructionObjectPtr BindEncodedInstructionObject(DecodedInstruction instruction) {
  const auto* match = FindEncodedGcnMatchRecord(instruction.words);
  if (match == nullptr || !match->known()) {
    throw std::invalid_argument("missing encoded instruction match record: " + instruction.mnemonic);
  }
  if (const auto* factory = FindInstructionFactory(match->encoding_def->mnemonic); factory != nullptr) {
    instruction.mnemonic = std::string(match->encoding_def->mnemonic);
    return factory->factory(std::move(instruction));
  }
  throw std::invalid_argument("missing encoded instruction factory: " +
                              std::string(match->encoding_def->mnemonic));
}

}  // namespace gpu_model
