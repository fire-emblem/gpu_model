#include "gpu_model/exec/raw_gcn_instruction_object.h"

#include <bitset>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include "gpu_model/decode/gcn_inst_encoding_def.h"
#include "gpu_model/decode/gcn_inst_decoder.h"

namespace gpu_model {

namespace {

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

class UnsupportedInstructionHandler final : public IRawGcnSemanticHandler {
 public:
  void Execute(const DecodedGcnInstruction& instruction, RawGcnWaveContext&) const override {
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

std::pair<uint32_t, uint32_t> RequireScalarRange(const DecodedGcnOperand& operand) {
  if (operand.kind != DecodedGcnOperandKind::ScalarRegRange || operand.info.reg_count == 0) {
    throw std::invalid_argument("expected scalar register range operand");
  }
  return {operand.info.reg_first, operand.info.reg_first + operand.info.reg_count - 1};
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

class SmrdInstructionBase : public RawGcnInstructionObject {
 public:
  using RawGcnInstructionObject::RawGcnInstructionObject;
  std::string_view op_type_name() const override { return "smrd"; }
};

class Sop1InstructionBase : public RawGcnInstructionObject {
 public:
  using RawGcnInstructionObject::RawGcnInstructionObject;
  std::string_view op_type_name() const override { return "sop1"; }
};

class Sop2InstructionBase : public RawGcnInstructionObject {
 public:
  using RawGcnInstructionObject::RawGcnInstructionObject;
  std::string_view op_type_name() const override { return "sop2"; }
};

class SopkInstructionBase : public RawGcnInstructionObject {
 public:
  using RawGcnInstructionObject::RawGcnInstructionObject;
  std::string_view op_type_name() const override { return "sopk"; }
};

class SopcInstructionBase : public RawGcnInstructionObject {
 public:
  using RawGcnInstructionObject::RawGcnInstructionObject;
  std::string_view op_type_name() const override { return "sopc"; }
};

class SoppInstructionBase : public RawGcnInstructionObject {
 public:
  using RawGcnInstructionObject::RawGcnInstructionObject;
  std::string_view op_type_name() const override { return "sopp"; }
};

class Vop1InstructionBase : public RawGcnInstructionObject {
 public:
  using RawGcnInstructionObject::RawGcnInstructionObject;
  std::string_view op_type_name() const override { return "vop1"; }
};

class Vop2InstructionBase : public RawGcnInstructionObject {
 public:
  using RawGcnInstructionObject::RawGcnInstructionObject;
  std::string_view op_type_name() const override { return "vop2"; }
};

class Vop3aInstructionBase : public RawGcnInstructionObject {
 public:
  using RawGcnInstructionObject::RawGcnInstructionObject;
  std::string_view op_type_name() const override { return "vop3a"; }
};

class Vop3bInstructionBase : public RawGcnInstructionObject {
 public:
  using RawGcnInstructionObject::RawGcnInstructionObject;
  std::string_view op_type_name() const override { return "vop3b"; }
};

class Vop3pInstructionBase : public RawGcnInstructionObject {
 public:
  using RawGcnInstructionObject::RawGcnInstructionObject;
  std::string_view op_type_name() const override { return "vop3p"; }
};

class VopcInstructionBase : public RawGcnInstructionObject {
 public:
  using RawGcnInstructionObject::RawGcnInstructionObject;
  std::string_view op_type_name() const override { return "vopc"; }
};

class FlatInstructionBase : public RawGcnInstructionObject {
 public:
  using RawGcnInstructionObject::RawGcnInstructionObject;
  std::string_view op_type_name() const override { return "flat"; }
};

class DsInstructionBase : public RawGcnInstructionObject {
 public:
  using RawGcnInstructionObject::RawGcnInstructionObject;
  std::string_view op_type_name() const override { return "ds"; }
};

class PlaceholderInstructionBase : public RawGcnInstructionObject {
 public:
  PlaceholderInstructionBase(DecodedGcnInstruction instruction,
                             const IRawGcnSemanticHandler& handler,
                             std::string_view op_type_name,
                             std::string_view class_name)
      : RawGcnInstructionObject(std::move(instruction), handler),
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

  void Execute(RawGcnWaveContext& context) const override {
    const auto& instruction = decoded();
    const auto [sdst, _] = RequireScalarRange(instruction.operands.at(0));
    const uint64_t exec_before = context.wave.exec.to_ullong();
    context.wave.sgpr.Write(sdst, static_cast<uint32_t>(exec_before & 0xffffffffu));
    context.wave.sgpr.Write(sdst + 1, static_cast<uint32_t>(exec_before >> 32u));
    const uint64_t mask = ResolveScalarPair(instruction.operands.at(1), context);
    context.wave.exec = context.wave.exec & MaskFromU64(mask);
    DebugLog("pc=0x%llx s_and_saveexec_b64 before=0x%llx mask=0x%llx after=0x%llx",
             static_cast<unsigned long long>(instruction.pc),
             static_cast<unsigned long long>(exec_before),
             static_cast<unsigned long long>(mask),
             static_cast<unsigned long long>(context.wave.exec.to_ullong()));
    context.wave.pc += instruction.size_bytes;
  }
};

class SNopInstruction final : public SoppInstructionBase {
 public:
  using SoppInstructionBase::SoppInstructionBase;

  std::string_view class_name() const override { return "s_nop"; }

  void Execute(RawGcnWaveContext& context) const override {
    context.wave.pc += decoded().size_bytes;
  }
};

class SWaitcntInstruction final : public SoppInstructionBase {
 public:
  using SoppInstructionBase::SoppInstructionBase;

  std::string_view class_name() const override { return "s_waitcnt"; }

  void Execute(RawGcnWaveContext& context) const override {
    context.wave.pc += decoded().size_bytes;
  }
};

class SEndpgmInstruction final : public SoppInstructionBase {
 public:
  using SoppInstructionBase::SoppInstructionBase;

  std::string_view class_name() const override { return "s_endpgm"; }

  void Execute(RawGcnWaveContext& context) const override {
    context.wave.status = WaveStatus::Exited;
    ++context.stats.wave_exits;
  }
};

class SBarrierInstruction final : public SoppInstructionBase {
 public:
  using SoppInstructionBase::SoppInstructionBase;

  std::string_view class_name() const override { return "s_barrier"; }

  void Execute(RawGcnWaveContext& context) const override {
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

  void BranchOrAdvance(RawGcnWaveContext& context, bool take_branch) const {
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

  void Execute(RawGcnWaveContext& context) const override {
    BranchOrAdvance(context, true);
  }
};

class SCbranchScc0Instruction final : public BranchInstructionBase {
 public:
  using BranchInstructionBase::BranchInstructionBase;

  std::string_view class_name() const override { return "s_cbranch_scc0"; }

  void Execute(RawGcnWaveContext& context) const override {
    BranchOrAdvance(context, !context.wave.ScalarMaskBit0());
  }
};

class SCbranchScc1Instruction final : public BranchInstructionBase {
 public:
  using BranchInstructionBase::BranchInstructionBase;

  std::string_view class_name() const override { return "s_cbranch_scc1"; }

  void Execute(RawGcnWaveContext& context) const override {
    BranchOrAdvance(context, context.wave.ScalarMaskBit0());
  }
};

class SCbranchVcczInstruction final : public BranchInstructionBase {
 public:
  using BranchInstructionBase::BranchInstructionBase;

  std::string_view class_name() const override { return "s_cbranch_vccz"; }

  void Execute(RawGcnWaveContext& context) const override {
    BranchOrAdvance(context, context.vcc == 0);
  }
};

class SCbranchExeczInstruction final : public BranchInstructionBase {
 public:
  using BranchInstructionBase::BranchInstructionBase;

  std::string_view class_name() const override { return "s_cbranch_execz"; }

  void Execute(RawGcnWaveContext& context) const override {
    BranchOrAdvance(context, context.wave.exec.none());
  }
};

class SCbranchExecnzInstruction final : public BranchInstructionBase {
 public:
  using BranchInstructionBase::BranchInstructionBase;

  std::string_view class_name() const override { return "s_cbranch_execnz"; }

  void Execute(RawGcnWaveContext& context) const override {
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
DEFINE_RAW_GCN_OPCODE_CLASS(SBcnt1I32B64Instruction, Sop1InstructionBase, "s_bcnt1_i32_b64");
DEFINE_RAW_GCN_OPCODE_CLASS(SMovkI32Instruction, SopkInstructionBase, "s_movk_i32");
DEFINE_RAW_GCN_OPCODE_CLASS(SCselectB64Instruction, Sop2InstructionBase, "s_cselect_b64");
DEFINE_RAW_GCN_OPCODE_CLASS(SAndn2B64Instruction, Sop2InstructionBase, "s_andn2_b64");
DEFINE_RAW_GCN_OPCODE_CLASS(SOrB64Instruction, Sop2InstructionBase, "s_or_b64");
DEFINE_RAW_GCN_OPCODE_CLASS(SAndB64Instruction, Sop2InstructionBase, "s_and_b64");
DEFINE_RAW_GCN_OPCODE_CLASS(SAndB32Instruction, Sop2InstructionBase, "s_and_b32");
DEFINE_RAW_GCN_OPCODE_CLASS(SMulI32Instruction, Sop2InstructionBase, "s_mul_i32");
DEFINE_RAW_GCN_OPCODE_CLASS(SAddI32Instruction, Sop2InstructionBase, "s_add_i32");
DEFINE_RAW_GCN_OPCODE_CLASS(SAddU32Instruction, Sop2InstructionBase, "s_add_u32");
DEFINE_RAW_GCN_OPCODE_CLASS(SAddcU32Instruction, Sop2InstructionBase, "s_addc_u32");
DEFINE_RAW_GCN_OPCODE_CLASS(SLshrB32Instruction, Sop2InstructionBase, "s_lshr_b32");
DEFINE_RAW_GCN_OPCODE_CLASS(SAshrI32Instruction, Sop2InstructionBase, "s_ashr_i32");
DEFINE_RAW_GCN_OPCODE_CLASS(SLshlB64Instruction, Sop2InstructionBase, "s_lshl_b64");

DEFINE_RAW_GCN_OPCODE_CLASS(SCmpLtI32Instruction, SopcInstructionBase, "s_cmp_lt_i32");
DEFINE_RAW_GCN_OPCODE_CLASS(SCmpEqU32Instruction, SopcInstructionBase, "s_cmp_eq_u32");
DEFINE_RAW_GCN_OPCODE_CLASS(SCmpGtU32Instruction, SopcInstructionBase, "s_cmp_gt_u32");
DEFINE_RAW_GCN_OPCODE_CLASS(SCmpLtU32Instruction, SopcInstructionBase, "s_cmp_lt_u32");

DEFINE_RAW_GCN_OPCODE_CLASS(VNotB32Instruction, Vop1InstructionBase, "v_not_b32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VMovB32Instruction, Vop1InstructionBase, "v_mov_b32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VRndneF32Instruction, Vop1InstructionBase, "v_rndne_f32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VCvtI32F32Instruction, Vop1InstructionBase, "v_cvt_i32_f32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VCvtF32I32Instruction, Vop1InstructionBase, "v_cvt_f32_i32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VExpF32Instruction, Vop1InstructionBase, "v_exp_f32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VRcpF32Instruction, Vop1InstructionBase, "v_rcp_f32_e32");

DEFINE_RAW_GCN_OPCODE_CLASS(VAddU32Instruction, Vop2InstructionBase, "v_add_u32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VAshrrevI32Instruction, Vop2InstructionBase, "v_ashrrev_i32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VLshlrevB32Instruction, Vop2InstructionBase, "v_lshlrev_b32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VAddCoU32E32Instruction, Vop2InstructionBase, "v_add_co_u32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VAddcCoU32E32Instruction, Vop2InstructionBase, "v_addc_co_u32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VAddF32Instruction, Vop2InstructionBase, "v_add_f32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VSubF32Instruction, Vop2InstructionBase, "v_sub_f32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VMulF32Instruction, Vop2InstructionBase, "v_mul_f32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VMaxF32Instruction, Vop2InstructionBase, "v_max_f32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VFmacF32Instruction, Vop2InstructionBase, "v_fmac_f32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VCndmaskB32E32Instruction, Vop2InstructionBase, "v_cndmask_b32_e32");

DEFINE_RAW_GCN_OPCODE_CLASS(VLshlrevB64Instruction, Vop3aInstructionBase, "v_lshlrev_b64");
DEFINE_RAW_GCN_OPCODE_CLASS(VLshlAddU32Instruction, Vop3aInstructionBase, "v_lshl_add_u32");
DEFINE_RAW_GCN_OPCODE_CLASS(VFmaF32Instruction, Vop3aInstructionBase, "v_fma_f32");
DEFINE_RAW_GCN_OPCODE_CLASS(VMbcntLoInstruction, Vop3aInstructionBase, "v_mbcnt_lo_u32_b32");
DEFINE_RAW_GCN_OPCODE_CLASS(VMbcntHiInstruction, Vop3aInstructionBase, "v_mbcnt_hi_u32_b32");
DEFINE_RAW_GCN_OPCODE_CLASS(VLdexpF32Instruction, Vop3aInstructionBase, "v_ldexp_f32");
DEFINE_RAW_GCN_OPCODE_CLASS(VDivFmasF32Instruction, Vop3aInstructionBase, "v_div_fmas_f32");
DEFINE_RAW_GCN_OPCODE_CLASS(VDivFixupF32Instruction, Vop3aInstructionBase, "v_div_fixup_f32");
DEFINE_RAW_GCN_OPCODE_CLASS(VCndmaskB32E64Instruction, Vop3aInstructionBase, "v_cndmask_b32_e64");
DEFINE_RAW_GCN_OPCODE_CLASS(VCmpGtI32E64Instruction, Vop3aInstructionBase, "v_cmp_gt_i32_e64");

DEFINE_RAW_GCN_OPCODE_CLASS(VAddCoU32E64Instruction, Vop3bInstructionBase, "v_add_co_u32_e64");
DEFINE_RAW_GCN_OPCODE_CLASS(VAddcCoU32E64Instruction, Vop3bInstructionBase, "v_addc_co_u32_e64");
DEFINE_RAW_GCN_OPCODE_CLASS(VDivScaleF32Instruction, Vop3bInstructionBase, "v_div_scale_f32");
DEFINE_RAW_GCN_OPCODE_CLASS(VMadU64U32Instruction, Vop3bInstructionBase, "v_mad_u64_u32");

DEFINE_RAW_GCN_OPCODE_CLASS(VMfmaF32Instruction, Vop3pInstructionBase, "v_mfma_f32_16x16x4f32");

DEFINE_RAW_GCN_OPCODE_CLASS(VCmpGtI32Instruction, VopcInstructionBase, "v_cmp_gt_i32");
DEFINE_RAW_GCN_OPCODE_CLASS(VCmpLeI32Instruction, VopcInstructionBase, "v_cmp_le_i32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VCmpLtI32Instruction, VopcInstructionBase, "v_cmp_lt_i32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VCmpGtU32Instruction, VopcInstructionBase, "v_cmp_gt_u32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VCmpEqU32Instruction, VopcInstructionBase, "v_cmp_eq_u32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VCmpNgtF32Instruction, VopcInstructionBase, "v_cmp_ngt_f32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VCmpNltF32Instruction, VopcInstructionBase, "v_cmp_nlt_f32_e32");

DEFINE_RAW_GCN_OPCODE_CLASS(GlobalLoadDwordInstruction, FlatInstructionBase, "global_load_dword");
DEFINE_RAW_GCN_OPCODE_CLASS(GlobalStoreDwordInstruction, FlatInstructionBase, "global_store_dword");
DEFINE_RAW_GCN_OPCODE_CLASS(GlobalAtomicAddInstruction, FlatInstructionBase, "global_atomic_add");

DEFINE_RAW_GCN_OPCODE_CLASS(DsWriteB32Instruction, DsInstructionBase, "ds_write_b32");
DEFINE_RAW_GCN_OPCODE_CLASS(DsReadB32Instruction, DsInstructionBase, "ds_read_b32");

#undef DEFINE_RAW_GCN_OPCODE_CLASS

RawGcnInstructionObjectPtr MakePlaceholderInstruction(DecodedGcnInstruction instruction,
                                                      std::string_view op_type_name,
                                                      std::string_view class_name) {
  static const UnsupportedInstructionHandler kUnsupportedHandler;
  return std::make_unique<PlaceholderInstructionBase>(std::move(instruction), kUnsupportedHandler,
                                                      op_type_name, class_name);
}

uint32_t InstructionSizeForFormat(const std::vector<uint32_t>& words,
                                  GcnInstFormatClass format_class) {
  const uint32_t low = words.empty() ? 0u : words[0];
  switch (format_class) {
    case GcnInstFormatClass::Sopp:
    case GcnInstFormatClass::Sopk:
      return 4;
    case GcnInstFormatClass::Sop2:
    case GcnInstFormatClass::Sopc:
      return ((low & 0xffu) == 255u || ((low >> 8u) & 0xffu) == 255u) ? 8u : 4u;
    case GcnInstFormatClass::Sop1:
      return (low & 0xffu) == 255u ? 8u : 4u;
    case GcnInstFormatClass::Vop2:
    case GcnInstFormatClass::Vopc:
      return (low & 0x1ffu) == 255u ? 8u : 4u;
    case GcnInstFormatClass::Vop1:
      return (low & 0x1ffu) == 255u ? 8u : 4u;
    case GcnInstFormatClass::Smrd:
    case GcnInstFormatClass::Smem:
    case GcnInstFormatClass::Vop3a:
    case GcnInstFormatClass::Vop3b:
    case GcnInstFormatClass::Vop3p:
    case GcnInstFormatClass::Ds:
    case GcnInstFormatClass::Flat:
    case GcnInstFormatClass::Mubuf:
    case GcnInstFormatClass::Mtbuf:
    case GcnInstFormatClass::Mimg:
    case GcnInstFormatClass::Exp:
      return 8;
    case GcnInstFormatClass::Vintrp:
      return (((low >> 26u) & 0x3fu) == 0x32u) ? 4u : 8u;
    case GcnInstFormatClass::Unknown:
      break;
  }
  throw std::runtime_error("failed to determine raw instruction size");
}

std::vector<uint32_t> ReadWords(std::span<const std::byte> bytes, size_t offset, uint32_t size_bytes) {
  std::vector<uint32_t> words;
  words.reserve(size_bytes / 4);
  for (uint32_t i = 0; i < size_bytes; i += 4) {
    uint32_t word = 0;
    std::memcpy(&word, bytes.data() + offset + i, sizeof(word));
    words.push_back(word);
  }
  return words;
}

RawGcnInstructionObjectPtr CreateScalarMemoryInstruction(const GcnIsaOpcodeDescriptor& descriptor,
                                                         DecodedGcnInstruction instruction) {
  if (descriptor.op_type == GcnIsaOpType::Smrd || descriptor.op_type == GcnIsaOpType::Smem) {
    switch (descriptor.opcode) {
      case 0x0:
        return std::make_unique<SLoadDwordInstruction>(
            std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
      case 0x1:
        return std::make_unique<SLoadDwordx2Instruction>(
            std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
      case 0x2:
        return std::make_unique<SLoadDwordx4Instruction>(
            std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
      default:
        break;
    }
  }
  return MakePlaceholderInstruction(std::move(instruction), "scalar_memory", "scalar_memory_placeholder");
}

RawGcnInstructionObjectPtr CreateScalarInstruction(const GcnIsaOpcodeDescriptor& descriptor,
                                                   DecodedGcnInstruction instruction) {
  switch (descriptor.op_type) {
    case GcnIsaOpType::Sop1:
      switch (descriptor.opcode) {
        case 0x0:
          return std::make_unique<SMovB32Instruction>(
              std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
        case 0x1:
          return std::make_unique<SMovB64Instruction>(
              std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
        case 0x0d:
          return std::make_unique<SBcnt1I32B64Instruction>(
              std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
        case 0x20:
          return std::make_unique<SAndSaveexecB64Instruction>(
              std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
        default:
          break;
      }
      return MakePlaceholderInstruction(std::move(instruction), "sop1", "sop1_placeholder");
    case GcnIsaOpType::Sop2:
      switch (descriptor.opcode) {
        case 0x0:
          return std::make_unique<SAddU32Instruction>(
              std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
        case 0x2:
          return std::make_unique<SAddI32Instruction>(
              std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
        case 0x4:
          return std::make_unique<SAddcU32Instruction>(
              std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
        case 0xb:
          return std::make_unique<SCselectB64Instruction>(
              std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
        case 0xc:
          return std::make_unique<SAndB32Instruction>(
              std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
        case 0xd:
          return std::make_unique<SAndB64Instruction>(
              std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
        case 0xf:
          return std::make_unique<SOrB64Instruction>(
              std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
        case 0x13:
          return std::make_unique<SAndn2B64Instruction>(
              std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
        case 0x1d:
          return std::make_unique<SLshlB64Instruction>(
              std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
        case 0x1e:
          return std::make_unique<SLshrB32Instruction>(
              std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
        case 0x20:
          return std::make_unique<SAshrI32Instruction>(
              std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
        case 0x24:
          return std::make_unique<SMulI32Instruction>(
              std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
        default:
          break;
      }
      return MakePlaceholderInstruction(std::move(instruction), "sop2", "sop2_placeholder");
    case GcnIsaOpType::Sopk:
      if (descriptor.opcode == 0x0) {
        return std::make_unique<SMovkI32Instruction>(
            std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
      }
      return MakePlaceholderInstruction(std::move(instruction), "sopk", "sopk_placeholder");
    case GcnIsaOpType::Sopc:
      switch (descriptor.opcode) {
        case 0x4:
          return std::make_unique<SCmpLtI32Instruction>(
              std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
        case 0x6:
          return std::make_unique<SCmpEqU32Instruction>(
              std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
        case 0x8:
          return std::make_unique<SCmpGtU32Instruction>(
              std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
        case 0xa:
          return std::make_unique<SCmpLtU32Instruction>(
              std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
        default:
          break;
      }
      return MakePlaceholderInstruction(std::move(instruction), "sopc", "sopc_placeholder");
    case GcnIsaOpType::Sopp:
      switch (descriptor.opcode) {
        case 0x0:
          return std::make_unique<SNopInstruction>(
              std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
        case 0x1:
          return std::make_unique<SEndpgmInstruction>(
              std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
        case 0x2:
          return std::make_unique<SBranchInstruction>(
              std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
        case 0x4:
          return std::make_unique<SCbranchScc0Instruction>(
              std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
        case 0x5:
          return std::make_unique<SCbranchScc1Instruction>(
              std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
        case 0x6:
          return std::make_unique<SCbranchVcczInstruction>(
              std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
        case 0x8:
          return std::make_unique<SCbranchExeczInstruction>(
              std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
        case 0x9:
          return std::make_unique<SCbranchExecnzInstruction>(
              std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
        case 0xa:
          return std::make_unique<SBarrierInstruction>(
              std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
        case 0xc:
          return std::make_unique<SWaitcntInstruction>(
              std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
        default:
          break;
      }
      return MakePlaceholderInstruction(std::move(instruction), "sopp", "sopp_placeholder");
    default:
      break;
  }
  return MakePlaceholderInstruction(std::move(instruction), "scalar", "scalar_placeholder");
}

RawGcnInstructionObjectPtr CreateVectorInstruction(const GcnIsaOpcodeDescriptor& descriptor,
                                                   DecodedGcnInstruction instruction) {
  switch (descriptor.op_type) {
    case GcnIsaOpType::Vop1:
      switch (descriptor.opcode) {
        case 0x1:
          return std::make_unique<VMovB32Instruction>(
              std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
        case 0x5:
          return std::make_unique<VCvtF32I32Instruction>(
              std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
        case 0x8:
          return std::make_unique<VCvtI32F32Instruction>(
              std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
        case 0x1e:
          return std::make_unique<VRndneF32Instruction>(
              std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
        case 0x20:
          return std::make_unique<VExpF32Instruction>(
              std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
        case 0x22:
          return std::make_unique<VRcpF32Instruction>(
              std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
        case 0x2b:
          return std::make_unique<VNotB32Instruction>(
              std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
        default:
          break;
      }
      return MakePlaceholderInstruction(std::move(instruction), "vop1", "vop1_placeholder");
    case GcnIsaOpType::Vop2:
      switch (descriptor.opcode) {
        case 0x0:
          return std::make_unique<VCndmaskB32E32Instruction>(
              std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
        case 0x1:
          return std::make_unique<VAddF32Instruction>(
              std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
        case 0x2:
          return std::make_unique<VSubF32Instruction>(
              std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
        case 0x5:
          return std::make_unique<VMulF32Instruction>(
              std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
        case 0xb:
          return std::make_unique<VMaxF32Instruction>(
              std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
        case 0x11:
          return std::make_unique<VAshrrevI32Instruction>(
              std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
        case 0x12:
          return std::make_unique<VLshlrevB32Instruction>(
              std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
        case 0x19:
          return std::make_unique<VAddCoU32E32Instruction>(
              std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
        case 0x1c:
          return std::make_unique<VAddcCoU32E32Instruction>(
              std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
        case 0x34:
          return std::make_unique<VAddU32Instruction>(
              std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
        case 0x3b:
          return std::make_unique<VFmacF32Instruction>(
              std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
        default:
          break;
      }
      return MakePlaceholderInstruction(std::move(instruction), "vop2", "vop2_placeholder");
    case GcnIsaOpType::Vop3a:
      switch (descriptor.opcode) {
        case 0x100:
          return std::make_unique<VCndmaskB32E64Instruction>(
              std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
        case 0x0c4:
          return std::make_unique<VCmpGtI32E64Instruction>(
              std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
        case 0x1cb:
          return std::make_unique<VFmaF32Instruction>(
              std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
        case 0x1de:
          return std::make_unique<VDivFixupF32Instruction>(
              std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
        case 0x1e2:
          return std::make_unique<VDivFmasF32Instruction>(
              std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
        case 0x1fd:
          return std::make_unique<VLshlAddU32Instruction>(
              std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
        case 0x288:
          return std::make_unique<VLdexpF32Instruction>(
              std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
        case 0x28c:
          return std::make_unique<VMbcntLoInstruction>(
              std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
        case 0x28d:
          return std::make_unique<VMbcntHiInstruction>(
              std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
        case 0x28f:
          return std::make_unique<VLshlrevB64Instruction>(
              std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
        default:
          break;
      }
      return MakePlaceholderInstruction(std::move(instruction), "vop3a", "vop3a_placeholder");
    case GcnIsaOpType::Vop3b:
      switch (descriptor.opcode) {
        case 0x119:
          return std::make_unique<VAddCoU32E64Instruction>(
              std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
        case 0x11c:
          return std::make_unique<VAddcCoU32E64Instruction>(
              std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
        case 0x1e0:
          return std::make_unique<VDivScaleF32Instruction>(
              std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
        case 0x1e8:
          return std::make_unique<VMadU64U32Instruction>(
              std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
        default:
          break;
      }
      return MakePlaceholderInstruction(std::move(instruction), "vop3b", "vop3b_placeholder");
    case GcnIsaOpType::Vop3p:
      if (descriptor.opcode == 0x45) {
        return std::make_unique<VMfmaF32Instruction>(
            std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
      }
      return MakePlaceholderInstruction(std::move(instruction), "vop3p", "vop3p_placeholder");
    case GcnIsaOpType::Vopc:
      switch (descriptor.opcode) {
        case 0x4b:
          return std::make_unique<VCmpNgtF32Instruction>(
              std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
        case 0x4e:
          return std::make_unique<VCmpNltF32Instruction>(
              std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
        case 0xc1:
          return std::make_unique<VCmpLtI32Instruction>(
              std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
        case 0xc3:
          return std::make_unique<VCmpLeI32Instruction>(
              std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
        case 0xc4:
          return std::make_unique<VCmpGtI32Instruction>(
              std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
        case 0xca:
          return std::make_unique<VCmpEqU32Instruction>(
              std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
        case 0xcc:
          return std::make_unique<VCmpGtU32Instruction>(
              std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
        default:
          break;
      }
      return MakePlaceholderInstruction(std::move(instruction), "vopc", "vopc_placeholder");
    default:
      break;
  }
  return MakePlaceholderInstruction(std::move(instruction), "vector", "vector_placeholder");
}

RawGcnInstructionObjectPtr CreateMemoryInstruction(const GcnIsaOpcodeDescriptor& descriptor,
                                                   DecodedGcnInstruction instruction) {
  if (descriptor.op_type == GcnIsaOpType::Flat) {
    if (descriptor.opname == std::string_view("global_load_dword")) {
      return std::make_unique<GlobalLoadDwordInstruction>(
          std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
    }
    if (descriptor.opname == std::string_view("global_store_dword")) {
      return std::make_unique<GlobalStoreDwordInstruction>(
          std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
    }
    if (descriptor.opname == std::string_view("global_atomic_add")) {
      return std::make_unique<GlobalAtomicAddInstruction>(
          std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
    }
    return MakePlaceholderInstruction(std::move(instruction), "flat", "flat_placeholder");
  }
  if (descriptor.op_type == GcnIsaOpType::Ds) {
    switch (descriptor.opcode) {
      case 0x0d:
        return std::make_unique<DsWriteB32Instruction>(
            std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
      case 0x36:
        return std::make_unique<DsReadB32Instruction>(
            std::move(instruction), RawGcnSemanticHandlerRegistry::Get(instruction));
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

RawGcnInstructionObjectPtr CreateInstructionObjectImpl(DecodedGcnInstruction instruction) {
  const auto* descriptor = FindGcnFallbackOpcodeDescriptor(instruction.words);
  if (descriptor == nullptr) {
    return MakePlaceholderInstruction(std::move(instruction), "unknown", "unknown_placeholder");
  }

  switch (descriptor->op_type) {
    case GcnIsaOpType::Smrd:
    case GcnIsaOpType::Smem:
      return CreateScalarMemoryInstruction(*descriptor, std::move(instruction));
    case GcnIsaOpType::Sop1:
    case GcnIsaOpType::Sop2:
    case GcnIsaOpType::Sopk:
    case GcnIsaOpType::Sopc:
    case GcnIsaOpType::Sopp:
      return CreateScalarInstruction(*descriptor, std::move(instruction));
    case GcnIsaOpType::Vop1:
    case GcnIsaOpType::Vop2:
    case GcnIsaOpType::Vopc:
    case GcnIsaOpType::Vop3a:
    case GcnIsaOpType::Vop3b:
    case GcnIsaOpType::Vop3p:
      return CreateVectorInstruction(*descriptor, std::move(instruction));
    case GcnIsaOpType::Ds:
    case GcnIsaOpType::Flat:
    case GcnIsaOpType::Mubuf:
    case GcnIsaOpType::Mtbuf:
    case GcnIsaOpType::Mimg:
    case GcnIsaOpType::Vintrp:
    case GcnIsaOpType::Exp:
      return CreateMemoryInstruction(*descriptor, std::move(instruction));
    case GcnIsaOpType::Unknown:
      break;
  }
  return MakePlaceholderInstruction(std::move(instruction), "unknown", "unknown_placeholder");
}

}  // namespace

RawGcnInstructionObject::RawGcnInstructionObject(DecodedGcnInstruction instruction,
                                                 const IRawGcnSemanticHandler& handler)
    : instruction_(std::move(instruction)), handler_(&handler) {}

void RawGcnInstructionObject::Execute(RawGcnWaveContext& context) const {
  handler_->Execute(instruction_, context);
}

std::vector<RawGcnInstructionObjectPtr> RawGcnInstructionArrayParser::Parse(
    const std::vector<DecodedGcnInstruction>& instructions) {
  std::vector<RawGcnInstructionObjectPtr> objects;
  objects.reserve(instructions.size());
  for (const auto& instruction : instructions) {
    objects.push_back(RawGcnInstructionFactory::Create(instruction));
  }
  return objects;
}

RawGcnParsedInstructionArray RawGcnInstructionArrayParser::Parse(
    const std::vector<RawGcnInstruction>& instructions) {
  RawGcnParsedInstructionArray result;
  result.raw_instructions = instructions;
  result.decoded_instructions.reserve(instructions.size());
  for (const auto& instruction : instructions) {
    result.decoded_instructions.push_back(GcnInstDecoder{}.Decode(instruction));
  }
  result.instruction_objects = Parse(result.decoded_instructions);
  return result;
}

RawGcnParsedInstructionArray RawGcnInstructionArrayParser::Parse(std::span<const std::byte> text_bytes,
                                                                 uint64_t start_pc) {
  RawGcnParsedInstructionArray result;
  size_t offset = 0;
  while (offset < text_bytes.size()) {
    if (offset + sizeof(uint32_t) > text_bytes.size()) {
      throw std::runtime_error("raw instruction exceeds text section bounds");
    }
    uint32_t low = 0;
    std::memcpy(&low, text_bytes.data() + offset, sizeof(low));
    const auto format_class = ClassifyGcnInstFormat({low});
    const uint32_t size_bytes = InstructionSizeForFormat({low}, format_class);
    if (offset + size_bytes > text_bytes.size()) {
      throw std::runtime_error("raw instruction exceeds text section bounds");
    }

    RawGcnInstruction instruction;
    instruction.pc = start_pc + offset;
    instruction.words = ReadWords(text_bytes, offset, size_bytes);
    instruction.size_bytes = size_bytes;
    instruction.format_class = format_class;
    if (const auto* def = FindGcnInstEncodingDef(instruction.words)) {
      instruction.encoding_id = def->id;
      instruction.mnemonic = std::string(def->mnemonic);
    } else {
      instruction.mnemonic = std::string(LookupGcnOpcodeName(instruction.words));
    }
    DecodeGcnOperands(instruction);
    result.raw_instructions.push_back(instruction);
    offset += size_bytes;
  }
  result.decoded_instructions.reserve(result.raw_instructions.size());
  for (const auto& instruction : result.raw_instructions) {
    result.decoded_instructions.push_back(GcnInstDecoder{}.Decode(instruction));
  }
  result.instruction_objects = Parse(result.decoded_instructions);
  return result;
}

RawGcnInstructionObjectPtr RawGcnInstructionFactory::Create(DecodedGcnInstruction instruction) {
  return CreateInstructionObjectImpl(std::move(instruction));
}

}  // namespace gpu_model
