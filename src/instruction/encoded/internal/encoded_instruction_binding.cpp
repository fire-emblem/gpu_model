#include "gpu_model/instruction/encoded/internal/encoded_instruction_binding.h"

#include <stdexcept>
#include <string>
#include <unordered_map>

#include "gpu_model/instruction/encoded/internal/encoded_gcn_encoding_def.h"
#include "gpu_model/instruction/encoded/internal/encoded_instruction_descriptor.h"
#include "gpu_model/execution/encoded_semantic_handler.h"
#include "gpu_model/execution/internal/encoded_handler_utils.h"
#include "gpu_model/execution/internal/float_utils.h"

namespace gpu_model {

namespace {

// Note: DebugEnabled/EncodedDebugLog, MaskFromU64, BranchTarget, RequireScalarRange,
// RequireScalarIndex, RequireVectorIndex, ResolveScalarPair are now provided by
// gpu_model/execution/internal/encoded_handler_utils.h

std::pair<std::string_view, std::string_view> PlaceholderNamesForFormatClass(
    EncodedGcnInstFormatClass format_class) {
  const auto op_type_name = ToString(format_class);
  switch (format_class) {
    case EncodedGcnInstFormatClass::Mimg:
      return {"mimg", "mimg_placeholder"};
    case EncodedGcnInstFormatClass::Exp:
      return {"exp", "exp_placeholder"};
    case EncodedGcnInstFormatClass::Smrd:
      return {"smrd", "smrd_placeholder"};
    case EncodedGcnInstFormatClass::Smem:
      return {"smem", "smem_placeholder"};
    case EncodedGcnInstFormatClass::Sop1:
      return {"sop1", "sop1_placeholder"};
    case EncodedGcnInstFormatClass::Sop2:
      return {"sop2", "sop2_placeholder"};
    case EncodedGcnInstFormatClass::Sopc:
      return {"sopc", "sopc_placeholder"};
    case EncodedGcnInstFormatClass::Sopp:
      return {"sopp", "sopp_placeholder"};
    case EncodedGcnInstFormatClass::Sopk:
      return {"sopk", "sopk_placeholder"};
    case EncodedGcnInstFormatClass::Vop1:
      return {"vop1", "vop1_placeholder"};
    case EncodedGcnInstFormatClass::Vop2:
      return {"vop2", "vop2_placeholder"};
    case EncodedGcnInstFormatClass::Vop3a:
      return {"vop3a", "vop3a_placeholder"};
    case EncodedGcnInstFormatClass::Vop3b:
      return {"vop3b", "vop3b_placeholder"};
    case EncodedGcnInstFormatClass::Vop3p:
      return {"vop3p", "vop3p_placeholder"};
    case EncodedGcnInstFormatClass::Vopc:
      return {"vopc", "vopc_placeholder"};
    case EncodedGcnInstFormatClass::Vintrp:
      return {"vintrp", "vintrp_placeholder"};
    case EncodedGcnInstFormatClass::Ds:
      return {"ds", "ds_placeholder"};
    case EncodedGcnInstFormatClass::Flat:
      return {"flat", "flat_placeholder"};
    case EncodedGcnInstFormatClass::Mubuf:
      return {"mubuf", "mubuf_placeholder"};
    case EncodedGcnInstFormatClass::Mtbuf:
      return {"mtbuf", "mtbuf_placeholder"};
    case EncodedGcnInstFormatClass::Unknown:
      return {"unknown", "unknown_placeholder"};
  }
  return {op_type_name, "unknown_placeholder"};
}

class UnsupportedInstructionHandler final : public IEncodedSemanticHandler {
 public:
  void Execute(const DecodedInstruction& instruction, EncodedWaveContext&) const override {
    throw std::invalid_argument("unsupported instantiated raw GCN opcode: " + instruction.mnemonic);
  }
};

class EncodedInstructionObject : public InstructionObject {
 public:
  using InstructionObject::InstructionObject;
  // Note: Execute() removed - actual execution goes through EncodedSemanticHandlerRegistry
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
};

class SAndn2SaveexecB64Instruction final : public Sop1InstructionBase {
 public:
  using Sop1InstructionBase::Sop1InstructionBase;

  std::string_view class_name() const override { return "s_andn2_saveexec_b64"; }
};

class SNopInstruction final : public SoppInstructionBase {
 public:
  using SoppInstructionBase::SoppInstructionBase;

  std::string_view class_name() const override { return "s_nop"; }
};

class SWaitcntInstruction final : public SoppInstructionBase {
 public:
  using SoppInstructionBase::SoppInstructionBase;

  std::string_view class_name() const override { return "s_waitcnt"; }
};

class SEndpgmInstruction final : public SoppInstructionBase {
 public:
  using SoppInstructionBase::SoppInstructionBase;

  std::string_view class_name() const override { return "s_endpgm"; }
};

class SBarrierInstruction final : public SoppInstructionBase {
 public:
  using SoppInstructionBase::SoppInstructionBase;

  std::string_view class_name() const override { return "s_barrier"; }
};

// Note: BranchInstructionBase removed - branch execution is handled by EncodedSemanticHandlerRegistry

class SBranchInstruction final : public SoppInstructionBase {
 public:
  using SoppInstructionBase::SoppInstructionBase;
  std::string_view class_name() const override { return "s_branch"; }
};

class SCbranchScc0Instruction final : public SoppInstructionBase {
 public:
  using SoppInstructionBase::SoppInstructionBase;
  std::string_view class_name() const override { return "s_cbranch_scc0"; }
};

class SCbranchScc1Instruction final : public SoppInstructionBase {
 public:
  using SoppInstructionBase::SoppInstructionBase;
  std::string_view class_name() const override { return "s_cbranch_scc1"; }
};

class SCbranchVcczInstruction final : public SoppInstructionBase {
 public:
  using SoppInstructionBase::SoppInstructionBase;
  std::string_view class_name() const override { return "s_cbranch_vccz"; }
};

class SCbranchExeczInstruction final : public SoppInstructionBase {
 public:
  using SoppInstructionBase::SoppInstructionBase;
  std::string_view class_name() const override { return "s_cbranch_execz"; }
};

class SCbranchExecnzInstruction final : public SoppInstructionBase {
 public:
  using SoppInstructionBase::SoppInstructionBase;
  std::string_view class_name() const override { return "s_cbranch_execnz"; }
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
DEFINE_RAW_GCN_OPCODE_CLASS(SLshlB32Instruction, Sop2InstructionBase, "s_lshl_b32");
DEFINE_RAW_GCN_OPCODE_CLASS(SLshrB32Instruction, Sop2InstructionBase, "s_lshr_b32");
DEFINE_RAW_GCN_OPCODE_CLASS(SAshrI32Instruction, Sop2InstructionBase, "s_ashr_i32");
DEFINE_RAW_GCN_OPCODE_CLASS(SLshlB64Instruction, Sop2InstructionBase, "s_lshl_b64");

DEFINE_RAW_GCN_OPCODE_CLASS(SCmpLtI32Instruction, SopcInstructionBase, "s_cmp_lt_i32");
DEFINE_RAW_GCN_OPCODE_CLASS(SCmpGtI32Instruction, SopcInstructionBase, "s_cmp_gt_i32");
DEFINE_RAW_GCN_OPCODE_CLASS(SCmpEqU32Instruction, SopcInstructionBase, "s_cmp_eq_u32");
DEFINE_RAW_GCN_OPCODE_CLASS(SCmpLgU32Instruction, SopcInstructionBase, "s_cmp_lg_u32");
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
DEFINE_RAW_GCN_OPCODE_CLASS(VPkMovB32Instruction, Vop3pInstructionBase, "v_pk_mov_b32");
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

// Factory registry using unordered_map for O(1) lookup
const std::unordered_map<std::string_view, InstructionFactoryFn>& GetInstructionFactoryMap() {
  static const std::unordered_map<std::string_view, InstructionFactoryFn> kFactoryMap = {
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
      {"s_lshl_b32", &MakeInstruction<SLshlB32Instruction>},
      {"s_lshr_b32", &MakeInstruction<SLshrB32Instruction>},
      {"s_ashr_i32", &MakeInstruction<SAshrI32Instruction>},
      {"s_lshl_b64", &MakeInstruction<SLshlB64Instruction>},
      {"s_cmp_lt_i32", &MakeInstruction<SCmpLtI32Instruction>},
      {"s_cmp_gt_i32", &MakeInstruction<SCmpGtI32Instruction>},
      {"s_cmp_eq_u32", &MakeInstruction<SCmpEqU32Instruction>},
      {"s_cmp_lg_u32", &MakeInstruction<SCmpLgU32Instruction>},
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
      {"v_pk_mov_b32", &MakeInstruction<VPkMovB32Instruction>},
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
  return kFactoryMap;
}

InstructionFactoryFn FindInstructionFactory(std::string_view mnemonic) {
  const auto& map = GetInstructionFactoryMap();
  const auto it = map.find(mnemonic);
  return it != map.end() ? it->second : nullptr;
}

}  // namespace

InstructionObjectPtr BindEncodedInstructionObject(DecodedInstruction instruction) {
  const auto* match = FindEncodedGcnMatchRecord(instruction.words);
  if (match == nullptr || !match->known()) {
    const auto [op_type_name, class_name] =
        PlaceholderNamesForFormatClass(instruction.format_class);
    EncodedDebugLog("BindEncodedInstruction: pc=0x%llx placeholder class=%s",
                    static_cast<unsigned long long>(instruction.pc),
                    class_name.data());
    return MakePlaceholderInstruction(std::move(instruction), op_type_name, class_name);
  }
  if (const auto factory = FindInstructionFactory(match->encoding_def->mnemonic); factory != nullptr) {
    instruction.mnemonic = std::string(match->encoding_def->mnemonic);
    EncodedDebugLog("BindEncodedInstruction: pc=0x%llx mnemonic=%s",
                    static_cast<unsigned long long>(instruction.pc),
                    instruction.mnemonic.c_str());
    return factory(std::move(instruction));
  }
  const auto desc = DescribeEncodedInstruction(instruction);
  EncodedDebugLog("BindEncodedInstruction: pc=0x%llx fallback placeholder class=%s",
                  static_cast<unsigned long long>(instruction.pc),
                  desc.placeholder_class_name.data());
  return MakePlaceholderInstruction(std::move(instruction),
                                    desc.placeholder_op_type_name,
                                    desc.placeholder_class_name);
}

}  // namespace gpu_model
