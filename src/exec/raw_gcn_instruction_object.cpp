#include "gpu_model/exec/raw_gcn_instruction_object.h"

#include <stdexcept>
#include <string>

#include "gpu_model/decode/gcn_inst_encoding_def.h"

namespace gpu_model {

namespace {

class UnsupportedInstructionHandler final : public IRawGcnSemanticHandler {
 public:
  void Execute(const DecodedGcnInstruction& instruction, RawGcnWaveContext&) const override {
    throw std::invalid_argument("unsupported instantiated raw GCN opcode: " + instruction.mnemonic);
  }
};

class ScalarMemoryInstructionBase : public RawGcnInstructionObject {
 public:
  using RawGcnInstructionObject::RawGcnInstructionObject;
  std::string_view op_type_name() const override { return "scalar_memory"; }
};

class ScalarAluInstructionBase : public RawGcnInstructionObject {
 public:
  using RawGcnInstructionObject::RawGcnInstructionObject;
  std::string_view op_type_name() const override { return "scalar_alu"; }
};

class ScalarCompareInstructionBase : public RawGcnInstructionObject {
 public:
  using RawGcnInstructionObject::RawGcnInstructionObject;
  std::string_view op_type_name() const override { return "scalar_compare"; }
};

class VectorAluInstructionBase : public RawGcnInstructionObject {
 public:
  using RawGcnInstructionObject::RawGcnInstructionObject;
  std::string_view op_type_name() const override { return "vector_alu"; }
};

class VectorCompareInstructionBase : public RawGcnInstructionObject {
 public:
  using RawGcnInstructionObject::RawGcnInstructionObject;
  std::string_view op_type_name() const override { return "vector_compare"; }
};

class FlatMemoryInstructionBase : public RawGcnInstructionObject {
 public:
  using RawGcnInstructionObject::RawGcnInstructionObject;
  std::string_view op_type_name() const override { return "vector_memory"; }
};

class SharedMemoryInstructionBase : public RawGcnInstructionObject {
 public:
  using RawGcnInstructionObject::RawGcnInstructionObject;
  std::string_view op_type_name() const override { return "lds"; }
};

class BranchInstructionBase : public RawGcnInstructionObject {
 public:
  using RawGcnInstructionObject::RawGcnInstructionObject;
  std::string_view op_type_name() const override { return "branch"; }
};

class SpecialInstructionBase : public RawGcnInstructionObject {
 public:
  using RawGcnInstructionObject::RawGcnInstructionObject;
  std::string_view op_type_name() const override { return "special"; }
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

#define DEFINE_RAW_GCN_OPCODE_CLASS(ClassName, BaseClass, TextName) \
  class ClassName final : public BaseClass {                        \
   public:                                                          \
    using BaseClass::BaseClass;                                     \
    std::string_view class_name() const override { return TextName; } \
  }

DEFINE_RAW_GCN_OPCODE_CLASS(SLoadDwordInstruction, ScalarMemoryInstructionBase, "s_load_dword");
DEFINE_RAW_GCN_OPCODE_CLASS(SLoadDwordx2Instruction, ScalarMemoryInstructionBase, "s_load_dwordx2");
DEFINE_RAW_GCN_OPCODE_CLASS(SLoadDwordx4Instruction, ScalarMemoryInstructionBase, "s_load_dwordx4");

DEFINE_RAW_GCN_OPCODE_CLASS(SAndSaveexecB64Instruction, ScalarAluInstructionBase, "s_and_saveexec_b64");
DEFINE_RAW_GCN_OPCODE_CLASS(SMovB32Instruction, ScalarAluInstructionBase, "s_mov_b32");
DEFINE_RAW_GCN_OPCODE_CLASS(SMovB64Instruction, ScalarAluInstructionBase, "s_mov_b64");
DEFINE_RAW_GCN_OPCODE_CLASS(SBcnt1I32B64Instruction, ScalarAluInstructionBase, "s_bcnt1_i32_b64");
DEFINE_RAW_GCN_OPCODE_CLASS(SMovkI32Instruction, ScalarAluInstructionBase, "s_movk_i32");
DEFINE_RAW_GCN_OPCODE_CLASS(SCselectB64Instruction, ScalarAluInstructionBase, "s_cselect_b64");
DEFINE_RAW_GCN_OPCODE_CLASS(SAndn2B64Instruction, ScalarAluInstructionBase, "s_andn2_b64");
DEFINE_RAW_GCN_OPCODE_CLASS(SOrB64Instruction, ScalarAluInstructionBase, "s_or_b64");
DEFINE_RAW_GCN_OPCODE_CLASS(SAndB64Instruction, ScalarAluInstructionBase, "s_and_b64");
DEFINE_RAW_GCN_OPCODE_CLASS(SAndB32Instruction, ScalarAluInstructionBase, "s_and_b32");
DEFINE_RAW_GCN_OPCODE_CLASS(SMulI32Instruction, ScalarAluInstructionBase, "s_mul_i32");
DEFINE_RAW_GCN_OPCODE_CLASS(SAddI32Instruction, ScalarAluInstructionBase, "s_add_i32");
DEFINE_RAW_GCN_OPCODE_CLASS(SAddU32Instruction, ScalarAluInstructionBase, "s_add_u32");
DEFINE_RAW_GCN_OPCODE_CLASS(SAddcU32Instruction, ScalarAluInstructionBase, "s_addc_u32");
DEFINE_RAW_GCN_OPCODE_CLASS(SLshrB32Instruction, ScalarAluInstructionBase, "s_lshr_b32");
DEFINE_RAW_GCN_OPCODE_CLASS(SAshrI32Instruction, ScalarAluInstructionBase, "s_ashr_i32");
DEFINE_RAW_GCN_OPCODE_CLASS(SLshlB64Instruction, ScalarAluInstructionBase, "s_lshl_b64");

DEFINE_RAW_GCN_OPCODE_CLASS(SCmpLtI32Instruction, ScalarCompareInstructionBase, "s_cmp_lt_i32");
DEFINE_RAW_GCN_OPCODE_CLASS(SCmpEqU32Instruction, ScalarCompareInstructionBase, "s_cmp_eq_u32");
DEFINE_RAW_GCN_OPCODE_CLASS(SCmpGtU32Instruction, ScalarCompareInstructionBase, "s_cmp_gt_u32");
DEFINE_RAW_GCN_OPCODE_CLASS(SCmpLtU32Instruction, ScalarCompareInstructionBase, "s_cmp_lt_u32");

DEFINE_RAW_GCN_OPCODE_CLASS(VNotB32Instruction, VectorAluInstructionBase, "v_not_b32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VAddU32Instruction, VectorAluInstructionBase, "v_add_u32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VAshrrevI32Instruction, VectorAluInstructionBase, "v_ashrrev_i32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VLshlrevB64Instruction, VectorAluInstructionBase, "v_lshlrev_b64");
DEFINE_RAW_GCN_OPCODE_CLASS(VMovB32Instruction, VectorAluInstructionBase, "v_mov_b32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VLshlrevB32Instruction, VectorAluInstructionBase, "v_lshlrev_b32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VLshlAddU32Instruction, VectorAluInstructionBase, "v_lshl_add_u32");
DEFINE_RAW_GCN_OPCODE_CLASS(VAddCoU32E32Instruction, VectorAluInstructionBase, "v_add_co_u32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VAddcCoU32E32Instruction, VectorAluInstructionBase, "v_addc_co_u32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VAddCoU32E64Instruction, VectorAluInstructionBase, "v_add_co_u32_e64");
DEFINE_RAW_GCN_OPCODE_CLASS(VAddcCoU32E64Instruction, VectorAluInstructionBase, "v_addc_co_u32_e64");
DEFINE_RAW_GCN_OPCODE_CLASS(VAddF32Instruction, VectorAluInstructionBase, "v_add_f32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VSubF32Instruction, VectorAluInstructionBase, "v_sub_f32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VMulF32Instruction, VectorAluInstructionBase, "v_mul_f32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VMaxF32Instruction, VectorAluInstructionBase, "v_max_f32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VFmacF32Instruction, VectorAluInstructionBase, "v_fmac_f32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VFmaF32Instruction, VectorAluInstructionBase, "v_fma_f32");
DEFINE_RAW_GCN_OPCODE_CLASS(VMfmaF32Instruction, VectorAluInstructionBase, "v_mfma_f32_16x16x4f32");
DEFINE_RAW_GCN_OPCODE_CLASS(VRndneF32Instruction, VectorAluInstructionBase, "v_rndne_f32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VCvtI32F32Instruction, VectorAluInstructionBase, "v_cvt_i32_f32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VCvtF32I32Instruction, VectorAluInstructionBase, "v_cvt_f32_i32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VMbcntLoInstruction, VectorAluInstructionBase, "v_mbcnt_lo_u32_b32");
DEFINE_RAW_GCN_OPCODE_CLASS(VMbcntHiInstruction, VectorAluInstructionBase, "v_mbcnt_hi_u32_b32");
DEFINE_RAW_GCN_OPCODE_CLASS(VExpF32Instruction, VectorAluInstructionBase, "v_exp_f32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VRcpF32Instruction, VectorAluInstructionBase, "v_rcp_f32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VLdexpF32Instruction, VectorAluInstructionBase, "v_ldexp_f32");
DEFINE_RAW_GCN_OPCODE_CLASS(VDivScaleF32Instruction, VectorAluInstructionBase, "v_div_scale_f32");
DEFINE_RAW_GCN_OPCODE_CLASS(VDivFmasF32Instruction, VectorAluInstructionBase, "v_div_fmas_f32");
DEFINE_RAW_GCN_OPCODE_CLASS(VDivFixupF32Instruction, VectorAluInstructionBase, "v_div_fixup_f32");
DEFINE_RAW_GCN_OPCODE_CLASS(VCndmaskB32E32Instruction, VectorAluInstructionBase, "v_cndmask_b32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VCndmaskB32E64Instruction, VectorAluInstructionBase, "v_cndmask_b32_e64");
DEFINE_RAW_GCN_OPCODE_CLASS(VMadU64U32Instruction, VectorAluInstructionBase, "v_mad_u64_u32");

DEFINE_RAW_GCN_OPCODE_CLASS(VCmpGtI32Instruction, VectorCompareInstructionBase, "v_cmp_gt_i32");
DEFINE_RAW_GCN_OPCODE_CLASS(VCmpLeI32Instruction, VectorCompareInstructionBase, "v_cmp_le_i32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VCmpLtI32Instruction, VectorCompareInstructionBase, "v_cmp_lt_i32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VCmpGtU32Instruction, VectorCompareInstructionBase, "v_cmp_gt_u32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VCmpEqU32Instruction, VectorCompareInstructionBase, "v_cmp_eq_u32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VCmpNgtF32Instruction, VectorCompareInstructionBase, "v_cmp_ngt_f32_e32");
DEFINE_RAW_GCN_OPCODE_CLASS(VCmpNltF32Instruction, VectorCompareInstructionBase, "v_cmp_nlt_f32_e32");

DEFINE_RAW_GCN_OPCODE_CLASS(GlobalLoadDwordInstruction, FlatMemoryInstructionBase, "global_load_dword");
DEFINE_RAW_GCN_OPCODE_CLASS(GlobalStoreDwordInstruction, FlatMemoryInstructionBase, "global_store_dword");
DEFINE_RAW_GCN_OPCODE_CLASS(GlobalAtomicAddInstruction, FlatMemoryInstructionBase, "global_atomic_add");

DEFINE_RAW_GCN_OPCODE_CLASS(DsWriteB32Instruction, SharedMemoryInstructionBase, "ds_write_b32");
DEFINE_RAW_GCN_OPCODE_CLASS(DsReadB32Instruction, SharedMemoryInstructionBase, "ds_read_b32");

DEFINE_RAW_GCN_OPCODE_CLASS(SAndSaveexecMaskInstruction, SpecialInstructionBase, "s_and_saveexec_b64");
DEFINE_RAW_GCN_OPCODE_CLASS(SBarrierInstruction, SpecialInstructionBase, "s_barrier");
DEFINE_RAW_GCN_OPCODE_CLASS(SNopInstruction, SpecialInstructionBase, "s_nop");
DEFINE_RAW_GCN_OPCODE_CLASS(SWaitcntInstruction, SpecialInstructionBase, "s_waitcnt");
DEFINE_RAW_GCN_OPCODE_CLASS(SEndpgmInstruction, SpecialInstructionBase, "s_endpgm");

DEFINE_RAW_GCN_OPCODE_CLASS(SBranchInstruction, BranchInstructionBase, "s_branch");
DEFINE_RAW_GCN_OPCODE_CLASS(SCbranchScc0Instruction, BranchInstructionBase, "s_cbranch_scc0");
DEFINE_RAW_GCN_OPCODE_CLASS(SCbranchScc1Instruction, BranchInstructionBase, "s_cbranch_scc1");
DEFINE_RAW_GCN_OPCODE_CLASS(SCbranchVcczInstruction, BranchInstructionBase, "s_cbranch_vccz");
DEFINE_RAW_GCN_OPCODE_CLASS(SCbranchExeczInstruction, BranchInstructionBase, "s_cbranch_execz");
DEFINE_RAW_GCN_OPCODE_CLASS(SCbranchExecnzInstruction, BranchInstructionBase, "s_cbranch_execnz");

#undef DEFINE_RAW_GCN_OPCODE_CLASS

RawGcnInstructionObjectPtr MakePlaceholderInstruction(DecodedGcnInstruction instruction,
                                                      std::string_view op_type_name,
                                                      std::string_view class_name) {
  static const UnsupportedInstructionHandler kUnsupportedHandler;
  return std::make_unique<PlaceholderInstructionBase>(std::move(instruction), kUnsupportedHandler,
                                                      op_type_name, class_name);
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

RawGcnInstructionObjectPtr CreateInstructionObject(DecodedGcnInstruction instruction) {
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
    objects.push_back(CreateInstructionObject(instruction));
  }
  return objects;
}

}  // namespace gpu_model
