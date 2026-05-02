#include "instruction/isa/instruction_builder.h"

#include <stdexcept>

namespace gpu_model {

Operand Operand::ScalarReg(uint32_t index) {
  return Operand{.kind = OperandKind::Register,
                 .reg = RegRef{.file = RegisterFile::Scalar, .index = index}};
}

Operand Operand::VectorReg(uint32_t index) {
  return Operand{.kind = OperandKind::Register,
                 .reg = RegRef{.file = RegisterFile::Vector, .index = index}};
}

Operand Operand::ScalarRegRange(uint32_t index, uint32_t count) {
  return Operand{.kind = OperandKind::RegisterRange,
                 .reg = RegRef{.file = RegisterFile::Scalar, .index = index},
                 .reg_count = count};
}

Operand Operand::VectorRegRange(uint32_t index, uint32_t count) {
  return Operand{.kind = OperandKind::RegisterRange,
                 .reg = RegRef{.file = RegisterFile::Vector, .index = index},
                 .reg_count = count};
}

Operand Operand::ImmediateU64(uint64_t value) {
  return Operand{.kind = OperandKind::Immediate, .immediate = value};
}

Operand Operand::Argument(uint32_t index) {
  return Operand{.kind = OperandKind::ArgumentIndex, .immediate = index};
}

Operand Operand::Branch(uint64_t target_pc) {
  return Operand{.kind = OperandKind::BranchTarget, .immediate = target_pc};
}

InstructionBuilder& InstructionBuilder::SLoadArg(std::string_view dest, uint32_t arg_index) {
  return AddInstruction(
      Opcode::SysLoadArg, {ParseRegOperand(dest), Operand::Argument(arg_index)});
}

InstructionBuilder& InstructionBuilder::SysGlobalIdX(std::string_view dest) {
  return AddInstruction(Opcode::SysGlobalIdX, {ParseRegOperand(dest)});
}

InstructionBuilder& InstructionBuilder::SysGlobalIdY(std::string_view dest) {
  return AddInstruction(Opcode::SysGlobalIdY, {ParseRegOperand(dest)});
}

InstructionBuilder& InstructionBuilder::SysGlobalIdZ(std::string_view dest) {
  return AddInstruction(Opcode::SysGlobalIdZ, {ParseRegOperand(dest)});
}

InstructionBuilder& InstructionBuilder::SysLocalIdX(std::string_view dest) {
  return AddInstruction(Opcode::SysLocalIdX, {ParseRegOperand(dest)});
}

InstructionBuilder& InstructionBuilder::SysLocalIdY(std::string_view dest) {
  return AddInstruction(Opcode::SysLocalIdY, {ParseRegOperand(dest)});
}

InstructionBuilder& InstructionBuilder::SysLocalIdZ(std::string_view dest) {
  return AddInstruction(Opcode::SysLocalIdZ, {ParseRegOperand(dest)});
}

InstructionBuilder& InstructionBuilder::SysBlockOffsetX(std::string_view dest) {
  return AddInstruction(Opcode::SysBlockOffsetX, {ParseRegOperand(dest)});
}

InstructionBuilder& InstructionBuilder::SysBlockIdxX(std::string_view dest) {
  return AddInstruction(Opcode::SysBlockIdxX, {ParseRegOperand(dest)});
}

InstructionBuilder& InstructionBuilder::SysBlockIdxY(std::string_view dest) {
  return AddInstruction(Opcode::SysBlockIdxY, {ParseRegOperand(dest)});
}

InstructionBuilder& InstructionBuilder::SysBlockIdxZ(std::string_view dest) {
  return AddInstruction(Opcode::SysBlockIdxZ, {ParseRegOperand(dest)});
}

InstructionBuilder& InstructionBuilder::SysBlockDimX(std::string_view dest) {
  return AddInstruction(Opcode::SysBlockDimX, {ParseRegOperand(dest)});
}

InstructionBuilder& InstructionBuilder::SysBlockDimY(std::string_view dest) {
  return AddInstruction(Opcode::SysBlockDimY, {ParseRegOperand(dest)});
}

InstructionBuilder& InstructionBuilder::SysBlockDimZ(std::string_view dest) {
  return AddInstruction(Opcode::SysBlockDimZ, {ParseRegOperand(dest)});
}

InstructionBuilder& InstructionBuilder::SysGridDimX(std::string_view dest) {
  return AddInstruction(Opcode::SysGridDimX, {ParseRegOperand(dest)});
}

InstructionBuilder& InstructionBuilder::SysGridDimY(std::string_view dest) {
  return AddInstruction(Opcode::SysGridDimY, {ParseRegOperand(dest)});
}

InstructionBuilder& InstructionBuilder::SysGridDimZ(std::string_view dest) {
  return AddInstruction(Opcode::SysGridDimZ, {ParseRegOperand(dest)});
}

InstructionBuilder& InstructionBuilder::SysLaneId(std::string_view dest) {
  return AddInstruction(Opcode::SysLaneId, {ParseRegOperand(dest)});
}

InstructionBuilder& InstructionBuilder::SMov(std::string_view dest, std::string_view src) {
  return AddInstruction(Opcode::SMov, {ParseRegOperand(dest), ParseRegOperand(src)});
}

InstructionBuilder& InstructionBuilder::SMov(std::string_view dest, uint64_t imm) {
  return AddInstruction(Opcode::SMov, {ParseRegOperand(dest), ImmediateOperand(imm)});
}

InstructionBuilder& InstructionBuilder::SAdd(std::string_view dest,
                                             std::string_view lhs,
                                             std::string_view rhs) {
  return AddInstruction(
      Opcode::SAdd, {ParseRegOperand(dest), ParseRegOperand(lhs), ParseRegOperand(rhs)});
}

InstructionBuilder& InstructionBuilder::SAdd(std::string_view dest,
                                             std::string_view lhs,
                                             uint64_t rhs) {
  return AddInstruction(
      Opcode::SAdd, {ParseRegOperand(dest), ParseRegOperand(lhs), ImmediateOperand(rhs)});
}

InstructionBuilder& InstructionBuilder::SSub(std::string_view dest,
                                             std::string_view lhs,
                                             std::string_view rhs) {
  return AddInstruction(
      Opcode::SSub, {ParseRegOperand(dest), ParseRegOperand(lhs), ParseRegOperand(rhs)});
}

InstructionBuilder& InstructionBuilder::SSub(std::string_view dest,
                                             std::string_view lhs,
                                             uint64_t rhs) {
  return AddInstruction(
      Opcode::SSub, {ParseRegOperand(dest), ParseRegOperand(lhs), ImmediateOperand(rhs)});
}

InstructionBuilder& InstructionBuilder::SMul(std::string_view dest,
                                             std::string_view lhs,
                                             std::string_view rhs) {
  return AddInstruction(
      Opcode::SMul, {ParseRegOperand(dest), ParseRegOperand(lhs), ParseRegOperand(rhs)});
}

InstructionBuilder& InstructionBuilder::SMul(std::string_view dest,
                                             std::string_view lhs,
                                             uint64_t rhs) {
  return AddInstruction(
      Opcode::SMul, {ParseRegOperand(dest), ParseRegOperand(lhs), ImmediateOperand(rhs)});
}

InstructionBuilder& InstructionBuilder::SDiv(std::string_view dest,
                                             std::string_view lhs,
                                             std::string_view rhs) {
  return AddInstruction(
      Opcode::SDiv, {ParseRegOperand(dest), ParseRegOperand(lhs), ParseRegOperand(rhs)});
}

InstructionBuilder& InstructionBuilder::SDiv(std::string_view dest,
                                             std::string_view lhs,
                                             uint64_t rhs) {
  return AddInstruction(
      Opcode::SDiv, {ParseRegOperand(dest), ParseRegOperand(lhs), ImmediateOperand(rhs)});
}

InstructionBuilder& InstructionBuilder::SRem(std::string_view dest,
                                             std::string_view lhs,
                                             std::string_view rhs) {
  return AddInstruction(
      Opcode::SRem, {ParseRegOperand(dest), ParseRegOperand(lhs), ParseRegOperand(rhs)});
}

InstructionBuilder& InstructionBuilder::SRem(std::string_view dest,
                                             std::string_view lhs,
                                             uint64_t rhs) {
  return AddInstruction(
      Opcode::SRem, {ParseRegOperand(dest), ParseRegOperand(lhs), ImmediateOperand(rhs)});
}

InstructionBuilder& InstructionBuilder::SAnd(std::string_view dest,
                                             std::string_view lhs,
                                             std::string_view rhs) {
  return AddInstruction(
      Opcode::SAnd, {ParseRegOperand(dest), ParseRegOperand(lhs), ParseRegOperand(rhs)});
}

InstructionBuilder& InstructionBuilder::SAnd(std::string_view dest,
                                             std::string_view lhs,
                                             uint64_t rhs) {
  return AddInstruction(
      Opcode::SAnd, {ParseRegOperand(dest), ParseRegOperand(lhs), ImmediateOperand(rhs)});
}

InstructionBuilder& InstructionBuilder::SOr(std::string_view dest,
                                            std::string_view lhs,
                                            std::string_view rhs) {
  return AddInstruction(
      Opcode::SOr, {ParseRegOperand(dest), ParseRegOperand(lhs), ParseRegOperand(rhs)});
}

InstructionBuilder& InstructionBuilder::SOr(std::string_view dest,
                                            std::string_view lhs,
                                            uint64_t rhs) {
  return AddInstruction(
      Opcode::SOr, {ParseRegOperand(dest), ParseRegOperand(lhs), ImmediateOperand(rhs)});
}

InstructionBuilder& InstructionBuilder::SXor(std::string_view dest,
                                             std::string_view lhs,
                                             std::string_view rhs) {
  return AddInstruction(
      Opcode::SXor, {ParseRegOperand(dest), ParseRegOperand(lhs), ParseRegOperand(rhs)});
}

InstructionBuilder& InstructionBuilder::SXor(std::string_view dest,
                                             std::string_view lhs,
                                             uint64_t rhs) {
  return AddInstruction(
      Opcode::SXor, {ParseRegOperand(dest), ParseRegOperand(lhs), ImmediateOperand(rhs)});
}

InstructionBuilder& InstructionBuilder::SShl(std::string_view dest,
                                             std::string_view lhs,
                                             std::string_view rhs) {
  return AddInstruction(
      Opcode::SShl, {ParseRegOperand(dest), ParseRegOperand(lhs), ParseRegOperand(rhs)});
}

InstructionBuilder& InstructionBuilder::SShl(std::string_view dest,
                                             std::string_view lhs,
                                             uint64_t rhs) {
  return AddInstruction(
      Opcode::SShl, {ParseRegOperand(dest), ParseRegOperand(lhs), ImmediateOperand(rhs)});
}

InstructionBuilder& InstructionBuilder::SShr(std::string_view dest,
                                             std::string_view lhs,
                                             std::string_view rhs) {
  return AddInstruction(
      Opcode::SShr, {ParseRegOperand(dest), ParseRegOperand(lhs), ParseRegOperand(rhs)});
}

InstructionBuilder& InstructionBuilder::SShr(std::string_view dest,
                                             std::string_view lhs,
                                             uint64_t rhs) {
  return AddInstruction(
      Opcode::SShr, {ParseRegOperand(dest), ParseRegOperand(lhs), ImmediateOperand(rhs)});
}

InstructionBuilder& InstructionBuilder::SWaitCnt(uint32_t global_count,
                                                 uint32_t shared_count,
                                                 uint32_t private_count,
                                                 uint32_t scalar_buffer_count) {
  return AddInstruction(Opcode::SWaitCnt,
                        {ImmediateOperand(global_count), ImmediateOperand(shared_count),
                         ImmediateOperand(private_count), ImmediateOperand(scalar_buffer_count)});
}

InstructionBuilder& InstructionBuilder::SBufferLoadDword(std::string_view dest,
                                                         std::string_view index,
                                                         uint32_t scale_bytes,
                                                         uint32_t offset_bytes) {
  return AddInstruction(Opcode::SBufferLoadDword,
                        {ParseRegOperand(dest), ParseRegOperand(index), ImmediateOperand(scale_bytes),
                         ImmediateOperand(offset_bytes)});
}

InstructionBuilder& InstructionBuilder::SCmpLt(std::string_view lhs, std::string_view rhs) {
  return AddInstruction(Opcode::SCmpLt, {ParseRegOperand(lhs), ParseRegOperand(rhs)});
}

InstructionBuilder& InstructionBuilder::SCmpLt(std::string_view lhs, uint64_t rhs) {
  return AddInstruction(Opcode::SCmpLt, {ParseRegOperand(lhs), ImmediateOperand(rhs)});
}

InstructionBuilder& InstructionBuilder::SCmpEq(std::string_view lhs, std::string_view rhs) {
  return AddInstruction(Opcode::SCmpEq, {ParseRegOperand(lhs), ParseRegOperand(rhs)});
}

InstructionBuilder& InstructionBuilder::SCmpEq(std::string_view lhs, uint64_t rhs) {
  return AddInstruction(Opcode::SCmpEq, {ParseRegOperand(lhs), ImmediateOperand(rhs)});
}

InstructionBuilder& InstructionBuilder::SCmpGt(std::string_view lhs, std::string_view rhs) {
  return AddInstruction(Opcode::SCmpGt, {ParseRegOperand(lhs), ParseRegOperand(rhs)});
}

InstructionBuilder& InstructionBuilder::SCmpGt(std::string_view lhs, uint64_t rhs) {
  return AddInstruction(Opcode::SCmpGt, {ParseRegOperand(lhs), ImmediateOperand(rhs)});
}

InstructionBuilder& InstructionBuilder::SCmpGe(std::string_view lhs, std::string_view rhs) {
  return AddInstruction(Opcode::SCmpGe, {ParseRegOperand(lhs), ParseRegOperand(rhs)});
}

InstructionBuilder& InstructionBuilder::SCmpGe(std::string_view lhs, uint64_t rhs) {
  return AddInstruction(Opcode::SCmpGe, {ParseRegOperand(lhs), ImmediateOperand(rhs)});
}

InstructionBuilder& InstructionBuilder::VMov(std::string_view dest, std::string_view src) {
  return AddInstruction(Opcode::VMov, {ParseRegOperand(dest), ParseRegOperand(src)});
}

InstructionBuilder& InstructionBuilder::VMov(std::string_view dest, uint64_t imm) {
  return AddInstruction(Opcode::VMov, {ParseRegOperand(dest), ImmediateOperand(imm)});
}

InstructionBuilder& InstructionBuilder::VAdd(std::string_view dest,
                                             std::string_view lhs,
                                             std::string_view rhs) {
  return AddInstruction(
      Opcode::VAdd, {ParseRegOperand(dest), ParseRegOperand(lhs), ParseRegOperand(rhs)});
}

InstructionBuilder& InstructionBuilder::VAnd(std::string_view dest,
                                             std::string_view lhs,
                                             std::string_view rhs) {
  return AddInstruction(
      Opcode::VAnd, {ParseRegOperand(dest), ParseRegOperand(lhs), ParseRegOperand(rhs)});
}

InstructionBuilder& InstructionBuilder::VOr(std::string_view dest,
                                            std::string_view lhs,
                                            std::string_view rhs) {
  return AddInstruction(
      Opcode::VOr, {ParseRegOperand(dest), ParseRegOperand(lhs), ParseRegOperand(rhs)});
}

InstructionBuilder& InstructionBuilder::VXor(std::string_view dest,
                                             std::string_view lhs,
                                             std::string_view rhs) {
  return AddInstruction(
      Opcode::VXor, {ParseRegOperand(dest), ParseRegOperand(lhs), ParseRegOperand(rhs)});
}

InstructionBuilder& InstructionBuilder::VShl(std::string_view dest,
                                             std::string_view lhs,
                                             std::string_view rhs) {
  return AddInstruction(
      Opcode::VShl, {ParseRegOperand(dest), ParseRegOperand(lhs), ParseRegOperand(rhs)});
}

InstructionBuilder& InstructionBuilder::VShr(std::string_view dest,
                                             std::string_view lhs,
                                             std::string_view rhs) {
  return AddInstruction(
      Opcode::VShr, {ParseRegOperand(dest), ParseRegOperand(lhs), ParseRegOperand(rhs)});
}

InstructionBuilder& InstructionBuilder::VSub(std::string_view dest,
                                             std::string_view lhs,
                                             std::string_view rhs) {
  return AddInstruction(
      Opcode::VSub, {ParseRegOperand(dest), ParseRegOperand(lhs), ParseRegOperand(rhs)});
}

InstructionBuilder& InstructionBuilder::VDiv(std::string_view dest,
                                             std::string_view lhs,
                                             std::string_view rhs) {
  return AddInstruction(
      Opcode::VDiv, {ParseRegOperand(dest), ParseRegOperand(lhs), ParseRegOperand(rhs)});
}

InstructionBuilder& InstructionBuilder::VRem(std::string_view dest,
                                             std::string_view lhs,
                                             std::string_view rhs) {
  return AddInstruction(
      Opcode::VRem, {ParseRegOperand(dest), ParseRegOperand(lhs), ParseRegOperand(rhs)});
}

InstructionBuilder& InstructionBuilder::VMul(std::string_view dest,
                                             std::string_view lhs,
                                             std::string_view rhs) {
  return AddInstruction(
      Opcode::VMul, {ParseRegOperand(dest), ParseRegOperand(lhs), ParseRegOperand(rhs)});
}

InstructionBuilder& InstructionBuilder::VAddF32(std::string_view dest,
                                                std::string_view lhs,
                                                std::string_view rhs) {
  return AddInstruction(
      Opcode::VAddF32, {ParseRegOperand(dest), ParseRegOperand(lhs), ParseRegOperand(rhs)});
}

InstructionBuilder& InstructionBuilder::VNotB32(std::string_view dest, std::string_view src) {
  return AddInstruction(Opcode::VNotB32, {ParseRegOperand(dest), ParseRegOperand(src)});
}

InstructionBuilder& InstructionBuilder::VCvtF32I32(std::string_view dest, std::string_view src) {
  return AddInstruction(Opcode::VCvtF32I32, {ParseRegOperand(dest), ParseRegOperand(src)});
}

InstructionBuilder& InstructionBuilder::VCvtI32F32(std::string_view dest, std::string_view src) {
  return AddInstruction(Opcode::VCvtI32F32, {ParseRegOperand(dest), ParseRegOperand(src)});
}

InstructionBuilder& InstructionBuilder::VMin(std::string_view dest,
                                             std::string_view lhs,
                                             std::string_view rhs) {
  return AddInstruction(
      Opcode::VMin, {ParseRegOperand(dest), ParseRegOperand(lhs), ParseRegOperand(rhs)});
}

InstructionBuilder& InstructionBuilder::VMax(std::string_view dest,
                                             std::string_view lhs,
                                             std::string_view rhs) {
  return AddInstruction(
      Opcode::VMax, {ParseRegOperand(dest), ParseRegOperand(lhs), ParseRegOperand(rhs)});
}

InstructionBuilder& InstructionBuilder::VFma(std::string_view dest,
                                             std::string_view lhs,
                                             std::string_view rhs,
                                             std::string_view addend) {
  return AddInstruction(Opcode::VFma,
                        {ParseRegOperand(dest), ParseRegOperand(lhs), ParseRegOperand(rhs),
                         ParseRegOperand(addend)});
}

InstructionBuilder& InstructionBuilder::VMadU64U32(std::string_view dest_lo,
                                                   std::string_view sdst_lo,
                                                   std::string_view lhs,
                                                   std::string_view rhs,
                                                   std::string_view acc_lo) {
  const auto dst = ParseRegisterName(dest_lo);
  const auto sdst = ParseRegisterName(sdst_lo);
  const auto acc = ParseRegisterName(acc_lo);
  if (dst.file != RegisterFile::Vector || sdst.file != RegisterFile::Scalar ||
      acc.file != RegisterFile::Vector) {
    throw std::invalid_argument("v_mad_u64_u32 expects vdst, sdst, and acc range operands");
  }
  return AddInstruction(Opcode::VMadU64U32,
                        {Operand::VectorRegRange(dst.index, 2), Operand::ScalarRegRange(sdst.index, 2),
                         ParseRegOperand(lhs), ParseRegOperand(rhs),
                         Operand::VectorRegRange(acc.index, 2)});
}

InstructionBuilder& InstructionBuilder::VMadU32U24(std::string_view dest,
                                                   std::string_view lhs,
                                                   std::string_view rhs,
                                                   std::string_view addend) {
  return AddInstruction(Opcode::VMadU32U24,
                        {ParseRegOperand(dest), ParseRegOperand(lhs), ParseRegOperand(rhs),
                         ParseRegOperand(addend)});
}

InstructionBuilder& InstructionBuilder::VCmpLtCmask(std::string_view lhs, std::string_view rhs) {
  return AddInstruction(Opcode::VCmpLtCmask, {ParseRegOperand(lhs), ParseRegOperand(rhs)});
}

InstructionBuilder& InstructionBuilder::VCmpEqCmask(std::string_view lhs, std::string_view rhs) {
  return AddInstruction(Opcode::VCmpEqCmask, {ParseRegOperand(lhs), ParseRegOperand(rhs)});
}

InstructionBuilder& InstructionBuilder::VCmpGeCmask(std::string_view lhs, std::string_view rhs) {
  return AddInstruction(Opcode::VCmpGeCmask, {ParseRegOperand(lhs), ParseRegOperand(rhs)});
}

InstructionBuilder& InstructionBuilder::VCmpGtCmask(std::string_view lhs, std::string_view rhs) {
  return AddInstruction(Opcode::VCmpGtCmask, {ParseRegOperand(lhs), ParseRegOperand(rhs)});
}

InstructionBuilder& InstructionBuilder::VSelectCmask(std::string_view dest,
                                                     std::string_view true_value,
                                                     std::string_view false_value) {
  return AddInstruction(Opcode::VSelectCmask,
                        {ParseRegOperand(dest), ParseRegOperand(true_value),
                         ParseRegOperand(false_value)});
}

InstructionBuilder& InstructionBuilder::MLoadGlobal(std::string_view dest,
                                                    std::string_view base,
                                                    std::string_view index,
                                                    uint32_t scale_bytes,
                                                    uint32_t offset_bytes) {
  return AddInstruction(Opcode::MLoadGlobal,
                        {ParseRegOperand(dest), ParseRegOperand(base), ParseRegOperand(index),
                         ImmediateOperand(scale_bytes), ImmediateOperand(offset_bytes)});
}

InstructionBuilder& InstructionBuilder::MStoreGlobal(std::string_view base,
                                                     std::string_view index,
                                                     std::string_view src,
                                                     uint32_t scale_bytes,
                                                     uint32_t offset_bytes) {
  return AddInstruction(Opcode::MStoreGlobal,
                        {ParseRegOperand(base), ParseRegOperand(index), ParseRegOperand(src),
                         ImmediateOperand(scale_bytes), ImmediateOperand(offset_bytes)});
}

InstructionBuilder& InstructionBuilder::MAtomicAddGlobal(std::string_view base,
                                                         std::string_view index,
                                                         std::string_view src,
                                                         uint32_t scale_bytes,
                                                         uint32_t offset_bytes) {
  return AddInstruction(Opcode::MAtomicAddGlobal,
                        {ParseRegOperand(base), ParseRegOperand(index), ParseRegOperand(src),
                         ImmediateOperand(scale_bytes), ImmediateOperand(offset_bytes)});
}

InstructionBuilder& InstructionBuilder::MAtomicMaxGlobal(std::string_view base,
                                                         std::string_view index,
                                                         std::string_view src,
                                                         uint32_t scale_bytes,
                                                         uint32_t offset_bytes) {
  return AddInstruction(Opcode::MAtomicMaxGlobal,
                        {ParseRegOperand(base), ParseRegOperand(index), ParseRegOperand(src),
                         ImmediateOperand(scale_bytes), ImmediateOperand(offset_bytes)});
}

InstructionBuilder& InstructionBuilder::MAtomicMinGlobal(std::string_view base,
                                                         std::string_view index,
                                                         std::string_view src,
                                                         uint32_t scale_bytes,
                                                         uint32_t offset_bytes) {
  return AddInstruction(Opcode::MAtomicMinGlobal,
                        {ParseRegOperand(base), ParseRegOperand(index), ParseRegOperand(src),
                         ImmediateOperand(scale_bytes), ImmediateOperand(offset_bytes)});
}

InstructionBuilder& InstructionBuilder::MAtomicExchGlobal(std::string_view base,
                                                          std::string_view index,
                                                          std::string_view src,
                                                          uint32_t scale_bytes,
                                                          uint32_t offset_bytes) {
  return AddInstruction(Opcode::MAtomicExchGlobal,
                        {ParseRegOperand(base), ParseRegOperand(index), ParseRegOperand(src),
                         ImmediateOperand(scale_bytes), ImmediateOperand(offset_bytes)});
}

InstructionBuilder& InstructionBuilder::MLoadGlobalAddr(std::string_view dest,
                                                        std::string_view addr_lo,
                                                        std::string_view addr_hi,
                                                        uint32_t offset_bytes) {
  return AddInstruction(Opcode::MLoadGlobalAddr,
                        {ParseRegOperand(dest), ParseRegOperand(addr_lo), ParseRegOperand(addr_hi),
                         ImmediateOperand(offset_bytes)});
}

InstructionBuilder& InstructionBuilder::MStoreGlobalAddr(std::string_view addr_lo,
                                                         std::string_view addr_hi,
                                                         std::string_view src,
                                                         uint32_t offset_bytes) {
  return AddInstruction(Opcode::MStoreGlobalAddr,
                        {ParseRegOperand(addr_lo), ParseRegOperand(addr_hi), ParseRegOperand(src),
                         ImmediateOperand(offset_bytes)});
}

InstructionBuilder& InstructionBuilder::MLoadShared(std::string_view dest,
                                                    std::string_view index,
                                                    uint32_t scale_bytes) {
  return AddInstruction(Opcode::MLoadShared,
                        {ParseRegOperand(dest), ParseRegOperand(index),
                         ImmediateOperand(scale_bytes)});
}

InstructionBuilder& InstructionBuilder::MStoreShared(std::string_view index,
                                                     std::string_view src,
                                                     uint32_t scale_bytes) {
  return AddInstruction(Opcode::MStoreShared,
                        {ParseRegOperand(index), ParseRegOperand(src),
                         ImmediateOperand(scale_bytes)});
}

InstructionBuilder& InstructionBuilder::MAtomicAddShared(std::string_view index,
                                                         std::string_view src,
                                                         uint32_t scale_bytes) {
  return AddInstruction(Opcode::MAtomicAddShared,
                        {ParseRegOperand(index), ParseRegOperand(src),
                         ImmediateOperand(scale_bytes)});
}

InstructionBuilder& InstructionBuilder::MAtomicMaxShared(std::string_view index,
                                                         std::string_view src,
                                                         uint32_t scale_bytes) {
  return AddInstruction(Opcode::MAtomicMaxShared,
                        {ParseRegOperand(index), ParseRegOperand(src),
                         ImmediateOperand(scale_bytes)});
}

InstructionBuilder& InstructionBuilder::MAtomicMinShared(std::string_view index,
                                                         std::string_view src,
                                                         uint32_t scale_bytes) {
  return AddInstruction(Opcode::MAtomicMinShared,
                        {ParseRegOperand(index), ParseRegOperand(src),
                         ImmediateOperand(scale_bytes)});
}

InstructionBuilder& InstructionBuilder::MAtomicExchShared(std::string_view index,
                                                          std::string_view src,
                                                          uint32_t scale_bytes) {
  return AddInstruction(Opcode::MAtomicExchShared,
                        {ParseRegOperand(index), ParseRegOperand(src),
                         ImmediateOperand(scale_bytes)});
}

InstructionBuilder& InstructionBuilder::MLoadPrivate(std::string_view dest,
                                                     std::string_view index,
                                                     uint32_t scale_bytes) {
  return AddInstruction(Opcode::MLoadPrivate,
                        {ParseRegOperand(dest), ParseRegOperand(index),
                         ImmediateOperand(scale_bytes)});
}

InstructionBuilder& InstructionBuilder::MStorePrivate(std::string_view index,
                                                      std::string_view src,
                                                      uint32_t scale_bytes) {
  return AddInstruction(Opcode::MStorePrivate,
                        {ParseRegOperand(index), ParseRegOperand(src),
                         ImmediateOperand(scale_bytes)});
}

InstructionBuilder& InstructionBuilder::MLoadConst(std::string_view dest,
                                                   std::string_view index,
                                                   uint32_t scale_bytes,
                                                   uint32_t offset_bytes) {
  return AddInstruction(Opcode::MLoadConst,
                        {ParseRegOperand(dest), ParseRegOperand(index),
                         ImmediateOperand(scale_bytes), ImmediateOperand(offset_bytes)});
}

InstructionBuilder& InstructionBuilder::MaskSaveExec(std::string_view dest) {
  return AddInstruction(Opcode::MaskSaveExec, {ParseRegOperand(dest)});
}

InstructionBuilder& InstructionBuilder::MaskRestoreExec(std::string_view src) {
  return AddInstruction(Opcode::MaskRestoreExec, {ParseRegOperand(src)});
}

InstructionBuilder& InstructionBuilder::MaskAndExecCmask() {
  return AddInstruction(Opcode::MaskAndExecCmask, {});
}

InstructionBuilder& InstructionBuilder::BBranch(std::string_view label) {
  return AddBranch(Opcode::BBranch, label);
}

InstructionBuilder& InstructionBuilder::BIfSmask(std::string_view label) {
  return AddBranch(Opcode::BIfSmask, label);
}

InstructionBuilder& InstructionBuilder::BIfNoexec(std::string_view label) {
  return AddBranch(Opcode::BIfNoexec, label);
}

InstructionBuilder& InstructionBuilder::SyncWaveBarrier() {
  return AddInstruction(Opcode::SyncWaveBarrier, {});
}

InstructionBuilder& InstructionBuilder::SyncBarrier() {
  return AddInstruction(Opcode::SyncBarrier, {});
}

InstructionBuilder& InstructionBuilder::BExit() {
  return AddInstruction(Opcode::BExit, {});
}

}  // namespace gpu_model
