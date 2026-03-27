#include "gpu_model/isa/instruction_builder.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace gpu_model {

Operand Operand::ScalarReg(uint32_t index) {
  return Operand{.kind = OperandKind::Register,
                 .reg = RegRef{.file = RegisterFile::Scalar, .index = index}};
}

Operand Operand::VectorReg(uint32_t index) {
  return Operand{.kind = OperandKind::Register,
                 .reg = RegRef{.file = RegisterFile::Vector, .index = index}};
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

RegRef ParseRegisterName(std::string_view text) {
  if (text.size() < 2) {
    throw std::invalid_argument("register name must have at least 2 characters");
  }

  const char prefix = text.front();
  const auto number = std::stoul(std::string(text.substr(1)));

  switch (prefix) {
    case 's':
      return RegRef{.file = RegisterFile::Scalar, .index = static_cast<uint32_t>(number)};
    case 'v':
      return RegRef{.file = RegisterFile::Vector, .index = static_cast<uint32_t>(number)};
    default:
      throw std::invalid_argument("unsupported register prefix");
  }
}

InstructionBuilder& InstructionBuilder::SetNextDebugLoc(std::string file, uint32_t line) {
  next_debug_loc_.file = std::move(file);
  next_debug_loc_.line = line;
  has_next_debug_loc_ = true;
  return *this;
}

InstructionBuilder& InstructionBuilder::Label(std::string name) {
  labels_[name] = instructions_.size();
  next_debug_loc_.label = std::move(name);
  has_next_debug_loc_ = true;
  return *this;
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

InstructionBuilder& InstructionBuilder::SysLocalIdX(std::string_view dest) {
  return AddInstruction(Opcode::SysLocalIdX, {ParseRegOperand(dest)});
}

InstructionBuilder& InstructionBuilder::SysLocalIdY(std::string_view dest) {
  return AddInstruction(Opcode::SysLocalIdY, {ParseRegOperand(dest)});
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

InstructionBuilder& InstructionBuilder::SysBlockDimX(std::string_view dest) {
  return AddInstruction(Opcode::SysBlockDimX, {ParseRegOperand(dest)});
}

InstructionBuilder& InstructionBuilder::SysBlockDimY(std::string_view dest) {
  return AddInstruction(Opcode::SysBlockDimY, {ParseRegOperand(dest)});
}

InstructionBuilder& InstructionBuilder::SysGridDimX(std::string_view dest) {
  return AddInstruction(Opcode::SysGridDimX, {ParseRegOperand(dest)});
}

InstructionBuilder& InstructionBuilder::SysGridDimY(std::string_view dest) {
  return AddInstruction(Opcode::SysGridDimY, {ParseRegOperand(dest)});
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
                                                         uint32_t scale_bytes) {
  return AddInstruction(Opcode::SBufferLoadDword,
                        {ParseRegOperand(dest), ParseRegOperand(index), ImmediateOperand(scale_bytes)});
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
                                                    uint32_t scale_bytes) {
  return AddInstruction(Opcode::MLoadGlobal,
                        {ParseRegOperand(dest), ParseRegOperand(base), ParseRegOperand(index),
                         ImmediateOperand(scale_bytes)});
}

InstructionBuilder& InstructionBuilder::MStoreGlobal(std::string_view base,
                                                     std::string_view index,
                                                     std::string_view src,
                                                     uint32_t scale_bytes) {
  return AddInstruction(Opcode::MStoreGlobal,
                        {ParseRegOperand(base), ParseRegOperand(index), ParseRegOperand(src),
                         ImmediateOperand(scale_bytes)});
}

InstructionBuilder& InstructionBuilder::MAtomicAddGlobal(std::string_view base,
                                                         std::string_view index,
                                                         std::string_view src,
                                                         uint32_t scale_bytes) {
  return AddInstruction(Opcode::MAtomicAddGlobal,
                        {ParseRegOperand(base), ParseRegOperand(index), ParseRegOperand(src),
                         ImmediateOperand(scale_bytes)});
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
                                                   uint32_t scale_bytes) {
  return AddInstruction(Opcode::MLoadConst,
                        {ParseRegOperand(dest), ParseRegOperand(index),
                         ImmediateOperand(scale_bytes)});
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

KernelProgram InstructionBuilder::Build(std::string name,
                                        MetadataBlob metadata,
                                        ConstSegment const_segment) {
  for (const auto& pending : pending_labels_) {
    const auto it = labels_.find(pending.label);
    if (it == labels_.end()) {
      throw std::invalid_argument("undefined branch label: " + pending.label);
    }
    instructions_[pending.instruction_index].operands[pending.operand_index] =
        Operand::Branch(it->second);
  }
  pending_labels_.clear();
  return KernelProgram(std::move(name), instructions_, labels_, std::move(metadata),
                       std::move(const_segment));
}

InstructionBuilder& InstructionBuilder::AddInstruction(Opcode opcode,
                                                       std::vector<Operand> operands) {
  instructions_.push_back(
      Instruction{.opcode = opcode, .operands = std::move(operands), .debug_loc = ConsumeNextDebugLoc()});
  return *this;
}

InstructionBuilder& InstructionBuilder::AddBranch(Opcode opcode, std::string_view label) {
  const size_t instruction_index = instructions_.size();
  instructions_.push_back(Instruction{
      .opcode = opcode,
      .operands = {Operand::Branch(0)},
      .debug_loc = ConsumeNextDebugLoc(),
  });
  pending_labels_.push_back(
      PendingLabelRef{.instruction_index = instruction_index, .operand_index = 0, .label = std::string(label)});
  return *this;
}

Operand InstructionBuilder::ParseRegOperand(std::string_view text) const {
  const auto reg = ParseRegisterName(text);
  return reg.file == RegisterFile::Scalar ? Operand::ScalarReg(reg.index)
                                          : Operand::VectorReg(reg.index);
}

Operand InstructionBuilder::ImmediateOperand(uint64_t value) const {
  return Operand::ImmediateU64(value);
}

DebugLoc InstructionBuilder::ConsumeNextDebugLoc() {
  DebugLoc loc;
  if (has_next_debug_loc_) {
    loc = next_debug_loc_;
    next_debug_loc_ = {};
    has_next_debug_loc_ = false;
  }
  return loc;
}

}  // namespace gpu_model
