#pragma once

#include <cstdint>
#include <string>
#include <string_view>

#include "gpu_model/isa/kernel_program_builder.h"

namespace gpu_model {

class InstructionBuilder : public KernelProgramBuilder {
 public:
  InstructionBuilder& SetNextDebugLoc(std::string file, uint32_t line) {
    KernelProgramBuilder::SetNextDebugLoc(std::move(file), line);
    return *this;
  }
  InstructionBuilder& Label(std::string name) {
    KernelProgramBuilder::Label(std::move(name));
    return *this;
  }

  InstructionBuilder& SLoadArg(std::string_view dest, uint32_t arg_index);
  InstructionBuilder& SysGlobalIdX(std::string_view dest);
  InstructionBuilder& SysGlobalIdY(std::string_view dest);
  InstructionBuilder& SysGlobalIdZ(std::string_view dest);
  InstructionBuilder& SysLocalIdX(std::string_view dest);
  InstructionBuilder& SysLocalIdY(std::string_view dest);
  InstructionBuilder& SysLocalIdZ(std::string_view dest);
  InstructionBuilder& SysBlockOffsetX(std::string_view dest);
  InstructionBuilder& SysBlockIdxX(std::string_view dest);
  InstructionBuilder& SysBlockIdxY(std::string_view dest);
  InstructionBuilder& SysBlockIdxZ(std::string_view dest);
  InstructionBuilder& SysBlockDimX(std::string_view dest);
  InstructionBuilder& SysBlockDimY(std::string_view dest);
  InstructionBuilder& SysBlockDimZ(std::string_view dest);
  InstructionBuilder& SysGridDimX(std::string_view dest);
  InstructionBuilder& SysGridDimY(std::string_view dest);
  InstructionBuilder& SysGridDimZ(std::string_view dest);
  InstructionBuilder& SysLaneId(std::string_view dest);

  InstructionBuilder& SMov(std::string_view dest, std::string_view src);
  InstructionBuilder& SMov(std::string_view dest, uint64_t imm);
  InstructionBuilder& SAdd(std::string_view dest, std::string_view lhs, std::string_view rhs);
  InstructionBuilder& SAdd(std::string_view dest, std::string_view lhs, uint64_t rhs);
  InstructionBuilder& SSub(std::string_view dest, std::string_view lhs, std::string_view rhs);
  InstructionBuilder& SSub(std::string_view dest, std::string_view lhs, uint64_t rhs);
  InstructionBuilder& SMul(std::string_view dest, std::string_view lhs, std::string_view rhs);
  InstructionBuilder& SMul(std::string_view dest, std::string_view lhs, uint64_t rhs);
  InstructionBuilder& SDiv(std::string_view dest, std::string_view lhs, std::string_view rhs);
  InstructionBuilder& SDiv(std::string_view dest, std::string_view lhs, uint64_t rhs);
  InstructionBuilder& SRem(std::string_view dest, std::string_view lhs, std::string_view rhs);
  InstructionBuilder& SRem(std::string_view dest, std::string_view lhs, uint64_t rhs);
  InstructionBuilder& SAnd(std::string_view dest, std::string_view lhs, std::string_view rhs);
  InstructionBuilder& SAnd(std::string_view dest, std::string_view lhs, uint64_t rhs);
  InstructionBuilder& SOr(std::string_view dest, std::string_view lhs, std::string_view rhs);
  InstructionBuilder& SOr(std::string_view dest, std::string_view lhs, uint64_t rhs);
  InstructionBuilder& SXor(std::string_view dest, std::string_view lhs, std::string_view rhs);
  InstructionBuilder& SXor(std::string_view dest, std::string_view lhs, uint64_t rhs);
  InstructionBuilder& SShl(std::string_view dest, std::string_view lhs, std::string_view rhs);
  InstructionBuilder& SShl(std::string_view dest, std::string_view lhs, uint64_t rhs);
  InstructionBuilder& SShr(std::string_view dest, std::string_view lhs, std::string_view rhs);
  InstructionBuilder& SShr(std::string_view dest, std::string_view lhs, uint64_t rhs);
  InstructionBuilder& SWaitCnt(uint32_t global_count,
                               uint32_t shared_count,
                               uint32_t private_count,
                               uint32_t scalar_buffer_count);
  InstructionBuilder& SBufferLoadDword(std::string_view dest,
                                       std::string_view index,
                                       uint32_t scale_bytes = 1,
                                       uint32_t offset_bytes = 0);
  InstructionBuilder& SCmpLt(std::string_view lhs, std::string_view rhs);
  InstructionBuilder& SCmpLt(std::string_view lhs, uint64_t rhs);
  InstructionBuilder& SCmpEq(std::string_view lhs, std::string_view rhs);
  InstructionBuilder& SCmpEq(std::string_view lhs, uint64_t rhs);
  InstructionBuilder& SCmpGt(std::string_view lhs, std::string_view rhs);
  InstructionBuilder& SCmpGt(std::string_view lhs, uint64_t rhs);
  InstructionBuilder& SCmpGe(std::string_view lhs, std::string_view rhs);
  InstructionBuilder& SCmpGe(std::string_view lhs, uint64_t rhs);

  InstructionBuilder& VMov(std::string_view dest, std::string_view src);
  InstructionBuilder& VMov(std::string_view dest, uint64_t imm);
  InstructionBuilder& VAdd(std::string_view dest, std::string_view lhs, std::string_view rhs);
  InstructionBuilder& VAnd(std::string_view dest, std::string_view lhs, std::string_view rhs);
  InstructionBuilder& VOr(std::string_view dest, std::string_view lhs, std::string_view rhs);
  InstructionBuilder& VXor(std::string_view dest, std::string_view lhs, std::string_view rhs);
  InstructionBuilder& VShl(std::string_view dest, std::string_view lhs, std::string_view rhs);
  InstructionBuilder& VShr(std::string_view dest, std::string_view lhs, std::string_view rhs);
  InstructionBuilder& VSub(std::string_view dest, std::string_view lhs, std::string_view rhs);
  InstructionBuilder& VDiv(std::string_view dest, std::string_view lhs, std::string_view rhs);
  InstructionBuilder& VRem(std::string_view dest, std::string_view lhs, std::string_view rhs);
  InstructionBuilder& VMul(std::string_view dest, std::string_view lhs, std::string_view rhs);
  InstructionBuilder& VAddF32(std::string_view dest, std::string_view lhs, std::string_view rhs);
  InstructionBuilder& VMin(std::string_view dest, std::string_view lhs, std::string_view rhs);
  InstructionBuilder& VMax(std::string_view dest, std::string_view lhs, std::string_view rhs);
  InstructionBuilder& VFma(std::string_view dest,
                           std::string_view lhs,
                           std::string_view rhs,
                           std::string_view addend);
  InstructionBuilder& VCmpLtCmask(std::string_view lhs, std::string_view rhs);
  InstructionBuilder& VCmpEqCmask(std::string_view lhs, std::string_view rhs);
  InstructionBuilder& VCmpGeCmask(std::string_view lhs, std::string_view rhs);
  InstructionBuilder& VCmpGtCmask(std::string_view lhs, std::string_view rhs);
  InstructionBuilder& VSelectCmask(std::string_view dest,
                                   std::string_view true_value,
                                   std::string_view false_value);

  InstructionBuilder& MLoadGlobal(std::string_view dest,
                                  std::string_view base,
                                  std::string_view index,
                                  uint32_t scale_bytes = 1,
                                  uint32_t offset_bytes = 0);
  InstructionBuilder& MStoreGlobal(std::string_view base,
                                   std::string_view index,
                                   std::string_view src,
                                   uint32_t scale_bytes = 1,
                                   uint32_t offset_bytes = 0);
  InstructionBuilder& MAtomicAddGlobal(std::string_view base,
                                       std::string_view index,
                                       std::string_view src,
                                       uint32_t scale_bytes = 1,
                                       uint32_t offset_bytes = 0);
  InstructionBuilder& MLoadGlobalAddr(std::string_view dest,
                                      std::string_view addr_lo,
                                      std::string_view addr_hi,
                                      uint32_t offset_bytes = 0);
  InstructionBuilder& MStoreGlobalAddr(std::string_view addr_lo,
                                       std::string_view addr_hi,
                                       std::string_view src,
                                       uint32_t offset_bytes = 0);
  InstructionBuilder& MLoadShared(std::string_view dest,
                                  std::string_view index,
                                  uint32_t scale_bytes = 1);
  InstructionBuilder& MStoreShared(std::string_view index,
                                   std::string_view src,
                                   uint32_t scale_bytes = 1);
  InstructionBuilder& MAtomicAddShared(std::string_view index,
                                       std::string_view src,
                                       uint32_t scale_bytes = 1);
  InstructionBuilder& MLoadPrivate(std::string_view dest,
                                   std::string_view index,
                                   uint32_t scale_bytes = 1);
  InstructionBuilder& MStorePrivate(std::string_view index,
                                    std::string_view src,
                                    uint32_t scale_bytes = 1);
  InstructionBuilder& MLoadConst(std::string_view dest,
                                 std::string_view index,
                                 uint32_t scale_bytes = 1,
                                 uint32_t offset_bytes = 0);

  InstructionBuilder& MaskSaveExec(std::string_view dest);
  InstructionBuilder& MaskRestoreExec(std::string_view src);
  InstructionBuilder& MaskAndExecCmask();

  InstructionBuilder& BBranch(std::string_view label);
  InstructionBuilder& BIfSmask(std::string_view label);
  InstructionBuilder& BIfNoexec(std::string_view label);
  InstructionBuilder& SyncWaveBarrier();
  InstructionBuilder& SyncBarrier();
  InstructionBuilder& BExit();

 private:
  InstructionBuilder& AddInstruction(Opcode opcode, std::vector<Operand> operands) {
    KernelProgramBuilder::AddInstruction(opcode, std::move(operands));
    return *this;
  }
  InstructionBuilder& AddBranch(Opcode opcode, std::string_view label) {
    KernelProgramBuilder::AddBranch(opcode, label);
    return *this;
  }
};

}  // namespace gpu_model
