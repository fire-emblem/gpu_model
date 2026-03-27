#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "gpu_model/isa/kernel_program.h"

namespace gpu_model {

class InstructionBuilder {
 public:
  InstructionBuilder& SetNextDebugLoc(std::string file, uint32_t line);
  InstructionBuilder& Label(std::string name);

  InstructionBuilder& SLoadArg(std::string_view dest, uint32_t arg_index);
  InstructionBuilder& SysGlobalIdX(std::string_view dest);
  InstructionBuilder& SysBlockIdxX(std::string_view dest);
  InstructionBuilder& SysBlockDimX(std::string_view dest);
  InstructionBuilder& SysLaneId(std::string_view dest);

  InstructionBuilder& SMov(std::string_view dest, std::string_view src);
  InstructionBuilder& SMov(std::string_view dest, uint64_t imm);
  InstructionBuilder& SAdd(std::string_view dest, std::string_view lhs, std::string_view rhs);
  InstructionBuilder& SAdd(std::string_view dest, std::string_view lhs, uint64_t rhs);
  InstructionBuilder& SMul(std::string_view dest, std::string_view lhs, std::string_view rhs);
  InstructionBuilder& SMul(std::string_view dest, std::string_view lhs, uint64_t rhs);
  InstructionBuilder& SCmpLt(std::string_view lhs, std::string_view rhs);
  InstructionBuilder& SCmpLt(std::string_view lhs, uint64_t rhs);
  InstructionBuilder& SCmpEq(std::string_view lhs, std::string_view rhs);
  InstructionBuilder& SCmpEq(std::string_view lhs, uint64_t rhs);

  InstructionBuilder& VMov(std::string_view dest, std::string_view src);
  InstructionBuilder& VMov(std::string_view dest, uint64_t imm);
  InstructionBuilder& VAdd(std::string_view dest, std::string_view lhs, std::string_view rhs);
  InstructionBuilder& VMul(std::string_view dest, std::string_view lhs, std::string_view rhs);
  InstructionBuilder& VFma(std::string_view dest,
                           std::string_view lhs,
                           std::string_view rhs,
                           std::string_view addend);
  InstructionBuilder& VCmpLtCmask(std::string_view lhs, std::string_view rhs);

  InstructionBuilder& MLoadGlobal(std::string_view dest,
                                  std::string_view base,
                                  std::string_view index,
                                  uint32_t scale_bytes = 1);
  InstructionBuilder& MStoreGlobal(std::string_view base,
                                   std::string_view index,
                                   std::string_view src,
                                   uint32_t scale_bytes = 1);
  InstructionBuilder& MLoadShared(std::string_view dest,
                                  std::string_view index,
                                  uint32_t scale_bytes = 1);
  InstructionBuilder& MStoreShared(std::string_view index,
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
                                 uint32_t scale_bytes = 1);

  InstructionBuilder& MaskSaveExec(std::string_view dest);
  InstructionBuilder& MaskRestoreExec(std::string_view src);
  InstructionBuilder& MaskAndExecCmask();

  InstructionBuilder& BBranch(std::string_view label);
  InstructionBuilder& BIfSmask(std::string_view label);
  InstructionBuilder& BIfNoexec(std::string_view label);
  InstructionBuilder& SyncBarrier();
  InstructionBuilder& BExit();

  KernelProgram Build(std::string name,
                      MetadataBlob metadata = {},
                      ConstSegment const_segment = {});

 private:
  struct PendingLabelRef {
    size_t instruction_index = 0;
    size_t operand_index = 0;
    std::string label;
  };

  InstructionBuilder& AddInstruction(Opcode opcode, std::vector<Operand> operands);
  InstructionBuilder& AddBranch(Opcode opcode, std::string_view label);
  Operand ParseRegOperand(std::string_view text) const;
  Operand ImmediateOperand(uint64_t value) const;
  DebugLoc ConsumeNextDebugLoc();

  std::vector<Instruction> instructions_;
  std::unordered_map<std::string, uint64_t> labels_;
  std::vector<PendingLabelRef> pending_labels_;
  DebugLoc next_debug_loc_;
  bool has_next_debug_loc_ = false;
};

}  // namespace gpu_model
