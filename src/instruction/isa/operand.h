#pragma once

#include <cstdint>
#include <string_view>

namespace gpu_model {

enum class RegisterFile {
  Scalar,
  Vector,
};

struct RegRef {
  RegisterFile file = RegisterFile::Scalar;
  uint32_t index = 0;
};

enum class OperandKind {
  None,
  Register,
  RegisterRange,
  Immediate,
  ArgumentIndex,
  BranchTarget,
};

struct Operand {
  OperandKind kind = OperandKind::None;
  RegRef reg{};
  uint32_t reg_count = 1;
  uint64_t immediate = 0;

  static Operand ScalarReg(uint32_t index);
  static Operand VectorReg(uint32_t index);
  static Operand ScalarRegRange(uint32_t index, uint32_t count);
  static Operand VectorRegRange(uint32_t index, uint32_t count);
  static Operand ImmediateU64(uint64_t value);
  static Operand Argument(uint32_t index);
  static Operand Branch(uint64_t target_pc);
};

RegRef ParseRegisterName(std::string_view text);

}  // namespace gpu_model
