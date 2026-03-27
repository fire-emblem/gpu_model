#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace gpu_model {

enum class GcnTextOperandKind {
  Register,
  RegisterRange,
  SpecialRegister,
  Immediate,
  Identifier,
  Off,
};

enum class GcnSpecialRegister {
  Vcc,
  Exec,
  Scc,
  M0,
};

struct GcnTextRegister {
  char prefix = '\0';
  uint32_t index = 0;
};

struct GcnTextRegisterRange {
  char prefix = '\0';
  uint32_t first = 0;
  uint32_t last = 0;
};

struct GcnTextOperand {
  GcnTextOperandKind kind = GcnTextOperandKind::Identifier;
  std::string text;
  std::optional<GcnTextRegister> reg;
  std::optional<GcnTextRegisterRange> reg_range;
  std::optional<GcnSpecialRegister> special_reg;
  std::optional<uint64_t> immediate;
};

struct GcnTextInstruction {
  std::string mnemonic;
  std::vector<GcnTextOperand> operands;
};

class GcnTextParser {
 public:
  static std::string StripComments(std::string_view line);
  static std::vector<std::string> SplitOperands(std::string_view text);
  static GcnTextOperand ParseOperand(std::string_view text);
  static GcnTextInstruction ParseInstruction(std::string_view line);
};

std::string_view ToString(GcnSpecialRegister reg);

}  // namespace gpu_model
