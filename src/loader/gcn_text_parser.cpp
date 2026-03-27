#include "gpu_model/loader/gcn_text_parser.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace gpu_model {

namespace {

std::string Trim(std::string_view text) {
  size_t begin = 0;
  size_t end = text.size();
  while (begin < end && std::isspace(static_cast<unsigned char>(text[begin])) != 0) {
    ++begin;
  }
  while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1])) != 0) {
    --end;
  }
  return std::string(text.substr(begin, end - begin));
}

std::string Lowercase(std::string_view text) {
  std::string lowered(text);
  std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  return lowered;
}

bool IsDecimalOrHexNumber(std::string_view text) {
  if (text.empty()) {
    return false;
  }
  size_t index = 0;
  if (text.front() == '-' || text.front() == '+') {
    index = 1;
  }
  if (index >= text.size()) {
    return false;
  }
  if (text.substr(index, 2) == "0x" || text.substr(index, 2) == "0X") {
    index += 2;
    if (index >= text.size()) {
      return false;
    }
    for (; index < text.size(); ++index) {
      if (std::isxdigit(static_cast<unsigned char>(text[index])) == 0) {
        return false;
      }
    }
    return true;
  }
  for (; index < text.size(); ++index) {
    if (std::isdigit(static_cast<unsigned char>(text[index])) == 0) {
      return false;
    }
  }
  return true;
}

}  // namespace

std::string GcnTextParser::StripComments(std::string_view line) {
  std::string stripped(line);
  const size_t slash_comment = stripped.find("//");
  const size_t hash_comment = stripped.find('#');
  size_t comment_pos = std::string::npos;
  if (slash_comment != std::string::npos) {
    comment_pos = slash_comment;
  }
  if (hash_comment != std::string::npos) {
    comment_pos = std::min(comment_pos, hash_comment);
  }
  if (comment_pos != std::string::npos) {
    stripped.resize(comment_pos);
  }
  return Trim(stripped);
}

std::vector<std::string> GcnTextParser::SplitOperands(std::string_view text) {
  std::vector<std::string> operands;
  std::string current;
  int bracket_depth = 0;
  for (const char ch : text) {
    if (ch == '[') {
      ++bracket_depth;
    } else if (ch == ']') {
      --bracket_depth;
    } else if (ch == ',' && bracket_depth == 0) {
      operands.push_back(Trim(current));
      current.clear();
      continue;
    }
    current.push_back(ch);
  }
  if (!current.empty()) {
    operands.push_back(Trim(current));
  }
  return operands;
}

GcnTextOperand GcnTextParser::ParseOperand(std::string_view text) {
  const std::string trimmed = Trim(text);
  const std::string lowered = Lowercase(trimmed);

  GcnTextOperand operand;
  operand.text = lowered;

  if (lowered == "off") {
    operand.kind = GcnTextOperandKind::Off;
    return operand;
  }
  if (lowered == "vcc") {
    operand.kind = GcnTextOperandKind::SpecialRegister;
    operand.special_reg = GcnSpecialRegister::Vcc;
    return operand;
  }
  if (lowered == "exec") {
    operand.kind = GcnTextOperandKind::SpecialRegister;
    operand.special_reg = GcnSpecialRegister::Exec;
    return operand;
  }
  if (lowered == "scc") {
    operand.kind = GcnTextOperandKind::SpecialRegister;
    operand.special_reg = GcnSpecialRegister::Scc;
    return operand;
  }
  if (lowered == "m0") {
    operand.kind = GcnTextOperandKind::SpecialRegister;
    operand.special_reg = GcnSpecialRegister::M0;
    return operand;
  }

  if (lowered.size() >= 4 && (lowered.front() == 's' || lowered.front() == 'v') &&
      lowered[1] == '[' && lowered.back() == ']') {
    const size_t colon = lowered.find(':', 2);
    if (colon == std::string::npos) {
      throw std::invalid_argument("invalid GCN register range operand: " + lowered);
    }
    operand.kind = GcnTextOperandKind::RegisterRange;
    operand.reg_range = GcnTextRegisterRange{
        .prefix = lowered.front(),
        .first = static_cast<uint32_t>(std::stoul(lowered.substr(2, colon - 2))),
        .last = static_cast<uint32_t>(std::stoul(lowered.substr(colon + 1, lowered.size() - colon - 2))),
    };
    return operand;
  }

  if (lowered.size() >= 2 && (lowered.front() == 's' || lowered.front() == 'v')) {
    bool all_digits = true;
    for (size_t i = 1; i < lowered.size(); ++i) {
      if (std::isdigit(static_cast<unsigned char>(lowered[i])) == 0) {
        all_digits = false;
        break;
      }
    }
    if (all_digits) {
      operand.kind = GcnTextOperandKind::Register;
      operand.reg = GcnTextRegister{
          .prefix = lowered.front(),
          .index = static_cast<uint32_t>(std::stoul(lowered.substr(1))),
      };
      return operand;
    }
  }

  if (IsDecimalOrHexNumber(lowered)) {
    operand.kind = GcnTextOperandKind::Immediate;
    operand.immediate = static_cast<uint64_t>(std::stoll(lowered, nullptr, 0));
    return operand;
  }

  operand.kind = GcnTextOperandKind::Identifier;
  return operand;
}

GcnTextInstruction GcnTextParser::ParseInstruction(std::string_view line) {
  const std::string stripped = StripComments(line);
  if (stripped.empty()) {
    throw std::invalid_argument("cannot parse empty GCN instruction line");
  }

  const size_t space = stripped.find_first_of(" \t");
  GcnTextInstruction instruction;
  instruction.mnemonic = Lowercase(space == std::string::npos ? stripped : stripped.substr(0, space));
  if (space == std::string::npos) {
    return instruction;
  }

  for (const auto& operand_text : SplitOperands(std::string_view(stripped).substr(space + 1))) {
    if (!operand_text.empty()) {
      instruction.operands.push_back(ParseOperand(operand_text));
    }
  }
  return instruction;
}

std::string_view ToString(GcnSpecialRegister reg) {
  switch (reg) {
    case GcnSpecialRegister::Vcc:
      return "vcc";
    case GcnSpecialRegister::Exec:
      return "exec";
    case GcnSpecialRegister::Scc:
      return "scc";
    case GcnSpecialRegister::M0:
      return "m0";
  }
  return "vcc";
}

}  // namespace gpu_model
