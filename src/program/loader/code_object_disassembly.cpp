#include "program/loader/code_object_disassembly.h"

#include <cctype>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string_view>

#include "instruction/decode/encoded/internal/encoded_gcn_encoding_def.h"
#include "program/loader/external_tool_executor.h"

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

bool HasLlvmMc() {
  static const bool available = ExternalToolExecutor::HasLlvmMc();
  return available;
}

std::string FormatHexByteStream(std::span<const std::byte> bytes) {
  std::ostringstream out;
  out << std::hex << std::setfill('0');
  for (size_t i = 0; i < bytes.size(); ++i) {
    if (i != 0) {
      out << ' ';
    }
    out << "0x" << std::setw(2) << std::to_integer<unsigned int>(bytes[i]);
  }
  return out.str();
}

std::string ExtractAsmOperands(const LlvmMcInstructionLine& line) {
  const size_t split = line.text.find_first_of(" \t");
  if (split == std::string::npos) {
    return {};
  }
  return Trim(std::string_view(line.text).substr(split + 1));
}

}  // namespace

std::vector<LlvmMcInstructionLine> DisassembleCodeSegmentWithLlvmMc(
    std::span<const std::byte> code_bytes,
    const ScopedTempDir& temp_dir) {
  std::vector<LlvmMcInstructionLine> lines;
  if (code_bytes.empty() || !HasLlvmMc()) {
    return lines;
  }

  const auto input_path = temp_dir.path() / "kernel_disasm_input.txt";
  {
    std::ofstream out(input_path);
    if (!out) {
      throw std::runtime_error("failed to create llvm-mc disassembly input");
    }
    out << FormatHexByteStream(code_bytes) << '\n';
  }

  const std::string output = ExternalToolExecutor::DisassembleHexByteStreamWithLlvmMc(input_path);
  std::istringstream input(output);
  std::string line;
  while (std::getline(input, line)) {
    const std::string trimmed = Trim(line);
    if (trimmed.empty() || trimmed == ".text") {
      continue;
    }
    const size_t split = trimmed.find_first_of(" \t");
    lines.push_back(LlvmMcInstructionLine{
        .op = split == std::string::npos ? trimmed : trimmed.substr(0, split),
        .text = trimmed,
    });
  }
  return lines;
}

void BindLlvmMcDisassembly(ParsedInstructionArray& parsed,
                           const std::vector<LlvmMcInstructionLine>& disassembly) {
  if (disassembly.empty()) {
    return;
  }
  if (disassembly.size() != parsed.raw_instructions.size() ||
      disassembly.size() != parsed.decoded_instructions.size()) {
    throw std::runtime_error("llvm-mc disassembly count does not match decoded instruction count");
  }

  for (size_t i = 0; i < disassembly.size(); ++i) {
    parsed.raw_instructions[i].asm_op = disassembly[i].op;
    parsed.raw_instructions[i].asm_text = disassembly[i].text;
    if (parsed.raw_instructions[i].operands.empty()) {
      parsed.raw_instructions[i].operands = ExtractAsmOperands(disassembly[i]);
    }
    parsed.decoded_instructions[i].asm_op = disassembly[i].op;
    parsed.decoded_instructions[i].asm_text = disassembly[i].text;
  }
}

std::string JoinLlvmMcAssemblyText(const std::vector<LlvmMcInstructionLine>& disassembly) {
  std::ostringstream out;
  for (size_t i = 0; i < disassembly.size(); ++i) {
    if (i != 0) {
      out << '\n';
    }
    out << disassembly[i].text;
  }
  return out.str();
}

void FillMissingOperandTextFromDecodedOperands(ProgramObject& code_object) {
  auto instructions = code_object.instructions();
  for (auto& instruction : instructions) {
    if (instruction.operands.empty() && !instruction.decoded_operands.empty()) {
      std::ostringstream operand_text;
      for (size_t i = 0; i < instruction.decoded_operands.size(); ++i) {
        if (i != 0) {
          operand_text << ", ";
        }
        operand_text << instruction.decoded_operands[i].text;
      }
      instruction.operands = operand_text.str();
    }
  }
  code_object.set_instructions(std::move(instructions));
}

}  // namespace gpu_model
