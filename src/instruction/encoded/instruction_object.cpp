#include "gpu_model/instruction/encoded/instruction_object.h"

#include <cstring>
#include <stdexcept>
#include <vector>

#include "gpu_model/instruction/encoded/internal/encoded_gcn_encoding_def.h"
#include "gpu_model/instruction/encoded/internal/encoded_instruction_binding.h"
#include "gpu_model/execution/encoded_semantic_handler.h"
#include "gpu_model/instruction/encoded/instruction_decoder.h"

namespace gpu_model {

namespace {

uint32_t InstructionSizeForFormat(const std::vector<uint32_t>& words,
                                  EncodedGcnInstFormatClass format_class) {
  const uint32_t low = words.empty() ? 0u : words[0];
  switch (format_class) {
    case EncodedGcnInstFormatClass::Sopp:
    case EncodedGcnInstFormatClass::Sopk:
      return 4;
    case EncodedGcnInstFormatClass::Sop2:
    case EncodedGcnInstFormatClass::Sopc:
      return ((low & 0xffu) == 255u || ((low >> 8u) & 0xffu) == 255u) ? 8u : 4u;
    case EncodedGcnInstFormatClass::Sop1:
      return (low & 0xffu) == 255u ? 8u : 4u;
    case EncodedGcnInstFormatClass::Vop2:
    case EncodedGcnInstFormatClass::Vopc:
      return (low & 0x1ffu) == 255u ? 8u : 4u;
    case EncodedGcnInstFormatClass::Vop1:
      return (low & 0x1ffu) == 255u ? 8u : 4u;
    case EncodedGcnInstFormatClass::Smrd:
    case EncodedGcnInstFormatClass::Smem:
    case EncodedGcnInstFormatClass::Vop3a:
    case EncodedGcnInstFormatClass::Vop3b:
    case EncodedGcnInstFormatClass::Vop3p:
    case EncodedGcnInstFormatClass::Ds:
    case EncodedGcnInstFormatClass::Flat:
    case EncodedGcnInstFormatClass::Mubuf:
    case EncodedGcnInstFormatClass::Mtbuf:
    case EncodedGcnInstFormatClass::Mimg:
    case EncodedGcnInstFormatClass::Exp:
      return 8;
    case EncodedGcnInstFormatClass::Vintrp:
      return (((low >> 26u) & 0x3fu) == 0x32u) ? 4u : 8u;
    case EncodedGcnInstFormatClass::Unknown:
      break;
  }
  throw std::runtime_error("failed to determine raw instruction size");
}

std::vector<uint32_t> ReadWords(std::span<const std::byte> bytes, size_t offset, uint32_t size_bytes) {
  std::vector<uint32_t> words;
  words.reserve(size_bytes / 4);
  for (uint32_t i = 0; i < size_bytes; i += 4) {
    uint32_t word = 0;
    std::memcpy(&word, bytes.data() + offset + i, sizeof(word));
    words.push_back(word);
  }
  return words;
}

std::vector<EncodedGcnInstruction> ParseRawInstructions(std::span<const std::byte> text_bytes,
                                                    uint64_t start_pc) {
  std::vector<EncodedGcnInstruction> instructions;
  size_t offset = 0;
  while (offset < text_bytes.size()) {
    if (offset + sizeof(uint32_t) > text_bytes.size()) {
      throw std::runtime_error("raw instruction exceeds text section bounds");
    }
    uint32_t low = 0;
    std::memcpy(&low, text_bytes.data() + offset, sizeof(low));
    const auto format_class = ClassifyGcnInstFormat({low});
    const uint32_t size_bytes = InstructionSizeForFormat({low}, format_class);
    if (offset + size_bytes > text_bytes.size()) {
      throw std::runtime_error("raw instruction exceeds text section bounds");
    }

    EncodedGcnInstruction instruction;
    instruction.pc = start_pc + offset;
    instruction.words = ReadWords(text_bytes, offset, size_bytes);
    instruction.size_bytes = size_bytes;
    instruction.format_class = format_class;
    if (const auto* def = FindEncodedGcnEncodingDef(instruction.words)) {
      instruction.encoding_id = def->id;
      instruction.mnemonic = std::string(def->mnemonic);
    } else {
      instruction.mnemonic = std::string(LookupEncodedGcnOpcodeName(instruction.words));
    }
    DecodeEncodedGcnOperands(instruction);
    instructions.push_back(instruction);
    offset += size_bytes;
  }
  return instructions;
}

}  // namespace

InstructionObject::InstructionObject(DecodedInstruction instruction,
                                     const InstructionSemanticHandler& handler)
    : instruction_(std::move(instruction)), handler_(&handler) {}

void InstructionObject::Execute(InstructionExecutionContext& context) const {
  handler_->Execute(instruction_, context);
}

std::vector<InstructionObjectPtr> InstructionArrayParser::Parse(
    const std::vector<DecodedInstruction>& instructions) {
  std::vector<InstructionObjectPtr> objects;
  objects.reserve(instructions.size());
  for (const auto& instruction : instructions) {
    objects.push_back(InstructionFactory::Create(instruction));
  }
  return objects;
}

ParsedInstructionArray InstructionArrayParser::Parse(std::span<const std::byte> text_bytes,
                                                     uint64_t start_pc) {
  ParsedInstructionArray result;
  result.raw_instructions = ParseRawInstructions(text_bytes, start_pc);
  result.decoded_instructions.reserve(result.raw_instructions.size());
  for (const auto& instruction : result.raw_instructions) {
    result.decoded_instructions.push_back(InstructionDecoder{}.Decode(instruction));
  }
  result.instruction_objects = Parse(result.decoded_instructions);
  return result;
}

ParsedInstructionArray InstructionArrayParser::Parse(const std::vector<EncodedGcnInstruction>& instructions) {
  ParsedInstructionArray result;
  result.raw_instructions = instructions;
  result.decoded_instructions.reserve(instructions.size());
  for (const auto& instruction : instructions) {
    result.decoded_instructions.push_back(InstructionDecoder{}.Decode(instruction));
  }
  result.instruction_objects = Parse(result.decoded_instructions);
  return result;
}

InstructionObjectPtr InstructionFactory::Create(DecodedInstruction instruction) {
  return BindEncodedInstructionObject(std::move(instruction));
}

}  // namespace gpu_model
