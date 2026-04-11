#include "gpu_model/instruction/encoded/internal/encoded_instruction_binding.h"

#include <stdexcept>
#include <string>

#include "gpu_model/instruction/encoded/internal/encoded_gcn_encoding_def.h"
#include "gpu_model/instruction/encoded/internal/encoded_instruction_descriptor.h"
#include "gpu_model/execution/encoded_semantic_handler.h"
#include "gpu_model/execution/internal/encoded_handler_utils.h"

namespace gpu_model {

namespace {

std::string InstructionDebugContext(const DecodedInstruction& instruction) {
  std::string message;
  const std::string hex_words = instruction.HexWords();
  if (!hex_words.empty()) {
    message += " binary=" + hex_words;
  }
  const std::string asm_text = instruction.BoundAsmText();
  if (!asm_text.empty()) {
    message += " asm=\"" + asm_text + "\"";
  }
  return message;
}

// Unsupported handler for unknown/placeholder instructions
class UnsupportedInstructionHandler final : public IEncodedSemanticHandler {
 public:
  void Execute(const DecodedInstruction& instruction, EncodedWaveContext&) const override {
    throw std::invalid_argument("unsupported instantiated raw GCN opcode: " + instruction.mnemonic +
                                InstructionDebugContext(instruction));
  }
};

static const UnsupportedInstructionHandler kUnsupportedHandler;

// Single unified instruction class - replaces 42 classes from previous design
class EncodedInstructionObject final : public InstructionObject {
 public:
  EncodedInstructionObject(DecodedInstruction instruction,
                           const IEncodedSemanticHandler& handler,
                           std::string op_type_name,
                           std::string class_name)
      : InstructionObject(std::move(instruction), handler),
        op_type_name_(std::move(op_type_name)),
        class_name_(std::move(class_name)) {}

  std::string_view op_type_name() const override { return op_type_name_; }
  std::string_view class_name() const override { return class_name_; }

 private:
  std::string op_type_name_;
  std::string class_name_;
};

}  // namespace

InstructionObjectPtr BindEncodedInstructionObject(DecodedInstruction instruction) {
  const auto* match = FindEncodedGcnMatchRecord(instruction.words);
  const std::string op_type_name(ToString(instruction.format_class));

  if (match == nullptr || !match->known()) {
    // Unknown instruction - create placeholder
    const std::string class_name = op_type_name + "_placeholder";
    return std::make_unique<EncodedInstructionObject>(
        std::move(instruction), kUnsupportedHandler, op_type_name, class_name);
  }

  // Known instruction - create with handler
  const std::string class_name(match->encoding_def->mnemonic);
  instruction.mnemonic = class_name;
  instruction.encoding_id = match->encoding_def->id;
  if (instruction.size_bytes == 0) {
    instruction.size_bytes = match->encoding_def->size_bytes;
  }
  const auto& handler = EncodedSemanticHandlerRegistry::Get(instruction);
  return std::make_unique<EncodedInstructionObject>(
      std::move(instruction), handler, op_type_name, class_name);
}

}  // namespace gpu_model
