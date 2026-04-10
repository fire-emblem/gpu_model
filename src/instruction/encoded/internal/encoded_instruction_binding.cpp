#include "gpu_model/instruction/encoded/internal/encoded_instruction_binding.h"

#include <stdexcept>
#include <string>

#include "gpu_model/instruction/encoded/internal/encoded_gcn_encoding_def.h"
#include "gpu_model/instruction/encoded/internal/encoded_instruction_descriptor.h"
#include "gpu_model/execution/encoded_semantic_handler.h"
#include "gpu_model/execution/internal/encoded_handler_utils.h"

namespace gpu_model {

namespace {

// Unsupported handler for unknown/placeholder instructions
class UnsupportedInstructionHandler final : public IEncodedSemanticHandler {
 public:
  void Execute(const DecodedInstruction& instruction, EncodedWaveContext&) const override {
    throw std::invalid_argument("unsupported instantiated raw GCN opcode: " + instruction.mnemonic);
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
    EncodedDebugLog("BindEncodedInstruction: pc=0x%llx placeholder class=%s",
                    static_cast<unsigned long long>(instruction.pc),
                    class_name.c_str());
    return std::make_unique<EncodedInstructionObject>(
        std::move(instruction), kUnsupportedHandler, op_type_name, class_name);
  }

  // Known instruction - create with handler
  const std::string class_name(match->encoding_def->mnemonic);
  instruction.mnemonic = class_name;
  const auto& handler = EncodedSemanticHandlerRegistry::Get(instruction);
  EncodedDebugLog("BindEncodedInstruction: pc=0x%llx mnemonic=%s",
                  static_cast<unsigned long long>(instruction.pc),
                  class_name.c_str());
  return std::make_unique<EncodedInstructionObject>(
      std::move(instruction), handler, op_type_name, class_name);
}

}  // namespace gpu_model
