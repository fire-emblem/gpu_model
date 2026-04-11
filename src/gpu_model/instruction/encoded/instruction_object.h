#pragma once

#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include "gpu_model/instruction/encoded/encoded_gcn_instruction.h"
#include "gpu_model/instruction/encoded/decoded_instruction.h"

namespace gpu_model {

struct InstructionExecutionContext {
  virtual ~InstructionExecutionContext() = default;
};

class InstructionSemanticHandler {
 public:
  virtual ~InstructionSemanticHandler() = default;
  virtual void Execute(const DecodedInstruction& instruction,
                       InstructionExecutionContext& context) const = 0;
};

class InstructionObject {
 public:
  InstructionObject(DecodedInstruction instruction, const InstructionSemanticHandler& handler);
  virtual ~InstructionObject() = default;

  const DecodedInstruction& decoded() const { return instruction_; }
  virtual std::string_view op_type_name() const = 0;
  virtual std::string_view class_name() const = 0;
  virtual void Execute(InstructionExecutionContext& context) const;

 private:
  DecodedInstruction instruction_;
  const InstructionSemanticHandler* handler_ = nullptr;
};

using InstructionObjectPtr = std::unique_ptr<InstructionObject>;

class InstructionFactory {
 public:
  static InstructionObjectPtr Create(DecodedInstruction instruction);
};

struct ParsedInstructionArray {
  std::vector<EncodedGcnInstruction> raw_instructions;
  std::vector<DecodedInstruction> decoded_instructions;
  std::vector<InstructionObjectPtr> instruction_objects;
};

class InstructionArrayParser {
 public:
  static std::vector<EncodedGcnInstruction> ParseRaw(std::span<const std::byte> text_bytes,
                                                     uint64_t start_pc);
  static std::vector<DecodedInstruction> Decode(
      const std::vector<EncodedGcnInstruction>& instructions);
  static ParsedInstructionArray Parse(std::span<const std::byte> text_bytes, uint64_t start_pc);
  static ParsedInstructionArray Parse(const std::vector<EncodedGcnInstruction>& instructions);
  static std::vector<InstructionObjectPtr> Parse(const std::vector<DecodedInstruction>& instructions);
};

}  // namespace gpu_model
