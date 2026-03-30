#include <gtest/gtest.h>

#include <memory>
#include <span>
#include <type_traits>

#include "gpu_model/instruction/encoded/decoded_instruction.h"
#include "gpu_model/instruction/encoded/instruction_decoder.h"
#include "gpu_model/instruction/encoded/instruction_object.h"
#include "gpu_model/instruction/modeled/lowering.h"
#include "gpu_model/program/executable_kernel.h"
#include "gpu_model/program/program_object.h"

namespace gpu_model {
namespace {

TEST(InstructionNamingTest, InstructionHeadersDeclarePrimaryTypes) {
  using DecodeSignature = DecodedInstruction (InstructionDecoder::*)(const InstructionEncoding&) const;
  static_assert(std::is_same_v<decltype(&InstructionDecoder::Decode), DecodeSignature>);

  static_assert(std::is_same_v<decltype(InstructionEncoding{}.words), std::vector<uint32_t>>);
  static_assert(std::is_same_v<decltype(InstructionEncoding{}.format_class), EncodedGcnInstFormatClass>);

  static_assert(std::is_enum_v<DecodedInstructionOperandKind>);
  static_assert(std::is_same_v<decltype(DecodedInstructionOperand{}.kind), DecodedInstructionOperandKind>);
  static_assert(std::is_same_v<decltype(DecodedInstruction{}.operands),
                               std::vector<DecodedInstructionOperand>>);

  using InstructionObjectDecodedSignature = const DecodedInstruction& (InstructionObject::*)() const;
  using InstructionObjectExecuteSignature =
      void (InstructionObject::*)(InstructionExecutionContext&) const;
  static_assert(std::is_abstract_v<InstructionObject>);
  static_assert(std::is_same_v<decltype(&InstructionObject::decoded), InstructionObjectDecodedSignature>);
  static_assert(std::is_same_v<decltype(&InstructionObject::Execute), InstructionObjectExecuteSignature>);
  static_assert(std::is_same_v<InstructionObjectPtr, std::unique_ptr<InstructionObject>>);

  using FactoryCreateSignature = InstructionObjectPtr (*)(DecodedInstruction);
  static_assert(std::is_same_v<decltype(&InstructionFactory::Create), FactoryCreateSignature>);

  static_assert(std::is_same_v<decltype(ParsedInstructionArray{}.decoded_instructions),
                               std::vector<DecodedInstruction>>);
  static_assert(std::is_same_v<decltype(ParsedInstructionArray{}.instruction_objects),
                               std::vector<InstructionObjectPtr>>);

  using ParseBytesSignature = ParsedInstructionArray (*)(std::span<const std::byte>, uint64_t);
  using ParseDecodedSignature = std::vector<InstructionObjectPtr> (*)(
      const std::vector<DecodedInstruction>&);
  static_assert(std::is_same_v<
                decltype(static_cast<ParseBytesSignature>(&InstructionArrayParser::Parse)),
                ParseBytesSignature>);
  static_assert(std::is_same_v<
                decltype(static_cast<ParseDecodedSignature>(&InstructionArrayParser::Parse)),
                ParseDecodedSignature>);

  static_assert(std::is_abstract_v<ModeledInstructionLowerer>);

  using RegistryGetSignature = const ModeledInstructionLowerer& (*)(TargetIsa);
  using RegistryLowerSignature = ExecutableKernel (*)(const ProgramObject&);
  static_assert(std::is_same_v<decltype(&ModeledInstructionLoweringRegistry::Get),
                               RegistryGetSignature>);
  static_assert(std::is_same_v<decltype(&ModeledInstructionLoweringRegistry::Lower),
                               RegistryLowerSignature>);
}

}  // namespace
}  // namespace gpu_model
