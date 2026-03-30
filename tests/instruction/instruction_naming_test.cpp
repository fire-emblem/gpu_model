#include <gtest/gtest.h>

#include <type_traits>

#include "gpu_model/instruction/encoded/decoded_instruction.h"
#include "gpu_model/instruction/encoded/instruction_decoder.h"
#include "gpu_model/instruction/encoded/instruction_object.h"
#include "gpu_model/instruction/modeled/lowering.h"
#include "gpu_model/program/executable_kernel.h"
#include "gpu_model/program/program_object.h"

namespace gpu_model {
namespace {

TEST(InstructionNamingTest, NewInstructionNamesAliasLegacyTypes) {
  static_assert(std::is_same_v<InstructionDecoder, GcnInstDecoder>);
  static_assert(std::is_same_v<DecodedInstruction, DecodedGcnInstruction>);
  static_assert(std::is_same_v<DecodedInstructionOperand, DecodedGcnOperand>);
  static_assert(std::is_same_v<DecodedInstructionOperandKind, DecodedGcnOperandKind>);
  static_assert(std::is_same_v<InstructionObject, RawGcnInstructionObject>);
  static_assert(std::is_same_v<InstructionObjectPtr, RawGcnInstructionObjectPtr>);
  static_assert(std::is_same_v<InstructionFactory, RawGcnInstructionFactory>);
  static_assert(std::is_same_v<ParsedInstructionArray, RawGcnParsedInstructionArray>);
  static_assert(std::is_same_v<InstructionArrayParser, RawGcnInstructionArrayParser>);
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
