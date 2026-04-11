#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

#include "gpu_model/instruction/encoded/internal/generated_encoded_gcn_full_opcode_table.h"
#include "gpu_model/instruction/encoded/encoded_gcn_inst_format.h"
#include "gpu_model/instruction/encoded/encoded_gcn_instruction.h"

namespace gpu_model {

enum class EncodedInstructionCategory {
  Unknown,
  ScalarMemory,
  Scalar,
  Vector,
  Memory,
};

struct EncodedGcnEncodingDef {
  uint32_t id = 0;
  EncodedGcnInstFormatClass format_class = EncodedGcnInstFormatClass::Unknown;
  uint32_t op = 0;
  uint32_t size_bytes = 0;
  std::string_view mnemonic;
};

enum class EncodedOperandDecoderKind {
  Unsupported,
  Generated,
  ViStyleScalarMemory,
  Vop3pMatrix,
  Vop3pAccvgprRead,
  Vop3pAccvgprWrite,
  DsRead2B32,
  DsReadB32,
  DsWriteB32,
  DsAtomicNoRtnB32,
  DsAtomicRtnB32,
  Sop1Scalar,
  Sop1ScalarPair,
  Sop1ScalarSrcPair,
  Sop2Scalar,
  Sop2ScalarPair,
  Sop2ScalarPairScalar,
  SopcScalar,
  SopcScalarPair,
  Vop1Generic,
  Vop2Generic,
  Vop3aGeneric,
  Vop3Readlane,
  Vop3CmpE64,
  VopcGeneric,
  Vop3MadU64U32,
};

struct EncodedGcnMatchRecord {
  const EncodedGcnEncodingDef* encoding_def = nullptr;
  const GcnIsaOpcodeDescriptor* opcode_descriptor = nullptr;
  EncodedInstructionCategory category = EncodedInstructionCategory::Unknown;
  EncodedOperandDecoderKind operand_decoder_kind = EncodedOperandDecoderKind::Unsupported;

  bool known() const { return encoding_def != nullptr && opcode_descriptor != nullptr; }
};

void DecodeEncodedGcnOperands(EncodedGcnInstruction& instruction);
const EncodedGcnMatchRecord* FindEncodedGcnMatchRecord(const std::vector<uint32_t>& words);
const EncodedGcnEncodingDef* FindEncodedGcnEncodingDef(const std::vector<uint32_t>& words);
const EncodedGcnEncodingDef* FindEncodedGcnEncodingDefByMnemonic(
    std::string_view mnemonic,
    EncodedGcnInstFormatClass format_class = EncodedGcnInstFormatClass::Unknown,
    uint32_t size_bytes = 0);
std::string_view LookupEncodedGcnOpcodeName(const std::vector<uint32_t>& words);

}  // namespace gpu_model
