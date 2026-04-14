#include "instruction/semantics/encoded_handler.h"

#include <stdexcept>
#include <string>
#include <string_view>

#include "instruction/semantics/internal/handler_support.h"
#include "instruction/decode/encoded/internal/encoded_gcn_db_lookup.h"

namespace gpu_model {

// Forward declarations for handlers extracted to separate compilation units
namespace semantics {
const IEncodedSemanticHandler& GetBranchHandler();
const IEncodedSemanticHandler& GetSpecialHandler();
const IEncodedSemanticHandler& GetScalarMemoryHandler();
const IEncodedSemanticHandler& GetScalarAluHandler();
const IEncodedSemanticHandler& GetScalarCompareHandler();
const IEncodedSemanticHandler& GetMaskHandler();
const IEncodedSemanticHandler& GetFlatMemoryHandler();
const IEncodedSemanticHandler& GetBufferMemoryHandler();
const IEncodedSemanticHandler& GetSharedMemoryHandler();
const IEncodedSemanticHandler& GetVectorCompareHandler();
}  // namespace semantics

namespace {

using handler_support::HandlerRegistry;
using handler_support::ThrowUnsupportedInstruction;

const IEncodedSemanticHandler* HandlerForSemanticFamily(std::string_view semantic_family,
                                                         std::string_view mnemonic) {
  if (semantic_family == "scalar_memory") {
    return &semantics::GetScalarMemoryHandler();
  }
  if (semantic_family == "scalar_alu") {
    return &semantics::GetScalarAluHandler();
  }
  if (semantic_family == "scalar_compare") {
    return &semantics::GetScalarCompareHandler();
  }
  if (semantic_family == "vector_compare") {
    return &semantics::GetVectorCompareHandler();
  }
  if (semantic_family == "vector_memory") {
    return &semantics::GetFlatMemoryHandler();
  }
  if (semantic_family == "lds") {
    return &semantics::GetSharedMemoryHandler();
  }
  if (semantic_family == "branch_or_sync") {
    if (mnemonic == "s_barrier" || mnemonic == "s_waitcnt" || mnemonic == "s_endpgm" ||
        mnemonic == "s_nop") {
      return &semantics::GetSpecialHandler();
    }
    return &semantics::GetBranchHandler();
  }
  return nullptr;
}

}  // namespace

const IEncodedSemanticHandler& EncodedSemanticHandlerRegistry::Get(std::string_view mnemonic) {
  // O(1) lookup via unified registry (all handlers self-register on load)
  if (const auto* handler = HandlerRegistry::Instance().Find(mnemonic)) {
    return *handler;
  }
  // Secondary lookup via generated semantic-family metadata for mnemonics without direct registry entries.
  if (const auto* def = FindGeneratedGcnInstDefByMnemonic(mnemonic); def != nullptr) {
    if (const auto* handler = HandlerForSemanticFamily(def->semantic_family, def->mnemonic)) {
      return *handler;
    }
  }
  throw std::invalid_argument("unsupported raw GCN opcode: " + std::string(mnemonic));
}

const IEncodedSemanticHandler& EncodedSemanticHandlerRegistry::Get(
    const DecodedInstruction& instruction) {
  // O(1) lookup via unified registry (all handlers self-register on load)
  if (const auto* handler = HandlerRegistry::Instance().Find(instruction.mnemonic)) {
    return *handler;
  }
  // Fallback: semantic family lookup
  if (instruction.encoding_id != 0) {
    if (const auto* def = FindGeneratedGcnInstDefById(instruction.encoding_id); def != nullptr) {
      if (const auto* handler =
              HandlerForSemanticFamily(def->semantic_family, instruction.mnemonic)) {
        return *handler;
      }
    }
  }
  ThrowUnsupportedInstruction("unsupported raw GCN opcode: ", instruction);
}

}  // namespace gpu_model
