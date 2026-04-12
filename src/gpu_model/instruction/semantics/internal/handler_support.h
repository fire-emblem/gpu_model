#pragma once

#include <bit>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "gpu_model/execution/encoded_semantic_handler.h"
#include "gpu_model/execution/internal/encoded_handler_utils.h"
#include "gpu_model/execution/internal/float_utils.h"
#include "gpu_model/instruction/encoded/internal/encoded_gcn_encoding_def.h"
#include "gpu_model/instruction/encoded/internal/encoded_gcn_db_lookup.h"
#include "gpu_model/instruction/operand/operand_accessors.h"
#include "gpu_model/utils/logging/log_macros.h"
#include "gpu_model/utils/math/bit_utils.h"
#include "gpu_model/utils/math/float_convert.h"

namespace gpu_model {
namespace handler_support {

// --- Debug context formatting ---

inline std::string InstructionDebugContext(const DecodedInstruction& instruction) {
  std::string message = " [pc=0x" + [](uint64_t pc) {
    std::ostringstream out;
    out << std::hex << pc;
    return out.str();
  }(instruction.pc);
  const std::string hex_words = instruction.HexWords();
  if (!hex_words.empty()) {
    message += ", binary=" + hex_words;
  }
  const std::string asm_text = instruction.BoundAsmText();
  if (!asm_text.empty()) {
    message += ", asm=\"" + asm_text + "\"";
  }
  message += "]";
  return message;
}

[[noreturn]] inline void ThrowUnsupportedInstruction(const std::string& prefix,
                                                      const DecodedInstruction& instruction) {
  throw std::invalid_argument(prefix + instruction.mnemonic + InstructionDebugContext(instruction));
}

[[noreturn]] inline void RethrowWithInstructionContext(const std::exception& error,
                                                       const DecodedInstruction& instruction) {
  std::string message = error.what();
  if (message.find(" [pc=0x") == std::string::npos) {
    message += InstructionDebugContext(instruction);
  }
  throw std::invalid_argument(std::move(message));
}

// --- Register operand helpers ---

inline std::pair<uint32_t, uint32_t> RequireVectorRange(const DecodedInstructionOperand& operand) {
  if (operand.kind != DecodedInstructionOperandKind::VectorRegRange || operand.info.reg_count == 0) {
    throw std::invalid_argument("expected vector register range operand");
  }
  return {operand.info.reg_first, operand.info.reg_first + operand.info.reg_count - 1};
}

// --- Scalar/vector value resolution ---

inline uint64_t ResolveScalarLike(const DecodedInstructionOperand& operand,
                                  const EncodedWaveContext& context) {
  if (operand.kind == DecodedInstructionOperandKind::Immediate ||
      operand.kind == DecodedInstructionOperandKind::BranchTarget) {
    if (!operand.info.has_immediate) {
      throw std::invalid_argument("immediate operand missing value");
    }
    return static_cast<uint64_t>(operand.info.immediate);
  }
  if (operand.kind == DecodedInstructionOperandKind::ScalarReg) {
    return context.wave.sgpr.Read(RequireScalarIndex(operand));
  }
  if (operand.kind == DecodedInstructionOperandKind::SpecialReg &&
      operand.info.special_reg == GcnSpecialReg::Vcc) {
    return context.vcc;
  }
  if (operand.kind == DecodedInstructionOperandKind::SpecialReg &&
      operand.info.special_reg == GcnSpecialReg::Exec) {
    return context.wave.exec.to_ullong();
  }
  throw std::invalid_argument("unsupported scalar-like raw operand");
}

inline void StoreScalarPair(const DecodedInstructionOperand& operand,
                            EncodedWaveContext& context, uint64_t value) {
  if (operand.kind == DecodedInstructionOperandKind::ScalarReg) {
    const uint32_t first = operand.info.reg_first;
    context.wave.sgpr.Write(first, static_cast<uint32_t>(value & 0xffffffffu));
    context.wave.sgpr.Write(first + 1, static_cast<uint32_t>(value >> 32u));
    return;
  }
  if (operand.kind == DecodedInstructionOperandKind::ScalarRegRange && operand.info.reg_count == 2) {
    const uint32_t first = operand.info.reg_first;
    context.wave.sgpr.Write(first, static_cast<uint32_t>(value & 0xffffffffu));
    context.wave.sgpr.Write(first + 1, static_cast<uint32_t>(value >> 32u));
    return;
  }
  if (operand.kind == DecodedInstructionOperandKind::SpecialReg &&
      operand.info.special_reg == GcnSpecialReg::Vcc) {
    context.vcc = value;
    return;
  }
  if (operand.kind == DecodedInstructionOperandKind::SpecialReg &&
      operand.info.special_reg == GcnSpecialReg::Exec) {
    context.wave.exec = MaskFromU64(value);
    return;
  }
  throw std::invalid_argument("unsupported scalar pair destination");
}

inline uint64_t ResolveVectorLane(const DecodedInstructionOperand& operand,
                                  const EncodedWaveContext& context,
                                  uint32_t lane) {
  if (operand.kind == DecodedInstructionOperandKind::Immediate) {
    if (!operand.info.has_immediate) {
      throw std::invalid_argument("immediate operand missing value");
    }
    return static_cast<uint64_t>(operand.info.immediate);
  }
  if (operand.kind == DecodedInstructionOperandKind::ScalarReg) {
    return context.wave.sgpr.Read(RequireScalarIndex(operand));
  }
  if (operand.kind == DecodedInstructionOperandKind::ScalarRegRange) {
    return ResolveScalarPair(operand, context);
  }
  if (operand.kind == DecodedInstructionOperandKind::SpecialReg) {
    switch (operand.info.special_reg) {
      case GcnSpecialReg::Vcc:
        return ((context.vcc >> lane) & 1ull) != 0 ? 1ull : 0ull;
      case GcnSpecialReg::Exec:
        return context.wave.exec.test(lane) ? 1ull : 0ull;
      default:
        break;
    }
  }
  if (operand.kind == DecodedInstructionOperandKind::VectorReg) {
    return context.wave.vgpr.Read(RequireVectorIndex(operand), lane);
  }
  throw std::invalid_argument("unsupported vector-lane raw operand kind=" +
                              std::to_string(static_cast<int>(operand.kind)) +
                              " text=" + operand.text);
}

// --- Bit manipulation ---

inline uint32_t ReverseBits32(uint32_t value) {
  value = ((value & 0x55555555u) << 1u) | ((value >> 1u) & 0x55555555u);
  value = ((value & 0x33333333u) << 2u) | ((value >> 2u) & 0x33333333u);
  value = ((value & 0x0f0f0f0fu) << 4u) | ((value >> 4u) & 0x0f0f0f0fu);
  value = ((value & 0x00ff00ffu) << 8u) | ((value >> 8u) & 0x00ff00ffu);
  return (value << 16u) | (value >> 16u);
}

// --- Opcode descriptor lookup ---

inline const GcnIsaOpcodeDescriptor& RequireCanonicalOpcode(const DecodedInstruction& instruction) {
  if (const auto* match = FindEncodedGcnMatchRecord(instruction.words); match != nullptr &&
      match->opcode_descriptor != nullptr) {
    return *match->opcode_descriptor;
  }
  if (const auto* descriptor = FindGcnIsaOpcodeDescriptorByName(instruction.mnemonic);
      descriptor != nullptr) {
    return *descriptor;
  }
  throw std::invalid_argument("missing canonical opcode descriptor: " + instruction.mnemonic +
                              InstructionDebugContext(instruction));
}

// --- Atomic memory operand structures ---

struct FlatAtomicOperands {
  const DecodedInstructionOperand* return_dest = nullptr;
  const DecodedInstructionOperand* address = nullptr;
  const DecodedInstructionOperand* data = nullptr;
  const DecodedInstructionOperand* scalar_base = nullptr;
  int64_t offset = 0;
};

struct SharedAtomicOperands {
  const DecodedInstructionOperand* return_dest = nullptr;
  const DecodedInstructionOperand* address = nullptr;
  const DecodedInstructionOperand* data = nullptr;
  uint32_t offset = 0;
};

inline SharedAtomicOperands ResolveSharedAtomicOperands(const DecodedInstruction& instruction) {
  SharedAtomicOperands operands;
  if (instruction.operands.size() < 2) {
    throw std::invalid_argument("shared atomic instruction is missing operands");
  }
  size_t cursor = 0;
  if (instruction.operands.size() >= 3 &&
      instruction.operands.at(0).kind == DecodedInstructionOperandKind::VectorReg &&
      instruction.operands.at(1).kind == DecodedInstructionOperandKind::VectorReg &&
      instruction.operands.at(2).kind == DecodedInstructionOperandKind::VectorReg) {
    operands.return_dest = &instruction.operands.at(0);
    cursor = 1;
  }
  if (instruction.operands.size() < cursor + 2) {
    throw std::invalid_argument("shared atomic instruction is missing address/data operands");
  }
  operands.address = &instruction.operands.at(cursor);
  operands.data = &instruction.operands.at(cursor + 1);
  if (instruction.operands.size() > cursor + 2) {
    const auto& tail = instruction.operands.at(cursor + 2);
    if (!tail.info.has_immediate) {
      throw std::invalid_argument("shared atomic offset operand is not immediate");
    }
    operands.offset = static_cast<uint32_t>(tail.info.immediate);
  }
  return operands;
}

inline void StoreSharedAtomicReturnValue(const SharedAtomicOperands& operands,
                                          EncodedWaveContext& context,
                                          uint32_t lane,
                                          uint32_t old_value) {
  if (operands.return_dest == nullptr) {
    return;
  }
  context.wave.vgpr.Write(RequireVectorIndex(*operands.return_dest), lane, old_value);
}

inline FlatAtomicOperands ResolveFlatAtomicOperands(const DecodedInstruction& instruction) {
  FlatAtomicOperands operands;
  if (!instruction.operands.empty() && instruction.operands.back().info.has_immediate) {
    operands.offset = instruction.operands.back().info.immediate;
  }
  if (instruction.operands.size() < 2) {
    throw std::invalid_argument("flat atomic instruction is missing operands");
  }
  const auto& first = instruction.operands.at(0);
  if (first.kind == DecodedInstructionOperandKind::VectorRegRange) {
    operands.address = &first;
    operands.data = &instruction.operands.at(1);
    return operands;
  }
  if (instruction.operands.size() >= 3 &&
      first.kind == DecodedInstructionOperandKind::VectorReg &&
      instruction.operands.at(1).kind == DecodedInstructionOperandKind::VectorReg &&
      instruction.operands.at(2).kind == DecodedInstructionOperandKind::ScalarRegRange) {
    operands.data = &instruction.operands.at(1);
    operands.scalar_base = &instruction.operands.at(2);
    if (!instruction.asm_text.empty()) {
      operands.address = &first;
    } else {
      operands.return_dest = &first;
    }
    return operands;
  }
  throw std::invalid_argument("unsupported flat atomic operand layout");
}

inline uint64_t ResolveFlatAtomicAddress(const FlatAtomicOperands& operands,
                                         const EncodedWaveContext& context,
                                         uint32_t lane) {
  if (operands.address != nullptr &&
      operands.address->kind == DecodedInstructionOperandKind::VectorRegRange) {
    const auto [first, _] = RequireVectorRange(*operands.address);
    const uint64_t lo = static_cast<uint32_t>(context.wave.vgpr.Read(first, lane));
    const uint64_t hi = static_cast<uint32_t>(context.wave.vgpr.Read(first + 1, lane));
    return static_cast<uint64_t>(static_cast<int64_t>((hi << 32u) | lo) + operands.offset);
  }
  if (operands.scalar_base != nullptr) {
    const auto [first, _] = RequireScalarRange(*operands.scalar_base);
    const uint64_t base = static_cast<uint64_t>(context.wave.sgpr.Read(first)) |
                          (static_cast<uint64_t>(context.wave.sgpr.Read(first + 1)) << 32u);
    if (operands.address != nullptr) {
      const int32_t lane_offset =
          static_cast<int32_t>(ResolveVectorLane(*operands.address, context, lane));
      return static_cast<uint64_t>(static_cast<int64_t>(base) + lane_offset + operands.offset);
    }
    return static_cast<uint64_t>(static_cast<int64_t>(base) + operands.offset);
  }
  throw std::invalid_argument("flat atomic address source is missing");
}

inline void StoreFlatAtomicReturnValue(const FlatAtomicOperands& operands,
                                       EncodedWaveContext& context,
                                       uint32_t lane,
                                       uint32_t old_value) {
  if (operands.return_dest == nullptr) {
    return;
  }
  context.wave.vgpr.Write(RequireVectorIndex(*operands.return_dest), lane, old_value);
}

// --- Handler base classes ---

class BaseHandler : public IEncodedSemanticHandler {
 public:
  void Execute(const DecodedInstruction& instruction, EncodedWaveContext& context) const override {
    if (context.on_execute) {
      context.on_execute(instruction, context, "start");
    }
    try {
      ExecuteImpl(instruction, context);
    } catch (const std::exception& error) {
      RethrowWithInstructionContext(error, instruction);
    }
    if (context.on_execute) {
      context.on_execute(instruction, context, "end");
    }
    context.wave.pc += instruction.size_bytes;
  }

 protected:
  virtual void ExecuteImpl(const DecodedInstruction& instruction,
                           EncodedWaveContext& context) const = 0;
};

template <typename Impl>
class VectorLaneHandler : public BaseHandler {
 protected:
  void ExecuteImpl(const DecodedInstruction& instruction,
                   EncodedWaveContext& context) const override {
    ForEachActiveLane(context, [&](uint32_t lane) {
      static_cast<const Impl*>(this)->ExecuteLane(instruction, context, lane);
    });
  }

 private:
  void ForEachActiveLane(EncodedWaveContext& context, auto&& fn) const {
    for (uint32_t lane = 0; lane < LaneCount(context); ++lane) {
      if (context.wave.exec.test(lane)) {
        fn(lane);
      }
    }
  }
};

// --- Handler registry ---

class HandlerRegistry {
 public:
  static HandlerRegistry& MutableInstance() {
    static HandlerRegistry kInstance;
    return kInstance;
  }
  static const HandlerRegistry& Instance() {
    return MutableInstance();
  }

  const IEncodedSemanticHandler* Find(std::string_view mnemonic) const {
    auto it = map_.find(mnemonic);
    if (it != map_.end()) {
      return it->second;
    }
    return nullptr;
  }

  void Register(std::string_view mnemonic, const IEncodedSemanticHandler* handler) {
    map_[mnemonic] = handler;
  }

 private:
  HandlerRegistry() = default;
  std::unordered_map<std::string_view, const IEncodedSemanticHandler*> map_;
};

}  // namespace handler_support
}  // namespace gpu_model
