#include "instruction/semantics/internal/handler_support.h"
#include "utils/math/bit_utils.h"
#include "state/wave/barrier_wait_ops.h"

namespace gpu_model {
namespace semantics {

using handler_support::ThrowUnsupportedInstruction;

class BranchHandler final : public IEncodedSemanticHandler {
 public:
  void Execute(const DecodedInstruction& instruction, EncodedWaveContext& context) const override {
    if (context.on_execute) {
      context.on_execute(instruction, context, "start");
    }

    switch (instruction.encoding_id) {
      case 10: {  // s_cbranch_execz
      if (context.wave.exec.none()) {
        context.wave.pc =
            BranchTarget(context.wave.pc, static_cast<int32_t>(instruction.operands.at(0).info.immediate));
      } else {
        context.wave.pc += instruction.size_bytes;
      }
      break;
      }
      case 22: {  // s_cbranch_scc1
      if (context.wave.ScalarMaskBit0()) {
        context.wave.pc =
            BranchTarget(context.wave.pc, static_cast<int32_t>(instruction.operands.at(0).info.immediate));
      } else {
        context.wave.pc += instruction.size_bytes;
      }
      break;
      }
      case 26: {  // s_cbranch_scc0
      if (!context.wave.ScalarMaskBit0()) {
        context.wave.pc =
            BranchTarget(context.wave.pc, static_cast<int32_t>(instruction.operands.at(0).info.immediate));
      } else {
        context.wave.pc += instruction.size_bytes;
      }
      break;
      }
      case 43: {  // s_cbranch_vccz
      if (context.vcc == 0) {
        context.wave.pc =
            BranchTarget(context.wave.pc, static_cast<int32_t>(instruction.operands.at(0).info.immediate));
      } else {
        context.wave.pc += instruction.size_bytes;
      }
      break;
      }
      case 74: {  // s_cbranch_execnz
      if (context.wave.exec.any()) {
        context.wave.pc =
            BranchTarget(context.wave.pc, static_cast<int32_t>(instruction.operands.at(0).info.immediate));
      } else {
        context.wave.pc += instruction.size_bytes;
      }
      break;
      }
      case 27: {  // s_branch
      context.wave.pc =
          BranchTarget(context.wave.pc, static_cast<int32_t>(instruction.operands.at(0).info.immediate));
      break;
      }
      default:
        ThrowUnsupportedInstruction("unsupported branch opcode: ", instruction);
    }

    if (context.on_execute) {
      context.on_execute(instruction, context, "end");
    }
  }
};

class SpecialHandler final : public IEncodedSemanticHandler {
 public:
  void Execute(const DecodedInstruction& instruction, EncodedWaveContext& context) const override {
    if (context.on_execute) {
      context.on_execute(instruction, context, "start");
    }

    switch (instruction.encoding_id) {
      case 29: {  // s_barrier
        ++context.stats.barriers;
        MarkWaveAtBarrier(context.wave,
                          context.block.barrier_generation,
                          context.block.barrier_arrivals,
                          false);
        break;
      }
      case 68:  // s_nop
        context.wave.pc += instruction.size_bytes;
        break;
      case 12:  // s_waitcnt
        context.wave.pc += instruction.size_bytes;
        break;
      case 1:  // s_endpgm
        context.wave.status = WaveStatus::Exited;
        ++context.stats.wave_exits;
        break;
      default:
        ThrowUnsupportedInstruction("unsupported special opcode: ", instruction);
    }

    if (context.on_execute) {
      context.on_execute(instruction, context, "end");
    }
  }
};

static const BranchHandler kBranchHandler;
static const SpecialHandler kSpecialHandler;

const IEncodedSemanticHandler& GetBranchHandler() { return kBranchHandler; }
const IEncodedSemanticHandler& GetSpecialHandler() { return kSpecialHandler; }

}  // namespace semantics
}  // namespace gpu_model
