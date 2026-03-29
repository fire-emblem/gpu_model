#include "gpu_model/exec/semantic_handler.h"

#include <algorithm>
#include <bit>
#include <bitset>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace gpu_model {

namespace {

int64_t AsSigned(uint64_t value) {
  return static_cast<int64_t>(value);
}

uint32_t RequireVectorReg(const Operand& operand) {
  if (operand.kind != OperandKind::Register || operand.reg.file != RegisterFile::Vector) {
    throw std::invalid_argument("expected vector register operand");
  }
  return operand.reg.index;
}

uint32_t RequireScalarReg(const Operand& operand) {
  if (operand.kind != OperandKind::Register || operand.reg.file != RegisterFile::Scalar) {
    throw std::invalid_argument("expected scalar register operand");
  }
  return operand.reg.index;
}

uint32_t LocalLinearId(const WaveState& wave, uint32_t lane) {
  return wave.wave_id * kWaveSize + lane;
}

uint32_t LocalIdY(const WaveState& wave, const LaunchConfig& config, uint32_t lane) {
  return (LocalLinearId(wave, lane) / config.block_dim_x) % config.block_dim_y;
}

uint32_t LocalIdZ(const WaveState& wave, const LaunchConfig& config, uint32_t lane) {
  return LocalLinearId(wave, lane) / (config.block_dim_x * config.block_dim_y);
}

uint64_t ReadScalarOperand(const Operand& operand, const WaveState& wave) {
  switch (operand.kind) {
    case OperandKind::Immediate:
    case OperandKind::ArgumentIndex:
    case OperandKind::BranchTarget:
      return operand.immediate;
    case OperandKind::Register:
      if (operand.reg.file != RegisterFile::Scalar) {
        throw std::invalid_argument("expected scalar register operand");
      }
      return wave.sgpr.Read(operand.reg.index);
    case OperandKind::None:
      break;
  }
  throw std::invalid_argument("unsupported scalar operand kind");
}

uint64_t ReadVectorLaneOperand(const Operand& operand, const WaveState& wave, uint32_t lane) {
  switch (operand.kind) {
    case OperandKind::Immediate:
    case OperandKind::ArgumentIndex:
    case OperandKind::BranchTarget:
      return operand.immediate;
    case OperandKind::Register:
      if (operand.reg.file == RegisterFile::Scalar) {
        return wave.sgpr.Read(operand.reg.index);
      }
      return wave.vgpr.Read(operand.reg.index, lane);
    case OperandKind::None:
      break;
  }
  throw std::invalid_argument("unsupported vector operand kind");
}

std::bitset<64> ThreadMask(const WaveState& wave) {
  std::bitset<64> mask;
  for (uint32_t lane = 0; lane < wave.thread_count && lane < kWaveSize; ++lane) {
    mask.set(lane);
  }
  return mask;
}

class BuiltinHandler final : public ISemanticHandler {
 public:
  SemanticFamily family() const override { return SemanticFamily::Builtin; }

  OpPlan Build(const Instruction& instruction,
               const WaveState& wave,
               const ExecutionContext& context) const override {
    OpPlan plan;
    switch (instruction.opcode) {
      case Opcode::SysLoadArg: {
        const uint32_t dest = RequireScalarReg(instruction.operands.at(0));
        const uint32_t arg_index = static_cast<uint32_t>(instruction.operands.at(1).immediate);
        plan.scalar_writes.push_back(
            ScalarWrite{.reg_index = dest, .value = context.args.GetU64(arg_index)});
        return plan;
      }
      case Opcode::SysGlobalIdX: {
        const uint32_t dest = RequireVectorReg(instruction.operands.at(0));
        VectorWrite write;
        write.reg_index = dest;
        write.mask = ThreadMask(wave);
        for (uint32_t lane = 0; lane < wave.thread_count && lane < kWaveSize; ++lane) {
          const uint32_t local_linear = LocalLinearId(wave, lane);
          const uint32_t local_x = local_linear % context.launch_config.block_dim_x;
          write.values[lane] = static_cast<uint64_t>(
              wave.block_idx_x * context.launch_config.block_dim_x + local_x);
        }
        plan.vector_writes.push_back(write);
        return plan;
      }
      case Opcode::SysGlobalIdY: {
        const uint32_t dest = RequireVectorReg(instruction.operands.at(0));
        VectorWrite write;
        write.reg_index = dest;
        write.mask = ThreadMask(wave);
        for (uint32_t lane = 0; lane < wave.thread_count && lane < kWaveSize; ++lane) {
          const uint32_t local_linear = LocalLinearId(wave, lane);
          const uint32_t local_y = (local_linear / context.launch_config.block_dim_x) %
                                   context.launch_config.block_dim_y;
          write.values[lane] = static_cast<uint64_t>(
              wave.block_idx_y * context.launch_config.block_dim_y + local_y);
        }
        plan.vector_writes.push_back(write);
        return plan;
      }
      case Opcode::SysGlobalIdZ: {
        const uint32_t dest = RequireVectorReg(instruction.operands.at(0));
        VectorWrite write;
        write.reg_index = dest;
        write.mask = ThreadMask(wave);
        for (uint32_t lane = 0; lane < wave.thread_count && lane < kWaveSize; ++lane) {
          const uint32_t local_z = LocalIdZ(wave, context.launch_config, lane);
          write.values[lane] = static_cast<uint64_t>(
              wave.block_idx_z * context.launch_config.block_dim_z + local_z);
        }
        plan.vector_writes.push_back(write);
        return plan;
      }
      case Opcode::SysLocalIdX: {
        const uint32_t dest = RequireVectorReg(instruction.operands.at(0));
        VectorWrite write;
        write.reg_index = dest;
        write.mask = ThreadMask(wave);
        for (uint32_t lane = 0; lane < wave.thread_count && lane < kWaveSize; ++lane) {
          write.values[lane] = static_cast<uint64_t>(
              LocalLinearId(wave, lane) % context.launch_config.block_dim_x);
        }
        plan.vector_writes.push_back(write);
        return plan;
      }
      case Opcode::SysLocalIdY: {
        const uint32_t dest = RequireVectorReg(instruction.operands.at(0));
        VectorWrite write;
        write.reg_index = dest;
        write.mask = ThreadMask(wave);
        for (uint32_t lane = 0; lane < wave.thread_count && lane < kWaveSize; ++lane) {
          write.values[lane] = static_cast<uint64_t>(LocalIdY(wave, context.launch_config, lane));
        }
        plan.vector_writes.push_back(write);
        return plan;
      }
      case Opcode::SysLocalIdZ: {
        const uint32_t dest = RequireVectorReg(instruction.operands.at(0));
        VectorWrite write;
        write.reg_index = dest;
        write.mask = ThreadMask(wave);
        for (uint32_t lane = 0; lane < wave.thread_count && lane < kWaveSize; ++lane) {
          write.values[lane] = static_cast<uint64_t>(LocalIdZ(wave, context.launch_config, lane));
        }
        plan.vector_writes.push_back(write);
        return plan;
      }
      case Opcode::SysBlockOffsetX:
      case Opcode::SysBlockIdxX:
      case Opcode::SysBlockIdxY:
      case Opcode::SysBlockIdxZ:
      case Opcode::SysBlockDimX:
      case Opcode::SysBlockDimY:
      case Opcode::SysBlockDimZ:
      case Opcode::SysGridDimX:
      case Opcode::SysGridDimY:
      case Opcode::SysGridDimZ:
      case Opcode::SysLaneId:
        break;
      default:
        throw std::invalid_argument("builtin handler received unsupported opcode");
    }

    const Operand& dest = instruction.operands.at(0);
    if (dest.kind != OperandKind::Register) {
      throw std::invalid_argument("builtin requires register destination");
    }

    auto emit_scalar_or_vector = [&](uint64_t value) {
      if (dest.reg.file == RegisterFile::Scalar) {
        plan.scalar_writes.push_back(ScalarWrite{.reg_index = dest.reg.index, .value = value});
      } else {
        VectorWrite write;
        write.reg_index = dest.reg.index;
        write.mask = ThreadMask(wave);
        for (uint32_t lane = 0; lane < wave.thread_count && lane < kWaveSize; ++lane) {
          write.values[lane] = value;
        }
        plan.vector_writes.push_back(write);
      }
    };

    switch (instruction.opcode) {
      case Opcode::SysBlockOffsetX:
        emit_scalar_or_vector(static_cast<uint64_t>(wave.block_idx_x) *
                              static_cast<uint64_t>(context.launch_config.block_dim_x));
        return plan;
      case Opcode::SysBlockIdxX:
        emit_scalar_or_vector(wave.block_idx_x);
        return plan;
      case Opcode::SysBlockIdxY:
        emit_scalar_or_vector(wave.block_idx_y);
        return plan;
      case Opcode::SysBlockIdxZ:
        emit_scalar_or_vector(wave.block_idx_z);
        return plan;
      case Opcode::SysBlockDimX:
        emit_scalar_or_vector(context.launch_config.block_dim_x);
        return plan;
      case Opcode::SysBlockDimY:
        emit_scalar_or_vector(context.launch_config.block_dim_y);
        return plan;
      case Opcode::SysBlockDimZ:
        emit_scalar_or_vector(context.launch_config.block_dim_z);
        return plan;
      case Opcode::SysGridDimX:
        emit_scalar_or_vector(context.launch_config.grid_dim_x);
        return plan;
      case Opcode::SysGridDimY:
        emit_scalar_or_vector(context.launch_config.grid_dim_y);
        return plan;
      case Opcode::SysGridDimZ:
        emit_scalar_or_vector(context.launch_config.grid_dim_z);
        return plan;
      case Opcode::SysLaneId: {
        if (dest.reg.file == RegisterFile::Scalar) {
          plan.scalar_writes.push_back(ScalarWrite{.reg_index = dest.reg.index, .value = 0});
        } else {
          VectorWrite write;
          write.reg_index = dest.reg.index;
          write.mask = ThreadMask(wave);
          for (uint32_t lane = 0; lane < wave.thread_count && lane < kWaveSize; ++lane) {
            write.values[lane] = lane;
          }
          plan.vector_writes.push_back(write);
        }
        return plan;
      }
      default:
        break;
    }
    throw std::invalid_argument("builtin handler fell through");
  }
};

class ScalarAluHandler final : public ISemanticHandler {
 public:
  SemanticFamily family() const override { return SemanticFamily::ScalarAlu; }

  OpPlan Build(const Instruction& instruction,
               const WaveState& wave,
               const ExecutionContext&) const override {
    OpPlan plan;
    const uint32_t dest = RequireScalarReg(instruction.operands.at(0));
    const uint64_t lhs = instruction.operands.size() > 1
                             ? ReadScalarOperand(instruction.operands.at(1), wave)
                             : 0;
    const uint64_t rhs = instruction.operands.size() > 2
                             ? ReadScalarOperand(instruction.operands.at(2), wave)
                             : 0;

    switch (instruction.opcode) {
      case Opcode::SMov:
        plan.scalar_writes.push_back(ScalarWrite{.reg_index = dest, .value = lhs});
        return plan;
      case Opcode::SAdd:
        plan.scalar_writes.push_back(ScalarWrite{.reg_index = dest, .value = lhs + rhs});
        return plan;
      case Opcode::SSub:
        plan.scalar_writes.push_back(ScalarWrite{.reg_index = dest, .value = lhs - rhs});
        return plan;
      case Opcode::SMul:
        plan.scalar_writes.push_back(ScalarWrite{.reg_index = dest, .value = lhs * rhs});
        return plan;
      case Opcode::SDiv: {
        const int64_t slhs = AsSigned(lhs);
        const int64_t srhs = AsSigned(rhs);
        plan.scalar_writes.push_back(
            ScalarWrite{.reg_index = dest, .value = srhs == 0 ? 0ULL : static_cast<uint64_t>(slhs / srhs)});
        return plan;
      }
      case Opcode::SRem: {
        const int64_t slhs = AsSigned(lhs);
        const int64_t srhs = AsSigned(rhs);
        plan.scalar_writes.push_back(
            ScalarWrite{.reg_index = dest, .value = srhs == 0 ? 0ULL : static_cast<uint64_t>(slhs % srhs)});
        return plan;
      }
      case Opcode::SAnd:
        plan.scalar_writes.push_back(ScalarWrite{.reg_index = dest, .value = lhs & rhs});
        return plan;
      case Opcode::SOr:
        plan.scalar_writes.push_back(ScalarWrite{.reg_index = dest, .value = lhs | rhs});
        return plan;
      case Opcode::SXor:
        plan.scalar_writes.push_back(ScalarWrite{.reg_index = dest, .value = lhs ^ rhs});
        return plan;
      case Opcode::SShl:
        plan.scalar_writes.push_back(
            ScalarWrite{.reg_index = dest, .value = lhs << (rhs & 63ULL)});
        return plan;
      case Opcode::SShr:
        plan.scalar_writes.push_back(
            ScalarWrite{.reg_index = dest, .value = lhs >> (rhs & 63ULL)});
        return plan;
      default:
        throw std::invalid_argument("scalar ALU handler received unsupported opcode");
    }
  }
};

class ScalarCompareHandler final : public ISemanticHandler {
 public:
  SemanticFamily family() const override { return SemanticFamily::ScalarCompare; }

  OpPlan Build(const Instruction& instruction,
               const WaveState& wave,
               const ExecutionContext&) const override {
    OpPlan plan;
    switch (instruction.opcode) {
      case Opcode::SWaitCnt:
        plan.wait_cnt = true;
        return plan;
      case Opcode::SBufferLoadDword: {
        MemoryRequest request;
        request.space = MemorySpace::Constant;
        request.kind = AccessKind::Load;
        request.dst = RegRef{.file = RegisterFile::Scalar,
                             .index = RequireScalarReg(instruction.operands.at(0))};
        request.block_id = wave.block_id;
        request.wave_id = wave.wave_id;
        const uint64_t scale = ReadScalarOperand(instruction.operands.at(2), wave);
        const uint64_t offset = instruction.operands.size() > 3
                                    ? ReadScalarOperand(instruction.operands.at(3), wave)
                                    : 0;
        const uint64_t index = ReadScalarOperand(instruction.operands.at(1), wave);
        request.lanes[0] = LaneAccess{
            .active = true,
            .addr = offset + index * scale,
            .bytes = static_cast<uint32_t>(scale),
        };
        plan.memory = request;
        return plan;
      }
      case Opcode::SCmpLt:
      case Opcode::SCmpEq:
      case Opcode::SCmpGt:
      case Opcode::SCmpGe: {
        const int64_t lhs = AsSigned(ReadScalarOperand(instruction.operands.at(0), wave));
        const int64_t rhs = AsSigned(ReadScalarOperand(instruction.operands.at(1), wave));
        bool result = false;
        if (instruction.opcode == Opcode::SCmpLt) {
          result = lhs < rhs;
        } else if (instruction.opcode == Opcode::SCmpEq) {
          result = lhs == rhs;
        } else if (instruction.opcode == Opcode::SCmpGt) {
          result = lhs > rhs;
        } else {
          result = lhs >= rhs;
        }
        plan.smask_write = result ? 1ULL : 0ULL;
        return plan;
      }
      default:
        throw std::invalid_argument("scalar compare handler received unsupported opcode");
    }
  }
};

class VectorAluHandler final : public ISemanticHandler {
 public:
  explicit VectorAluHandler(SemanticFamily family) : family_(family) {}

  SemanticFamily family() const override { return family_; }

  OpPlan Build(const Instruction& instruction,
               const WaveState& wave,
               const ExecutionContext&) const override {
    OpPlan plan;
    auto make_write = [&](uint32_t reg_index) {
      VectorWrite write;
      write.reg_index = reg_index;
      write.mask = wave.exec;
      return write;
    };

    switch (instruction.opcode) {
      case Opcode::VMov: {
        VectorWrite write = make_write(RequireVectorReg(instruction.operands.at(0)));
        for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
          if (wave.exec.test(lane)) {
            write.values[lane] = ReadVectorLaneOperand(instruction.operands.at(1), wave, lane);
          }
        }
        plan.vector_writes.push_back(write);
        return plan;
      }
      case Opcode::VAdd:
      case Opcode::VAnd:
      case Opcode::VOr:
      case Opcode::VXor:
      case Opcode::VShl:
      case Opcode::VShr:
      case Opcode::VSub:
      case Opcode::VDiv:
      case Opcode::VRem:
      case Opcode::VMul:
      case Opcode::VMin:
      case Opcode::VMax:
      case Opcode::VAddF32: {
        VectorWrite write = make_write(RequireVectorReg(instruction.operands.at(0)));
        for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
          if (!wave.exec.test(lane)) {
            continue;
          }
          const uint64_t lhs = ReadVectorLaneOperand(instruction.operands.at(1), wave, lane);
          const uint64_t rhs = ReadVectorLaneOperand(instruction.operands.at(2), wave, lane);
          switch (instruction.opcode) {
            case Opcode::VAdd:
              write.values[lane] = static_cast<uint64_t>(AsSigned(lhs) + AsSigned(rhs));
              break;
            case Opcode::VAnd:
              write.values[lane] = lhs & rhs;
              break;
            case Opcode::VOr:
              write.values[lane] = lhs | rhs;
              break;
            case Opcode::VXor:
              write.values[lane] = lhs ^ rhs;
              break;
            case Opcode::VShl:
              write.values[lane] = lhs << (rhs & 63ULL);
              break;
            case Opcode::VShr:
              write.values[lane] = lhs >> (rhs & 63ULL);
              break;
            case Opcode::VSub:
              write.values[lane] = static_cast<uint64_t>(AsSigned(lhs) - AsSigned(rhs));
              break;
            case Opcode::VDiv:
              write.values[lane] =
                  AsSigned(rhs) == 0 ? 0ULL : static_cast<uint64_t>(AsSigned(lhs) / AsSigned(rhs));
              break;
            case Opcode::VRem:
              write.values[lane] =
                  AsSigned(rhs) == 0 ? 0ULL : static_cast<uint64_t>(AsSigned(lhs) % AsSigned(rhs));
              break;
            case Opcode::VMul:
              write.values[lane] = static_cast<uint64_t>(AsSigned(lhs) * AsSigned(rhs));
              break;
            case Opcode::VMin:
              write.values[lane] = static_cast<uint64_t>(std::min(AsSigned(lhs), AsSigned(rhs)));
              break;
            case Opcode::VMax:
              write.values[lane] = static_cast<uint64_t>(std::max(AsSigned(lhs), AsSigned(rhs)));
              break;
            case Opcode::VAddF32: {
              const float flhs = std::bit_cast<float>(static_cast<uint32_t>(lhs));
              const float frhs = std::bit_cast<float>(static_cast<uint32_t>(rhs));
              write.values[lane] = std::bit_cast<uint32_t>(flhs + frhs);
              break;
            }
            default:
              break;
          }
        }
        plan.vector_writes.push_back(write);
        return plan;
      }
      case Opcode::VFma: {
        VectorWrite write = make_write(RequireVectorReg(instruction.operands.at(0)));
        for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
          if (!wave.exec.test(lane)) {
            continue;
          }
          const int64_t lhs = AsSigned(ReadVectorLaneOperand(instruction.operands.at(1), wave, lane));
          const int64_t rhs = AsSigned(ReadVectorLaneOperand(instruction.operands.at(2), wave, lane));
          const int64_t addend =
              AsSigned(ReadVectorLaneOperand(instruction.operands.at(3), wave, lane));
          write.values[lane] = static_cast<uint64_t>(lhs * rhs + addend);
        }
        plan.vector_writes.push_back(write);
        return plan;
      }
      case Opcode::VSelectCmask: {
        VectorWrite write = make_write(RequireVectorReg(instruction.operands.at(0)));
        for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
          if (!wave.exec.test(lane)) {
            continue;
          }
          const Operand& selected = wave.cmask.test(lane) ? instruction.operands.at(1)
                                                          : instruction.operands.at(2);
          write.values[lane] = ReadVectorLaneOperand(selected, wave, lane);
        }
        plan.vector_writes.push_back(write);
        return plan;
      }
      case Opcode::VCmpLtCmask:
      case Opcode::VCmpEqCmask:
      case Opcode::VCmpGeCmask:
      case Opcode::VCmpGtCmask: {
        std::bitset<64> cmask;
        for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
          if (!wave.exec.test(lane)) {
            continue;
          }
          const int64_t lhs = AsSigned(ReadVectorLaneOperand(instruction.operands.at(0), wave, lane));
          const int64_t rhs = AsSigned(ReadVectorLaneOperand(instruction.operands.at(1), wave, lane));
          bool result = false;
          if (instruction.opcode == Opcode::VCmpLtCmask) {
            result = lhs < rhs;
          } else if (instruction.opcode == Opcode::VCmpEqCmask) {
            result = lhs == rhs;
          } else if (instruction.opcode == Opcode::VCmpGeCmask) {
            result = lhs >= rhs;
          } else {
            result = lhs > rhs;
          }
          if (result) {
            cmask.set(lane);
          }
        }
        plan.cmask_write = cmask;
        return plan;
      }
      default:
        throw std::invalid_argument("vector ALU handler received unsupported opcode");
    }
  }

 private:
  SemanticFamily family_;
};

class VectorMemoryHandler final : public ISemanticHandler {
 public:
  explicit VectorMemoryHandler(SemanticFamily family) : family_(family) {}

  SemanticFamily family() const override { return family_; }

  OpPlan Build(const Instruction& instruction,
               const WaveState& wave,
               const ExecutionContext&) const override {
    OpPlan plan;
    MemoryRequest request;
    request.exec_snapshot = wave.exec;
    request.block_id = wave.block_id;
    request.wave_id = wave.wave_id;

    switch (instruction.opcode) {
      case Opcode::MLoadGlobal: {
        request.space = MemorySpace::Global;
        request.kind = AccessKind::Load;
        request.dst = RegRef{.file = RegisterFile::Vector,
                             .index = RequireVectorReg(instruction.operands.at(0))};
        const uint64_t scale = ReadScalarOperand(instruction.operands.at(3), wave);
        const uint64_t offset = instruction.operands.size() > 4
                                    ? ReadScalarOperand(instruction.operands.at(4), wave)
                                    : 0;
        for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
          if (!wave.exec.test(lane)) {
            continue;
          }
          const uint64_t base = ReadScalarOperand(instruction.operands.at(1), wave);
          const uint64_t index = ReadVectorLaneOperand(instruction.operands.at(2), wave, lane);
          request.lanes[lane] = LaneAccess{
              .active = true,
              .addr = base + offset + index * scale,
              .bytes = static_cast<uint32_t>(scale),
          };
        }
        plan.memory = request;
        return plan;
      }
      case Opcode::MLoadGlobalAddr: {
        request.space = MemorySpace::Global;
        request.kind = AccessKind::Load;
        request.dst = RegRef{.file = RegisterFile::Vector,
                             .index = RequireVectorReg(instruction.operands.at(0))};
        const uint64_t offset = instruction.operands.size() > 3
                                    ? ReadScalarOperand(instruction.operands.at(3), wave)
                                    : 0;
        for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
          if (!wave.exec.test(lane)) {
            continue;
          }
          const uint64_t lo = ReadVectorLaneOperand(instruction.operands.at(1), wave, lane) &
                              0xffffffffULL;
          const uint64_t hi = ReadVectorLaneOperand(instruction.operands.at(2), wave, lane) &
                              0xffffffffULL;
          request.lanes[lane] = LaneAccess{
              .active = true,
              .addr = ((hi << 32ULL) | lo) + offset,
              .bytes = 4,
          };
        }
        plan.memory = request;
        return plan;
      }
      case Opcode::MStoreGlobal: {
        request.space = MemorySpace::Global;
        request.kind = AccessKind::Store;
        const uint64_t scale = ReadScalarOperand(instruction.operands.at(3), wave);
        const uint64_t offset = instruction.operands.size() > 4
                                    ? ReadScalarOperand(instruction.operands.at(4), wave)
                                    : 0;
        for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
          if (!wave.exec.test(lane)) {
            continue;
          }
          const uint64_t base = ReadScalarOperand(instruction.operands.at(0), wave);
          const uint64_t index = ReadVectorLaneOperand(instruction.operands.at(1), wave, lane);
          const uint64_t value = ReadVectorLaneOperand(instruction.operands.at(2), wave, lane);
          request.lanes[lane] = LaneAccess{
              .active = true,
              .addr = base + offset + index * scale,
              .bytes = static_cast<uint32_t>(scale),
              .value = value,
          };
        }
        plan.memory = request;
        return plan;
      }
      case Opcode::MStoreGlobalAddr: {
        request.space = MemorySpace::Global;
        request.kind = AccessKind::Store;
        const uint64_t offset = instruction.operands.size() > 3
                                    ? ReadScalarOperand(instruction.operands.at(3), wave)
                                    : 0;
        for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
          if (!wave.exec.test(lane)) {
            continue;
          }
          const uint64_t lo = ReadVectorLaneOperand(instruction.operands.at(0), wave, lane) &
                              0xffffffffULL;
          const uint64_t hi = ReadVectorLaneOperand(instruction.operands.at(1), wave, lane) &
                              0xffffffffULL;
          const uint64_t value = ReadVectorLaneOperand(instruction.operands.at(2), wave, lane);
          request.lanes[lane] = LaneAccess{
              .active = true,
              .addr = ((hi << 32ULL) | lo) + offset,
              .bytes = 4,
              .value = value,
          };
        }
        plan.memory = request;
        return plan;
      }
      case Opcode::MAtomicAddGlobal: {
        request.space = MemorySpace::Global;
        request.kind = AccessKind::Atomic;
        const uint64_t scale = ReadScalarOperand(instruction.operands.at(3), wave);
        const uint64_t offset = instruction.operands.size() > 4
                                    ? ReadScalarOperand(instruction.operands.at(4), wave)
                                    : 0;
        for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
          if (!wave.exec.test(lane)) {
            continue;
          }
          const uint64_t base = ReadScalarOperand(instruction.operands.at(0), wave);
          const uint64_t index = ReadVectorLaneOperand(instruction.operands.at(1), wave, lane);
          const uint64_t value = ReadVectorLaneOperand(instruction.operands.at(2), wave, lane);
          request.lanes[lane] = LaneAccess{
              .active = true,
              .addr = base + offset + index * scale,
              .bytes = static_cast<uint32_t>(scale),
              .value = value,
          };
        }
        plan.memory = request;
        return plan;
      }
      case Opcode::MLoadShared:
      case Opcode::MLoadPrivate:
      case Opcode::MLoadConst: {
        request.kind = AccessKind::Load;
        request.dst = RegRef{.file = RegisterFile::Vector,
                             .index = RequireVectorReg(instruction.operands.at(0))};
        request.space = instruction.opcode == Opcode::MLoadShared   ? MemorySpace::Shared
                        : instruction.opcode == Opcode::MLoadPrivate ? MemorySpace::Private
                                                                    : MemorySpace::Constant;
        const uint64_t scale = ReadScalarOperand(instruction.operands.at(2), wave);
        const uint64_t offset = (instruction.opcode == Opcode::MLoadShared &&
                                 instruction.operands.size() > 3)
                                    ? ReadScalarOperand(instruction.operands.at(3), wave)
                                    : 0;
        for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
          if (!wave.exec.test(lane)) {
            continue;
          }
          const uint64_t index = ReadVectorLaneOperand(instruction.operands.at(1), wave, lane);
          request.lanes[lane] = LaneAccess{
              .active = true,
              .addr = offset + index * scale,
              .bytes = static_cast<uint32_t>(scale),
          };
        }
        plan.memory = request;
        return plan;
      }
      case Opcode::MStoreShared:
      case Opcode::MAtomicAddShared:
      case Opcode::MStorePrivate: {
        request.kind = instruction.opcode == Opcode::MAtomicAddShared ? AccessKind::Atomic
                                                                      : AccessKind::Store;
        request.space = instruction.opcode == Opcode::MStorePrivate ? MemorySpace::Private
                                                                    : MemorySpace::Shared;
        const uint64_t scale = ReadScalarOperand(instruction.operands.at(2), wave);
        for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
          if (!wave.exec.test(lane)) {
            continue;
          }
          const uint64_t index = ReadVectorLaneOperand(instruction.operands.at(0), wave, lane);
          const uint64_t value = ReadVectorLaneOperand(instruction.operands.at(1), wave, lane);
          request.lanes[lane] = LaneAccess{
              .active = true,
              .addr = index * scale,
              .bytes = static_cast<uint32_t>(scale),
              .value = value,
          };
        }
        plan.memory = request;
        return plan;
      }
      default:
        throw std::invalid_argument("vector memory handler received unsupported opcode");
    }
  }

 private:
  SemanticFamily family_;
};

class MaskHandler final : public ISemanticHandler {
 public:
  SemanticFamily family() const override { return SemanticFamily::Mask; }

  OpPlan Build(const Instruction& instruction,
               const WaveState& wave,
               const ExecutionContext&) const override {
    OpPlan plan;
    switch (instruction.opcode) {
      case Opcode::MaskSaveExec: {
        const uint32_t dest = RequireScalarReg(instruction.operands.at(0));
        plan.scalar_writes.push_back(
            ScalarWrite{.reg_index = dest, .value = wave.exec.to_ullong()});
        return plan;
      }
      case Opcode::MaskRestoreExec: {
        const uint32_t src = RequireScalarReg(instruction.operands.at(0));
        plan.exec_write = std::bitset<64>(wave.sgpr.Read(src));
        return plan;
      }
      case Opcode::MaskAndExecCmask:
        plan.exec_write = wave.exec & wave.cmask;
        return plan;
      default:
        throw std::invalid_argument("mask handler received unsupported opcode");
    }
  }
};

class BranchHandler final : public ISemanticHandler {
 public:
  SemanticFamily family() const override { return SemanticFamily::Branch; }

  OpPlan Build(const Instruction& instruction,
               const WaveState& wave,
               const ExecutionContext&) const override {
    OpPlan plan;
    switch (instruction.opcode) {
      case Opcode::BBranch:
        plan.branch_target = instruction.operands.at(0).immediate;
        return plan;
      case Opcode::BIfSmask:
        if (wave.ScalarMaskBit0()) {
          plan.branch_target = instruction.operands.at(0).immediate;
        }
        return plan;
      case Opcode::BIfNoexec:
        if (wave.exec.none()) {
          plan.branch_target = instruction.operands.at(0).immediate;
        }
        return plan;
      default:
        throw std::invalid_argument("branch handler received unsupported opcode");
    }
  }
};

class SyncSpecialHandler final : public ISemanticHandler {
 public:
  explicit SyncSpecialHandler(SemanticFamily family) : family_(family) {}

  SemanticFamily family() const override { return family_; }

  OpPlan Build(const Instruction& instruction,
               const WaveState&,
               const ExecutionContext&) const override {
    OpPlan plan;
    switch (instruction.opcode) {
      case Opcode::SWaitCnt:
        plan.wait_cnt = true;
        return plan;
      case Opcode::SyncWaveBarrier:
        plan.sync_wave_barrier = true;
        return plan;
      case Opcode::SyncBarrier:
        plan.sync_barrier = true;
        plan.advance_pc = false;
        return plan;
      case Opcode::BExit:
        plan.exit_wave = true;
        return plan;
      default:
        throw std::invalid_argument("sync/special handler received unsupported opcode");
    }
  }

 private:
  SemanticFamily family_;
};

struct HandlerBinding {
  SemanticFamily family = SemanticFamily::Special;
  const ISemanticHandler* handler = nullptr;
};

const std::vector<HandlerBinding>& HandlerBindings() {
  static const BuiltinHandler kBuiltinHandler;
  static const ScalarAluHandler kScalarAluHandler;
  static const ScalarCompareHandler kScalarCompareHandler;
  static const VectorAluHandler kVectorAluIntHandler(SemanticFamily::VectorAluInt);
  static const VectorAluHandler kVectorAluFloatHandler(SemanticFamily::VectorAluFloat);
  static const VectorAluHandler kVectorCompareHandler(SemanticFamily::VectorCompare);
  static const VectorMemoryHandler kVectorMemoryHandler(SemanticFamily::VectorMemory);
  static const VectorMemoryHandler kLocalDataShareHandler(SemanticFamily::LocalDataShare);
  static const MaskHandler kMaskHandler;
  static const BranchHandler kBranchHandler;
  static const SyncSpecialHandler kSyncHandler(SemanticFamily::Sync);
  static const SyncSpecialHandler kSpecialHandler(SemanticFamily::Special);
  static const std::vector<HandlerBinding> kBindings = {
      {.family = SemanticFamily::Builtin, .handler = &kBuiltinHandler},
      {.family = SemanticFamily::ScalarAlu, .handler = &kScalarAluHandler},
      {.family = SemanticFamily::ScalarCompare, .handler = &kScalarCompareHandler},
      {.family = SemanticFamily::ScalarMemory, .handler = &kScalarCompareHandler},
      {.family = SemanticFamily::VectorAluInt, .handler = &kVectorAluIntHandler},
      {.family = SemanticFamily::VectorAluFloat, .handler = &kVectorAluFloatHandler},
      {.family = SemanticFamily::VectorCompare, .handler = &kVectorCompareHandler},
      {.family = SemanticFamily::VectorMemory, .handler = &kVectorMemoryHandler},
      {.family = SemanticFamily::LocalDataShare, .handler = &kLocalDataShareHandler},
      {.family = SemanticFamily::Mask, .handler = &kMaskHandler},
      {.family = SemanticFamily::Branch, .handler = &kBranchHandler},
      {.family = SemanticFamily::Sync, .handler = &kSyncHandler},
      {.family = SemanticFamily::Special, .handler = &kSpecialHandler},
  };
  return kBindings;
}

}  // namespace

const ISemanticHandler& SemanticHandlerRegistry::Get(SemanticFamily family) {
  for (const auto& binding : HandlerBindings()) {
    if (binding.family == family) {
      return *binding.handler;
    }
  }
  throw std::invalid_argument("missing semantic handler binding");
}

OpPlan SemanticHandlerRegistry::Build(const Instruction& instruction,
                                      const WaveState& wave,
                                      const ExecutionContext& context) {
  return Get(GetOpcodeExecutionInfo(instruction.opcode).family).Build(instruction, wave, context);
}

}  // namespace gpu_model
