#include "gpu_model/exec/semantics.h"

#include <algorithm>
#include <bitset>
#include <cstdint>
#include <stdexcept>

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

}  // namespace

OpPlan Semantics::BuildPlan(const Instruction& instruction,
                            const WaveState& wave,
                            const ExecutionContext& context) const {
  OpPlan plan;

  switch (instruction.opcode) {
    case Opcode::SysLoadArg: {
      const uint32_t dest = RequireScalarReg(instruction.operands.at(0));
      const uint32_t arg_index = static_cast<uint32_t>(instruction.operands.at(1).immediate);
      plan.scalar_writes.push_back(ScalarWrite{.reg_index = dest, .value = context.args.GetU64(arg_index)});
      return plan;
    }
    case Opcode::SysGlobalIdX: {
      const uint32_t dest = RequireVectorReg(instruction.operands.at(0));
      VectorWrite write;
      write.reg_index = dest;
      write.mask = ThreadMask(wave);
      for (uint32_t lane = 0; lane < wave.thread_count && lane < kWaveSize; ++lane) {
        write.values[lane] =
            static_cast<uint64_t>(wave.block_id * context.launch_config.block_dim_x + wave.wave_id * kWaveSize + lane);
      }
      plan.vector_writes.push_back(write);
      return plan;
    }
    case Opcode::SysBlockIdxX: {
      const Operand& dest = instruction.operands.at(0);
      if (dest.kind != OperandKind::Register) {
        throw std::invalid_argument("sys_block_idx_x requires register destination");
      }
      if (dest.reg.file == RegisterFile::Scalar) {
        plan.scalar_writes.push_back(
            ScalarWrite{.reg_index = dest.reg.index, .value = wave.block_id});
      } else {
        VectorWrite write;
        write.reg_index = dest.reg.index;
        write.mask = ThreadMask(wave);
        for (uint32_t lane = 0; lane < wave.thread_count && lane < kWaveSize; ++lane) {
          write.values[lane] = wave.block_id;
        }
        plan.vector_writes.push_back(write);
      }
      return plan;
    }
    case Opcode::SysBlockDimX: {
      const Operand& dest = instruction.operands.at(0);
      if (dest.kind != OperandKind::Register) {
        throw std::invalid_argument("sys_block_dim_x requires register destination");
      }
      if (dest.reg.file == RegisterFile::Scalar) {
        plan.scalar_writes.push_back(
            ScalarWrite{.reg_index = dest.reg.index, .value = context.launch_config.block_dim_x});
      } else {
        VectorWrite write;
        write.reg_index = dest.reg.index;
        write.mask = ThreadMask(wave);
        for (uint32_t lane = 0; lane < wave.thread_count && lane < kWaveSize; ++lane) {
          write.values[lane] = context.launch_config.block_dim_x;
        }
        plan.vector_writes.push_back(write);
      }
      return plan;
    }
    case Opcode::SysLaneId: {
      const Operand& dest = instruction.operands.at(0);
      if (dest.kind != OperandKind::Register) {
        throw std::invalid_argument("sys_lane_id requires register destination");
      }
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
    case Opcode::SMov: {
      const uint32_t dest = RequireScalarReg(instruction.operands.at(0));
      const uint64_t src = ReadScalarOperand(instruction.operands.at(1), wave);
      plan.scalar_writes.push_back(ScalarWrite{.reg_index = dest, .value = src});
      return plan;
    }
    case Opcode::SAdd: {
      const uint32_t dest = RequireScalarReg(instruction.operands.at(0));
      const uint64_t lhs = ReadScalarOperand(instruction.operands.at(1), wave);
      const uint64_t rhs = ReadScalarOperand(instruction.operands.at(2), wave);
      plan.scalar_writes.push_back(ScalarWrite{.reg_index = dest, .value = lhs + rhs});
      return plan;
    }
    case Opcode::SMul: {
      const uint32_t dest = RequireScalarReg(instruction.operands.at(0));
      const uint64_t lhs = ReadScalarOperand(instruction.operands.at(1), wave);
      const uint64_t rhs = ReadScalarOperand(instruction.operands.at(2), wave);
      plan.scalar_writes.push_back(ScalarWrite{.reg_index = dest, .value = lhs * rhs});
      return plan;
    }
    case Opcode::SCmpLt: {
      const int64_t lhs = AsSigned(ReadScalarOperand(instruction.operands.at(0), wave));
      const int64_t rhs = AsSigned(ReadScalarOperand(instruction.operands.at(1), wave));
      plan.smask_write = lhs < rhs ? 1ULL : 0ULL;
      return plan;
    }
    case Opcode::SCmpEq: {
      const uint64_t lhs = ReadScalarOperand(instruction.operands.at(0), wave);
      const uint64_t rhs = ReadScalarOperand(instruction.operands.at(1), wave);
      plan.smask_write = lhs == rhs ? 1ULL : 0ULL;
      return plan;
    }
    case Opcode::VMov: {
      VectorWrite write;
      write.reg_index = RequireVectorReg(instruction.operands.at(0));
      write.mask = wave.exec;
      for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
        if (wave.exec.test(lane)) {
          write.values[lane] = ReadVectorLaneOperand(instruction.operands.at(1), wave, lane);
        }
      }
      plan.vector_writes.push_back(write);
      return plan;
    }
    case Opcode::VAdd: {
      VectorWrite write;
      write.reg_index = RequireVectorReg(instruction.operands.at(0));
      write.mask = wave.exec;
      for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
        if (wave.exec.test(lane)) {
          const int64_t lhs = AsSigned(ReadVectorLaneOperand(instruction.operands.at(1), wave, lane));
          const int64_t rhs = AsSigned(ReadVectorLaneOperand(instruction.operands.at(2), wave, lane));
          write.values[lane] = static_cast<uint64_t>(lhs + rhs);
        }
      }
      plan.vector_writes.push_back(write);
      return plan;
    }
    case Opcode::VMul: {
      VectorWrite write;
      write.reg_index = RequireVectorReg(instruction.operands.at(0));
      write.mask = wave.exec;
      for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
        if (wave.exec.test(lane)) {
          const int64_t lhs = AsSigned(ReadVectorLaneOperand(instruction.operands.at(1), wave, lane));
          const int64_t rhs = AsSigned(ReadVectorLaneOperand(instruction.operands.at(2), wave, lane));
          write.values[lane] = static_cast<uint64_t>(lhs * rhs);
        }
      }
      plan.vector_writes.push_back(write);
      return plan;
    }
    case Opcode::VFma: {
      VectorWrite write;
      write.reg_index = RequireVectorReg(instruction.operands.at(0));
      write.mask = wave.exec;
      for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
        if (wave.exec.test(lane)) {
          const int64_t lhs =
              AsSigned(ReadVectorLaneOperand(instruction.operands.at(1), wave, lane));
          const int64_t rhs =
              AsSigned(ReadVectorLaneOperand(instruction.operands.at(2), wave, lane));
          const int64_t addend =
              AsSigned(ReadVectorLaneOperand(instruction.operands.at(3), wave, lane));
          write.values[lane] = static_cast<uint64_t>(lhs * rhs + addend);
        }
      }
      plan.vector_writes.push_back(write);
      return plan;
    }
    case Opcode::VCmpLtCmask: {
      std::bitset<64> cmask;
      for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
        if (!wave.exec.test(lane)) {
          continue;
        }
        const int64_t lhs = AsSigned(ReadVectorLaneOperand(instruction.operands.at(0), wave, lane));
        const int64_t rhs = AsSigned(ReadVectorLaneOperand(instruction.operands.at(1), wave, lane));
        if (lhs < rhs) {
          cmask.set(lane);
        }
      }
      plan.cmask_write = cmask;
      return plan;
    }
    case Opcode::MLoadGlobal: {
      MemoryRequest request;
      request.space = MemorySpace::Global;
      request.kind = AccessKind::Load;
      request.exec_snapshot = wave.exec;
      request.dst = RegRef{.file = RegisterFile::Vector,
                           .index = RequireVectorReg(instruction.operands.at(0))};
      request.block_id = wave.block_id;
      request.wave_id = wave.wave_id;

      const uint64_t scale = ReadScalarOperand(instruction.operands.at(3), wave);
      for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
        if (!wave.exec.test(lane)) {
          continue;
        }
        const uint64_t base = ReadScalarOperand(instruction.operands.at(1), wave);
        const uint64_t index = ReadVectorLaneOperand(instruction.operands.at(2), wave, lane);
        request.lanes[lane] = LaneAccess{
            .active = true,
            .addr = base + index * scale,
            .bytes = static_cast<uint32_t>(scale),
        };
      }
      plan.memory = request;
      return plan;
    }
    case Opcode::MStoreGlobal: {
      MemoryRequest request;
      request.space = MemorySpace::Global;
      request.kind = AccessKind::Store;
      request.exec_snapshot = wave.exec;
      request.block_id = wave.block_id;
      request.wave_id = wave.wave_id;

      const uint64_t scale = ReadScalarOperand(instruction.operands.at(3), wave);
      for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
        if (!wave.exec.test(lane)) {
          continue;
        }
        const uint64_t base = ReadScalarOperand(instruction.operands.at(0), wave);
        const uint64_t index = ReadVectorLaneOperand(instruction.operands.at(1), wave, lane);
        const uint64_t value = ReadVectorLaneOperand(instruction.operands.at(2), wave, lane);
        request.lanes[lane] = LaneAccess{
            .active = true,
            .addr = base + index * scale,
            .bytes = static_cast<uint32_t>(scale),
            .value = value,
        };
      }
      plan.memory = request;
      return plan;
    }
    case Opcode::MLoadShared: {
      MemoryRequest request;
      request.space = MemorySpace::Shared;
      request.kind = AccessKind::Load;
      request.exec_snapshot = wave.exec;
      request.dst = RegRef{.file = RegisterFile::Vector,
                           .index = RequireVectorReg(instruction.operands.at(0))};
      request.block_id = wave.block_id;
      request.wave_id = wave.wave_id;

      const uint64_t scale = ReadScalarOperand(instruction.operands.at(2), wave);
      for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
        if (!wave.exec.test(lane)) {
          continue;
        }
        const uint64_t index = ReadVectorLaneOperand(instruction.operands.at(1), wave, lane);
        request.lanes[lane] = LaneAccess{
            .active = true,
            .addr = index * scale,
            .bytes = static_cast<uint32_t>(scale),
        };
      }
      plan.memory = request;
      return plan;
    }
    case Opcode::MStoreShared: {
      MemoryRequest request;
      request.space = MemorySpace::Shared;
      request.kind = AccessKind::Store;
      request.exec_snapshot = wave.exec;
      request.block_id = wave.block_id;
      request.wave_id = wave.wave_id;

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
    case Opcode::MLoadPrivate: {
      MemoryRequest request;
      request.space = MemorySpace::Private;
      request.kind = AccessKind::Load;
      request.exec_snapshot = wave.exec;
      request.dst = RegRef{.file = RegisterFile::Vector,
                           .index = RequireVectorReg(instruction.operands.at(0))};
      request.block_id = wave.block_id;
      request.wave_id = wave.wave_id;

      const uint64_t scale = ReadScalarOperand(instruction.operands.at(2), wave);
      for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
        if (!wave.exec.test(lane)) {
          continue;
        }
        const uint64_t index = ReadVectorLaneOperand(instruction.operands.at(1), wave, lane);
        request.lanes[lane] = LaneAccess{
            .active = true,
            .addr = index * scale,
            .bytes = static_cast<uint32_t>(scale),
        };
      }
      plan.memory = request;
      return plan;
    }
    case Opcode::MStorePrivate: {
      MemoryRequest request;
      request.space = MemorySpace::Private;
      request.kind = AccessKind::Store;
      request.exec_snapshot = wave.exec;
      request.block_id = wave.block_id;
      request.wave_id = wave.wave_id;

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
    case Opcode::MLoadConst: {
      MemoryRequest request;
      request.space = MemorySpace::Constant;
      request.kind = AccessKind::Load;
      request.exec_snapshot = wave.exec;
      request.dst = RegRef{.file = RegisterFile::Vector,
                           .index = RequireVectorReg(instruction.operands.at(0))};
      request.block_id = wave.block_id;
      request.wave_id = wave.wave_id;

      const uint64_t scale = ReadScalarOperand(instruction.operands.at(2), wave);
      for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
        if (!wave.exec.test(lane)) {
          continue;
        }
        const uint64_t index = ReadVectorLaneOperand(instruction.operands.at(1), wave, lane);
        request.lanes[lane] = LaneAccess{
            .active = true,
            .addr = index * scale,
            .bytes = static_cast<uint32_t>(scale),
        };
      }
      plan.memory = request;
      return plan;
    }
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
    case Opcode::MaskAndExecCmask: {
      plan.exec_write = wave.exec & wave.cmask;
      return plan;
    }
    case Opcode::BBranch: {
      plan.branch_target = instruction.operands.at(0).immediate;
      return plan;
    }
    case Opcode::BIfSmask: {
      if (wave.ScalarMaskBit0()) {
        plan.branch_target = instruction.operands.at(0).immediate;
      }
      return plan;
    }
    case Opcode::BIfNoexec: {
      if (wave.exec.none()) {
        plan.branch_target = instruction.operands.at(0).immediate;
      }
      return plan;
    }
    case Opcode::SyncBarrier: {
      plan.sync_barrier = true;
      plan.advance_pc = false;
      return plan;
    }
    case Opcode::BExit: {
      plan.exit_wave = true;
      return plan;
    }
  }

  throw std::invalid_argument("unsupported opcode");
}

uint64_t Semantics::ReadScalarOperand(const Operand& operand, const WaveState& wave) const {
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

uint64_t Semantics::ReadVectorLaneOperand(const Operand& operand,
                                          const WaveState& wave,
                                          uint32_t lane) const {
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

std::bitset<64> Semantics::ThreadMask(const WaveState& wave) const {
  std::bitset<64> mask;
  for (uint32_t lane = 0; lane < wave.thread_count && lane < kWaveSize; ++lane) {
    mask.set(lane);
  }
  return mask;
}

std::array<uint64_t, 64> Semantics::BroadcastScalar(const WaveState& wave, uint64_t value) const {
  std::array<uint64_t, 64> values{};
  for (uint32_t lane = 0; lane < wave.thread_count && lane < kWaveSize; ++lane) {
    values[lane] = value;
  }
  return values;
}

}  // namespace gpu_model
