#include "debug/trace/instruction_trace.h"

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <string_view>

#include "instruction/isa/instruction.h"
#include "instruction/isa/opcode.h"
#include "instruction/isa/operand.h"

namespace gpu_model {

namespace {

std::string HexU64(uint64_t value, int width = 0) {
  std::ostringstream out;
  out << "0x" << std::hex << std::nouppercase;
  if (width > 0) {
    out << std::setfill('0') << std::setw(width);
  }
  out << value;
  return out.str();
}

std::string RegisterName(const RegRef& reg) {
  const char prefix = reg.file == RegisterFile::Scalar ? 's' : 'v';
  return std::string(1, prefix) + std::to_string(reg.index);
}

std::string FormatVectorValue(uint32_t reg_index, const WaveContext& wave) {
  std::ostringstream out;
  out << '\n';
  bool emitted = false;
  for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
    if (!wave.exec.test(lane)) {
      continue;
    }
    emitted = true;
    out << "    lane[" << HexU64(lane, 2) << "] = "
        << HexU64(wave.vgpr.Read(reg_index, lane)) << '\n';
  }
  if (!emitted) {
    out << "    <no active lanes>\n";
  }
  return out.str();
}

std::string FormatOperand(const Operand& operand, const WaveContext& wave) {
  switch (operand.kind) {
    case OperandKind::Register:
      if (operand.reg.file == RegisterFile::Scalar) {
        return RegisterName(operand.reg) + " = " + HexU64(wave.sgpr.Read(operand.reg.index));
      }
      return RegisterName(operand.reg) + ":" + FormatVectorValue(operand.reg.index, wave);
    case OperandKind::RegisterRange: {
      const std::string name = Instruction::DumpOperand(operand);
      if (operand.reg.file == RegisterFile::Scalar) {
        std::ostringstream out;
        out << name << " =";
        for (uint32_t i = 0; i < operand.reg_count; ++i) {
          out << " " << HexU64(wave.sgpr.Read(operand.reg.index + i));
        }
        return out.str();
      }
      std::ostringstream out;
      out << name << ":";
      for (uint32_t i = 0; i < operand.reg_count; ++i) {
        out << "\n  " << RegisterName(RegRef{.file = operand.reg.file, .index = operand.reg.index + i})
            << ":" << FormatVectorValue(operand.reg.index + i, wave);
      }
      return out.str();
    }
    case OperandKind::Immediate:
      return HexU64(operand.immediate);
    case OperandKind::ArgumentIndex:
      return "arg[" + HexU64(operand.immediate) + "]";
    case OperandKind::BranchTarget:
      return "pc = " + HexU64(operand.immediate);
    case OperandKind::None:
      break;
  }
  return "_";
}

}  // namespace

std::string FormatWaveStepMessage(const Instruction& instruction, const WaveContext& wave) {
  std::ostringstream out;
  out << "pc=" << HexU64(wave.pc) << " op=" << ToString(instruction.opcode)
      << " exec_lanes=" << HexU64(wave.exec.count());
  if (instruction.operands.empty()) {
    return out.str();
  }
  out << "\n  operands:";
  for (size_t i = 0; i < instruction.operands.size(); ++i) {
    out << "\n  [" << HexU64(i, 2) << "] " << FormatOperand(instruction.operands[i], wave);
  }
  return out.str();
}

std::string FormatAssemblyText(const Instruction& instruction) {
  return instruction.Dump();
}

TraceWaveStepDetail BuildWaveStepDetail(const Instruction& instruction, const WaveContext& wave) {
  TraceWaveStepDetail detail;

  // Assembly text using Instruction::Dump()
  detail.asm_text = instruction.Dump();

  // Process operands to extract reads and writes
  // Convention: first operand is often destination (write), rest are sources (reads)
  // Note: This is a simplified model; real ISA may have different conventions
  for (size_t i = 0; i < instruction.operands.size(); ++i) {
    const Operand& op = instruction.operands[i];

    if (op.kind != OperandKind::Register && op.kind != OperandKind::RegisterRange) {
      continue;
    }

    std::string reg_value;
    if (op.reg.file == RegisterFile::Scalar) {
      if (op.kind == OperandKind::RegisterRange) {
        std::ostringstream sout;
        sout << Instruction::DumpOperand(op) << "=";
        for (uint32_t r = 0; r < op.reg_count; ++r) {
          if (r > 0) {
            sout << ",";
          }
          sout << HexU64(wave.sgpr.Read(op.reg.index + r));
        }
        reg_value = sout.str();
      } else {
        reg_value = Instruction::DumpOperand(op) + "=" + HexU64(wave.sgpr.Read(op.reg.index));
      }
    } else {
      // For vector registers, show sampled values
      std::ostringstream vout;
      vout << Instruction::DumpOperand(op) << "[step=4]:\n";
      bool emitted = false;
      for (uint32_t lane = 0; lane < kWaveSize; lane += 4) {
        if (!wave.exec.test(lane)) {
          continue;
        }
        emitted = true;
        vout << "        lane " << std::setw(4) << lane;
        for (uint32_t r = 0; r < op.reg_count; ++r) {
          vout << "  " << HexU64(wave.vgpr.Read(op.reg.index + r, lane));
        }
        vout << "\n";
      }
      if (!emitted) {
        vout << "        <no active lanes>\n";
      }
      reg_value = vout.str();
    }

    // First register operand is typically destination (write)
    // Note: This is ISA-specific; may need adjustment
    if (i == 0 && instruction.operands.size() > 1) {
      // Destination operand
      if (op.reg.file == RegisterFile::Scalar) {
        detail.scalar_writes.push_back(reg_value);
      } else {
        detail.vector_writes.push_back(reg_value);
      }
    } else {
      // Source operands (reads)
      if (op.reg.file == RegisterFile::Scalar) {
        detail.scalar_reads.push_back(reg_value);
      } else {
        detail.vector_reads.push_back(reg_value);
      }
    }
  }

  // Exec mask
  std::ostringstream exec_out;
  exec_out << "0x" << std::hex << wave.exec.to_ullong();
  detail.exec_before = exec_out.str();
  detail.exec_after = exec_out.str();  // May be updated by instruction

  // The detail schema reserves a compact memory summary even when a step does not touch memory.
  detail.mem_summary = "none";

  // Waitcnt/state summary is emitted when the producer attaches structured state details.
  detail.state_summary = "";

  return detail;
}

}  // namespace gpu_model
