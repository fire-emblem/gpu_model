#include "gpu_model/loader/asm_parser.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "gpu_model/isa/instruction_builder.h"

namespace gpu_model {

namespace {

std::string Trim(std::string_view text) {
  size_t begin = 0;
  size_t end = text.size();
  while (begin < end && std::isspace(static_cast<unsigned char>(text[begin])) != 0) {
    ++begin;
  }
  while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1])) != 0) {
    --end;
  }
  return std::string(text.substr(begin, end - begin));
}

std::vector<std::string> SplitOperands(std::string_view text) {
  std::vector<std::string> operands;
  std::string current;
  for (const char ch : text) {
    if (ch == ',') {
      operands.push_back(Trim(current));
      current.clear();
      continue;
    }
    current.push_back(ch);
  }
  if (!current.empty()) {
    operands.push_back(Trim(current));
  }
  return operands;
}

bool IsRegister(std::string_view operand) {
  return !operand.empty() && (operand.front() == 's' || operand.front() == 'v');
}

uint64_t ParseImmediate(std::string_view text) {
  return static_cast<uint64_t>(std::stoll(std::string(text), nullptr, 0));
}

void RequireOperandCount(std::string_view opcode,
                         const std::vector<std::string>& operands,
                         size_t expected) {
  if (operands.size() != expected) {
    throw std::invalid_argument("opcode " + std::string(opcode) + " expects " +
                                std::to_string(expected) + " operands");
  }
}

}  // namespace

KernelProgram AsmParser::Parse(const ProgramImage& image) const {
  InstructionBuilder builder;
  MetadataBlob metadata = image.metadata();

  std::istringstream input(image.assembly_text());
  std::string line;
  uint32_t line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    const auto comment_pos = line.find('#');
    if (comment_pos != std::string::npos) {
      line.resize(comment_pos);
    }

    const std::string trimmed = Trim(line);
    if (trimmed.empty()) {
      continue;
    }

    if (trimmed.rfind(".meta ", 0) == 0) {
      const std::string body = Trim(std::string_view(trimmed).substr(6));
      const size_t equals = body.find('=');
      if (equals == std::string::npos) {
        throw std::invalid_argument("invalid .meta directive at line " +
                                    std::to_string(line_number));
      }
      metadata.values[Trim(std::string_view(body).substr(0, equals))] =
          Trim(std::string_view(body).substr(equals + 1));
      continue;
    }

    if (trimmed.back() == ':') {
      builder.Label(trimmed.substr(0, trimmed.size() - 1));
      continue;
    }

    builder.SetNextDebugLoc(image.kernel_name() + ".asm", line_number);

    const size_t space = trimmed.find_first_of(" \t");
    const std::string opcode = space == std::string::npos ? trimmed : trimmed.substr(0, space);
    const std::vector<std::string> operands =
        space == std::string::npos ? std::vector<std::string>{}
                                   : SplitOperands(std::string_view(trimmed).substr(space + 1));

    if (opcode == "sys_load_arg" || opcode == "s_load_arg") {
      RequireOperandCount(opcode, operands, 2);
      builder.SLoadArg(operands[0], static_cast<uint32_t>(ParseImmediate(operands[1])));
    } else if (opcode == "sys_global_id_x") {
      RequireOperandCount(opcode, operands, 1);
      builder.SysGlobalIdX(operands[0]);
    } else if (opcode == "sys_local_id_x") {
      RequireOperandCount(opcode, operands, 1);
      builder.SysLocalIdX(operands[0]);
    } else if (opcode == "sys_block_offset_x") {
      RequireOperandCount(opcode, operands, 1);
      builder.SysBlockOffsetX(operands[0]);
    } else if (opcode == "sys_block_idx_x") {
      RequireOperandCount(opcode, operands, 1);
      builder.SysBlockIdxX(operands[0]);
    } else if (opcode == "sys_block_dim_x") {
      RequireOperandCount(opcode, operands, 1);
      builder.SysBlockDimX(operands[0]);
    } else if (opcode == "sys_lane_id") {
      RequireOperandCount(opcode, operands, 1);
      builder.SysLaneId(operands[0]);
    } else if (opcode == "s_mov") {
      RequireOperandCount(opcode, operands, 2);
      if (IsRegister(operands[1])) {
        builder.SMov(operands[0], operands[1]);
      } else {
        builder.SMov(operands[0], ParseImmediate(operands[1]));
      }
    } else if (opcode == "s_add") {
      RequireOperandCount(opcode, operands, 3);
      if (IsRegister(operands[2])) {
        builder.SAdd(operands[0], operands[1], operands[2]);
      } else {
        builder.SAdd(operands[0], operands[1], ParseImmediate(operands[2]));
      }
    } else if (opcode == "s_mul") {
      RequireOperandCount(opcode, operands, 3);
      if (IsRegister(operands[2])) {
        builder.SMul(operands[0], operands[1], operands[2]);
      } else {
        builder.SMul(operands[0], operands[1], ParseImmediate(operands[2]));
      }
    } else if (opcode == "s_cmp_lt") {
      RequireOperandCount(opcode, operands, 2);
      if (IsRegister(operands[1])) {
        builder.SCmpLt(operands[0], operands[1]);
      } else {
        builder.SCmpLt(operands[0], ParseImmediate(operands[1]));
      }
    } else if (opcode == "s_cmp_eq") {
      RequireOperandCount(opcode, operands, 2);
      if (IsRegister(operands[1])) {
        builder.SCmpEq(operands[0], operands[1]);
      } else {
        builder.SCmpEq(operands[0], ParseImmediate(operands[1]));
      }
    } else if (opcode == "v_mov") {
      RequireOperandCount(opcode, operands, 2);
      if (IsRegister(operands[1])) {
        builder.VMov(operands[0], operands[1]);
      } else {
        builder.VMov(operands[0], ParseImmediate(operands[1]));
      }
    } else if (opcode == "v_add") {
      RequireOperandCount(opcode, operands, 3);
      builder.VAdd(operands[0], operands[1], operands[2]);
    } else if (opcode == "v_and") {
      RequireOperandCount(opcode, operands, 3);
      builder.VAnd(operands[0], operands[1], operands[2]);
    } else if (opcode == "v_or") {
      RequireOperandCount(opcode, operands, 3);
      builder.VOr(operands[0], operands[1], operands[2]);
    } else if (opcode == "v_xor") {
      RequireOperandCount(opcode, operands, 3);
      builder.VXor(operands[0], operands[1], operands[2]);
    } else if (opcode == "v_shl") {
      RequireOperandCount(opcode, operands, 3);
      builder.VShl(operands[0], operands[1], operands[2]);
    } else if (opcode == "v_shr") {
      RequireOperandCount(opcode, operands, 3);
      builder.VShr(operands[0], operands[1], operands[2]);
    } else if (opcode == "v_sub") {
      RequireOperandCount(opcode, operands, 3);
      builder.VSub(operands[0], operands[1], operands[2]);
    } else if (opcode == "v_mul") {
      RequireOperandCount(opcode, operands, 3);
      builder.VMul(operands[0], operands[1], operands[2]);
    } else if (opcode == "v_min") {
      RequireOperandCount(opcode, operands, 3);
      builder.VMin(operands[0], operands[1], operands[2]);
    } else if (opcode == "v_max") {
      RequireOperandCount(opcode, operands, 3);
      builder.VMax(operands[0], operands[1], operands[2]);
    } else if (opcode == "v_fma") {
      RequireOperandCount(opcode, operands, 4);
      builder.VFma(operands[0], operands[1], operands[2], operands[3]);
    } else if (opcode == "v_cmp_lt_cmask") {
      RequireOperandCount(opcode, operands, 2);
      builder.VCmpLtCmask(operands[0], operands[1]);
    } else if (opcode == "v_cmp_eq_cmask") {
      RequireOperandCount(opcode, operands, 2);
      builder.VCmpEqCmask(operands[0], operands[1]);
    } else if (opcode == "v_cmp_gt_cmask") {
      RequireOperandCount(opcode, operands, 2);
      builder.VCmpGtCmask(operands[0], operands[1]);
    } else if (opcode == "v_select_cmask") {
      RequireOperandCount(opcode, operands, 3);
      builder.VSelectCmask(operands[0], operands[1], operands[2]);
    } else if (opcode == "m_load_global") {
      RequireOperandCount(opcode, operands, 4);
      builder.MLoadGlobal(operands[0], operands[1], operands[2],
                          static_cast<uint32_t>(ParseImmediate(operands[3])));
    } else if (opcode == "m_store_global") {
      RequireOperandCount(opcode, operands, 4);
      builder.MStoreGlobal(operands[0], operands[1], operands[2],
                           static_cast<uint32_t>(ParseImmediate(operands[3])));
    } else if (opcode == "m_atomic_add_global") {
      RequireOperandCount(opcode, operands, 4);
      builder.MAtomicAddGlobal(operands[0], operands[1], operands[2],
                               static_cast<uint32_t>(ParseImmediate(operands[3])));
    } else if (opcode == "m_load_shared") {
      RequireOperandCount(opcode, operands, 3);
      builder.MLoadShared(operands[0], operands[1],
                          static_cast<uint32_t>(ParseImmediate(operands[2])));
    } else if (opcode == "m_store_shared") {
      RequireOperandCount(opcode, operands, 3);
      builder.MStoreShared(operands[0], operands[1],
                           static_cast<uint32_t>(ParseImmediate(operands[2])));
    } else if (opcode == "m_atomic_add_shared") {
      RequireOperandCount(opcode, operands, 3);
      builder.MAtomicAddShared(operands[0], operands[1],
                               static_cast<uint32_t>(ParseImmediate(operands[2])));
    } else if (opcode == "m_load_private") {
      RequireOperandCount(opcode, operands, 3);
      builder.MLoadPrivate(operands[0], operands[1],
                           static_cast<uint32_t>(ParseImmediate(operands[2])));
    } else if (opcode == "m_load_const") {
      RequireOperandCount(opcode, operands, 3);
      builder.MLoadConst(operands[0], operands[1],
                         static_cast<uint32_t>(ParseImmediate(operands[2])));
    } else if (opcode == "m_store_private") {
      RequireOperandCount(opcode, operands, 3);
      builder.MStorePrivate(operands[0], operands[1],
                            static_cast<uint32_t>(ParseImmediate(operands[2])));
    } else if (opcode == "mask_save_exec") {
      RequireOperandCount(opcode, operands, 1);
      builder.MaskSaveExec(operands[0]);
    } else if (opcode == "mask_restore_exec") {
      RequireOperandCount(opcode, operands, 1);
      builder.MaskRestoreExec(operands[0]);
    } else if (opcode == "mask_and_exec_cmask") {
      RequireOperandCount(opcode, operands, 0);
      builder.MaskAndExecCmask();
    } else if (opcode == "b_branch") {
      RequireOperandCount(opcode, operands, 1);
      builder.BBranch(operands[0]);
    } else if (opcode == "b_if_smask") {
      RequireOperandCount(opcode, operands, 1);
      builder.BIfSmask(operands[0]);
    } else if (opcode == "b_if_noexec") {
      RequireOperandCount(opcode, operands, 1);
      builder.BIfNoexec(operands[0]);
    } else if (opcode == "sync_wave_barrier") {
      RequireOperandCount(opcode, operands, 0);
      builder.SyncWaveBarrier();
    } else if (opcode == "sync_barrier") {
      RequireOperandCount(opcode, operands, 0);
      builder.SyncBarrier();
    } else if (opcode == "b_exit") {
      RequireOperandCount(opcode, operands, 0);
      builder.BExit();
    } else {
      throw std::invalid_argument("unsupported opcode in asm parser: " + opcode);
    }
  }

  return builder.Build(image.kernel_name(), std::move(metadata), image.const_segment());
}

}  // namespace gpu_model
