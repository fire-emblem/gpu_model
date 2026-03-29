#include "gpu_model/loader/asm_parser.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "gpu_model/isa/kernel_program_builder.h"

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

struct WaitCntThresholds {
  uint32_t global = UINT32_MAX;
  uint32_t shared = UINT32_MAX;
  uint32_t private_mem = UINT32_MAX;
  uint32_t scalar_buffer = UINT32_MAX;
};

WaitCntThresholds ParseWaitCnt(std::string_view text) {
  WaitCntThresholds thresholds;
  std::string body = Trim(text);
  if (body.empty()) {
    return thresholds;
  }

  const auto parse_named = [&](std::string_view term, std::string_view name) -> std::optional<uint32_t> {
    const std::string trimmed = Trim(term);
    if (trimmed.rfind(name, 0) != 0) {
      return std::nullopt;
    }
    const size_t open = trimmed.find('(');
    const size_t close = trimmed.find(')');
    if (open == std::string::npos || close == std::string::npos || close <= open + 1) {
      throw std::invalid_argument("invalid s_waitcnt term: " + trimmed);
    }
    return static_cast<uint32_t>(ParseImmediate(
        std::string_view(trimmed).substr(open + 1, close - open - 1)));
  };

  std::vector<std::string> terms;
  std::string current;
  for (const char ch : body) {
    if (ch == '&') {
      terms.push_back(Trim(current));
      current.clear();
      continue;
    }
    current.push_back(ch);
  }
  if (!current.empty()) {
    terms.push_back(Trim(current));
  }

  if (terms.size() == 4 && terms[0].find('(') == std::string::npos) {
    thresholds.global = static_cast<uint32_t>(ParseImmediate(terms[0]));
    thresholds.shared = static_cast<uint32_t>(ParseImmediate(terms[1]));
    thresholds.private_mem = static_cast<uint32_t>(ParseImmediate(terms[2]));
    thresholds.scalar_buffer = static_cast<uint32_t>(ParseImmediate(terms[3]));
    return thresholds;
  }

  for (const auto& term : terms) {
    if (term.empty()) {
      continue;
    }
    if (const auto value = parse_named(term, "vmcnt")) {
      thresholds.global = *value;
      continue;
    }
    if (const auto value = parse_named(term, "lgkmcnt")) {
      thresholds.shared = *value;
      thresholds.private_mem = *value;
      thresholds.scalar_buffer = *value;
      continue;
    }
    if (const auto value = parse_named(term, "sharedcnt")) {
      thresholds.shared = *value;
      continue;
    }
    if (const auto value = parse_named(term, "privatecnt")) {
      thresholds.private_mem = *value;
      continue;
    }
    if (const auto value = parse_named(term, "scalarcnt")) {
      thresholds.scalar_buffer = *value;
      continue;
    }
    if (parse_named(term, "expcnt")) {
      continue;
    }
    throw std::invalid_argument("unsupported s_waitcnt term: " + term);
  }

  return thresholds;
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
  KernelProgramBuilder builder;
  MetadataBlob metadata = image.metadata();
  const auto reg = [&](std::string_view text) { return builder.ParseRegOperand(text); };
  const auto imm_value = [&](std::string_view text) { return ParseImmediate(text); };
  const auto imm = [&](std::string_view text) { return builder.ImmediateOperand(ParseImmediate(text)); };
  const auto reg_or_imm = [&](std::string_view text) {
    return IsRegister(text) ? reg(text) : imm(text);
  };

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

    if (opcode == "s_load_kernarg") {
      RequireOperandCount(opcode, operands, 2);
      builder.AddInstruction(
          Opcode::SysLoadArg, {reg(operands[0]), Operand::Argument(static_cast<uint32_t>(imm_value(operands[1])))});
    } else if (opcode == "v_get_global_id_x") {
      RequireOperandCount(opcode, operands, 1);
      builder.AddInstruction(Opcode::SysGlobalIdX, {reg(operands[0])});
    } else if (opcode == "v_get_global_id_y") {
      RequireOperandCount(opcode, operands, 1);
      builder.AddInstruction(Opcode::SysGlobalIdY, {reg(operands[0])});
    } else if (opcode == "v_get_local_id_x") {
      RequireOperandCount(opcode, operands, 1);
      builder.AddInstruction(Opcode::SysLocalIdX, {reg(operands[0])});
    } else if (opcode == "v_get_local_id_y") {
      RequireOperandCount(opcode, operands, 1);
      builder.AddInstruction(Opcode::SysLocalIdY, {reg(operands[0])});
    } else if (opcode == "s_get_block_offset_x") {
      RequireOperandCount(opcode, operands, 1);
      builder.AddInstruction(Opcode::SysBlockOffsetX, {reg(operands[0])});
    } else if (opcode == "s_get_block_id_x") {
      RequireOperandCount(opcode, operands, 1);
      builder.AddInstruction(Opcode::SysBlockIdxX, {reg(operands[0])});
    } else if (opcode == "s_get_block_id_y") {
      RequireOperandCount(opcode, operands, 1);
      builder.AddInstruction(Opcode::SysBlockIdxY, {reg(operands[0])});
    } else if (opcode == "s_get_block_dim_x") {
      RequireOperandCount(opcode, operands, 1);
      builder.AddInstruction(Opcode::SysBlockDimX, {reg(operands[0])});
    } else if (opcode == "s_get_block_dim_y") {
      RequireOperandCount(opcode, operands, 1);
      builder.AddInstruction(Opcode::SysBlockDimY, {reg(operands[0])});
    } else if (opcode == "s_get_grid_dim_x") {
      RequireOperandCount(opcode, operands, 1);
      builder.AddInstruction(Opcode::SysGridDimX, {reg(operands[0])});
    } else if (opcode == "s_get_grid_dim_y") {
      RequireOperandCount(opcode, operands, 1);
      builder.AddInstruction(Opcode::SysGridDimY, {reg(operands[0])});
    } else if (opcode == "v_lane_id_u32") {
      RequireOperandCount(opcode, operands, 1);
      builder.AddInstruction(Opcode::SysLaneId, {reg(operands[0])});
    } else if (opcode == "s_mov_b32") {
      RequireOperandCount(opcode, operands, 2);
      builder.AddInstruction(Opcode::SMov, {reg(operands[0]), reg_or_imm(operands[1])});
    } else if (opcode == "s_add_u32") {
      RequireOperandCount(opcode, operands, 3);
      builder.AddInstruction(Opcode::SAdd, {reg(operands[0]), reg(operands[1]), reg_or_imm(operands[2])});
    } else if (opcode == "s_sub_u32") {
      RequireOperandCount(opcode, operands, 3);
      builder.AddInstruction(Opcode::SSub, {reg(operands[0]), reg(operands[1]), reg_or_imm(operands[2])});
    } else if (opcode == "s_mul_i32") {
      RequireOperandCount(opcode, operands, 3);
      builder.AddInstruction(Opcode::SMul, {reg(operands[0]), reg(operands[1]), reg_or_imm(operands[2])});
    } else if (opcode == "s_div_i32") {
      RequireOperandCount(opcode, operands, 3);
      builder.AddInstruction(Opcode::SDiv, {reg(operands[0]), reg(operands[1]), reg_or_imm(operands[2])});
    } else if (opcode == "s_rem_i32") {
      RequireOperandCount(opcode, operands, 3);
      builder.AddInstruction(Opcode::SRem, {reg(operands[0]), reg(operands[1]), reg_or_imm(operands[2])});
    } else if (opcode == "s_and_b32") {
      RequireOperandCount(opcode, operands, 3);
      builder.AddInstruction(Opcode::SAnd, {reg(operands[0]), reg(operands[1]), reg_or_imm(operands[2])});
    } else if (opcode == "s_or_b32") {
      RequireOperandCount(opcode, operands, 3);
      builder.AddInstruction(Opcode::SOr, {reg(operands[0]), reg(operands[1]), reg_or_imm(operands[2])});
    } else if (opcode == "s_xor_b32") {
      RequireOperandCount(opcode, operands, 3);
      builder.AddInstruction(Opcode::SXor, {reg(operands[0]), reg(operands[1]), reg_or_imm(operands[2])});
    } else if (opcode == "s_lshl_b32") {
      RequireOperandCount(opcode, operands, 3);
      builder.AddInstruction(Opcode::SShl, {reg(operands[0]), reg(operands[1]), reg_or_imm(operands[2])});
    } else if (opcode == "s_lshr_b32") {
      RequireOperandCount(opcode, operands, 3);
      builder.AddInstruction(Opcode::SShr, {reg(operands[0]), reg(operands[1]), reg_or_imm(operands[2])});
    } else if (opcode == "s_waitcnt") {
      const auto thresholds =
          ParseWaitCnt(space == std::string::npos ? std::string_view{} :
                                                     std::string_view(trimmed).substr(space + 1));
      builder.AddInstruction(Opcode::SWaitCnt,
                             {builder.ImmediateOperand(thresholds.global),
                              builder.ImmediateOperand(thresholds.shared),
                              builder.ImmediateOperand(thresholds.private_mem),
                              builder.ImmediateOperand(thresholds.scalar_buffer)});
    } else if (opcode == "s_buffer_load_dword") {
      if (operands.size() != 3 && operands.size() != 4) {
        throw std::invalid_argument("opcode " + opcode + " expects 3 or 4 operands");
      }
      builder.AddInstruction(Opcode::SBufferLoadDword,
                             {reg(operands[0]), reg(operands[1]), imm(operands[2]),
                              operands.size() == 4 ? imm(operands[3]) : builder.ImmediateOperand(0)});
    } else if (opcode == "s_cmp_lt_i32") {
      RequireOperandCount(opcode, operands, 2);
      builder.AddInstruction(Opcode::SCmpLt, {reg(operands[0]), reg_or_imm(operands[1])});
    } else if (opcode == "s_cmp_eq_u32") {
      RequireOperandCount(opcode, operands, 2);
      builder.AddInstruction(Opcode::SCmpEq, {reg(operands[0]), reg_or_imm(operands[1])});
    } else if (opcode == "s_cmp_gt_i32") {
      RequireOperandCount(opcode, operands, 2);
      builder.AddInstruction(Opcode::SCmpGt, {reg(operands[0]), reg_or_imm(operands[1])});
    } else if (opcode == "s_cmp_ge_i32") {
      RequireOperandCount(opcode, operands, 2);
      builder.AddInstruction(Opcode::SCmpGe, {reg(operands[0]), reg_or_imm(operands[1])});
    } else if (opcode == "v_mov_b32") {
      RequireOperandCount(opcode, operands, 2);
      builder.AddInstruction(Opcode::VMov, {reg(operands[0]), reg_or_imm(operands[1])});
    } else if (opcode == "v_add_i32") {
      RequireOperandCount(opcode, operands, 3);
      builder.AddInstruction(Opcode::VAdd, {reg(operands[0]), reg(operands[1]), reg(operands[2])});
    } else if (opcode == "v_and_b32") {
      RequireOperandCount(opcode, operands, 3);
      builder.AddInstruction(Opcode::VAnd, {reg(operands[0]), reg(operands[1]), reg(operands[2])});
    } else if (opcode == "v_or_b32") {
      RequireOperandCount(opcode, operands, 3);
      builder.AddInstruction(Opcode::VOr, {reg(operands[0]), reg(operands[1]), reg(operands[2])});
    } else if (opcode == "v_xor_b32") {
      RequireOperandCount(opcode, operands, 3);
      builder.AddInstruction(Opcode::VXor, {reg(operands[0]), reg(operands[1]), reg(operands[2])});
    } else if (opcode == "v_lshl_b32") {
      RequireOperandCount(opcode, operands, 3);
      builder.AddInstruction(Opcode::VShl, {reg(operands[0]), reg(operands[1]), reg(operands[2])});
    } else if (opcode == "v_lshr_b32") {
      RequireOperandCount(opcode, operands, 3);
      builder.AddInstruction(Opcode::VShr, {reg(operands[0]), reg(operands[1]), reg(operands[2])});
    } else if (opcode == "v_sub_i32") {
      RequireOperandCount(opcode, operands, 3);
      builder.AddInstruction(Opcode::VSub, {reg(operands[0]), reg(operands[1]), reg(operands[2])});
    } else if (opcode == "v_div_i32") {
      RequireOperandCount(opcode, operands, 3);
      builder.AddInstruction(Opcode::VDiv, {reg(operands[0]), reg(operands[1]), reg(operands[2])});
    } else if (opcode == "v_rem_i32") {
      RequireOperandCount(opcode, operands, 3);
      builder.AddInstruction(Opcode::VRem, {reg(operands[0]), reg(operands[1]), reg(operands[2])});
    } else if (opcode == "v_mul_lo_i32") {
      RequireOperandCount(opcode, operands, 3);
      builder.AddInstruction(Opcode::VMul, {reg(operands[0]), reg(operands[1]), reg(operands[2])});
    } else if (opcode == "v_add_f32" || opcode == "v_add_f32_e32") {
      RequireOperandCount(opcode, operands, 3);
      builder.AddInstruction(Opcode::VAddF32, {reg(operands[0]), reg(operands[1]), reg(operands[2])});
    } else if (opcode == "v_min_i32") {
      RequireOperandCount(opcode, operands, 3);
      builder.AddInstruction(Opcode::VMin, {reg(operands[0]), reg(operands[1]), reg(operands[2])});
    } else if (opcode == "v_max_i32") {
      RequireOperandCount(opcode, operands, 3);
      builder.AddInstruction(Opcode::VMax, {reg(operands[0]), reg(operands[1]), reg(operands[2])});
    } else if (opcode == "v_mad_i32") {
      RequireOperandCount(opcode, operands, 4);
      builder.AddInstruction(Opcode::VFma,
                             {reg(operands[0]), reg(operands[1]), reg(operands[2]), reg(operands[3])});
    } else if (opcode == "v_cmp_lt_i32_cmask") {
      RequireOperandCount(opcode, operands, 2);
      builder.AddInstruction(Opcode::VCmpLtCmask, {reg(operands[0]), reg(operands[1])});
    } else if (opcode == "v_cmp_eq_i32_cmask") {
      RequireOperandCount(opcode, operands, 2);
      builder.AddInstruction(Opcode::VCmpEqCmask, {reg(operands[0]), reg(operands[1])});
    } else if (opcode == "v_cmp_ge_i32_cmask") {
      RequireOperandCount(opcode, operands, 2);
      builder.AddInstruction(Opcode::VCmpGeCmask, {reg(operands[0]), reg(operands[1])});
    } else if (opcode == "v_cmp_gt_i32_cmask") {
      RequireOperandCount(opcode, operands, 2);
      builder.AddInstruction(Opcode::VCmpGtCmask, {reg(operands[0]), reg(operands[1])});
    } else if (opcode == "v_cndmask_b32") {
      RequireOperandCount(opcode, operands, 3);
      builder.AddInstruction(Opcode::VSelectCmask, {reg(operands[0]), reg(operands[1]), reg(operands[2])});
    } else if (opcode == "buffer_load_dword") {
      if (operands.size() != 4 && operands.size() != 5) {
        throw std::invalid_argument("opcode " + opcode + " expects 4 or 5 operands");
      }
      builder.AddInstruction(Opcode::MLoadGlobal,
                             {reg(operands[0]), reg(operands[1]), reg(operands[2]), imm(operands[3]),
                              operands.size() == 5 ? imm(operands[4]) : builder.ImmediateOperand(0)});
    } else if (opcode == "buffer_store_dword") {
      if (operands.size() != 4 && operands.size() != 5) {
        throw std::invalid_argument("opcode " + opcode + " expects 4 or 5 operands");
      }
      builder.AddInstruction(Opcode::MStoreGlobal,
                             {reg(operands[0]), reg(operands[1]), reg(operands[2]), imm(operands[3]),
                              operands.size() == 5 ? imm(operands[4]) : builder.ImmediateOperand(0)});
    } else if (opcode == "buffer_atomic_add_u32") {
      if (operands.size() != 4 && operands.size() != 5) {
        throw std::invalid_argument("opcode " + opcode + " expects 4 or 5 operands");
      }
      builder.AddInstruction(Opcode::MAtomicAddGlobal,
                             {reg(operands[0]), reg(operands[1]), reg(operands[2]), imm(operands[3]),
                              operands.size() == 5 ? imm(operands[4]) : builder.ImmediateOperand(0)});
    } else if (opcode == "global_load_dword_addr") {
      if (operands.size() != 3 && operands.size() != 4) {
        throw std::invalid_argument("opcode " + opcode + " expects 3 or 4 operands");
      }
      builder.AddInstruction(Opcode::MLoadGlobalAddr,
                             {reg(operands[0]), reg(operands[1]), reg(operands[2]),
                              operands.size() == 4 ? imm(operands[3]) : builder.ImmediateOperand(0)});
    } else if (opcode == "global_store_dword_addr") {
      if (operands.size() != 3 && operands.size() != 4) {
        throw std::invalid_argument("opcode " + opcode + " expects 3 or 4 operands");
      }
      builder.AddInstruction(Opcode::MStoreGlobalAddr,
                             {reg(operands[0]), reg(operands[1]), reg(operands[2]),
                              operands.size() == 4 ? imm(operands[3]) : builder.ImmediateOperand(0)});
    } else if (opcode == "ds_read_b32") {
      RequireOperandCount(opcode, operands, 3);
      builder.AddInstruction(Opcode::MLoadShared, {reg(operands[0]), reg(operands[1]), imm(operands[2])});
    } else if (opcode == "ds_write_b32") {
      RequireOperandCount(opcode, operands, 3);
      builder.AddInstruction(Opcode::MStoreShared, {reg(operands[0]), reg(operands[1]), imm(operands[2])});
    } else if (opcode == "ds_add_u32") {
      RequireOperandCount(opcode, operands, 3);
      builder.AddInstruction(Opcode::MAtomicAddShared, {reg(operands[0]), reg(operands[1]), imm(operands[2])});
    } else if (opcode == "scratch_load_dword") {
      RequireOperandCount(opcode, operands, 3);
      builder.AddInstruction(Opcode::MLoadPrivate, {reg(operands[0]), reg(operands[1]), imm(operands[2])});
    } else if (opcode == "scalar_buffer_load_dword") {
      if (operands.size() != 3 && operands.size() != 4) {
        throw std::invalid_argument("opcode " + opcode + " expects 3 or 4 operands");
      }
      builder.AddInstruction(Opcode::MLoadConst,
                             {reg(operands[0]), reg(operands[1]), imm(operands[2]),
                              operands.size() == 4 ? imm(operands[3]) : builder.ImmediateOperand(0)});
    } else if (opcode == "scratch_store_dword") {
      RequireOperandCount(opcode, operands, 3);
      builder.AddInstruction(Opcode::MStorePrivate, {reg(operands[0]), reg(operands[1]), imm(operands[2])});
    } else if (opcode == "s_saveexec_b64") {
      RequireOperandCount(opcode, operands, 1);
      builder.AddInstruction(Opcode::MaskSaveExec, {reg(operands[0])});
    } else if (opcode == "s_restoreexec_b64") {
      RequireOperandCount(opcode, operands, 1);
      builder.AddInstruction(Opcode::MaskRestoreExec, {reg(operands[0])});
    } else if (opcode == "s_and_exec_cmask_b64") {
      RequireOperandCount(opcode, operands, 0);
      builder.AddInstruction(Opcode::MaskAndExecCmask, {});
    } else if (opcode == "s_branch") {
      RequireOperandCount(opcode, operands, 1);
      builder.AddBranch(Opcode::BBranch, operands[0]);
    } else if (opcode == "s_cbranch_scc1") {
      RequireOperandCount(opcode, operands, 1);
      builder.AddBranch(Opcode::BIfSmask, operands[0]);
    } else if (opcode == "s_cbranch_execz") {
      RequireOperandCount(opcode, operands, 1);
      builder.AddBranch(Opcode::BIfNoexec, operands[0]);
    } else if (opcode == "s_wave_barrier") {
      RequireOperandCount(opcode, operands, 0);
      builder.AddInstruction(Opcode::SyncWaveBarrier, {});
    } else if (opcode == "s_barrier") {
      RequireOperandCount(opcode, operands, 0);
      builder.AddInstruction(Opcode::SyncBarrier, {});
    } else if (opcode == "s_endpgm") {
      RequireOperandCount(opcode, operands, 0);
      builder.AddInstruction(Opcode::BExit, {});
    } else {
      throw std::invalid_argument("unsupported opcode in asm parser: " + opcode);
    }
  }

  return builder.Build(image.kernel_name(), std::move(metadata), image.const_segment());
}

}  // namespace gpu_model
