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

    if (opcode == "s_load_kernarg") {
      RequireOperandCount(opcode, operands, 2);
      builder.SLoadArg(operands[0], static_cast<uint32_t>(ParseImmediate(operands[1])));
    } else if (opcode == "v_get_global_id_x") {
      RequireOperandCount(opcode, operands, 1);
      builder.SysGlobalIdX(operands[0]);
    } else if (opcode == "v_get_global_id_y") {
      RequireOperandCount(opcode, operands, 1);
      builder.SysGlobalIdY(operands[0]);
    } else if (opcode == "v_get_local_id_x") {
      RequireOperandCount(opcode, operands, 1);
      builder.SysLocalIdX(operands[0]);
    } else if (opcode == "v_get_local_id_y") {
      RequireOperandCount(opcode, operands, 1);
      builder.SysLocalIdY(operands[0]);
    } else if (opcode == "s_get_block_offset_x") {
      RequireOperandCount(opcode, operands, 1);
      builder.SysBlockOffsetX(operands[0]);
    } else if (opcode == "s_get_block_id_x") {
      RequireOperandCount(opcode, operands, 1);
      builder.SysBlockIdxX(operands[0]);
    } else if (opcode == "s_get_block_id_y") {
      RequireOperandCount(opcode, operands, 1);
      builder.SysBlockIdxY(operands[0]);
    } else if (opcode == "s_get_block_dim_x") {
      RequireOperandCount(opcode, operands, 1);
      builder.SysBlockDimX(operands[0]);
    } else if (opcode == "s_get_block_dim_y") {
      RequireOperandCount(opcode, operands, 1);
      builder.SysBlockDimY(operands[0]);
    } else if (opcode == "s_get_grid_dim_x") {
      RequireOperandCount(opcode, operands, 1);
      builder.SysGridDimX(operands[0]);
    } else if (opcode == "s_get_grid_dim_y") {
      RequireOperandCount(opcode, operands, 1);
      builder.SysGridDimY(operands[0]);
    } else if (opcode == "v_lane_id_u32") {
      RequireOperandCount(opcode, operands, 1);
      builder.SysLaneId(operands[0]);
    } else if (opcode == "s_mov_b32") {
      RequireOperandCount(opcode, operands, 2);
      if (IsRegister(operands[1])) {
        builder.SMov(operands[0], operands[1]);
      } else {
        builder.SMov(operands[0], ParseImmediate(operands[1]));
      }
    } else if (opcode == "s_add_u32") {
      RequireOperandCount(opcode, operands, 3);
      if (IsRegister(operands[2])) {
        builder.SAdd(operands[0], operands[1], operands[2]);
      } else {
        builder.SAdd(operands[0], operands[1], ParseImmediate(operands[2]));
      }
    } else if (opcode == "s_sub_u32") {
      RequireOperandCount(opcode, operands, 3);
      if (IsRegister(operands[2])) {
        builder.SSub(operands[0], operands[1], operands[2]);
      } else {
        builder.SSub(operands[0], operands[1], ParseImmediate(operands[2]));
      }
    } else if (opcode == "s_mul_i32") {
      RequireOperandCount(opcode, operands, 3);
      if (IsRegister(operands[2])) {
        builder.SMul(operands[0], operands[1], operands[2]);
      } else {
        builder.SMul(operands[0], operands[1], ParseImmediate(operands[2]));
      }
    } else if (opcode == "s_div_i32") {
      RequireOperandCount(opcode, operands, 3);
      if (IsRegister(operands[2])) {
        builder.SDiv(operands[0], operands[1], operands[2]);
      } else {
        builder.SDiv(operands[0], operands[1], ParseImmediate(operands[2]));
      }
    } else if (opcode == "s_rem_i32") {
      RequireOperandCount(opcode, operands, 3);
      if (IsRegister(operands[2])) {
        builder.SRem(operands[0], operands[1], operands[2]);
      } else {
        builder.SRem(operands[0], operands[1], ParseImmediate(operands[2]));
      }
    } else if (opcode == "s_and_b32") {
      RequireOperandCount(opcode, operands, 3);
      if (IsRegister(operands[2])) {
        builder.SAnd(operands[0], operands[1], operands[2]);
      } else {
        builder.SAnd(operands[0], operands[1], ParseImmediate(operands[2]));
      }
    } else if (opcode == "s_or_b32") {
      RequireOperandCount(opcode, operands, 3);
      if (IsRegister(operands[2])) {
        builder.SOr(operands[0], operands[1], operands[2]);
      } else {
        builder.SOr(operands[0], operands[1], ParseImmediate(operands[2]));
      }
    } else if (opcode == "s_xor_b32") {
      RequireOperandCount(opcode, operands, 3);
      if (IsRegister(operands[2])) {
        builder.SXor(operands[0], operands[1], operands[2]);
      } else {
        builder.SXor(operands[0], operands[1], ParseImmediate(operands[2]));
      }
    } else if (opcode == "s_lshl_b32") {
      RequireOperandCount(opcode, operands, 3);
      if (IsRegister(operands[2])) {
        builder.SShl(operands[0], operands[1], operands[2]);
      } else {
        builder.SShl(operands[0], operands[1], ParseImmediate(operands[2]));
      }
    } else if (opcode == "s_lshr_b32") {
      RequireOperandCount(opcode, operands, 3);
      if (IsRegister(operands[2])) {
        builder.SShr(operands[0], operands[1], operands[2]);
      } else {
        builder.SShr(operands[0], operands[1], ParseImmediate(operands[2]));
      }
    } else if (opcode == "s_waitcnt") {
      const auto thresholds =
          ParseWaitCnt(space == std::string::npos ? std::string_view{} :
                                                     std::string_view(trimmed).substr(space + 1));
      builder.SWaitCnt(thresholds.global, thresholds.shared, thresholds.private_mem,
                       thresholds.scalar_buffer);
    } else if (opcode == "s_buffer_load_dword") {
      if (operands.size() != 3 && operands.size() != 4) {
        throw std::invalid_argument("opcode " + opcode + " expects 3 or 4 operands");
      }
      builder.SBufferLoadDword(operands[0], operands[1],
                               static_cast<uint32_t>(ParseImmediate(operands[2])),
                               operands.size() == 4 ? static_cast<uint32_t>(ParseImmediate(operands[3])) : 0);
    } else if (opcode == "s_cmp_lt_i32") {
      RequireOperandCount(opcode, operands, 2);
      if (IsRegister(operands[1])) {
        builder.SCmpLt(operands[0], operands[1]);
      } else {
        builder.SCmpLt(operands[0], ParseImmediate(operands[1]));
      }
    } else if (opcode == "s_cmp_eq_u32") {
      RequireOperandCount(opcode, operands, 2);
      if (IsRegister(operands[1])) {
        builder.SCmpEq(operands[0], operands[1]);
      } else {
        builder.SCmpEq(operands[0], ParseImmediate(operands[1]));
      }
    } else if (opcode == "s_cmp_gt_i32") {
      RequireOperandCount(opcode, operands, 2);
      if (IsRegister(operands[1])) {
        builder.SCmpGt(operands[0], operands[1]);
      } else {
        builder.SCmpGt(operands[0], ParseImmediate(operands[1]));
      }
    } else if (opcode == "s_cmp_ge_i32") {
      RequireOperandCount(opcode, operands, 2);
      if (IsRegister(operands[1])) {
        builder.SCmpGe(operands[0], operands[1]);
      } else {
        builder.SCmpGe(operands[0], ParseImmediate(operands[1]));
      }
    } else if (opcode == "v_mov_b32") {
      RequireOperandCount(opcode, operands, 2);
      if (IsRegister(operands[1])) {
        builder.VMov(operands[0], operands[1]);
      } else {
        builder.VMov(operands[0], ParseImmediate(operands[1]));
      }
    } else if (opcode == "v_add_i32") {
      RequireOperandCount(opcode, operands, 3);
      builder.VAdd(operands[0], operands[1], operands[2]);
    } else if (opcode == "v_and_b32") {
      RequireOperandCount(opcode, operands, 3);
      builder.VAnd(operands[0], operands[1], operands[2]);
    } else if (opcode == "v_or_b32") {
      RequireOperandCount(opcode, operands, 3);
      builder.VOr(operands[0], operands[1], operands[2]);
    } else if (opcode == "v_xor_b32") {
      RequireOperandCount(opcode, operands, 3);
      builder.VXor(operands[0], operands[1], operands[2]);
    } else if (opcode == "v_lshl_b32") {
      RequireOperandCount(opcode, operands, 3);
      builder.VShl(operands[0], operands[1], operands[2]);
    } else if (opcode == "v_lshr_b32") {
      RequireOperandCount(opcode, operands, 3);
      builder.VShr(operands[0], operands[1], operands[2]);
    } else if (opcode == "v_sub_i32") {
      RequireOperandCount(opcode, operands, 3);
      builder.VSub(operands[0], operands[1], operands[2]);
    } else if (opcode == "v_div_i32") {
      RequireOperandCount(opcode, operands, 3);
      builder.VDiv(operands[0], operands[1], operands[2]);
    } else if (opcode == "v_rem_i32") {
      RequireOperandCount(opcode, operands, 3);
      builder.VRem(operands[0], operands[1], operands[2]);
    } else if (opcode == "v_mul_lo_i32") {
      RequireOperandCount(opcode, operands, 3);
      builder.VMul(operands[0], operands[1], operands[2]);
    } else if (opcode == "v_min_i32") {
      RequireOperandCount(opcode, operands, 3);
      builder.VMin(operands[0], operands[1], operands[2]);
    } else if (opcode == "v_max_i32") {
      RequireOperandCount(opcode, operands, 3);
      builder.VMax(operands[0], operands[1], operands[2]);
    } else if (opcode == "v_mad_i32") {
      RequireOperandCount(opcode, operands, 4);
      builder.VFma(operands[0], operands[1], operands[2], operands[3]);
    } else if (opcode == "v_cmp_lt_i32_cmask") {
      RequireOperandCount(opcode, operands, 2);
      builder.VCmpLtCmask(operands[0], operands[1]);
    } else if (opcode == "v_cmp_eq_i32_cmask") {
      RequireOperandCount(opcode, operands, 2);
      builder.VCmpEqCmask(operands[0], operands[1]);
    } else if (opcode == "v_cmp_ge_i32_cmask") {
      RequireOperandCount(opcode, operands, 2);
      builder.VCmpGeCmask(operands[0], operands[1]);
    } else if (opcode == "v_cmp_gt_i32_cmask") {
      RequireOperandCount(opcode, operands, 2);
      builder.VCmpGtCmask(operands[0], operands[1]);
    } else if (opcode == "v_cndmask_b32") {
      RequireOperandCount(opcode, operands, 3);
      builder.VSelectCmask(operands[0], operands[1], operands[2]);
    } else if (opcode == "buffer_load_dword") {
      if (operands.size() != 4 && operands.size() != 5) {
        throw std::invalid_argument("opcode " + opcode + " expects 4 or 5 operands");
      }
      builder.MLoadGlobal(operands[0], operands[1], operands[2],
                          static_cast<uint32_t>(ParseImmediate(operands[3])),
                          operands.size() == 5 ? static_cast<uint32_t>(ParseImmediate(operands[4])) : 0);
    } else if (opcode == "buffer_store_dword") {
      if (operands.size() != 4 && operands.size() != 5) {
        throw std::invalid_argument("opcode " + opcode + " expects 4 or 5 operands");
      }
      builder.MStoreGlobal(operands[0], operands[1], operands[2],
                           static_cast<uint32_t>(ParseImmediate(operands[3])),
                           operands.size() == 5 ? static_cast<uint32_t>(ParseImmediate(operands[4])) : 0);
    } else if (opcode == "buffer_atomic_add_u32") {
      if (operands.size() != 4 && operands.size() != 5) {
        throw std::invalid_argument("opcode " + opcode + " expects 4 or 5 operands");
      }
      builder.MAtomicAddGlobal(operands[0], operands[1], operands[2],
                               static_cast<uint32_t>(ParseImmediate(operands[3])),
                               operands.size() == 5 ? static_cast<uint32_t>(ParseImmediate(operands[4])) : 0);
    } else if (opcode == "ds_read_b32") {
      RequireOperandCount(opcode, operands, 3);
      builder.MLoadShared(operands[0], operands[1],
                          static_cast<uint32_t>(ParseImmediate(operands[2])));
    } else if (opcode == "ds_write_b32") {
      RequireOperandCount(opcode, operands, 3);
      builder.MStoreShared(operands[0], operands[1],
                           static_cast<uint32_t>(ParseImmediate(operands[2])));
    } else if (opcode == "ds_add_u32") {
      RequireOperandCount(opcode, operands, 3);
      builder.MAtomicAddShared(operands[0], operands[1],
                               static_cast<uint32_t>(ParseImmediate(operands[2])));
    } else if (opcode == "scratch_load_dword") {
      RequireOperandCount(opcode, operands, 3);
      builder.MLoadPrivate(operands[0], operands[1],
                           static_cast<uint32_t>(ParseImmediate(operands[2])));
    } else if (opcode == "scalar_buffer_load_dword") {
      if (operands.size() != 3 && operands.size() != 4) {
        throw std::invalid_argument("opcode " + opcode + " expects 3 or 4 operands");
      }
      builder.MLoadConst(operands[0], operands[1],
                         static_cast<uint32_t>(ParseImmediate(operands[2])),
                         operands.size() == 4 ? static_cast<uint32_t>(ParseImmediate(operands[3])) : 0);
    } else if (opcode == "scratch_store_dword") {
      RequireOperandCount(opcode, operands, 3);
      builder.MStorePrivate(operands[0], operands[1],
                            static_cast<uint32_t>(ParseImmediate(operands[2])));
    } else if (opcode == "s_saveexec_b64") {
      RequireOperandCount(opcode, operands, 1);
      builder.MaskSaveExec(operands[0]);
    } else if (opcode == "s_restoreexec_b64") {
      RequireOperandCount(opcode, operands, 1);
      builder.MaskRestoreExec(operands[0]);
    } else if (opcode == "s_and_exec_cmask_b64") {
      RequireOperandCount(opcode, operands, 0);
      builder.MaskAndExecCmask();
    } else if (opcode == "s_branch") {
      RequireOperandCount(opcode, operands, 1);
      builder.BBranch(operands[0]);
    } else if (opcode == "s_cbranch_scc1") {
      RequireOperandCount(opcode, operands, 1);
      builder.BIfSmask(operands[0]);
    } else if (opcode == "s_cbranch_execz") {
      RequireOperandCount(opcode, operands, 1);
      builder.BIfNoexec(operands[0]);
    } else if (opcode == "s_wave_barrier") {
      RequireOperandCount(opcode, operands, 0);
      builder.SyncWaveBarrier();
    } else if (opcode == "s_barrier") {
      RequireOperandCount(opcode, operands, 0);
      builder.SyncBarrier();
    } else if (opcode == "s_endpgm") {
      RequireOperandCount(opcode, operands, 0);
      builder.BExit();
    } else {
      throw std::invalid_argument("unsupported opcode in asm parser: " + opcode);
    }
  }

  return builder.Build(image.kernel_name(), std::move(metadata), image.const_segment());
}

}  // namespace gpu_model
