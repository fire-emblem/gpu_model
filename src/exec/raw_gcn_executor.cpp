#include "gpu_model/exec/raw_gcn_executor.h"

#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <unordered_map>

#include "gpu_model/loader/gcn_text_parser.h"
#include "gpu_model/runtime/mapper.h"
#include "gpu_model/state/wave_state.h"

namespace gpu_model {

namespace {

struct RawWave {
  WaveState wave;
  uint64_t vcc = 0;
};

uint32_t LaneCount(const RawWave& raw_wave) {
  return raw_wave.wave.thread_count < kWaveSize ? raw_wave.wave.thread_count : kWaveSize;
}

std::bitset<64> MaskFromU64(uint64_t value) {
  return std::bitset<64>(value);
}

std::vector<std::byte> BuildKernargBytes(const KernelArgPack& args, const LaunchConfig& config) {
  std::vector<std::byte> bytes(128, std::byte{0});
  for (size_t i = 0; i < args.values().size(); ++i) {
    const uint64_t value = args.values()[i];
    if (i < 3) {
      std::memcpy(bytes.data() + i * 8, &value, sizeof(uint64_t));
    } else if (i == 3) {
      const uint32_t narrowed = static_cast<uint32_t>(value);
      std::memcpy(bytes.data() + 24, &narrowed, sizeof(uint32_t));
    }
  }
  const uint32_t block_count_x = config.grid_dim_x;
  const uint32_t block_count_y = config.grid_dim_y;
  const uint32_t block_count_z = 1;
  const uint16_t group_size_x = static_cast<uint16_t>(config.block_dim_x);
  const uint16_t group_size_y = static_cast<uint16_t>(config.block_dim_y);
  const uint16_t group_size_z = 1;
  std::memcpy(bytes.data() + 32, &block_count_x, sizeof(block_count_x));
  std::memcpy(bytes.data() + 36, &block_count_y, sizeof(block_count_y));
  std::memcpy(bytes.data() + 40, &block_count_z, sizeof(block_count_z));
  std::memcpy(bytes.data() + 44, &group_size_x, sizeof(group_size_x));
  std::memcpy(bytes.data() + 46, &group_size_y, sizeof(group_size_y));
  std::memcpy(bytes.data() + 48, &group_size_z, sizeof(group_size_z));
  return bytes;
}

uint32_t LoadU32(const std::vector<std::byte>& bytes, uint32_t offset) {
  uint32_t value = 0;
  std::memcpy(&value, bytes.data() + offset, sizeof(value));
  return value;
}

void StoreU32(std::vector<std::byte>& bytes, uint32_t offset, uint32_t value) {
  std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

uint64_t LoadKernarg64(const std::vector<std::byte>& bytes, uint32_t offset) {
  uint64_t value = 0;
  std::memcpy(&value, bytes.data() + offset, sizeof(value));
  return value;
}

uint32_t LoadKernarg32(const std::vector<std::byte>& bytes, uint32_t offset) {
  return LoadU32(bytes, offset);
}

uint64_t BranchTarget(uint64_t pc, uint32_t simm16) {
  return pc + 4ull + static_cast<uint64_t>(simm16) * 4ull;
}

float U32AsFloat(uint32_t bits) {
  return std::bit_cast<float>(bits);
}

uint32_t FloatAsU32(float value) {
  return std::bit_cast<uint32_t>(value);
}

bool DebugEnabled() {
  return std::getenv("GPU_MODEL_RAW_GCN_DEBUG") != nullptr;
}

void DebugLog(const char* fmt, ...) {
  if (!DebugEnabled()) {
    return;
  }
  va_list args;
  va_start(args, fmt);
  std::fputs("[gpu_model_raw_gcn] ", stderr);
  std::vfprintf(stderr, fmt, args);
  std::fputc('\n', stderr);
  va_end(args);
}

const GcnTextInstruction& ParseText(const RawGcnInstruction& inst) {
  thread_local GcnTextInstruction parsed;
  parsed = GcnTextParser::ParseInstruction(
      inst.operands.empty() ? inst.mnemonic : inst.mnemonic + " " + inst.operands);
  return parsed;
}

uint32_t RequireScalarIndex(const GcnTextOperand& operand) {
  if (!operand.reg.has_value() || operand.reg->prefix != 's') {
    throw std::invalid_argument("expected scalar register operand");
  }
  return operand.reg->index;
}

uint32_t RequireVectorIndex(const GcnTextOperand& operand) {
  if (!operand.reg.has_value() || operand.reg->prefix != 'v') {
    throw std::invalid_argument("expected vector register operand");
  }
  return operand.reg->index;
}

std::pair<uint32_t, uint32_t> RequireScalarRange(const GcnTextOperand& operand) {
  if (!operand.reg_range.has_value() || operand.reg_range->prefix != 's') {
    throw std::invalid_argument("expected scalar register range operand");
  }
  return {operand.reg_range->first, operand.reg_range->last};
}

std::pair<uint32_t, uint32_t> RequireVectorRange(const GcnTextOperand& operand) {
  if (!operand.reg_range.has_value() || operand.reg_range->prefix != 'v') {
    throw std::invalid_argument("expected vector register range operand");
  }
  return {operand.reg_range->first, operand.reg_range->last};
}

uint64_t ResolveScalarLike(const GcnTextOperand& operand, const RawWave& raw_wave) {
  if (operand.kind == GcnTextOperandKind::Immediate && operand.immediate.has_value()) {
    return *operand.immediate;
  }
  if (operand.kind == GcnTextOperandKind::Register && operand.reg->prefix == 's') {
    return raw_wave.wave.sgpr.Read(operand.reg->index);
  }
  throw std::invalid_argument("unsupported scalar-like raw operand");
}

uint64_t ResolveVectorLane(const GcnTextOperand& operand, const RawWave& raw_wave, uint32_t lane) {
  if (operand.kind == GcnTextOperandKind::Immediate && operand.immediate.has_value()) {
    return *operand.immediate;
  }
  if (operand.kind == GcnTextOperandKind::Register) {
    if (operand.reg->prefix == 's') {
      return raw_wave.wave.sgpr.Read(operand.reg->index);
    }
    if (operand.reg->prefix == 'v') {
      return raw_wave.wave.vgpr.Read(operand.reg->index, lane);
    }
  }
  throw std::invalid_argument("unsupported vector-lane raw operand");
}

}  // namespace

LaunchResult RawGcnExecutor::Run(const AmdgpuCodeObjectImage& image,
                                 const GpuArchSpec& spec,
                                 const LaunchConfig& config,
                                 const KernelArgPack& args,
                                 MemorySystem& memory,
                                 TraceSink& trace) const {
  if (config.grid_dim_y != 1 || config.block_dim_y != 1) {
    throw std::invalid_argument("raw GCN executor currently supports only 1D launches");
  }

  LaunchResult result;
  result.ok = false;
  result.placement = Mapper::Place(spec, config);
  trace.OnEvent(TraceEvent{
      .kind = TraceEventKind::Launch,
      .cycle = 0,
      .message = "raw_kernel=" + image.kernel_name + " arch=" + spec.name,
  });

  std::unordered_map<uint64_t, size_t> pc_to_index;
  for (size_t i = 0; i < image.instructions.size(); ++i) {
    pc_to_index[image.instructions[i].pc] = i;
  }
  const auto kernarg = BuildKernargBytes(args, config);

  for (const auto& block : result.placement.blocks) {
    std::vector<RawWave> waves;
    waves.reserve(block.waves.size());
    for (const auto& wave_placement : block.waves) {
      RawWave raw_wave;
      raw_wave.wave.block_id = block.block_id;
      raw_wave.wave.block_idx_x = block.block_idx_x;
      raw_wave.wave.block_idx_y = block.block_idx_y;
      raw_wave.wave.dpc_id = block.dpc_id;
      raw_wave.wave.wave_id = wave_placement.wave_id;
      raw_wave.wave.peu_id = wave_placement.peu_id;
      raw_wave.wave.ap_id = block.ap_id;
      raw_wave.wave.thread_count = wave_placement.lane_count;
      raw_wave.wave.ResetInitialExec();
      raw_wave.wave.pc = image.instructions.front().pc;
      raw_wave.wave.sgpr.Write(4, 0);
      raw_wave.wave.sgpr.Write(5, 0);
      raw_wave.wave.sgpr.Write(6, block.block_idx_x);
      for (uint32_t lane = 0; lane < LaneCount(raw_wave); ++lane) {
        raw_wave.wave.vgpr.Write(0, lane, raw_wave.wave.wave_id * kWaveSize + lane);
      }
      waves.push_back(std::move(raw_wave));
    }

    for (auto& raw_wave : waves) {
      while (raw_wave.wave.status == WaveStatus::Active) {
        const auto it = pc_to_index.find(raw_wave.wave.pc);
        if (it == pc_to_index.end()) {
          throw std::out_of_range("raw GCN wave pc out of range");
        }
        const auto& inst = image.instructions[it->second];
        DebugLog("exec pc=0x%llx %s %s",
                 static_cast<unsigned long long>(inst.pc), inst.mnemonic.c_str(),
                 inst.operands.c_str());
        const auto& parsed = ParseText(inst);
        ++result.stats.wave_steps;
        ++result.stats.instructions_issued;
        try {
        if (inst.mnemonic == "s_load_dword") {
          const uint32_t offset = static_cast<uint32_t>(parsed.operands.at(2).immediate.value_or(0));
          const uint32_t sdst = RequireScalarIndex(parsed.operands.at(0));
          raw_wave.wave.sgpr.Write(sdst, LoadKernarg32(kernarg, offset));
          raw_wave.wave.pc += inst.size_bytes;
        } else if (inst.mnemonic == "s_load_dwordx2") {
          const uint32_t offset = static_cast<uint32_t>(parsed.operands.at(2).immediate.value_or(0));
          const auto [sdst, _] = RequireScalarRange(parsed.operands.at(0));
          raw_wave.wave.sgpr.Write(sdst, LoadKernarg64(kernarg, offset) & 0xffffffffu);
          raw_wave.wave.sgpr.Write(sdst + 1, LoadKernarg64(kernarg, offset) >> 32u);
          raw_wave.wave.pc += inst.size_bytes;
        } else if (inst.mnemonic == "s_load_dwordx4") {
          const uint32_t offset = static_cast<uint32_t>(parsed.operands.at(2).immediate.value_or(0));
          const auto [sdst, _] = RequireScalarRange(parsed.operands.at(0));
          raw_wave.wave.sgpr.Write(sdst + 0, LoadKernarg32(kernarg, offset + 0));
          raw_wave.wave.sgpr.Write(sdst + 1, LoadKernarg32(kernarg, offset + 4));
          raw_wave.wave.sgpr.Write(sdst + 2, LoadKernarg32(kernarg, offset + 8));
          raw_wave.wave.sgpr.Write(sdst + 3, LoadKernarg32(kernarg, offset + 12));
          raw_wave.wave.pc += inst.size_bytes;
        } else if (inst.mnemonic == "s_waitcnt") {
          raw_wave.wave.pc += inst.size_bytes;
        } else if (inst.mnemonic == "s_and_b32") {
          const uint32_t sdst = RequireScalarIndex(parsed.operands.at(0));
          const uint32_t lhs = static_cast<uint32_t>(ResolveScalarLike(parsed.operands.at(1), raw_wave));
          const uint32_t rhs = static_cast<uint32_t>(ResolveScalarLike(parsed.operands.at(2), raw_wave));
          raw_wave.wave.sgpr.Write(sdst, lhs & rhs);
          raw_wave.wave.pc += inst.size_bytes;
        } else if (inst.mnemonic == "s_mul_i32") {
          const uint32_t sdst = RequireScalarIndex(parsed.operands.at(0));
          const uint32_t ssrc0 = static_cast<uint32_t>(ResolveScalarLike(parsed.operands.at(1), raw_wave));
          const uint32_t ssrc1 = static_cast<uint32_t>(ResolveScalarLike(parsed.operands.at(2), raw_wave));
          raw_wave.wave.sgpr.Write(
              sdst, ssrc0 * ssrc1);
          raw_wave.wave.pc += inst.size_bytes;
        } else if (inst.mnemonic == "v_add_u32_e32") {
          const uint32_t vdst = RequireVectorIndex(parsed.operands.at(0));
          for (uint32_t lane = 0; lane < LaneCount(raw_wave); ++lane) {
            const uint32_t lhs = static_cast<uint32_t>(ResolveVectorLane(parsed.operands.at(1), raw_wave, lane));
            const uint32_t rhs = static_cast<uint32_t>(ResolveVectorLane(parsed.operands.at(2), raw_wave, lane));
            raw_wave.wave.vgpr.Write(vdst, lane, lhs + rhs);
          }
          raw_wave.wave.pc += inst.size_bytes;
        } else if (inst.mnemonic == "v_cmp_gt_i32_e32") {
          raw_wave.vcc = 0;
          for (uint32_t lane = 0; lane < LaneCount(raw_wave); ++lane) {
            const int32_t lhs = static_cast<int32_t>(ResolveVectorLane(parsed.operands.at(1), raw_wave, lane));
            const int32_t rhs = static_cast<int32_t>(ResolveVectorLane(parsed.operands.at(2), raw_wave, lane));
            if (lhs > rhs) {
              raw_wave.vcc |= (1ull << lane);
            }
          }
          DebugLog("pc=0x%llx v_cmp_gt_i32_e32 vcc=0x%llx",
                   static_cast<unsigned long long>(inst.pc),
                   static_cast<unsigned long long>(raw_wave.vcc));
          raw_wave.wave.pc += inst.size_bytes;
        } else if (inst.mnemonic == "s_and_saveexec_b64") {
          const auto [sdst, _] = RequireScalarRange(parsed.operands.at(0));
          const uint64_t exec_before = raw_wave.wave.exec.to_ullong();
          raw_wave.wave.sgpr.Write(sdst, static_cast<uint32_t>(exec_before & 0xffffffffu));
          raw_wave.wave.sgpr.Write(sdst + 1, static_cast<uint32_t>(exec_before >> 32u));
          raw_wave.wave.exec = raw_wave.wave.exec & MaskFromU64(raw_wave.vcc);
          DebugLog("pc=0x%llx s_and_saveexec_b64 exec=0x%llx",
                   static_cast<unsigned long long>(inst.pc),
                   static_cast<unsigned long long>(raw_wave.wave.exec.to_ullong()));
          raw_wave.wave.pc += inst.size_bytes;
        } else if (inst.mnemonic == "s_cbranch_execz") {
          if (raw_wave.wave.exec.none()) {
            raw_wave.wave.pc = BranchTarget(raw_wave.wave.pc, static_cast<uint32_t>(parsed.operands.at(0).immediate.value_or(0)));
          } else {
            raw_wave.wave.pc += inst.size_bytes;
          }
        } else if (inst.mnemonic == "v_ashrrev_i32_e32") {
          const uint32_t vdst = RequireVectorIndex(parsed.operands.at(0));
          for (uint32_t lane = 0; lane < LaneCount(raw_wave); ++lane) {
            const uint32_t imm = static_cast<uint32_t>(ResolveVectorLane(parsed.operands.at(1), raw_wave, lane));
            const int32_t rhs = static_cast<int32_t>(ResolveVectorLane(parsed.operands.at(2), raw_wave, lane));
            raw_wave.wave.vgpr.Write(vdst, lane, static_cast<uint32_t>(rhs >> (imm & 31u)));
          }
          raw_wave.wave.pc += inst.size_bytes;
        } else if (inst.mnemonic == "v_lshlrev_b64") {
          const auto [vdst, _vdst_hi] = RequireVectorRange(parsed.operands.at(0));
          const uint32_t shift = static_cast<uint32_t>(ResolveVectorLane(parsed.operands.at(1), raw_wave, 0));
          const auto [src_pair, _src_pair_hi] = RequireVectorRange(parsed.operands.at(2));
          for (uint32_t lane = 0; lane < LaneCount(raw_wave); ++lane) {
            const uint64_t lo = static_cast<uint32_t>(raw_wave.wave.vgpr.Read(src_pair, lane));
            const uint64_t hi = static_cast<uint32_t>(raw_wave.wave.vgpr.Read(src_pair + 1, lane));
            const uint64_t value = ((hi << 32u) | lo) << (shift & 63u);
            raw_wave.wave.vgpr.Write(vdst, lane, static_cast<uint32_t>(value & 0xffffffffu));
            raw_wave.wave.vgpr.Write(vdst + 1, lane, static_cast<uint32_t>(value >> 32u));
          }
          raw_wave.wave.pc += inst.size_bytes;
        } else if (inst.mnemonic == "v_mov_b32_e32") {
          const uint32_t vdst = RequireVectorIndex(parsed.operands.at(0));
          for (uint32_t lane = 0; lane < LaneCount(raw_wave); ++lane) {
            raw_wave.wave.vgpr.Write(vdst, lane,
                                     static_cast<uint32_t>(ResolveVectorLane(parsed.operands.at(1), raw_wave, lane)));
          }
          raw_wave.wave.pc += inst.size_bytes;
        } else if (inst.mnemonic == "v_add_co_u32_e32") {
          const uint32_t vdst = RequireVectorIndex(parsed.operands.at(0));
          raw_wave.vcc = 0;
          for (uint32_t lane = 0; lane < LaneCount(raw_wave); ++lane) {
            const uint64_t lhs = static_cast<uint32_t>(ResolveVectorLane(parsed.operands.at(2), raw_wave, lane));
            const uint64_t rhs = static_cast<uint32_t>(ResolveVectorLane(parsed.operands.at(3), raw_wave, lane));
            const uint64_t sum = lhs + rhs;
            raw_wave.wave.vgpr.Write(vdst, lane, static_cast<uint32_t>(sum));
            if ((sum >> 32u) != 0) {
              raw_wave.vcc |= (1ull << lane);
            }
          }
          raw_wave.wave.pc += inst.size_bytes;
        } else if (inst.mnemonic == "v_addc_co_u32_e32") {
          const uint32_t vdst = RequireVectorIndex(parsed.operands.at(0));
          uint64_t next_vcc = 0;
          for (uint32_t lane = 0; lane < LaneCount(raw_wave); ++lane) {
            const uint64_t carry_in = (raw_wave.vcc >> lane) & 1ull;
            const uint64_t lhs = static_cast<uint32_t>(ResolveVectorLane(parsed.operands.at(2), raw_wave, lane));
            const uint64_t rhs = static_cast<uint32_t>(ResolveVectorLane(parsed.operands.at(3), raw_wave, lane));
            const uint64_t sum = lhs + rhs + carry_in;
            raw_wave.wave.vgpr.Write(vdst, lane, static_cast<uint32_t>(sum));
            if ((sum >> 32u) != 0) {
              next_vcc |= (1ull << lane);
            }
          }
          raw_wave.vcc = next_vcc;
          raw_wave.wave.pc += inst.size_bytes;
        } else if (inst.mnemonic == "global_load_dword") {
          const uint32_t vdst = RequireVectorIndex(parsed.operands.at(0));
          const auto [addr, _addr_hi] = RequireVectorRange(parsed.operands.at(1));
          for (uint32_t lane = 0; lane < LaneCount(raw_wave); ++lane) {
            const uint64_t lo = static_cast<uint32_t>(raw_wave.wave.vgpr.Read(addr, lane));
            const uint64_t hi = static_cast<uint32_t>(raw_wave.wave.vgpr.Read(addr + 1, lane));
            const uint64_t address = (hi << 32u) | lo;
            raw_wave.wave.vgpr.Write(vdst, lane, memory.LoadGlobalValue<uint32_t>(address));
            if (lane == 0) {
              DebugLog("pc=0x%llx global_load addr=0x%llx -> v%u=0x%llx",
                       static_cast<unsigned long long>(inst.pc),
                       static_cast<unsigned long long>(address), vdst,
                       static_cast<unsigned long long>(raw_wave.wave.vgpr.Read(vdst, lane)));
            }
          }
          ++result.stats.global_loads;
          raw_wave.wave.pc += inst.size_bytes;
        } else if (inst.mnemonic == "v_add_f32_e32") {
          const uint32_t vdst = RequireVectorIndex(parsed.operands.at(0));
          for (uint32_t lane = 0; lane < LaneCount(raw_wave); ++lane) {
            const float lhs = U32AsFloat(static_cast<uint32_t>(ResolveVectorLane(parsed.operands.at(1), raw_wave, lane)));
            const float rhs = U32AsFloat(static_cast<uint32_t>(ResolveVectorLane(parsed.operands.at(2), raw_wave, lane)));
            raw_wave.wave.vgpr.Write(vdst, lane, FloatAsU32(lhs + rhs));
          }
          raw_wave.wave.pc += inst.size_bytes;
        } else if (inst.mnemonic == "global_store_dword") {
          const auto [addr, _addr_hi] = RequireVectorRange(parsed.operands.at(0));
          const uint32_t data = RequireVectorIndex(parsed.operands.at(1));
          for (uint32_t lane = 0; lane < LaneCount(raw_wave); ++lane) {
            const uint64_t lo = static_cast<uint32_t>(raw_wave.wave.vgpr.Read(addr, lane));
            const uint64_t hi = static_cast<uint32_t>(raw_wave.wave.vgpr.Read(addr + 1, lane));
            const uint64_t address = (hi << 32u) | lo;
            memory.StoreGlobalValue<uint32_t>(
                address, static_cast<uint32_t>(raw_wave.wave.vgpr.Read(data, lane)));
            if (lane == 0) {
              DebugLog("pc=0x%llx global_store addr=0x%llx value=0x%llx",
                       static_cast<unsigned long long>(inst.pc),
                       static_cast<unsigned long long>(address),
                       static_cast<unsigned long long>(raw_wave.wave.vgpr.Read(data, lane)));
            }
          }
          ++result.stats.global_stores;
          raw_wave.wave.pc += inst.size_bytes;
        } else if (inst.mnemonic == "s_endpgm") {
          raw_wave.wave.status = WaveStatus::Exited;
          ++result.stats.wave_exits;
        } else {
          throw std::invalid_argument("unsupported raw GCN opcode: " + inst.mnemonic);
        }
        } catch (const std::exception& ex) {
          throw std::runtime_error(inst.mnemonic + ": " + ex.what());
        }
      }
    }
  }

  result.ok = true;
  return result;
}

}  // namespace gpu_model
