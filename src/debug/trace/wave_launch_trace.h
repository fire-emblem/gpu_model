#pragma once

#include <cstdint>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "state/wave/wave_runtime_state.h"

namespace gpu_model {

struct NamedLaneSample {
  std::string name;
  uint32_t vgpr_index = 0;
};

struct WaveLaunchAbiSummary {
  std::vector<std::pair<std::string, uint64_t>> sgpr_fields;
  std::vector<NamedLaneSample> vgpr_fields;
};

inline std::string HexWaveTraceU64(uint64_t value) {
  std::ostringstream out;
  out << "0x" << std::hex << std::nouppercase << value;
  return out.str();
}

inline std::string FormatWaveLaunchTraceMessage(const WaveContext& wave,
                                                const WaveLaunchAbiSummary* abi_summary,
                                                uint32_t scalar_regs = 4,
                                                uint32_t vector_regs = 2,
                                                uint32_t lanes_per_vector = 2) {
  std::ostringstream out;
  out << "block_xyz=(" << HexWaveTraceU64(wave.block_idx_x) << ","
      << HexWaveTraceU64(wave.block_idx_y) << ","
      << HexWaveTraceU64(wave.block_idx_z) << ")"
      << " dpc=" << HexWaveTraceU64(wave.dpc_id)
      << " ap=" << HexWaveTraceU64(wave.ap_id)
      << " peu=" << HexWaveTraceU64(wave.peu_id)
      << " lanes=" << HexWaveTraceU64(wave.thread_count)
      << " exec=" << HexWaveTraceU64(wave.exec.to_ullong())
      << " cmask=" << HexWaveTraceU64(wave.cmask.to_ullong())
      << " smask=" << HexWaveTraceU64(wave.smask);

  out << " sgpr={";
  if (abi_summary != nullptr && !abi_summary->sgpr_fields.empty()) {
    for (size_t i = 0; i < abi_summary->sgpr_fields.size(); ++i) {
      if (i != 0) {
        out << ",";
      }
      out << abi_summary->sgpr_fields[i].first << "="
          << HexWaveTraceU64(abi_summary->sgpr_fields[i].second);
    }
  } else {
    for (uint32_t i = 0; i < scalar_regs; ++i) {
      if (i != 0) {
        out << ",";
      }
      out << "s" << i << "=" << HexWaveTraceU64(wave.sgpr.Read(i));
    }
  }
  out << "} vgpr={";
  if (abi_summary != nullptr && !abi_summary->vgpr_fields.empty()) {
    for (size_t reg = 0; reg < abi_summary->vgpr_fields.size(); ++reg) {
      if (reg != 0) {
        out << ",";
      }
      out << abi_summary->vgpr_fields[reg].name << "[";
      for (uint32_t lane = 0; lane < lanes_per_vector; ++lane) {
        if (lane != 0) {
          out << ",";
        }
        out << lane;
      }
      out << "]={";
      for (uint32_t lane = 0; lane < lanes_per_vector; ++lane) {
        if (lane != 0) {
          out << ",";
        }
        out << HexWaveTraceU64(
            wave.vgpr.Read(abi_summary->vgpr_fields[reg].vgpr_index, lane));
      }
      out << "}";
    }
  } else {
    for (uint32_t reg = 0; reg < vector_regs; ++reg) {
      if (reg != 0) {
        out << ",";
      }
      out << "v" << reg << "[";
      for (uint32_t lane = 0; lane < lanes_per_vector; ++lane) {
        if (lane != 0) {
          out << ",";
        }
        out << lane;
      }
      out << "]={";
      for (uint32_t lane = 0; lane < lanes_per_vector; ++lane) {
        if (lane != 0) {
          out << ",";
        }
        out << HexWaveTraceU64(wave.vgpr.Read(reg, lane));
      }
      out << "}";
    }
  }
  out << "}";
  if (wave.tensor_agpr_count != 0 || wave.tensor_accum_offset != 0) {
    out << " tensor={agpr_count=" << HexWaveTraceU64(wave.tensor_agpr_count)
        << ",accum_offset=" << HexWaveTraceU64(wave.tensor_accum_offset) << "}";
  }
  return out.str();
}

inline std::string FormatWaveLaunchTraceMessage(const WaveContext& wave,
                                                uint32_t scalar_regs = 4,
                                                uint32_t vector_regs = 2,
                                                uint32_t lanes_per_vector = 2) {
  return FormatWaveLaunchTraceMessage(wave, nullptr, scalar_regs, vector_regs, lanes_per_vector);
}

}  // namespace gpu_model
