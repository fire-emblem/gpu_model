#pragma once

#include <bitset>
#include <sstream>
#include <string>

#include "gpu_model/execution/wave_context.h"

namespace gpu_model {

inline std::string HexWaveTraceU64(uint64_t value) {
  std::ostringstream out;
  out << "0x" << std::hex << std::nouppercase << value;
  return out.str();
}

inline std::string FormatWaveLaunchTraceMessage(const WaveContext& wave,
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
  for (uint32_t i = 0; i < scalar_regs; ++i) {
    if (i != 0) {
      out << ",";
    }
    out << "s" << i << "=" << HexWaveTraceU64(wave.sgpr.Read(i));
  }
  out << "} vgpr={";
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
  out << "}";
  if (wave.tensor_agpr_count != 0 || wave.tensor_accum_offset != 0) {
    out << " tensor={agpr_count=" << HexWaveTraceU64(wave.tensor_agpr_count)
        << ",accum_offset=" << HexWaveTraceU64(wave.tensor_accum_offset) << "}";
  }
  return out.str();
}

}  // namespace gpu_model
