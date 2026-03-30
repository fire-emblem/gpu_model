#pragma once

#include <cstdint>

namespace gpu_model {

enum class GcnSpecialReg {
  None,
  Vcc,
  Exec,
};

struct GcnOperandInfo {
  uint32_t reg_first = 0;
  uint32_t reg_count = 0;
  int64_t immediate = 0;
  bool has_immediate = false;
  GcnSpecialReg special_reg = GcnSpecialReg::None;
  uint8_t wait_vmcnt = 0;
  uint8_t wait_expcnt = 0;
  uint8_t wait_lgkmcnt = 0;
  bool has_waitcnt = false;
};

}  // namespace gpu_model
