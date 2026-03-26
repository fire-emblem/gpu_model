#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>

#include "gpu_model/isa/kernel_program.h"

namespace gpu_model {

struct KernelDebugInfo {
  std::string kernel_name;
  std::unordered_map<uint64_t, DebugLoc> pc_to_debug_loc;

  static KernelDebugInfo FromKernel(const KernelProgram& kernel) {
    KernelDebugInfo info;
    info.kernel_name = kernel.name();
    for (uint64_t pc = 0; pc < kernel.instructions().size(); ++pc) {
      info.pc_to_debug_loc.emplace(pc, kernel.instructions()[pc].debug_loc);
    }
    return info;
  }
};

}  // namespace gpu_model
