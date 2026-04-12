#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>

#include "program/executable/executable_kernel.h"

namespace gpu_model {

struct KernelDebugInfo {
  std::string kernel_name;
  std::unordered_map<uint64_t, DebugLoc> pc_to_debug_loc;

  static KernelDebugInfo FromKernel(const ExecutableKernel& kernel) {
    KernelDebugInfo info;
    info.kernel_name = kernel.name();
    for (const auto& [pc, instruction] : kernel.instructions_by_pc()) {
      info.pc_to_debug_loc.emplace(pc, instruction.debug_loc);
    }
    return info;
  }
};

}  // namespace gpu_model
