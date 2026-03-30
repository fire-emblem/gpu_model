#include "gpu_model/execution/plan_apply.h"

#include <sstream>

namespace gpu_model {

void ApplyExecutionPlanRegisterWrites(const OpPlan& plan, WaveContext& wave) {
  for (const auto& write : plan.scalar_writes) {
    wave.sgpr.Write(write.reg_index, write.value);
  }
  for (const auto& write : plan.vector_writes) {
    for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
      if (write.mask.test(lane)) {
        wave.vgpr.Write(write.reg_index, lane, write.values[lane]);
      }
    }
  }
  if (plan.cmask_write.has_value()) {
    wave.cmask = *plan.cmask_write;
  }
  if (plan.smask_write.has_value()) {
    wave.smask = *plan.smask_write;
  }
  if (plan.exec_write.has_value()) {
    wave.exec = *plan.exec_write;
  }
}

std::optional<std::string> MaybeFormatExecutionMaskUpdate(const OpPlan& plan,
                                                          const WaveContext& wave) {
  if (!plan.exec_write.has_value()) {
    return std::nullopt;
  }
  std::ostringstream mask_text;
  mask_text << wave.exec;
  return mask_text.str();
}

void ApplyExecutionPlanControlFlow(const OpPlan& plan,
                                   WaveContext& wave,
                                   bool set_valid_entry,
                                   bool clear_branch_pending) {
  if (plan.exit_wave) {
    wave.status = WaveStatus::Exited;
    return;
  }
  if (plan.branch_target.has_value()) {
    wave.pc = *plan.branch_target;
  } else if (plan.advance_pc) {
    ++wave.pc;
  }
  if (clear_branch_pending) {
    wave.branch_pending = false;
  }
  if (set_valid_entry) {
    wave.valid_entry = true;
  }
}

}  // namespace gpu_model
