#pragma once

#include <cstdint>
#include <unordered_map>

#include "runtime/program_cycle_stats.h"

namespace gpu_model {

enum class ExecutedStepClass {
  ScalarAlu,    // SOP1, SOP2, SOPC, SOPK (non-branch, non-sync)
  ScalarMem,    // SMRD, SMEM
  VectorAlu,    // VOP1, VOP2, VOP3, VOPC, VINTRP
  VectorMem,    // FLAT, MUBUF, MTBUF, MIMG, DS (global/shared/private)
  Branch,       // SOPP branch instructions (s_branch, s_cbranch_*, etc.)
  Sync,         // s_barrier, s_waitcnt
  Tensor,       // VOP3P tensor operations
  Other,        // s_endpgm, s_nop, mask instructions, etc.
};

class ProgramCycleTracker {
 public:
  ProgramCycleTracker() = default;

  void BeginWaveWork(uint32_t wave_id,
                     ExecutedStepClass step_class,
                     uint64_t cost_cycles,
                     uint64_t work_weight = 1);
  void MarkWaveWaiting(uint32_t wave_id,
                       ExecutedStepClass wait_class,
                       uint64_t cost_cycles,
                       uint64_t work_weight = 1);
  void MarkWaveRunnable(uint32_t wave_id);
  void MarkWaveCompleted(uint32_t wave_id);
  void MarkWaveLaunched(uint32_t wave_id);
  void AdvanceOneTick();

  bool Done() const;
  ProgramCycleStats Finish() const;

 private:
  enum class WaveLifecycle {
    Runnable,
    Active,
  };

  struct WaveState {
    WaveLifecycle lifecycle = WaveLifecycle::Runnable;
    ExecutedStepClass step_class = ExecutedStepClass::ScalarAlu;
    uint64_t remaining_cycles = 0;
    uint64_t work_weight = 1;
  };

  void AssignWaveWork(uint32_t wave_id,
                      ExecutedStepClass step_class,
                      uint64_t cost_cycles,
                      uint64_t work_weight);

  ProgramCycleStats stats_;
  std::unordered_map<uint32_t, WaveState> waves_;
};

struct ProgramCycleTickSource {
  virtual bool Done() const = 0;
  virtual void AdvanceOneTick(ProgramCycleTracker& agg) = 0;
  virtual ~ProgramCycleTickSource() = default;
};

}  // namespace gpu_model
