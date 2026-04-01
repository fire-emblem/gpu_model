#pragma once

#include <cstdint>
#include <unordered_map>

#include "gpu_model/runtime/program_cycle_stats.h"

namespace gpu_model {

enum class ExecutedStepClass {
  ScalarAlu,
  VectorAlu,
  Tensor,
  SharedMem,
  ScalarMem,
  GlobalMem,
  PrivateMem,
  Barrier,
  Wait,
};

class ProgramCycleTracker {
 public:
  explicit ProgramCycleTracker(ProgramCycleStatsConfig config);

  void BeginWaveWork(uint32_t wave_id, ExecutedStepClass step_class, uint64_t cost_cycles);
  void MarkWaveWaiting(uint32_t wave_id, ExecutedStepClass wait_class, uint64_t cost_cycles);
  void MarkWaveRunnable(uint32_t wave_id);
  void MarkWaveCompleted(uint32_t wave_id);
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
  };

  void AssignWaveWork(uint32_t wave_id, ExecutedStepClass step_class, uint64_t cost_cycles);

  ProgramCycleStatsConfig config_;
  ProgramCycleStats stats_;
  std::unordered_map<uint32_t, WaveState> waves_;
};

struct ProgramCycleTickSource {
  virtual bool Done() const = 0;
  virtual void AdvanceOneTick(ProgramCycleTracker& agg) = 0;
  virtual ~ProgramCycleTickSource() = default;
};

}  // namespace gpu_model
