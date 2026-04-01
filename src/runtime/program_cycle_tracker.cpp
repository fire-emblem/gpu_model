#include "gpu_model/runtime/program_cycle_tracker.h"

namespace gpu_model {
namespace {

void AccumulateStepCycle(ProgramCycleStats& stats, ExecutedStepClass step_class) {
  ++stats.total_issued_work_cycles;
  switch (step_class) {
    case ExecutedStepClass::ScalarAlu:
      ++stats.scalar_alu_cycles;
      return;
    case ExecutedStepClass::VectorAlu:
      ++stats.vector_alu_cycles;
      return;
    case ExecutedStepClass::Tensor:
      ++stats.tensor_cycles;
      return;
    case ExecutedStepClass::SharedMem:
      ++stats.shared_mem_cycles;
      return;
    case ExecutedStepClass::ScalarMem:
      ++stats.scalar_mem_cycles;
      return;
    case ExecutedStepClass::GlobalMem:
      ++stats.global_mem_cycles;
      return;
    case ExecutedStepClass::PrivateMem:
      ++stats.private_mem_cycles;
      return;
    case ExecutedStepClass::Barrier:
      ++stats.barrier_cycles;
      return;
    case ExecutedStepClass::Wait:
      ++stats.wait_cycles;
      return;
  }
}

}  // namespace

ProgramCycleTracker::ProgramCycleTracker(ProgramCycleStatsConfig config)
    : config_(config) {}

void ProgramCycleTracker::BeginWaveWork(uint32_t wave_id,
                                           ExecutedStepClass step_class,
                                           uint64_t cost_cycles) {
  AssignWaveWork(wave_id, step_class, cost_cycles);
}

void ProgramCycleTracker::MarkWaveWaiting(uint32_t wave_id,
                                             ExecutedStepClass wait_class,
                                             uint64_t cost_cycles) {
  AssignWaveWork(wave_id, wait_class, cost_cycles);
}

void ProgramCycleTracker::MarkWaveRunnable(uint32_t wave_id) {
  auto& wave = waves_[wave_id];
  wave.lifecycle = WaveLifecycle::Runnable;
  wave.remaining_cycles = 0;
}

void ProgramCycleTracker::MarkWaveCompleted(uint32_t wave_id) {
  waves_.erase(wave_id);
}

void ProgramCycleTracker::AdvanceOneTick() {
  if (Done()) {
    return;
  }

  ++stats_.total_cycles;
  for (auto& [wave_id, wave] : waves_) {
    (void)wave_id;
    if (wave.lifecycle != WaveLifecycle::Active || wave.remaining_cycles == 0) {
      continue;
    }
    AccumulateStepCycle(stats_, wave.step_class);
    --wave.remaining_cycles;
    if (wave.remaining_cycles == 0) {
      wave.lifecycle = WaveLifecycle::Runnable;
    }
  }
}

bool ProgramCycleTracker::Done() const {
  return waves_.empty();
}

ProgramCycleStats ProgramCycleTracker::Finish() const {
  return stats_;
}

void ProgramCycleTracker::AssignWaveWork(uint32_t wave_id,
                                            ExecutedStepClass step_class,
                                            uint64_t cost_cycles) {
  if (cost_cycles == 0) {
    MarkWaveRunnable(wave_id);
    return;
  }

  auto& wave = waves_[wave_id];
  wave.lifecycle = WaveLifecycle::Active;
  wave.step_class = step_class;
  wave.remaining_cycles = cost_cycles;
}

}  // namespace gpu_model
