#include "gpu_model/runtime/program_cycle_tracker.h"

#include <cstdio>

namespace gpu_model {
namespace {

void AccumulateStepCycle(ProgramCycleStats& stats,
                         ExecutedStepClass step_class,
                         uint64_t work_weight) {
  stats.total_issued_work_cycles += work_weight;
  switch (step_class) {
    case ExecutedStepClass::ScalarAlu:
      stats.scalar_alu_cycles += work_weight;
      return;
    case ExecutedStepClass::VectorAlu:
      stats.vector_alu_cycles += work_weight;
      return;
    case ExecutedStepClass::Tensor:
      stats.tensor_cycles += work_weight;
      return;
    case ExecutedStepClass::SharedMem:
      stats.shared_mem_cycles += work_weight;
      return;
    case ExecutedStepClass::ScalarMem:
      stats.scalar_mem_cycles += work_weight;
      return;
    case ExecutedStepClass::GlobalMem:
      stats.global_mem_cycles += work_weight;
      return;
    case ExecutedStepClass::PrivateMem:
      stats.private_mem_cycles += work_weight;
      return;
    case ExecutedStepClass::Barrier:
      stats.barrier_cycles += work_weight;
      return;
    case ExecutedStepClass::Wait:
      stats.wait_cycles += work_weight;
      return;
  }
}

}  // namespace

void ProgramCycleTracker::BeginWaveWork(uint32_t wave_id,
                                        ExecutedStepClass step_class,
                                        uint64_t cost_cycles,
                                        uint64_t work_weight) {
  AssignWaveWork(wave_id, step_class, cost_cycles, work_weight);
  // Count instructions when work begins
  ++stats_.instructions_executed;
  switch (step_class) {
    case ExecutedStepClass::ScalarAlu:
      ++stats_.scalar_alu_insts;
      break;
    case ExecutedStepClass::VectorAlu:
      ++stats_.vector_alu_insts;
      break;
    case ExecutedStepClass::Tensor:
      ++stats_.tensor_insts;
      break;
    case ExecutedStepClass::Barrier:
      ++stats_.barrier_insts;
      break;
    case ExecutedStepClass::GlobalMem:
      // Memory instructions counted separately via loads/stores
      break;
    case ExecutedStepClass::SharedMem:
    case ExecutedStepClass::ScalarMem:
    case ExecutedStepClass::PrivateMem:
    case ExecutedStepClass::Wait:
      break;
  }
}

void ProgramCycleTracker::MarkWaveWaiting(uint32_t wave_id,
                                          ExecutedStepClass wait_class,
                                          uint64_t cost_cycles,
                                          uint64_t work_weight) {
  AssignWaveWork(wave_id, wait_class, cost_cycles, work_weight);
}

void ProgramCycleTracker::MarkWaveRunnable(uint32_t wave_id) {
  const auto it = waves_.find(wave_id);
  if (it == waves_.end()) {
    return;
  }
  auto& wave = it->second;
  wave.lifecycle = WaveLifecycle::Runnable;
  wave.remaining_cycles = 0;
  wave.work_weight = 1;
}

void ProgramCycleTracker::MarkWaveCompleted(uint32_t wave_id) {
  const auto it = waves_.find(wave_id);
  if (it != waves_.end()) {
    ++stats_.waves_completed;
    waves_.erase(it);
  }
}

void ProgramCycleTracker::MarkWaveLaunched(uint32_t wave_id) {
  // Ensure wave exists in tracking
  (void)waves_[wave_id];
  ++stats_.waves_launched;
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
    AccumulateStepCycle(stats_, wave.step_class, wave.work_weight);
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
                                         uint64_t cost_cycles,
                                         uint64_t work_weight) {
  if (cost_cycles == 0) {
    const auto it = waves_.find(wave_id);
    if (it != waves_.end()) {
      MarkWaveRunnable(wave_id);
    }
    return;
  }

  auto& wave = waves_[wave_id];
  wave.lifecycle = WaveLifecycle::Active;
  wave.step_class = step_class;
  wave.remaining_cycles = cost_cycles;
  wave.work_weight = work_weight;
}

}  // namespace gpu_model
