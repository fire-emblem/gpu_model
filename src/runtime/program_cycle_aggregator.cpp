#include "gpu_model/runtime/program_cycle_aggregator.h"

namespace gpu_model {
namespace {

void AccumulateStepCycle(ProgramCycleEstimate& estimate, ExecutedStepClass step_class) {
  ++estimate.total_issued_work_cycles;
  switch (step_class) {
    case ExecutedStepClass::ScalarAlu:
      ++estimate.scalar_alu_cycles;
      return;
    case ExecutedStepClass::VectorAlu:
      ++estimate.vector_alu_cycles;
      return;
    case ExecutedStepClass::Tensor:
      ++estimate.tensor_cycles;
      return;
    case ExecutedStepClass::SharedMem:
      ++estimate.shared_mem_cycles;
      return;
    case ExecutedStepClass::ScalarMem:
      ++estimate.scalar_mem_cycles;
      return;
    case ExecutedStepClass::GlobalMem:
      ++estimate.global_mem_cycles;
      return;
    case ExecutedStepClass::PrivateMem:
      ++estimate.private_mem_cycles;
      return;
    case ExecutedStepClass::Barrier:
      ++estimate.barrier_cycles;
      return;
    case ExecutedStepClass::Wait:
      ++estimate.wait_cycles;
      return;
  }
}

}  // namespace

ProgramCycleAggregator::ProgramCycleAggregator(ProgramCycleEstimatorConfig config)
    : config_(config) {}

void ProgramCycleAggregator::BeginWaveWork(uint32_t wave_id,
                                           ExecutedStepClass step_class,
                                           uint64_t cost_cycles) {
  AssignWaveWork(wave_id, step_class, cost_cycles);
}

void ProgramCycleAggregator::MarkWaveWaiting(uint32_t wave_id,
                                             ExecutedStepClass wait_class,
                                             uint64_t cost_cycles) {
  AssignWaveWork(wave_id, wait_class, cost_cycles);
}

void ProgramCycleAggregator::MarkWaveRunnable(uint32_t wave_id) {
  auto& wave = waves_[wave_id];
  wave.lifecycle = WaveLifecycle::Runnable;
  wave.remaining_cycles = 0;
}

void ProgramCycleAggregator::MarkWaveCompleted(uint32_t wave_id) {
  waves_.erase(wave_id);
}

void ProgramCycleAggregator::AdvanceOneTick() {
  if (Done()) {
    return;
  }

  ++estimate_.total_cycles;
  for (auto& [wave_id, wave] : waves_) {
    (void)wave_id;
    if (wave.lifecycle != WaveLifecycle::Active || wave.remaining_cycles == 0) {
      continue;
    }
    AccumulateStepCycle(estimate_, wave.step_class);
    --wave.remaining_cycles;
    if (wave.remaining_cycles == 0) {
      wave.lifecycle = WaveLifecycle::Runnable;
    }
  }
}

bool ProgramCycleAggregator::Done() const {
  return waves_.empty();
}

ProgramCycleEstimate ProgramCycleAggregator::Finish() const {
  return estimate_;
}

void ProgramCycleAggregator::AssignWaveWork(uint32_t wave_id,
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
