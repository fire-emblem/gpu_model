#include "gpu_model/exec/parallel_wave_executor.h"

#include <thread>

#include "gpu_model/execution/functional_exec_engine.h"

namespace gpu_model {

namespace {

uint32_t DefaultMarlWorkerThreadCount() {
  const uint32_t cpu_count = std::max(1u, std::thread::hardware_concurrency());
  return std::max(1u, (cpu_count * 2u) / 3u);
}

}  // namespace

uint64_t ParallelWaveExecutor::Run(ExecutionContext& context) {
  FunctionalExecEngine core(context);
  const uint32_t workers =
      config_.worker_threads == 0 ? DefaultMarlWorkerThreadCount() : config_.worker_threads;
  return core.RunParallelBlocks(workers);
}

}  // namespace gpu_model
