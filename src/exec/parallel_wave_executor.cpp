#include "gpu_model/exec/parallel_wave_executor.h"

#include <exception>

#ifdef GPU_MODEL_HAS_MARL
#include "marl/scheduler.h"
#include "marl/waitgroup.h"
#endif

namespace gpu_model {

uint64_t ParallelWaveExecutor::Run(ExecutionContext& context) {
#ifdef GPU_MODEL_HAS_MARL
  marl::Scheduler::Config scheduler_config;
  if (config_.worker_threads == 0) {
    scheduler_config = marl::Scheduler::Config::allCores();
  } else {
    scheduler_config.setWorkerThreadCount(static_cast<int>(config_.worker_threads));
  }

  marl::Scheduler scheduler(scheduler_config);
  scheduler.bind();

  marl::WaitGroup done(1);
  uint64_t total_cycles = 0;
  std::exception_ptr error;
  marl::schedule([&] {
    try {
      FunctionalExecutor executor;
      total_cycles = executor.Run(context);
    } catch (...) {
      error = std::current_exception();
    }
    done.done();
  });
  done.wait();
  marl::Scheduler::unbind();
  if (error != nullptr) {
    std::rethrow_exception(error);
  }
  return total_cycles;
#else
  (void)config_;
  FunctionalExecutor executor;
  return executor.Run(context);
#endif
}

}  // namespace gpu_model
