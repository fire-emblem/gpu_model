#include "gpu_model/exec/functional_executor.h"

#include "gpu_model/execution/functional_exec_engine.h"

namespace gpu_model {

uint64_t FunctionalExecutor::Run(ExecutionContext& context) {
  FunctionalExecEngine core(context);
  return core.RunSequential();
}

}  // namespace gpu_model
