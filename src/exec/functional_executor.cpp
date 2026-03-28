#include "gpu_model/exec/functional_executor.h"

#include "gpu_model/exec/functional_execution_core.h"

namespace gpu_model {

uint64_t FunctionalExecutor::Run(ExecutionContext& context) {
  FunctionalExecutionCore core(context);
  return core.RunSequential();
}

}  // namespace gpu_model
