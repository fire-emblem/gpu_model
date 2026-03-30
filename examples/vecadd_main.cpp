#include <cstdint>
#include <iostream>

#include "gpu_model/isa/instruction_builder.h"
#include "gpu_model/runtime/runtime_engine.h"

namespace gpu_model {

KernelProgram BuildVecAddKernel() {
  InstructionBuilder builder;
  builder.SLoadArg("s0", 0);
  builder.SLoadArg("s1", 1);
  builder.SLoadArg("s2", 2);
  builder.SLoadArg("s3", 3);
  builder.SysGlobalIdX("v0");
  builder.VCmpLtCmask("v0", "s3");
  builder.MaskSaveExec("s10");
  builder.MaskAndExecCmask();
  builder.BIfNoexec("exit");
  builder.MLoadGlobal("v1", "s0", "v0", 4);
  builder.MLoadGlobal("v2", "s1", "v0", 4);
  builder.VAdd("v3", "v1", "v2");
  builder.MStoreGlobal("s2", "v0", "v3", 4);
  builder.Label("exit");
  builder.MaskRestoreExec("s10");
  builder.BExit();
  return builder.Build("vecadd");
}

}  // namespace gpu_model

int main() {
  constexpr uint32_t n = 16;
  gpu_model::RuntimeEngine runtime;

  const uint64_t a_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  const uint64_t b_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));
  const uint64_t c_addr = runtime.memory().AllocateGlobal(n * sizeof(int32_t));

  for (uint32_t i = 0; i < n; ++i) {
    runtime.memory().StoreGlobalValue<int32_t>(a_addr + i * sizeof(int32_t),
                                               static_cast<int32_t>(i));
    runtime.memory().StoreGlobalValue<int32_t>(b_addr + i * sizeof(int32_t),
                                               static_cast<int32_t>(10 + i));
  }

  const auto kernel = gpu_model::BuildVecAddKernel();

  gpu_model::LaunchRequest request;
  request.kernel = &kernel;
  request.config.grid_dim_x = 1;
  request.config.block_dim_x = 64;
  request.args.PushU64(a_addr);
  request.args.PushU64(b_addr);
  request.args.PushU64(c_addr);
  request.args.PushU32(n);

  const auto result = runtime.Launch(request);
  if (!result.ok) {
    std::cerr << "launch failed: " << result.error_message << '\n';
    return 1;
  }

  for (uint32_t i = 0; i < n; ++i) {
    const int32_t value =
        runtime.memory().LoadGlobalValue<int32_t>(c_addr + i * sizeof(int32_t));
    std::cout << "c[" << i << "] = " << value << '\n';
  }

  return 0;
}
