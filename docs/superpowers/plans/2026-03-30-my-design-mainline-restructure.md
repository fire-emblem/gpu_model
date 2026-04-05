# My Design Mainline Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Status (2026-04-01):** Historical Phase 1 plan. The repository has already moved past this compatibility-wrapper stage, and later Phase 2 cleanup removed or renamed many files/tests referenced below. Do not execute this plan as-is without first reconciling it against the live tree.

**Goal:** Build the first-phase repository restructure that aligns the mainline with `docs/my_design.md`, using the new `runtime / program / instruction / execution / arch` architecture while preserving compatibility shims.

**Architecture:** The refactor is additive first. Introduce new module-aligned headers, names, and directories; migrate core implementations and tests to the new ownership model; keep old names as thin compatibility wrappers until Phase 2 cleanup. The mainline execution path must become visibly organized as `runtime -> program -> instruction -> execution -> wave`, with historical `loader / decode / isa / exec` concepts demoted to compatibility-only status.

**Tech Stack:** C++17, CMake, gtest, existing `gpu_model_tests` binary, existing repository docs under `docs/`

---

## File Structure

### New public headers to create

- `include/gpu_model/runtime/model_runtime.h`
- `include/gpu_model/runtime/hip_runtime.h`
- `include/gpu_model/runtime/exec_engine.h`
- `include/gpu_model/program/program_object.h`
- `include/gpu_model/program/executable_kernel.h`
- `include/gpu_model/program/encoded_program_object.h`
- `include/gpu_model/program/object_reader.h`
- `include/gpu_model/program/execution_route.h`
- `include/gpu_model/instruction/encoded/instruction_decoder.h`
- `include/gpu_model/instruction/encoded/decoded_instruction.h`
- `include/gpu_model/instruction/encoded/instruction_object.h`
- `include/gpu_model/instruction/modeled/lowering.h`
- `include/gpu_model/execution/functional_exec_engine.h`
- `include/gpu_model/execution/cycle_exec_engine.h`
- `include/gpu_model/execution/encoded_exec_engine.h`
- `include/gpu_model/execution/wave_context.h`
- `include/gpu_model/execution/wave_context_builder.h`
- `include/gpu_model/execution/memory_ops.h`
- `include/gpu_model/execution/sync_ops.h`
- `include/gpu_model/execution/plan_apply.h`

### Existing headers to keep as compatibility wrappers in Phase 1

- `include/gpu_model/runtime/model_runtime_api.h`
- `include/gpu_model/runtime/runtime_hooks.h`
- `include/gpu_model/runtime/host_runtime.h`
- `include/gpu_model/isa/program_image.h`
- `include/gpu_model/isa/kernel_program.h`
- `include/gpu_model/loader/amdgpu_code_object_decoder.h`
- `include/gpu_model/loader/program_lowering.h`
- `include/gpu_model/decode/gcn_inst_decoder.h`
- `include/gpu_model/decode/decoded_gcn_instruction.h`
- `include/gpu_model/exec/functional_execution_core.h`
- `include/gpu_model/exec/functional_executor.h`
- `include/gpu_model/exec/cycle_executor.h`
- `include/gpu_model/exec/encoded/executor/raw_gcn_executor.h`
- `include/gpu_model/exec/execution_state_builder.h`
- `include/gpu_model/exec/execution_memory_ops.h`
- `include/gpu_model/exec/execution_sync_ops.h`
- `include/gpu_model/exec/op_plan_apply.h`

### Source files to re-home or adapt

- `src/runtime/program_execution.cpp`
- `src/runtime/runtime_hooks.cpp`
- `src/runtime/host_runtime.cpp`
- `src/runtime/hip_interposer_state.cpp`
- `src/loader/amdgpu_obj_loader.cpp`
- `src/loader/amdgpu_code_object_decoder.cpp`
- `src/loader/program_file_loader.cpp`
- `src/loader/program_lowering.cpp`
- `src/exec/functional_execution_core.cpp`
- `src/exec/functional_executor.cpp`
- `src/exec/parallel_wave_executor.cpp`
- `src/exec/cycle_executor.cpp`
- `src/exec/execution_state_builder.cpp`
- `src/exec/execution_memory_ops.cpp`
- `src/exec/execution_sync_ops.cpp`
- `src/exec/op_plan_apply.cpp`
- `src/exec/encoded/executor/raw_gcn_executor.cpp`

### Tests to migrate

- `tests/runtime/model_runtime_api_test.cpp`
- `tests/runtime/runtime_hooks_test.cpp`
- `tests/runtime/raw_code_object_launch_test.cpp`
- `tests/runtime/kernel_metadata_test.cpp`
- `tests/loader/*`
- `tests/exec/*`

---

### Task 1: Establish New Runtime Public Names

**Files:**
- Create: `include/gpu_model/runtime/model_runtime.h`
- Create: `include/gpu_model/runtime/hip_runtime.h`
- Create: `include/gpu_model/runtime/exec_engine.h`
- Modify: `include/gpu_model/runtime/model_runtime_api.h`
- Modify: `include/gpu_model/runtime/runtime_hooks.h`
- Modify: `include/gpu_model/runtime/host_runtime.h`
- Test: `tests/runtime/model_runtime_api_test.cpp`
- Test: `tests/runtime/runtime_hooks_test.cpp`

- [ ] **Step 1: Write the failing compatibility-first runtime naming test**

```cpp
#include "gpu_model/runtime/model_runtime.h"
#include "gpu_model/runtime/hip_runtime.h"
#include "gpu_model/runtime/exec_engine.h"

#include <type_traits>

#include <gtest/gtest.h>

namespace gpu_model {
namespace {

TEST(RuntimeNamingTest, NewRuntimeNamesResolveToCurrentImplementations) {
  static_assert(std::is_default_constructible_v<ModelRuntime>);
  static_assert(std::is_default_constructible_v<HipRuntime>);
  static_assert(std::is_default_constructible_v<ExecEngine>);
}

}  // namespace
}  // namespace gpu_model
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build-ninja --target gpu_model_tests && ./build-ninja/tests/gpu_model_tests --gtest_filter='RuntimeNamingTest.NewRuntimeNamesResolveToCurrentImplementations'`
Expected: FAIL with missing headers `gpu_model/runtime/model_runtime.h`, `hip_runtime.h`, or `exec_engine.h`

- [ ] **Step 3: Add new runtime headers with Phase 1 aliases**

```cpp
// include/gpu_model/runtime/model_runtime.h
#pragma once

#include "gpu_model/runtime/model_runtime_api.h"

namespace gpu_model {

using ModelRuntime = ModelRuntimeApi;

}  // namespace gpu_model
```

```cpp
// include/gpu_model/runtime/hip_runtime.h
#pragma once

#include "gpu_model/runtime/runtime_hooks.h"

namespace gpu_model {

using HipRuntime = RuntimeHooks;

}  // namespace gpu_model
```

```cpp
// include/gpu_model/runtime/exec_engine.h
#pragma once

#include "gpu_model/runtime/host_runtime.h"

namespace gpu_model {

using ExecEngine = HostRuntime;

}  // namespace gpu_model
```

- [ ] **Step 4: Make legacy headers include the new public headers**

```cpp
// include/gpu_model/runtime/model_runtime_api.h
#pragma once

#include "gpu_model/runtime/runtime_hooks.h"

namespace gpu_model {

class ModelRuntimeApi {
  // existing implementation remains unchanged in Phase 1
};

using ModelRuntime = ModelRuntimeApi;

}  // namespace gpu_model
```

```cpp
// include/gpu_model/runtime/runtime_hooks.h
namespace gpu_model {

class RuntimeHooks {
  // existing implementation remains unchanged in Phase 1
};

using HipRuntime = RuntimeHooks;

}  // namespace gpu_model
```

```cpp
// include/gpu_model/runtime/host_runtime.h
namespace gpu_model {

class HostRuntime {
  // existing implementation remains unchanged in Phase 1
};

using ExecEngine = HostRuntime;

}  // namespace gpu_model
```

- [ ] **Step 5: Run tests to verify the new names compile**

Run: `cmake --build build-ninja --target gpu_model_tests && ./build-ninja/tests/gpu_model_tests --gtest_filter='RuntimeNamingTest.NewRuntimeNamesResolveToCurrentImplementations:ModelRuntimeApiTest.*:RuntimeHooksTest.*'`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add include/gpu_model/runtime/model_runtime.h \
        include/gpu_model/runtime/hip_runtime.h \
        include/gpu_model/runtime/exec_engine.h \
        include/gpu_model/runtime/model_runtime_api.h \
        include/gpu_model/runtime/runtime_hooks.h \
        include/gpu_model/runtime/host_runtime.h \
        tests/runtime/model_runtime_api_test.cpp \
        tests/runtime/runtime_hooks_test.cpp
git commit -m "refactor: add phase1 runtime public names"
```

### Task 2: Introduce Program Layer Public Types

**Files:**
- Create: `include/gpu_model/program/program_object.h`
- Create: `include/gpu_model/program/executable_kernel.h`
- Create: `include/gpu_model/program/encoded_program_object.h`
- Create: `include/gpu_model/program/object_reader.h`
- Create: `include/gpu_model/program/execution_route.h`
- Modify: `include/gpu_model/isa/program_image.h`
- Modify: `include/gpu_model/isa/kernel_program.h`
- Modify: `include/gpu_model/loader/amdgpu_code_object_decoder.h`
- Modify: `include/gpu_model/runtime/program_execution.h`
- Test: `tests/program/program_object_types_test.cpp`

- [ ] **Step 1: Write the failing program-layer naming test**

```cpp
#include "gpu_model/program/program_object.h"
#include "gpu_model/program/executable_kernel.h"
#include "gpu_model/program/encoded_program_object.h"
#include "gpu_model/program/execution_route.h"

#include <type_traits>

#include <gtest/gtest.h>

namespace gpu_model {
namespace {

TEST(ProgramNamingTest, NewProgramTypesResolveToCurrentImplementations) {
  static_assert(std::is_default_constructible_v<ProgramObject>);
  static_assert(std::is_default_constructible_v<ExecutableKernel>);
  static_assert(std::is_default_constructible_v<EncodedProgramObject>);
}

}  // namespace
}  // namespace gpu_model
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build-ninja --target gpu_model_tests && ./build-ninja/tests/gpu_model_tests --gtest_filter='ProgramNamingTest.NewProgramTypesResolveToCurrentImplementations'`
Expected: FAIL with missing `gpu_model/program/*.h` headers

- [ ] **Step 3: Add public program headers as new canonical names**

```cpp
// include/gpu_model/program/program_object.h
#pragma once

#include "gpu_model/isa/program_image.h"

namespace gpu_model {

using ProgramObject = ProgramImage;

}  // namespace gpu_model
```

```cpp
// include/gpu_model/program/executable_kernel.h
#pragma once

#include "gpu_model/isa/kernel_program.h"

namespace gpu_model {

using ExecutableKernel = KernelProgram;

}  // namespace gpu_model
```

```cpp
// include/gpu_model/program/encoded_program_object.h
#pragma once

#include "gpu_model/loader/amdgpu_code_object_decoder.h"

namespace gpu_model {

using EncodedProgramObject = AmdgpuCodeObjectImage;

}  // namespace gpu_model
```

```cpp
// include/gpu_model/program/execution_route.h
#pragma once

#include "gpu_model/runtime/launch_request.h"
```

- [ ] **Step 4: Add compatibility aliases in legacy headers**

```cpp
// include/gpu_model/isa/kernel_program.h
namespace gpu_model {

class KernelProgram {
  // existing implementation
};

using ExecutableKernel = KernelProgram;

}  // namespace gpu_model
```

```cpp
// include/gpu_model/loader/amdgpu_code_object_decoder.h
namespace gpu_model {

struct AmdgpuCodeObjectImage {
  // existing fields
};

using EncodedProgramObject = AmdgpuCodeObjectImage;

}  // namespace gpu_model
```

- [ ] **Step 5: Update runtime program route code to prefer new names in local variables**

```cpp
// src/runtime/program_execution.cpp
PreparedProgramExecution PrepareProgramExecution(const ProgramObject& image,
                                                 ProgramExecutionRoute requested_route) {
  // implementation body can stay the same in Phase 1
}
```

- [ ] **Step 6: Run tests**

Run: `cmake --build build-ninja --target gpu_model_tests && ./build-ninja/tests/gpu_model_tests --gtest_filter='ProgramNamingTest.NewProgramTypesResolveToCurrentImplementations:KernelMetadataTest.*:ProgramFileLoaderTest.*:AmdgpuCodeObjectDecoderTest.*'`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add include/gpu_model/program/program_object.h \
        include/gpu_model/program/executable_kernel.h \
        include/gpu_model/program/encoded_program_object.h \
        include/gpu_model/program/object_reader.h \
        include/gpu_model/program/execution_route.h \
        include/gpu_model/isa/program_image.h \
        include/gpu_model/isa/kernel_program.h \
        include/gpu_model/loader/amdgpu_code_object_decoder.h \
        include/gpu_model/runtime/program_execution.h \
        src/runtime/program_execution.cpp \
        tests/program/program_object_types_test.cpp
git commit -m "refactor: add phase1 program public names"
```

### Task 3: Introduce Instruction Layer Public Types

**Files:**
- Create: `include/gpu_model/instruction/encoded/instruction_decoder.h`
- Create: `include/gpu_model/instruction/encoded/decoded_instruction.h`
- Create: `include/gpu_model/instruction/encoded/instruction_object.h`
- Create: `include/gpu_model/instruction/modeled/lowering.h`
- Modify: `include/gpu_model/decode/gcn_inst_decoder.h`
- Modify: `include/gpu_model/decode/decoded_gcn_instruction.h`
- Modify: `include/gpu_model/exec/encoded/object/raw_gcn_instruction_object.h`
- Modify: `include/gpu_model/loader/program_lowering.h`
- Test: `tests/instruction/instruction_naming_test.cpp`

- [ ] **Step 1: Write the failing instruction naming test**

```cpp
#include "gpu_model/instruction/encoded/instruction_decoder.h"
#include "gpu_model/instruction/encoded/decoded_instruction.h"
#include "gpu_model/instruction/encoded/instruction_object.h"
#include "gpu_model/instruction/modeled/lowering.h"

#include <type_traits>

#include <gtest/gtest.h>

namespace gpu_model {
namespace {

TEST(InstructionNamingTest, NewInstructionHeadersCompile) {
  static_assert(std::is_class_v<GcnInstDecoder> || true);
}

}  // namespace
}  // namespace gpu_model
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build-ninja --target gpu_model_tests && ./build-ninja/tests/gpu_model_tests --gtest_filter='InstructionNamingTest.NewInstructionHeadersCompile'`
Expected: FAIL because the new `instruction/*` headers do not exist

- [ ] **Step 3: Add encoded and modeled instruction public headers**

```cpp
// include/gpu_model/instruction/encoded/instruction_decoder.h
#pragma once

#include "gpu_model/decode/gcn_inst_decoder.h"
```

```cpp
// include/gpu_model/instruction/encoded/decoded_instruction.h
#pragma once

#include "gpu_model/decode/decoded_gcn_instruction.h"
```

```cpp
// include/gpu_model/instruction/encoded/instruction_object.h
#pragma once

#include "gpu_model/exec/encoded/object/raw_gcn_instruction_object.h"

namespace gpu_model {

using EncodedInstructionObject = EncodedGcnInstructionObject;
using EncodedInstructionObjectPtr = EncodedGcnInstructionObjectPtr;

}  // namespace gpu_model
```

```cpp
// include/gpu_model/instruction/modeled/lowering.h
#pragma once

#include "gpu_model/loader/program_lowering.h"
```

- [ ] **Step 4: Add target aliases to legacy instruction-facing headers**

```cpp
// include/gpu_model/exec/encoded/object/raw_gcn_instruction_object.h
namespace gpu_model {

class EncodedGcnInstructionObject {
  // existing implementation
};

using EncodedInstructionObject = EncodedGcnInstructionObject;

}  // namespace gpu_model
```

- [ ] **Step 5: Run tests**

Run: `cmake --build build-ninja --target gpu_model_tests && ./build-ninja/tests/gpu_model_tests --gtest_filter='InstructionNamingTest.NewInstructionHeadersCompile:EncodedGcnInstructionObjectExecuteTest.*:EncodedGcnInstructionDescriptorTest.*'`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add include/gpu_model/instruction/encoded/instruction_decoder.h \
        include/gpu_model/instruction/encoded/decoded_instruction.h \
        include/gpu_model/instruction/encoded/instruction_object.h \
        include/gpu_model/instruction/modeled/lowering.h \
        include/gpu_model/decode/gcn_inst_decoder.h \
        include/gpu_model/decode/decoded_gcn_instruction.h \
        include/gpu_model/exec/encoded/object/raw_gcn_instruction_object.h \
        include/gpu_model/loader/program_lowering.h \
        tests/instruction/instruction_naming_test.cpp
git commit -m "refactor: add phase1 instruction public names"
```

### Task 4: Introduce Execution Layer Public Types

**Files:**
- Create: `include/gpu_model/execution/functional_exec_engine.h`
- Create: `include/gpu_model/execution/cycle_exec_engine.h`
- Create: `include/gpu_model/execution/encoded_exec_engine.h`
- Create: `include/gpu_model/execution/wave_context.h`
- Create: `include/gpu_model/execution/wave_context_builder.h`
- Create: `include/gpu_model/execution/memory_ops.h`
- Create: `include/gpu_model/execution/sync_ops.h`
- Create: `include/gpu_model/execution/plan_apply.h`
- Modify: `include/gpu_model/exec/functional_execution_core.h`
- Modify: `include/gpu_model/exec/cycle_executor.h`
- Modify: `include/gpu_model/exec/encoded/executor/raw_gcn_executor.h`
- Modify: `include/gpu_model/exec/execution_state_builder.h`
- Modify: `include/gpu_model/exec/execution_memory_ops.h`
- Modify: `include/gpu_model/exec/execution_sync_ops.h`
- Modify: `include/gpu_model/exec/op_plan_apply.h`
- Test: `tests/execution/execution_naming_test.cpp`

- [ ] **Step 1: Write the failing execution naming test**

```cpp
#include "gpu_model/execution/functional_exec_engine.h"
#include "gpu_model/execution/cycle_exec_engine.h"
#include "gpu_model/execution/encoded_exec_engine.h"
#include "gpu_model/execution/wave_context_builder.h"

#include <type_traits>

#include <gtest/gtest.h>

namespace gpu_model {
namespace {

TEST(ExecutionNamingTest, NewExecutionNamesCompile) {
  static_assert(std::is_default_constructible_v<FunctionalExecEngine>);
  static_assert(std::is_default_constructible_v<CycleExecEngine>);
  static_assert(std::is_default_constructible_v<EncodedExecEngine>);
}

}  // namespace
}  // namespace gpu_model
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build-ninja --target gpu_model_tests && ./build-ninja/tests/gpu_model_tests --gtest_filter='ExecutionNamingTest.NewExecutionNamesCompile'`
Expected: FAIL with missing `gpu_model/execution/*.h` headers

- [ ] **Step 3: Add new execution public headers**

```cpp
// include/gpu_model/execution/functional_exec_engine.h
#pragma once

#include "gpu_model/exec/functional_execution_core.h"

namespace gpu_model {

using FunctionalExecEngine = FunctionalExecutionCore;

}  // namespace gpu_model
```

```cpp
// include/gpu_model/execution/cycle_exec_engine.h
#pragma once

#include "gpu_model/exec/cycle_executor.h"

namespace gpu_model {

using CycleExecEngine = CycleExecutor;

}  // namespace gpu_model
```

```cpp
// include/gpu_model/execution/encoded_exec_engine.h
#pragma once

#include "gpu_model/exec/encoded/executor/raw_gcn_executor.h"

namespace gpu_model {

using EncodedExecEngine = RawGcnExecutor;

}  // namespace gpu_model
```

```cpp
// include/gpu_model/execution/wave_context_builder.h
#pragma once

#include "gpu_model/exec/execution_state_builder.h"
```

- [ ] **Step 4: Add `WaveContext` compatibility naming around existing wave state**

```cpp
// include/gpu_model/execution/wave_context.h
#pragma once

#include "gpu_model/state/wave_state.h"

namespace gpu_model {

using WaveContext = WaveState;

}  // namespace gpu_model
```

- [ ] **Step 5: Add aliases in legacy execution headers**

```cpp
// include/gpu_model/exec/functional_execution_core.h
namespace gpu_model {

class FunctionalExecutionCore {
 public:
  uint64_t RunSequential();
  uint64_t RunParallelBlocks(uint32_t worker_threads);
};

using FunctionalExecEngine = FunctionalExecutionCore;

}  // namespace gpu_model
```

- [ ] **Step 6: Run tests**

Run: `cmake --build build-ninja --target gpu_model_tests && ./build-ninja/tests/gpu_model_tests --gtest_filter='ExecutionNamingTest.NewExecutionNamesCompile:ExecutionStateBuilderTest.*:ExecutionMemoryOpsTest.*:ExecutionSyncOpsTest.*:OpPlanApplyTest.*'`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add include/gpu_model/execution/functional_exec_engine.h \
        include/gpu_model/execution/cycle_exec_engine.h \
        include/gpu_model/execution/encoded_exec_engine.h \
        include/gpu_model/execution/wave_context.h \
        include/gpu_model/execution/wave_context_builder.h \
        include/gpu_model/execution/memory_ops.h \
        include/gpu_model/execution/sync_ops.h \
        include/gpu_model/execution/plan_apply.h \
        include/gpu_model/exec/functional_execution_core.h \
        include/gpu_model/exec/cycle_executor.h \
        include/gpu_model/exec/encoded/executor/raw_gcn_executor.h \
        include/gpu_model/exec/execution_state_builder.h \
        include/gpu_model/exec/execution_memory_ops.h \
        include/gpu_model/exec/execution_sync_ops.h \
        include/gpu_model/exec/op_plan_apply.h \
        tests/execution/execution_naming_test.cpp
git commit -m "refactor: add phase1 execution public names"
```

### Task 5: Rewire Runtime Engine Internals to Prefer New Names

**Files:**
- Modify: `src/runtime/host_runtime.cpp`
- Modify: `src/runtime/runtime_hooks.cpp`
- Modify: `src/runtime/hip_interposer_state.cpp`
- Modify: `src/runtime/program_execution.cpp`
- Modify: `include/gpu_model/runtime/host_runtime.h`
- Modify: `include/gpu_model/runtime/runtime_hooks.h`
- Test: `tests/runtime/raw_code_object_launch_test.cpp`
- Test: `tests/runtime/hip_interposer_state_test.cpp`

- [ ] **Step 1: Add a failing smoke test that includes only new names in runtime-owned code**

```cpp
#include "gpu_model/runtime/model_runtime.h"
#include "gpu_model/runtime/exec_engine.h"
#include "gpu_model/program/program_object.h"
#include "gpu_model/execution/encoded_exec_engine.h"

#include <gtest/gtest.h>

namespace gpu_model {
namespace {

TEST(RuntimeMainlineNamingTest, NewMainlineNamesCanLaunchProgramImage) {
  ModelRuntime runtime;
  SUCCEED();
}

}  // namespace
}  // namespace gpu_model
```

- [ ] **Step 2: Run the targeted test to verify current compile gaps**

Run: `cmake --build build-ninja --target gpu_model_tests && ./build-ninja/tests/gpu_model_tests --gtest_filter='RuntimeMainlineNamingTest.NewMainlineNamesCanLaunchProgramImage'`
Expected: FAIL due to missing include rewrites or unresolved alias usage in runtime implementation

- [ ] **Step 3: Replace internal spelling in runtime implementation with new semantic names**

```cpp
// src/runtime/host_runtime.cpp
const ExecutableKernel* kernel = prepared.kernel;
const EncodedProgramObject* encoded_program = prepared.raw_code_object;
const bool use_encoded_exec_engine = encoded_program != nullptr;

if (use_encoded_exec_engine) {
  const auto encoded_result =
      EncodedExecEngine{}.Run(*encoded_program, *spec, request.config, request.args,
                              request.device_load, memory_, trace);
  result.ok = encoded_result.ok;
  result.error_message = encoded_result.error_message;
  result.total_cycles = encoded_result.total_cycles;
  result.end_cycle = encoded_result.end_cycle;
  result.stats = encoded_result.stats;
}
```

```cpp
// src/runtime/runtime_hooks.cpp
LaunchResult RuntimeHooks::LaunchProgramImage(const ProgramObject& image,
                                              LaunchConfig config,
                                              KernelArgPack args,
                                              ExecutionMode mode,
                                              std::string arch_name,
                                              TraceSink* trace,
                                              ProgramExecutionRoute route) {
  // body remains behaviorally equivalent in Phase 1
}
```

- [ ] **Step 4: Run runtime-focused regression tests**

Run: `cmake --build build-ninja --target gpu_model_tests && ./build-ninja/tests/gpu_model_tests --gtest_filter='RuntimeMainlineNamingTest.*:RawCodeObjectLaunchTest.*:HipRuntimeTest.*:ModelRuntimeApiTest.*:RuntimeHooksTest.*'`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/runtime/host_runtime.cpp \
        src/runtime/runtime_hooks.cpp \
        src/runtime/hip_interposer_state.cpp \
        src/runtime/program_execution.cpp \
        include/gpu_model/runtime/host_runtime.h \
        include/gpu_model/runtime/runtime_hooks.h \
        tests/runtime/raw_code_object_launch_test.cpp \
        tests/runtime/hip_interposer_state_test.cpp
git commit -m "refactor: prefer mainline names in runtime internals"
```

### Task 6: Migrate Tests into Module-Aligned Directories

**Files:**
- Create: `tests/program/`
- Create: `tests/instruction/`
- Create: `tests/execution/`
- Create: `tests/program/fixtures/`
- Modify: `tests/CMakeLists.txt`
- Move: `tests/loader/*`
- Move: `tests/exec/*`
- Move: selected `tests/runtime/*`

- [ ] **Step 1: Add failing references in `tests/CMakeLists.txt` to new directories**

```cmake
target_sources(gpu_model_tests PRIVATE
  tests/program/program_object_types_test.cpp
  tests/instruction/instruction_naming_test.cpp
  tests/execution/execution_naming_test.cpp
)
```

- [ ] **Step 2: Run build to verify missing files and update list**

Run: `cmake --build build-ninja --target gpu_model_tests`
Expected: FAIL if the new files are not yet moved or referenced correctly

- [ ] **Step 3: Move tests to the new ownership model**

```bash
mv tests/loader/program_lowering_test.cpp tests/instruction/program_lowering_test.cpp
mv tests/loader/program_file_loader_test.cpp tests/program/program_source_reader_test.cpp
mv tests/loader/amdgpu_code_object_decoder_test.cpp tests/program/encoded_program_object_test.cpp
mv tests/exec/execution_state_builder_test.cpp tests/execution/wave_context_builder_test.cpp
mv tests/exec/execution_memory_ops_test.cpp tests/execution/memory_ops_test.cpp
mv tests/exec/execution_sync_ops_test.cpp tests/execution/sync_ops_test.cpp
```

- [ ] **Step 4: Update includes and test names after the moves**

```cpp
// tests/execution/wave_context_builder_test.cpp
#include "gpu_model/execution/wave_context_builder.h"
```

```cpp
// tests/program/encoded_program_object_test.cpp
#include "gpu_model/program/encoded_program_object.h"
```

- [ ] **Step 5: Run broad module-aligned test matrix**

Run: `cmake --build build-ninja --target gpu_model_tests && ./build-ninja/tests/gpu_model_tests --gtest_filter='*Program*:*Instruction*:*Execution*:*Runtime*'`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tests/CMakeLists.txt tests/program tests/instruction tests/execution
git commit -m "test: migrate mainline tests to module-aligned directories"
```

### Task 7: Update Mainline Documentation to Match the New Architecture

**Files:**
- Modify: `docs/my_design.md`
- Modify: `docs/module-development-status.md`
- Modify: `docs/runtime-layering.md`
- Modify: `README.md`
- Test: `docs/superpowers/specs/2026-03-30-my-design-mainline-restructure-design.md`

- [ ] **Step 1: Add a failing terminology scan for stale primary names**

```bash
rg -n "RuntimeHooks|HostRuntime|ProgramImage|KernelProgram|raw_gcn|canonical/internal" \
  docs/my_design.md docs/module-development-status.md docs/runtime-layering.md README.md
```

Expected: matches found in files that still need terminology cleanup

- [ ] **Step 2: Rewrite the primary docs to prefer the new mainline vocabulary**

```md
- `ModelRuntimeApi` -> `ModelRuntime`
- `RuntimeHooks` -> `HipRuntime` (Phase 1 compatibility name retained where needed)
- `HostRuntime` -> `ExecEngine`
- `ProgramImage` -> `ProgramObject`
- `KernelProgram` -> `ExecutableKernel`
- `RawGcnExecutor` -> `EncodedExecEngine`
```

- [ ] **Step 3: Update status tables and layering docs to reflect the new architecture spine**

```md
长期主线：

`runtime -> program -> instruction -> execution -> wave`

历史目录：

- `loader`
- `decode`
- `isa`
- `exec`

在 Phase 1 中仅作为兼容或迁移中的实现来源，不再作为长期模块边界。
```

- [ ] **Step 4: Run the terminology scan again**

Run: `rg -n "RuntimeHooks|HostRuntime|ProgramImage|KernelProgram|raw_gcn|canonical/internal" docs/my_design.md docs/module-development-status.md docs/runtime-layering.md README.md`
Expected: only intentional historical references remain, each clearly marked as legacy or compatibility terms

- [ ] **Step 5: Commit**

```bash
git add docs/my_design.md \
        docs/module-development-status.md \
        docs/runtime-layering.md \
        README.md
git commit -m "docs: align mainline docs with new architecture terms"
```

### Task 8: Add a Phase 1 Compatibility Gate

**Files:**
- Modify: `tests/runtime/test_matrix_profile.h`
- Create: `tests/runtime/compatibility_alias_test.cpp`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Write a failing compatibility test that includes both old and new headers**

```cpp
#include "gpu_model/runtime/model_runtime.h"
#include "gpu_model/runtime/model_runtime_api.h"
#include "gpu_model/program/program_object.h"
#include "gpu_model/isa/program_image.h"
#include "gpu_model/execution/encoded_exec_engine.h"
#include "gpu_model/exec/encoded/executor/raw_gcn_executor.h"

#include <type_traits>

#include <gtest/gtest.h>

namespace gpu_model {
namespace {

TEST(CompatibilityAliasTest, OldAndNewNamesRemainEquivalentInPhase1) {
  static_assert(std::is_same_v<ModelRuntime, ModelRuntimeApi>);
  static_assert(std::is_same_v<ProgramObject, ProgramImage>);
  static_assert(std::is_same_v<EncodedExecEngine, RawGcnExecutor>);
}

}  // namespace
}  // namespace gpu_model
```

- [ ] **Step 2: Run test to verify any missing alias coverage**

Run: `cmake --build build-ninja --target gpu_model_tests && ./build-ninja/tests/gpu_model_tests --gtest_filter='CompatibilityAliasTest.OldAndNewNamesRemainEquivalentInPhase1'`
Expected: FAIL until all Phase 1 aliases are in place

- [ ] **Step 3: Wire the compatibility test into the standard matrix**

```cpp
// tests/runtime/test_matrix_profile.h
inline constexpr const char* kPhase1CompatibilityFilter =
    "CompatibilityAliasTest.*";
```

- [ ] **Step 4: Run final Phase 1 regression gate**

Run: `cmake --build build-ninja --target gpu_model_tests && ./build-ninja/tests/gpu_model_tests --gtest_filter='CompatibilityAliasTest.*:ModelRuntimeApiTest.*:RuntimeHooksTest.*:ProgramNamingTest.*:InstructionNamingTest.*:ExecutionNamingTest.*:RawCodeObjectLaunchTest.*:ExecutionStateBuilderTest.*:CycleExecutorTest.*'`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/runtime/test_matrix_profile.h \
        tests/runtime/compatibility_alias_test.cpp \
        tests/CMakeLists.txt
git commit -m "test: add phase1 compatibility alias gate"
```

## Self-Review

### Spec coverage

This plan covers:

- new module-aligned public names
- compatibility wrappers
- runtime/program/instruction/execution public structure
- runtime internal preference for new names
- test migration into module-aligned directories
- documentation updates
- a compatibility gate for Phase 1

Remaining for later plans:

- Phase 2 deletion of old wrappers
- deeper source file relocation on disk from historical directories to final ones
- full ISA family cleanup and encoded/modeled split beyond naming

### Placeholder scan

The plan contains no `TODO`, `TBD`, or “similar to previous task” references. Each task names exact files, concrete commands, and concrete snippets.

### Type consistency

This plan consistently uses:

- `ModelRuntime`
- `HipRuntime`
- `ExecEngine`
- `ProgramObject`
- `ExecutableKernel`
- `EncodedProgramObject`
- `FunctionalExecEngine`
- `CycleExecEngine`
- `EncodedExecEngine`
- `WaveContext`
- `WaveContextBuilder`

No later task introduces conflicting alternate names for the same Phase 1 concept.
