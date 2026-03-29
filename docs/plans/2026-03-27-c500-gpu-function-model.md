# C500 GPU Function Model Implementation Plan

> **For Claude:** Use `${SUPERPOWERS_SKILLS_ROOT}/skills/collaboration/executing-plans/SKILL.md` to implement this plan task-by-task.

**Goal:** Build a C++20 GPU functional model for the `c500` architecture with a custom AMD-style instruction IR, functional execution, trace/debug support, core integration tests, and a naive cycle model that reuses the same instruction semantics.

**Architecture:** The codebase is split into `arch`, `isa`, `state`, `memory`, `runtime`, `exec`, and `debug` layers. `KernelProgram` is the single canonical kernel representation; `FunctionalExecutor` and `CycleExecutor` both consume the same `Instruction` stream through a shared `Semantics -> OpPlan` interface. `c500` is described only by `GpuArchSpec`, so future GPUs are added by registering a new spec instead of forking execution logic.

**Tech Stack:** C++20, CMake, GoogleTest, STL only

---

## Phase Scope

- `V1.0` in this plan:
  - project scaffold
  - `c500` architecture spec and registry
  - custom instruction IR
  - runtime state and mapper
  - functional execution
  - trace + debug info
  - tests for multi-block multi-thread `vecadd`, `if` with `EXEC/CMASK/SMASK`, and placement
- `V2.0` in this plan:
  - `OpPlan`, `Scoreboard`, `EventQueue`
  - naive cycle executor
  - fixed `4 cycle` issue for normal instructions
  - `M_*` as `4 cycle issue + async arrive`
  - smoke tests for cycle ordering
- Out of scope for this plan:
  - assembler parser
  - binary loader
  - CUDA frontend
  - shared/private memory execution correctness
  - barrier/sync execution correctness
  - cache timing
  - MMA semantics

## Repository Layout To Create

```text
/data/gpu_model/
  CMakeLists.txt
  cmake/
    warnings.cmake
  include/gpu_model/
    arch/
      gpu_arch_spec.h
      arch_registry.h
    isa/
      opcode.h
      operand.h
      instruction.h
      instruction_builder.h
      kernel_program.h
    state/
      register_file.h
      wave_state.h
      peu_state.h
      ap_state.h
      dpc_state.h
      gpu_state.h
    memory/
      memory_space.h
      memory_system.h
      memory_request.h
    runtime/
      launch_config.h
      kernel_arg_pack.h
      launch_request.h
      mapper.h
      host_runtime.h
    exec/
      op_plan.h
      semantics.h
      execution_engine.h
      functional_executor.h
      scoreboard.h
      event_queue.h
      cycle_executor.h
    debug/
      debug_info.h
      trace_event.h
      trace_sink.h
  src/
    arch/
      arch_registry.cpp
      c500_spec.cpp
    isa/
      instruction_builder.cpp
      kernel_program.cpp
    memory/
      memory_system.cpp
    runtime/
      mapper.cpp
      host_runtime.cpp
    exec/
      semantics.cpp
      functional_executor.cpp
      scoreboard.cpp
      event_queue.cpp
      cycle_executor.cpp
    debug/
      trace_sink.cpp
  tests/
    arch/
      arch_registry_test.cpp
    functional/
      vecadd_functional_test.cpp
      predicated_if_functional_test.cpp
      mapper_test.cpp
    cycle/
      cycle_smoke_test.cpp
      async_memory_cycle_test.cpp
```

## Build And Test Conventions

- Configure: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug`
- Build: `cmake --build build`
- Run all tests: `./build/tests/gpu_model_tests`
- Run one test binary: `./build/tests/gpu_model_tests --gtest_filter=Suite.Name`

## Task 1: Bootstrap The CMake Project

**Files:**
- Create: `/data/gpu_model/CMakeLists.txt`
- Create: `/data/gpu_model/cmake/warnings.cmake`
- Create: `/data/gpu_model/tests/CMakeLists.txt`
- Create: `/data/gpu_model/tests/arch/arch_registry_test.cpp`

**Step 1: Write the failing test**

Create a minimal test that includes the future registry header and expects `c500` to be present:

```cpp
#include <gtest/gtest.h>
#include "gpu_model/arch/arch_registry.h"

TEST(ArchRegistryTest, C500SpecExists) {
  auto spec = gpu_model::ArchRegistry::Get("c500");
  EXPECT_NE(spec, nullptr);
}
```

**Step 2: Run test to verify it fails**

Run:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```

Expected: configure or build fails because the library targets and headers do not exist yet.

**Step 3: Write minimal implementation**

Create the root `CMakeLists.txt` with one static library and one test executable:

```cmake
cmake_minimum_required(VERSION 3.25)
project(gpu_model LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
)
FetchContent_MakeAvailable(googletest)


add_library(gpu_model STATIC)
target_include_directories(gpu_model PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_subdirectory(tests)
```

Create `tests/CMakeLists.txt`:

```cmake
add_executable(gpu_model_tests
  arch/arch_registry_test.cpp
)
target_link_libraries(gpu_model_tests PRIVATE gpu_model GTest::gtest_main)
```

**Step 4: Run test to verify it now fails for the right reason**

Run:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```

Expected: build fails on missing `gpu_model/arch/arch_registry.h`, which is the next intended failure.

**Step 5: Commit**

```bash
git add CMakeLists.txt cmake/warnings.cmake tests/CMakeLists.txt tests/arch/arch_registry_test.cpp
git commit -m "build: bootstrap cmake and gtest harness"
```

## Task 2: Add `GpuArchSpec` And Register `c500`

**Files:**
- Create: `/data/gpu_model/include/gpu_model/arch/gpu_arch_spec.h`
- Create: `/data/gpu_model/include/gpu_model/arch/arch_registry.h`
- Create: `/data/gpu_model/src/arch/arch_registry.cpp`
- Create: `/data/gpu_model/src/arch/c500_spec.cpp`
- Modify: `/data/gpu_model/CMakeLists.txt`
- Test: `/data/gpu_model/tests/arch/arch_registry_test.cpp`

**Step 1: Write the failing test**

Expand the registry test:

```cpp
TEST(ArchRegistryTest, C500FieldsMatchDesign) {
  auto spec = gpu_model::ArchRegistry::Get("c500");
  ASSERT_NE(spec, nullptr);
  EXPECT_EQ(spec->wave_size, 64u);
  EXPECT_EQ(spec->dpc_count, 8u);
  EXPECT_EQ(spec->ap_per_dpc, 13u);
  EXPECT_EQ(spec->peu_per_ap, 4u);
  EXPECT_EQ(spec->max_resident_waves, 8u);
  EXPECT_EQ(spec->max_issuable_waves, 4u);
  EXPECT_EQ(spec->default_issue_cycles, 4u);
}
```

**Step 2: Run test to verify it fails**

Run:

```bash
cmake --build build
```

Expected: missing headers and missing symbols for `ArchRegistry`.

**Step 3: Write minimal implementation**

Use this shape:

```cpp
namespace gpu_model {

struct FeatureFlags {
  bool sync = false;
  bool barrier = false;
  bool mma = false;
  bool l1_cache = false;
  bool l2_cache = false;
};

struct GpuArchSpec {
  std::string name;
  uint32_t wave_size = 64;
  uint32_t dpc_count = 0;
  uint32_t ap_per_dpc = 0;
  uint32_t peu_per_ap = 0;
  uint32_t max_resident_waves = 0;
  uint32_t max_issuable_waves = 0;
  uint32_t default_issue_cycles = 4;
  FeatureFlags features;
};

class ArchRegistry {
public:
  static std::shared_ptr<const GpuArchSpec> Get(std::string_view name);
};

}  // namespace gpu_model
```

Back it with a simple static map in `arch_registry.cpp`, and register exactly one built-in spec in `c500_spec.cpp`.

**Step 4: Run test to verify it passes**

Run:

```bash
cmake --build build
./build/tests/gpu_model_tests --gtest_filter=ArchRegistryTest.*
```

Expected: `ArchRegistryTest.C500SpecExists` and `ArchRegistryTest.C500FieldsMatchDesign` pass.

**Step 5: Commit**

```bash
git add include/gpu_model/arch src/arch CMakeLists.txt tests/arch/arch_registry_test.cpp
git commit -m "feat: add c500 architecture spec and registry"
```

## Task 3: Define The Canonical Instruction IR

**Files:**
- Create: `/data/gpu_model/include/gpu_model/isa/opcode.h`
- Create: `/data/gpu_model/include/gpu_model/isa/operand.h`
- Create: `/data/gpu_model/include/gpu_model/isa/instruction.h`
- Create: `/data/gpu_model/include/gpu_model/isa/instruction_builder.h`
- Create: `/data/gpu_model/include/gpu_model/isa/kernel_program.h`
- Create: `/data/gpu_model/src/isa/instruction_builder.cpp`
- Create: `/data/gpu_model/src/isa/kernel_program.cpp`
- Modify: `/data/gpu_model/CMakeLists.txt`
- Test: `/data/gpu_model/tests/functional/predicated_if_functional_test.cpp`

**Step 1: Write the failing test**

Add a new test that only builds a kernel and checks its label lookup:

```cpp
#include <gtest/gtest.h>
#include "gpu_model/isa/instruction_builder.h"

TEST(KernelProgramTest, ResolvesLabelsAndInstructionCount) {
  gpu_model::InstructionBuilder b;
  b.SLoadArg("s0", 0);
  b.Label("exit");
  b.BExit();

  auto kernel = b.Build("tiny_kernel");
  EXPECT_EQ(kernel.instructions().size(), 2u);
  EXPECT_EQ(kernel.ResolveLabel("exit"), 1u);
}
```

**Step 2: Run test to verify it fails**

Run:

```bash
cmake --build build
```

Expected: missing `instruction_builder.h` and `KernelProgram`.

**Step 3: Write minimal implementation**

Define the minimal opcode set for `V1.0`:

```cpp
enum class Opcode {
  SysLoadArg,
  SysGlobalIdX,
  SysBlockIdxX,
  SysBlockDimX,
  SysLaneId,
  SMov,
  SAdd,
  SMul,
  SCmpLt,
  SCmpEq,
  VMov,
  VAdd,
  VMul,
  VCmpLtCmask,
  MLoadGlobal,
  MStoreGlobal,
  MaskSaveExec,
  MaskRestoreExec,
  MaskAndExecCmask,
  BBranch,
  BIfSmask,
  BIfNoexec,
  BExit
};
```

Use small value objects:

```cpp
struct DebugLoc {
  std::string file;
  uint32_t line = 0;
  std::string label;
};

struct Instruction {
  Opcode opcode;
  std::vector<Operand> operands;
  DebugLoc debug_loc;
};
```

`InstructionBuilder` should only do three things:
- append instructions
- record labels
- build a `KernelProgram`

**Step 4: Run test to verify it passes**

Run:

```bash
cmake --build build
./build/tests/gpu_model_tests --gtest_filter=KernelProgramTest.*
```

Expected: label resolution and instruction count test pass.

**Step 5: Commit**

```bash
git add include/gpu_model/isa src/isa tests/functional/predicated_if_functional_test.cpp CMakeLists.txt
git commit -m "feat: add canonical instruction IR and builder"
```

## Task 4: Model Register Files And Wave Runtime State

**Files:**
- Create: `/data/gpu_model/include/gpu_model/state/register_file.h`
- Create: `/data/gpu_model/include/gpu_model/state/wave_state.h`
- Create: `/data/gpu_model/include/gpu_model/state/peu_state.h`
- Create: `/data/gpu_model/include/gpu_model/state/ap_state.h`
- Create: `/data/gpu_model/include/gpu_model/state/dpc_state.h`
- Create: `/data/gpu_model/include/gpu_model/state/gpu_state.h`
- Modify: `/data/gpu_model/CMakeLists.txt`
- Test: `/data/gpu_model/tests/functional/mapper_test.cpp`

**Step 1: Write the failing test**

Add a state-level test that constructs one `WaveState` and checks mask defaults:

```cpp
TEST(WaveStateTest, InitializesExecAndPredicateMasks) {
  gpu_model::WaveState wave;
  wave.thread_count = 10;
  wave.ResetInitialExec();

  EXPECT_EQ(wave.exec.count(), 10u);
  EXPECT_EQ(wave.cmask.count(), 0u);
  EXPECT_EQ(wave.smask, 0u);
}
```

**Step 2: Run test to verify it fails**

Run:

```bash
cmake --build build
```

Expected: missing runtime state headers and symbols.

**Step 3: Write minimal implementation**

Use these core shapes:

```cpp
enum class WaveStatus { Active, Exited, Stalled };

struct SGPRFile {
  std::vector<uint64_t> regs;
};

struct VGPRFile {
  std::vector<std::array<uint64_t, 64>> regs;
};

struct WaveState {
  uint32_t block_id = 0;
  uint32_t wave_id = 0;
  uint32_t peu_id = 0;
  uint32_t ap_id = 0;
  uint64_t pc = 0;
  WaveStatus status = WaveStatus::Active;
  std::bitset<64> exec;
  std::bitset<64> cmask;
  uint64_t smask = 0;
  uint32_t thread_count = 0;
  SGPRFile sgpr;
  VGPRFile vgpr;

  void ResetInitialExec();
};
```

`ResetInitialExec()` must set bits `[0, thread_count)` and clear the rest.

**Step 4: Run test to verify it passes**

Run:

```bash
cmake --build build
./build/tests/gpu_model_tests --gtest_filter=WaveStateTest.*
```

Expected: the new wave state test passes.

**Step 5: Commit**

```bash
git add include/gpu_model/state CMakeLists.txt tests/functional/mapper_test.cpp
git commit -m "feat: add register files and runtime wave state"
```

## Task 5: Add Launch Config, Argument Packing, Memory System, And Mapper

**Files:**
- Create: `/data/gpu_model/include/gpu_model/memory/memory_space.h`
- Create: `/data/gpu_model/include/gpu_model/memory/memory_request.h`
- Create: `/data/gpu_model/include/gpu_model/memory/memory_system.h`
- Create: `/data/gpu_model/src/memory/memory_system.cpp`
- Create: `/data/gpu_model/include/gpu_model/runtime/launch_config.h`
- Create: `/data/gpu_model/include/gpu_model/runtime/kernel_arg_pack.h`
- Create: `/data/gpu_model/include/gpu_model/runtime/launch_request.h`
- Create: `/data/gpu_model/include/gpu_model/runtime/mapper.h`
- Create: `/data/gpu_model/src/runtime/mapper.cpp`
- Modify: `/data/gpu_model/CMakeLists.txt`
- Test: `/data/gpu_model/tests/functional/mapper_test.cpp`

**Step 1: Write the failing test**

Add a placement test:

```cpp
TEST(MapperTest, MapsBlocksToApsAndWavesToPeusForC500) {
  auto spec = gpu_model::ArchRegistry::Get("c500");
  gpu_model::LaunchConfig cfg{.grid_dim_x = 2, .block_dim_x = 128};

  auto placement = gpu_model::Mapper::Place(*spec, cfg);

  ASSERT_EQ(placement.blocks.size(), 2u);
  EXPECT_EQ(placement.blocks[0].dpc_id, 0u);
  EXPECT_EQ(placement.blocks[0].ap_id, 0u);
  EXPECT_EQ(placement.blocks[0].waves.size(), 2u);
  EXPECT_EQ(placement.blocks[0].waves[0].peu_id, 0u);
  EXPECT_EQ(placement.blocks[0].waves[1].peu_id, 1u);
}
```

**Step 2: Run test to verify it fails**

Run:

```bash
cmake --build build
```

Expected: missing mapper, launch config, and placement structures.

**Step 3: Write minimal implementation**

Use a simple linear placement for `V1.0`:

```cpp
struct LaunchConfig {
  uint32_t grid_dim_x = 1;
  uint32_t block_dim_x = 1;
};

struct WavePlacement {
  uint32_t wave_id;
  uint32_t peu_id;
  uint32_t lane_count;
};

struct BlockPlacement {
  uint32_t block_id;
  uint32_t dpc_id;
  uint32_t ap_id;
  std::vector<WavePlacement> waves;
};

struct PlacementMap {
  std::vector<BlockPlacement> blocks;
};
```

Placement rules:
- block `i` maps to `global_ap = i % (dpc_count * ap_per_dpc)`
- `dpc_id = global_ap / ap_per_dpc`
- `ap_local = global_ap % ap_per_dpc`
- wave `j` maps to `peu_id = j % peu_per_ap`

Also add a byte-addressable `MemorySystem` with:
- `WriteGlobal(uint64_t addr, std::span<const std::byte>)`
- `ReadGlobal(uint64_t addr, std::span<std::byte>)`

**Step 4: Run test to verify it passes**

Run:

```bash
cmake --build build
./build/tests/gpu_model_tests --gtest_filter=MapperTest.*
```

Expected: placement test passes and confirms the basic `c500` mapping rules.

**Step 5: Commit**

```bash
git add include/gpu_model/memory include/gpu_model/runtime src/memory src/runtime CMakeLists.txt tests/functional/mapper_test.cpp
git commit -m "feat: add launch config, memory system, and c500 mapper"
```

## Task 6: Add Trace, Debug Info, And Host Runtime Shell

**Files:**
- Create: `/data/gpu_model/include/gpu_model/debug/debug_info.h`
- Create: `/data/gpu_model/include/gpu_model/debug/trace_event.h`
- Create: `/data/gpu_model/include/gpu_model/debug/trace_sink.h`
- Create: `/data/gpu_model/src/debug/trace_sink.cpp`
- Create: `/data/gpu_model/include/gpu_model/runtime/host_runtime.h`
- Create: `/data/gpu_model/src/runtime/host_runtime.cpp`
- Modify: `/data/gpu_model/CMakeLists.txt`
- Test: `/data/gpu_model/tests/functional/vecadd_functional_test.cpp`

**Step 1: Write the failing test**

Create a test that launches a trivial kernel and asserts trace events were emitted:

```cpp
TEST(TraceTest, EmitsLaunchAndWaveStepEvents) {
  gpu_model::CollectingTraceSink trace;
  gpu_model::HostRuntime runtime(&trace);

  auto result = runtime.Launch(/* request filled in later */);
  EXPECT_FALSE(trace.events().empty());
}
```

**Step 2: Run test to verify it fails**

Run:

```bash
cmake --build build
```

Expected: missing `HostRuntime`, `TraceEvent`, and trace sink types.

**Step 3: Write minimal implementation**

Use:

```cpp
enum class TraceEventKind {
  Launch,
  BlockPlaced,
  WaveStep,
  ExecMaskUpdate,
  MemoryAccess,
  WaveExit,
  Stall,
  Arrive
};

struct TraceEvent {
  TraceEventKind kind;
  uint64_t cycle = 0;
  uint32_t block_id = 0;
  uint32_t wave_id = 0;
  uint64_t pc = 0;
  std::string message;
};

class TraceSink {
public:
  virtual ~TraceSink() = default;
  virtual void OnEvent(const TraceEvent&) = 0;
};
```

Provide:
- `NullTraceSink`
- `CollectingTraceSink`
- `HostRuntime` shell that validates the request, maps blocks, emits placement trace, and returns a placeholder result

**Step 4: Run test to verify it passes**

Run:

```bash
cmake --build build
./build/tests/gpu_model_tests --gtest_filter=TraceTest.*
```

Expected: trace test passes with non-empty events.

**Step 5: Commit**

```bash
git add include/gpu_model/debug src/debug include/gpu_model/runtime/host_runtime.h src/runtime/host_runtime.cpp CMakeLists.txt tests/functional/vecadd_functional_test.cpp
git commit -m "feat: add trace infrastructure and host runtime shell"
```

## Task 7: Implement Shared Semantics And The Functional Executor

**Files:**
- Create: `/data/gpu_model/include/gpu_model/exec/op_plan.h`
- Create: `/data/gpu_model/include/gpu_model/exec/semantics.h`
- Create: `/data/gpu_model/include/gpu_model/exec/execution_engine.h`
- Create: `/data/gpu_model/include/gpu_model/exec/functional_executor.h`
- Create: `/data/gpu_model/src/exec/semantics.cpp`
- Create: `/data/gpu_model/src/exec/functional_executor.cpp`
- Modify: `/data/gpu_model/src/runtime/host_runtime.cpp`
- Modify: `/data/gpu_model/CMakeLists.txt`
- Test: `/data/gpu_model/tests/functional/vecadd_functional_test.cpp`

**Step 1: Write the failing test**

Write the full multi-block `vecadd` integration test:

```cpp
TEST(FunctionalVecAddTest, RunsMultiBlockMultiThreadKernel) {
  constexpr uint32_t n = 300;
  std::vector<float> a(n), b(n), c(n, -1.0f);
  for (uint32_t i = 0; i < n; ++i) {
    a[i] = static_cast<float>(i);
    b[i] = static_cast<float>(2 * i);
  }

  auto kernel = BuildVecAddKernel();
  auto result = LaunchVecAdd(kernel, a, b, c, n, /*grid=*/3, /*block=*/128);

  for (uint32_t i = 0; i < n; ++i) {
    EXPECT_FLOAT_EQ(c[i], a[i] + b[i]);
  }
}
```

**Step 2: Run test to verify it fails**

Run:

```bash
cmake --build build
./build/tests/gpu_model_tests --gtest_filter=FunctionalVecAddTest.*
```

Expected: the new test fails because no executor exists.

**Step 3: Write minimal implementation**

`OpPlan` should describe one instruction's effect:

```cpp
struct MemoryWriteback {
  uint32_t vgpr_index = 0;
  std::array<uint64_t, 64> values{};
  std::bitset<64> exec_snapshot;
};

struct OpPlan {
  uint32_t issue_cycles = 4;
  bool exit_wave = false;
  bool advance_pc = true;
  std::optional<uint64_t> branch_target;
  std::vector<std::function<void()>> immediate_actions;
};
```

`Semantics` must support exactly these `V1.0` opcodes:
- `SysLoadArg`
- `SysGlobalIdX`
- `SMov`, `SAdd`, `SMul`, `SCmpLt`, `SCmpEq`
- `VMov`, `VAdd`, `VMul`, `VCmpLtCmask`
- `MLoadGlobal`, `MStoreGlobal`
- `MaskSaveExec`, `MaskRestoreExec`, `MaskAndExecCmask`
- `BBranch`, `BIfSmask`, `BIfNoexec`, `BExit`

Functional rules:
- `S_*` ignore `EXEC`
- `V_*` and `M_*` only act on `EXEC=1`
- `VCmpLtCmask` writes `CMASK`
- `SCmp*` writes `SMASK bit0`
- `MaskAndExecCmask` updates `EXEC &= CMASK`
- `MLoadGlobal` and `MStoreGlobal` are synchronous in `V1.0`

`FunctionalExecutor` must:
- walk mapped blocks in deterministic order
- execute waves until every wave is `Exited`
- emit `WaveStep`, `ExecMaskUpdate`, `MemoryAccess`, and `WaveExit` trace events

**Step 4: Run test to verify it passes**

Run:

```bash
cmake --build build
./build/tests/gpu_model_tests --gtest_filter=FunctionalVecAddTest.*
```

Expected: `vecadd` passes for `N=300`, `grid=3`, `block=128`.

**Step 5: Commit**

```bash
git add include/gpu_model/exec src/exec src/runtime/host_runtime.cpp CMakeLists.txt tests/functional/vecadd_functional_test.cpp
git commit -m "feat: add shared semantics and functional executor"
```

## Task 8: Validate `EXEC/CMASK/SMASK` With A Predicated `if` Kernel

**Files:**
- Modify: `/data/gpu_model/tests/functional/predicated_if_functional_test.cpp`
- Modify: `/data/gpu_model/src/exec/semantics.cpp`
- Modify: `/data/gpu_model/src/exec/functional_executor.cpp`

**Step 1: Write the failing test**

Add a kernel test that keeps only positive elements:

```cpp
TEST(PredicatedIfFunctionalTest, UsesCmaskAndExecWithoutImplicitReconvergence) {
  std::vector<int32_t> in{3, -1, 0, 7, -8, 5};
  std::vector<int32_t> out(in.size(), -99);

  auto kernel = BuildPositiveCopyKernel();
  auto result = LaunchPositiveCopy(kernel, in, out, static_cast<uint32_t>(in.size()), 1, 64);

  EXPECT_EQ(out[0], 3);
  EXPECT_EQ(out[1], -99);
  EXPECT_EQ(out[2], -99);
  EXPECT_EQ(out[3], 7);
  EXPECT_EQ(out[4], -99);
  EXPECT_EQ(out[5], 5);
}
```

The kernel should explicitly do:
- compute `gid`
- bounds compare into `CMASK`
- save `EXEC`
- `EXEC &= CMASK`
- branch if `noexec`
- load input
- compare input against zero into `CMASK`
- `EXEC &= CMASK`
- perform store
- restore `EXEC`
- exit

**Step 2: Run test to verify it fails**

Run:

```bash
cmake --build build
./build/tests/gpu_model_tests --gtest_filter=PredicatedIfFunctionalTest.*
```

Expected: test fails because one or more mask instructions are incorrect.

**Step 3: Write minimal implementation**

Tighten mask semantics:
- `MaskSaveExec` copies `EXEC` into an SGPR slot
- `MaskRestoreExec` restores `EXEC` from SGPR
- `BIfNoexec` branches when `EXEC.none()`
- trace `ExecMaskUpdate` on every `EXEC` mutation

Do not add reconvergence stacks or hidden branch logic.

**Step 4: Run test to verify it passes**

Run:

```bash
cmake --build build
./build/tests/gpu_model_tests --gtest_filter=PredicatedIfFunctionalTest.*
```

Expected: the positive-copy kernel passes and verifies the explicit mask flow.

**Step 5: Commit**

```bash
git add src/exec tests/functional/predicated_if_functional_test.cpp
git commit -m "test: validate exec cmask and smask control flow"
```

## Task 9: Finish The Functional Placement And Trace Tests

**Files:**
- Modify: `/data/gpu_model/tests/functional/mapper_test.cpp`
- Modify: `/data/gpu_model/tests/functional/vecadd_functional_test.cpp`
- Modify: `/data/gpu_model/src/runtime/host_runtime.cpp`
- Modify: `/data/gpu_model/src/exec/functional_executor.cpp`

**Step 1: Write the failing test**

Add explicit assertions on trace content:

```cpp
TEST(FunctionalVecAddTest, EmitsMemoryAndWaveExitTrace) {
  gpu_model::CollectingTraceSink trace;
  auto result = LaunchVecAddWithTrace(trace);

  EXPECT_TRUE(ContainsEvent(trace.events(), gpu_model::TraceEventKind::MemoryAccess));
  EXPECT_TRUE(ContainsEvent(trace.events(), gpu_model::TraceEventKind::WaveExit));
}
```

Also expand mapper coverage for non-multiple-of-64 tail waves:

```cpp
TEST(MapperTest, TailWaveUsesPartialLaneCount) {
  auto spec = gpu_model::ArchRegistry::Get("c500");
  gpu_model::LaunchConfig cfg{.grid_dim_x = 1, .block_dim_x = 96};
  auto placement = gpu_model::Mapper::Place(*spec, cfg);
  EXPECT_EQ(placement.blocks[0].waves[1].lane_count, 32u);
}
```

**Step 2: Run test to verify it fails**

Run:

```bash
cmake --build build
./build/tests/gpu_model_tests --gtest_filter="FunctionalVecAddTest.*:MapperTest.*"
```

Expected: at least one trace or tail-wave assertion fails.

**Step 3: Write minimal implementation**

Make sure:
- mapper computes `lane_count = min(64, block_dim_x - wave_id * 64)`
- `HostRuntime` emits `BlockPlaced`
- `FunctionalExecutor` emits `MemoryAccess` for each load/store instruction
- `FunctionalExecutor` emits `WaveExit` once per wave

**Step 4: Run test to verify it passes**

Run:

```bash
cmake --build build
./build/tests/gpu_model_tests --gtest_filter="FunctionalVecAddTest.*:MapperTest.*"
```

Expected: placement and trace coverage both pass.

**Step 5: Commit**

```bash
git add src/runtime/host_runtime.cpp src/exec/functional_executor.cpp tests/functional/mapper_test.cpp tests/functional/vecadd_functional_test.cpp
git commit -m "test: complete functional placement and trace coverage"
```

## Task 10: Add `Scoreboard`, `EventQueue`, And `CycleExecutor` Skeleton

**Files:**
- Create: `/data/gpu_model/include/gpu_model/exec/scoreboard.h`
- Create: `/data/gpu_model/include/gpu_model/exec/event_queue.h`
- Create: `/data/gpu_model/include/gpu_model/exec/cycle_executor.h`
- Create: `/data/gpu_model/src/exec/scoreboard.cpp`
- Create: `/data/gpu_model/src/exec/event_queue.cpp`
- Create: `/data/gpu_model/src/exec/cycle_executor.cpp`
- Modify: `/data/gpu_model/CMakeLists.txt`
- Test: `/data/gpu_model/tests/cycle/cycle_smoke_test.cpp`

**Step 1: Write the failing test**

Create a cycle smoke test with zero memory traffic:

```cpp
TEST(CycleSmokeTest, ScalarAndVectorOpsConsumeFourCyclesEach) {
  auto kernel = BuildTinyArithmeticKernel();
  auto result = LaunchInCycleMode(kernel, /*grid=*/1, /*block=*/64);

  EXPECT_EQ(result.total_cycles, 12u); // 3 issued instructions * 4 cycles
}
```

**Step 2: Run test to verify it fails**

Run:

```bash
cmake --build build
./build/tests/gpu_model_tests --gtest_filter=CycleSmokeTest.*
```

Expected: missing cycle executor components.

**Step 3: Write minimal implementation**

`Scoreboard` only needs `MarkReady(reg, cycle)` and `IsReady(reg, cycle)`.

`EventQueue` only needs:

```cpp
struct TimedEvent {
  uint64_t cycle = 0;
  std::function<void()> action;
};
```

`CycleExecutor` should:
- reuse `Semantics`
- issue one instruction per PEU per cycle
- assign `issue_cycles = 4` to all normal instructions
- mark result registers and masks ready at `now + 4`
- advance `pc` only after the instruction commits

**Step 4: Run test to verify it passes**

Run:

```bash
cmake --build build
./build/tests/gpu_model_tests --gtest_filter=CycleSmokeTest.*
```

Expected: the simple arithmetic kernel reports `12` cycles.

**Step 5: Commit**

```bash
git add include/gpu_model/exec src/exec CMakeLists.txt tests/cycle/cycle_smoke_test.cpp
git commit -m "feat: add scoreboard event queue and cycle executor skeleton"
```

## Task 11: Add Async Memory `arrive` Handling For The Cycle Model

**Files:**
- Modify: `/data/gpu_model/include/gpu_model/memory/memory_request.h`
- Modify: `/data/gpu_model/include/gpu_model/exec/op_plan.h`
- Modify: `/data/gpu_model/src/exec/semantics.cpp`
- Modify: `/data/gpu_model/src/exec/cycle_executor.cpp`
- Create: `/data/gpu_model/tests/cycle/async_memory_cycle_test.cpp`

**Step 1: Write the failing test**

Create a test with fixed synthetic memory latency:

```cpp
TEST(AsyncMemoryCycleTest, LoadUsesIssuePlusArriveLatency) {
  auto kernel = BuildOneLoadKernel();
  auto result = LaunchInCycleModeWithFixedGlobalLatency(kernel, /*latency=*/20);

  EXPECT_EQ(result.total_cycles, 24u); // 4 issue + 20 arrive
}
```

Also assert that a dependent `v_add` cannot issue before the load arrives.

**Step 2: Run test to verify it fails**

Run:

```bash
cmake --build build
./build/tests/gpu_model_tests --gtest_filter=AsyncMemoryCycleTest.*
```

Expected: load completion is treated as immediate or the dependency ordering is wrong.

**Step 3: Write minimal implementation**

Update the model so that:
- `M_*` still consumes `4` issue cycles
- `MLoadGlobal` creates a `MemoryRequest` with a captured `exec_snapshot`
- `CycleExecutor` schedules a completion event at `issue_end + configured_latency`
- the destination VGPR is not ready until the arrive event fires
- `TraceEventKind::Arrive` is emitted on response

Use a trivial fixed-latency timing policy first:

```cpp
uint64_t GlobalLoadLatency() const { return fixed_global_latency_; }
```

**Step 4: Run test to verify it passes**

Run:

```bash
cmake --build build
./build/tests/gpu_model_tests --gtest_filter=AsyncMemoryCycleTest.*
```

Expected: total cycle count and dependency ordering both pass.

**Step 5: Commit**

```bash
git add include/gpu_model/memory/memory_request.h include/gpu_model/exec/op_plan.h src/exec/semantics.cpp src/exec/cycle_executor.cpp tests/cycle/async_memory_cycle_test.cpp
git commit -m "feat: add async memory arrive handling to cycle model"
```

## Task 12: Clean Up Public APIs And Add One End-To-End Example

**Files:**
- Modify: `/data/gpu_model/include/gpu_model/runtime/host_runtime.h`
- Modify: `/data/gpu_model/include/gpu_model/runtime/launch_request.h`
- Modify: `/data/gpu_model/tests/functional/vecadd_functional_test.cpp`
- Create: `/data/gpu_model/examples/vecadd_main.cpp`
- Modify: `/data/gpu_model/CMakeLists.txt`

**Step 1: Write the failing test**

Add one API-level test that uses only public headers:

```cpp
TEST(PublicApiTest, HostRuntimeLaunchesKernelUsingOnlyPublicInterfaces) {
  auto runtime = gpu_model::HostRuntime{};
  auto kernel = BuildVecAddKernel();
  auto result = runtime.Launch(MakeVecAddRequest(kernel));
  EXPECT_TRUE(result.ok);
}
```

**Step 2: Run test to verify it fails**

Run:

```bash
cmake --build build
./build/tests/gpu_model_tests --gtest_filter=PublicApiTest.*
```

Expected: one or more required public structs are still too internal or awkward to construct.

**Step 3: Write minimal implementation**

Polish the API so this works cleanly:

```cpp
struct LaunchRequest {
  std::string arch_name = "c500";
  const KernelProgram* kernel = nullptr;
  LaunchConfig config;
  KernelArgPack args;
  ExecutionMode mode = ExecutionMode::Functional;
  TraceSink* trace = nullptr;
};
```

Add `examples/vecadd_main.cpp` that:
- allocates host vectors
- builds a `vecadd` kernel with `InstructionBuilder`
- launches in functional mode
- prints the first few results

**Step 4: Run tests and example**

Run:

```bash
cmake --build build
./build/tests/gpu_model_tests
./build/examples/vecadd_main
```

Expected: all tests pass and the example prints correct `a + b` results.

**Step 5: Commit**

```bash
git add include/gpu_model/runtime tests/functional/vecadd_functional_test.cpp examples/vecadd_main.cpp CMakeLists.txt
git commit -m "chore: finalize public runtime api and example"
```

## Acceptance Checklist

- `ArchRegistry::Get("c500")` returns the expected static spec.
- `InstructionBuilder` can create kernels with labels and debug locations.
- `WaveState` contains `EXEC`, `CMASK`, `SMASK`, `PC`, SGPRs, and VGPRs.
- `Mapper` implements `block -> AP` and `wave -> PEU` for `c500`.
- `FunctionalExecutor` runs `vecadd` correctly across multiple blocks and waves.
- `FunctionalExecutor` handles explicit mask-driven `if` control flow without reconvergence logic.
- Trace output contains `Launch`, `BlockPlaced`, `WaveStep`, `ExecMaskUpdate`, `MemoryAccess`, and `WaveExit`.
- `CycleExecutor` reuses the same semantics and applies fixed `4 cycle` issue cost to normal instructions.
- `MLoadGlobal` in cycle mode uses `4 cycle` issue and asynchronous `arrive`.
- All tests pass under `./build/tests/gpu_model_tests`.

## Follow-On Work After This Plan

- `V2.1`: shared/private timing, barrier wait behavior, async wait instructions
- `V2.2`: L1/L2 timing model and shared bank conflict penalties
- `V3.0`: assembly parser, metadata loader, constant section loader, binary decoder
- `V3.1`: runtime API hook layer
- `V3.2`: restricted CUDA frontend if the project chooses to add it later
