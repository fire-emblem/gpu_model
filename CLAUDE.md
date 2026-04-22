# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`gpu_model` is a lightweight C++ functional and naive cycle model for AMD/GCN-style GPU kernels. It enables execution and analysis of HIP kernels without real GPU hardware, supporting:
- Operator library optimization
- Compiler codegen comparison
- Hardware parameter evaluation
- HIP/AMDGPU kernel behavior verification

## Build Commands

```bash
# Recommended: using presets (faster, Ninja backend)
cmake --preset dev-fast
cmake --build --preset dev-fast

# Standard CMake flow
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
```

The `dev-fast` preset uses Ninja and outputs to `build-ninja/`. Examples and scripts auto-detect build dir (`build-ninja/` > `build/`, overridable via `GPU_MODEL_BUILD_DIR`).

## Test Commands

```bash
# Run all tests (use build-ninja path if using preset)
./build-ninja/tests/gpu_model_tests

# Run full test matrix
GPU_MODEL_TEST_PROFILE=full ./build-ninja/tests/gpu_model_tests

# Run single test
./build-ninja/tests/gpu_model_tests --gtest_filter=HipRuntimeTest.LaunchesHipVecAddExecutableAndValidatesOutput

# Run specific test pattern
./build-ninja/tests/gpu_model_tests --gtest_filter=*waitcnt*
```

## Examples

Examples are numbered by complexity in `examples/01-vecadd-basic/` through `examples/13-algorithm-comparison/`. Non-comparison examples default to `mt` mode only; comparison/visualization examples explicitly run multiple modes.

```bash
# Run a specific example
./examples/01-vecadd-basic/run.sh

# Disable hipcc compilation cache (enabled by default via tools/hipcc_cache.sh)
GPU_MODEL_USE_HIPCC_CACHE=0 ./examples/01-vecadd-basic/run.sh
```

## Key Scripts

```bash
# Light push gate (fast smoke tests, recommended for daily use)
./scripts/run_push_gate_light.sh

# Full push gate
./scripts/run_push_gate.sh

# Execution checks
./scripts/run_exec_checks.sh

# Shared-heavy regression
./scripts/run_shared_heavy_regression.sh

# Real HIP kernel regression
./scripts/run_real_hip_kernel_regression.sh

# ABI regression
./scripts/run_abi_regression.sh

# Verify trace isolation (GPU_MODEL_DISABLE_TRACE=1)
./scripts/run_disable_trace_smoke.sh

# Install git hooks (pre-push runs light gate)
./scripts/install_git_hooks.sh
```

## Architecture

The codebase follows a five-layer architecture:

```
runtime -> program -> instruction -> execution -> wave
```

The runtime layer itself has an internal structure:

```
HipRuntime (C ABI / LD_PRELOAD) -> ModelRuntime (core) -> ExecEngine (execution chain)
```

`ExecEngine` is part of `ModelRuntime`'s execution chain, not a peer layer. See `docs/runtime-layering.md` for the canonical reference.

### Source Directory Structure

```
src/
├── arch/          # Architecture specs (GpuArchSpec, device topology)
├── debug/         # Trace output (text, jsonl, perfetto)
├── execution/     # Execution engines (functional/cycle/encoded)
├── gpu_model/     # Main library entry point
├── instruction/   # Instruction objects and semantic dispatch
├── isa/           # GCN ISA definitions and encodings
├── loader/        # AMDGPU object / HIP artifact loading
├── memory/        # Memory pools (global/shared/private/constant/kernarg)
├── program/       # ProgramObject, ExecutableKernel, EncodedProgramObject
├── runtime/
│   ├── config/    # Launch request configuration
│   ├── exec_engine/
│   ├── hip_runtime/  # HipRuntime (C ABI / LD_PRELOAD entry)
│   └── model_runtime/
│       ├── core/     # ModelRuntime facade
│       ├── compat/   # Compatibility layering (abi/launch/session)
│       ├── module/   # Module loading
│       └── stats/    # Runtime statistics
└── spec/          # Engineering reference materials
```

### Runtime Layer
- `HipRuntime`: HIP compatibility layer (C ABI entry, LD_PRELOAD interposition)
- `ModelRuntime`: Core implementation layer
- `ExecEngine`: Execution controller (part of ModelRuntime's chain, not a peer layer)

Key files:
- `src/runtime/hip_runtime/hip_ld_preload.cpp` - C ABI / LD_PRELOAD entry
- `src/runtime/model_runtime/core/` - ModelRuntime facade

### Program Layer
- `ProgramObject`: Static program representation
- `ExecutableKernel`: Launch-ready kernel
- `EncodedProgramObject`: Encoded code object representation

Key directories:
- `src/loader/` - AMDGPU object / HIP artifact loading
- `src/program/` - Program object types

### Instruction Layer
- Instruction decoding and semantic dispatch
- GCN encoding definitions

Key directories:
- `src/instruction/` - Instruction objects
- `src/isa/` - ISA definitions

### Execution Layer
- `FunctionalExecEngine`: Functional execution (st/mt modes)
- `CycleExecEngine`: Naive cycle-level model
- `ProgramObjectExecEngine`: Encoded binary execution
- `WaveContext`: Wave-level execution state

Key files:
- `src/execution/functional_exec_engine.cpp`
- `src/execution/cycle_exec_engine.cpp`
- `src/execution/program_object_exec_engine.cpp`

### Architecture Layer
- `GpuArchSpec`: Architecture parameters
- Device topology modeling

Key directory:
- `src/arch/`

### Memory System
- Global / Shared / Private / Constant / Kernarg pools

Key directory:
- `src/memory/`

### Debug/Trace
- Trace output (text, jsonl, perfetto)
- Timeline visualization

Key directory:
- `src/debug/`

## Engineering Constraints (from AGENTS.md)

**Important:** See `AGENTS.md` for the complete engineering constraints. Key points:

### Trace vs Business Logic
- Trace only consumes events; it does NOT participate in business logic decisions
- Behavior must be identical with or without trace enabled
- Disable trace with: `GPU_MODEL_DISABLE_TRACE=1`
- Principle: state change first -> typed event -> trace consumes. Never reverse.

### Cycle Field Semantics
- The `cycle` field in trace outputs is **model time**, NOT real hardware time
- It represents relative ordering, wait intervals, and dependencies
- Do not interpret as physical execution time without calibration
- Functional `cycle` = virtual counter for ordering; Cycle `cycle` = model counter, still not physical time

### Runtime Layering
- `HipRuntime` -> `ModelRuntime` -> `ExecEngine` only
- No independent "interposer module" concept; `hip_interposer.cpp` is historical naming for C ABI entry
- `ModelRuntime` must never depend back on `HipRuntime`

### Functional vs Cycle Models
- Functional: allows st/mt host execution strategies
- Cycle: single unified timing model (no "cycle st" / "cycle mt" variants)
- Cycle model must be tick-driven state machine
- `global_cycle` is the single scheduling time; `wave_cycle` only tracks per-wave accumulation

### Resume / Ready Semantics
- `arrive_resume` = wave is eligible, NOT guaranteed to issue same cycle
- `WaveStep` = actual issue. Gap between resume and step is real scheduling delay
- Functional st: next issue quantum after resume. Functional mt: allows preemption

## Dependencies

Vendored in `third_party/`:
- `loguru` - Logging
- `marl` - Multithreading (fiber-based parallelism)
- `gem5` - Reference material (not linked)
- `miaow` - Reference material
- `llvm-project` - Reference material

## Output Files

- `trace.txt` / `trace.jsonl` - Execution traces
- `timeline.perfetto.json` - Timeline for Perfetto UI
- Results written to `results/` or `examples/*/results/`

## Environment Variables

- `GPU_MODEL_DISABLE_TRACE=1` - Disable trace output
- `GPU_MODEL_TEST_PROFILE=full` - Run full test matrix
- `GPU_MODEL_GATE_LIGHT_GTEST_FILTER=...` - Override push gate filter
- `GPU_MODEL_BUILD_DIR=...` - Override build directory detection
- `GPU_MODEL_USE_HIPCC_CACHE=0` - Disable hipcc compilation cache

## Renamed Symbols (for reading old docs/code)

- `ModelRuntimeApi` -> `ModelRuntime`
- `RuntimeHooks` -> `HipRuntime`
- `HostRuntime` -> `ExecEngine`
- `HipInterposerState` -> deleted (merged into `HipRuntime`)
- `hip_interposer.cpp` -> `hip_ld_preload.cpp`
