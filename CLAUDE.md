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
# Configure and build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j

# Using presets (recommended for development)
cmake --preset dev-fast
cmake --build --preset dev-fast

# Build with ASan (default for Debug builds)
cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug
cmake --build build-asan -j
```

## Test Commands

```bash
# Run all tests
./build/tests/gpu_model_tests

# Run full test matrix
GPU_MODEL_TEST_PROFILE=full ./build/tests/gpu_model_tests

# Run single test
./build/tests/gpu_model_tests --gtest_filter=HipRuntimeTest.LaunchesHipVecAddExecutableAndValidatesOutput

# Run specific test pattern
./build/tests/gpu_model_tests --gtest_filter=*waitcnt*
```

## Examples

Examples are numbered by complexity in `examples/01-vecadd-basic/` through `examples/11-perfetto-waitcnt-slots/`:

```bash
# Run a specific example
./examples/01-vecadd-basic/run.sh

# Run with specific execution mode
# Each example runs st/mt/cycle modes by default
```

## Key Scripts

```bash
# Light push gate (fast smoke tests)
./scripts/run_push_gate_light.sh

# Full push gate
./scripts/run_push_gate.sh

# Execution checks
./scripts/run_exec_checks.sh

# Shared-heavy regression
./scripts/run_shared_heavy_regression.sh

# Real HIP kernel regression
./scripts/run_real_hip_kernel_regression.sh
```

## Architecture

The codebase follows a five-layer architecture:

```
runtime -> program -> instruction -> execution -> wave
```

### Runtime Layer
- `HipRuntime`: HIP compatibility layer (C ABI entry, LD_PRELOAD interposition)
- `ModelRuntime`: Core implementation layer
- `ExecEngine`: Execution controller

Key files:
- `src/runtime/hip_runtime_abi.cpp` - C ABI / LD_PRELOAD entry
- `src/runtime/hip_runtime.cpp` - HIP compatibility
- `src/runtime/exec_engine.cpp` - Execution engine

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

### Trace vs Business Logic
- Trace only consumes events; it does NOT participate in business logic decisions
- Behavior must be identical with or without trace enabled
- Disable trace with: `GPU_MODEL_DISABLE_TRACE=1`

### Cycle Field Semantics
- The `cycle` field in trace outputs is **model time**, NOT real hardware time
- It represents relative ordering, wait intervals, and dependencies
- Do not interpret as physical execution time without calibration

### Runtime Layering
- `HipRuntime` -> `ModelRuntime` -> `ExecEngine` only
- No independent "interposer module" concept
- `src/runtime/hip_interposer.cpp` is just a C ABI entry point, not a separate module

### Functional vs Cycle Models
- Functional: allows st/mt host execution strategies
- Cycle: single unified timing model (no "cycle st" / "cycle mt" variants)
- Cycle model must be tick-driven state machine

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
