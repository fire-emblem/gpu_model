# Building and Testing gpu_model

This skill covers the standard build and test workflows for the gpu_model project.

## Prerequisites

```bash
# Set up environment (ROCM_PATH, PATH, LD_LIBRARY_PATH)
source scripts/setup_env.sh
```

## Build Commands

### Quick Build (recommended)
```bash
cmake --preset dev-fast
cmake --build --preset dev-fast
```

### Standard CMake Build
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
```

### Build Outputs
- Library: `libgpu_model.a`
- Tests: `build-ninja/tests/gpu_model_tests`
- Demo: `build-ninja/gpu_model_perfetto_waitcnt_slots_demo`

## Test Commands

### Run All Tests
```bash
./build-ninja/tests/gpu_model_tests
```

### Run Full Test Matrix
```bash
GPU_MODEL_TEST_PROFILE=full ./build-ninja/tests/gpu_model_tests
```

### Run Specific Test
```bash
./build-ninja/tests/gpu_model_tests --gtest_filter=TestName.*
```

### Run HIP-related Tests
```bash
./build-ninja/tests/gpu_model_tests --gtest_filter=*Hip*:*hip*
```

## Push Gate Scripts

```bash
# Light gate (fast, recommended for daily use)
./scripts/run_push_gate_light.sh

# Full gate (comprehensive, three parallel pipelines)
./scripts/run_push_gate.sh

# Specific regression tests
./scripts/run_shared_heavy_regression.sh
./scripts/run_real_hip_kernel_regression.sh
./scripts/run_abi_regression.sh
./scripts/run_scaling_regression.sh
```

## Example Scripts

Examples are in `examples/01-vecadd-basic/` through `examples/13-algorithm-comparison/`:

```bash
./examples/01-vecadd-basic/run.sh
```

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `GPU_MODEL_DISABLE_TRACE=1` | Disable trace output |
| `GPU_MODEL_TEST_PROFILE=full` | Run full test matrix |
| `GPU_MODEL_BUILD_DIR=...` | Override build directory |
| `GPU_MODEL_USE_HIPCC_CACHE=0` | Disable hipcc cache |
| `ROCM_PATH=...` | Custom ROCm installation path |

## Troubleshooting

### Build Fails: "hip_runtime_api.h not found"
- Ensure `ROCM_PATH` is set correctly
- Check HIP headers exist at `$ROCM_PATH/include/hip/`

### Tests Fail: "unsupported raw GCN opcode"
- Instruction is missing semantic handler
- See `.claude/skills/adding-gcn-instructions.md` to add support

### hipcc Fails: "cannot find ROCm device library"
- Check `HIP_DEVICE_LIB_PATH` points to bitcode directory
- Verify `$ROCM_PATH/amdgcn/bitcode/` contains `*.bc` files
