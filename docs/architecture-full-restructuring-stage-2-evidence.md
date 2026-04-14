# Architecture Full Restructuring — Stage 2 Evidence

## Stage 2 cleanup plan

1. Replace remaining true bridge includes with canonical ownership headers:
   - `state/peu_state.h` -> `state/peu/peu_runtime_state.h`
   - `execution/wave_context.h` -> `state/wave/wave_runtime_state.h`
   - `execution/internal/opcode_execution_info.h` -> `instruction/isa/opcode_info.h`
   - `execution/internal/memory_arrive_kind.h` -> `gpu_arch/memory/memory_arrive_kind.h`
2. Confirm `execution/internal/issue_model.h` is already consumer-free and leave it transitional only until Stage 3 deletion.
3. Re-evaluate `gpu_arch/chip_config/{gpu_arch_spec,amdgpu_target_config}.h` against the approved artifacts and current code. If they are already canonical owners rather than redirect bridges, document that exception explicitly and avoid destructive relocation without design evidence.
4. Run Stage 2 targeted tests, residual include/dependency-direction proof, then the full frozen gate before commit/push.

## Canonical landing-zone determination

### True bridge surfaces migrated in Stage 2

- `src/state/peu_state.h` is a redirect to `state/peu/peu_runtime_state.h`.
- `src/execution/wave_context.h` is a redirect to `state/wave/wave_runtime_state.h`.
- `src/execution/internal/opcode_execution_info.h` is a redirect to `instruction/isa/opcode_info.h`.
- `src/execution/internal/memory_arrive_kind.h` is a redirect to `gpu_arch/memory/memory_arrive_kind.h`.
- `src/execution/internal/issue_model.h` is a redirect to `gpu_arch/issue_config/issue_config.h` and had no remaining `src/` or `tests/` consumers.

### Chip-config ambiguity resolution

`gpu_arch/chip_config/gpu_arch_spec.h` and `gpu_arch/chip_config/amdgpu_target_config.h` are treated as current canonical owners, not Stage 2 move/delete candidates:

- `src/gpu_arch/chip_config/gpu_arch_spec.h` contains the real `GpuArchSpec` definitions and matches the architecture plan's target ownership under `gpu_arch/chip_config/`.
- `src/gpu_arch/chip_config/amdgpu_target_config.h` contains the real target constants/helpers and the approved artifacts define no alternate landing zone.

Stage 2 therefore migrates consumers off the true bridges while keeping both chip-config headers in place as canonical owners. Their appearance in the PRD/test-spec Stage 2/3 header lists is interpreted as stale residue inventory, not authorization to delete real owning headers without a revised plan.

## Files changed for Stage 2

### Source / test include cutovers

- `src/state/ap/ap_runtime_state.h`
- `src/execution/internal/semantic_handler.h`
- `src/execution/internal/cycle_types.cpp`
- `src/execution/internal/wave_state.h`
- `src/execution/functional/functional_exec_engine.cpp`
- `src/execution/cycle/cycle_exec_engine.cpp`
- `src/execution/encoded/program_object_exec_engine.cpp`
- `tests/instruction/instruction_object_execute_test.cpp`
- `tests/isa/opcode_descriptor_test.cpp`

### Stage 2 proof tightening

- `tests/arch/arch_registry_test.cpp`
- `tests/arch/amdgpu_target_config_test.cpp`
- `tests/CMakeLists.txt`

### Evidence

- `docs/architecture-full-restructuring-stage-2-evidence.md`

## Verification evidence

### Targeted Stage 2 regression

Command:

```bash
./build-gate-release/tests/gpu_model_tests \
  --gtest_filter='ExecutionNamingTest.*:WaveContextBuilderTest.*:IssueEligibilityTest.*:ArchRegistryTest.*:GpuArchSpecTest.*:AmdgpuTargetConfigTest.*'
```

Result:

- `21 tests from 6 test suites`
- all passed
- transcript: `/tmp/stage2-evidence/targeted-tests-postgate.txt`

### Residual bridge-include proof

Command:

```bash
rg -n '#include "state/peu_state.h"|#include "execution/wave_context.h"|#include "execution/internal/(opcode_execution_info|memory_arrive_kind|issue_model)\.h"' src tests
```

Result:

- no matches
- transcript: `/tmp/stage2-evidence/searches-postgate.txt`

### Canonical owner snapshot for chip-config headers

Command:

```bash
rg -n '#include "gpu_arch/chip_config/(gpu_arch_spec|amdgpu_target_config)\.h"' src tests
```

Result:

- expected matches remain for canonical owners only
- `gpu_arch_spec.h` continues to serve architecture/config consumers
- `amdgpu_target_config.h` continues to serve program/loader + test/tooling consumers
- transcript: `/tmp/stage2-evidence/searches-postgate.txt`

### Dependency-direction proof

Command:

```bash
rg -n '#include "execution/' src/gpu_arch src/state src/debug src/instruction
```

Result:

- no matches
- transcript: `/tmp/stage2-evidence/searches-postgate.txt`

### Full frozen gate

Command:

```bash
./scripts/run_push_gate.sh
```

Result:

- passed cleanly
- terminal proof ends with:

```text
[push-gate] ok
- debug+asan tests passed
- release tests passed
- all examples passed
```

- transcript: `/tmp/run_push_gate_stage2_clean.log`
- exit status file: `/tmp/run_push_gate_stage2_clean.exit`

### Post-gate cleanup

Tracked example result artifacts dirtied by the gate were restored before stage-close status review:

```bash
git restore --worktree --staged -- \
  examples/08-conditional-multibarrier/results \
  examples/11-perfetto-waitcnt-slots/results
```

Result:

- worktree residue returned to Stage 2 code/test/evidence files only
