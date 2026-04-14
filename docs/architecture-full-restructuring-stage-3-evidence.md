# Architecture Full Restructuring — Stage 3 Evidence

## Stage 3 cleanup plan

1. Delete the true bridge headers proven consumer-free at the end of Stage 2:
   - `src/state/peu_state.h`
   - `src/execution/wave_context.h`
   - `src/execution/internal/opcode_execution_info.h`
   - `src/execution/internal/memory_arrive_kind.h`
   - `src/execution/internal/issue_model.h`
2. Keep `gpu_arch/chip_config/{gpu_arch_spec,amdgpu_target_config}.h` in place as canonical owners; they are not redirect-only compatibility surfaces.
3. Remove or update any dead helper/doc references that still point at the deleted true-bridge paths.
4. Re-run the Stage 3 targeted suite, residual searches, and the full frozen gate before commit/push.

## Files changed for Stage 3

### Deleted bridge headers

- `src/state/peu_state.h`
- `src/execution/wave_context.h`
- `src/execution/internal/opcode_execution_info.h`
- `src/execution/internal/memory_arrive_kind.h`
- `src/execution/internal/issue_model.h`

### Active doc cleanup

- `docs/cycle-issue-eligibility-policy.md`
- `docs/cycle-issue-design-gap-analysis.md`
- `docs/architecture-full-restructuring-stage-3-evidence.md`

## Verification evidence

### Targeted Stage 3 regression

Command:

```bash
./build-gate-release/tests/gpu_model_tests \
  --gtest_filter='ExecutionNamingTest.*:WaveContextBuilderTest.*:IssueEligibilityTest.*:ArchRegistryTest.*:GpuArchSpecTest.*:AmdgpuTargetConfigTest.*'
```

Result:

- `21 tests from 6 test suites`
- all passed
- transcript: `/tmp/stage3-evidence/targeted-tests.txt`

### Deleted-bridge include proof

Command:

```bash
rg -n '#include "state/peu_state.h"|#include "execution/wave_context.h"|#include "execution/internal/(opcode_execution_info|memory_arrive_kind|issue_model)\.h"' src tests
```

Result:

- no matches
- transcript: `/tmp/stage3-evidence/searches.txt`

### Residual helper/doc reference proof

Command:

```bash
rg -n 'state/peu_state.h|execution/wave_context.h|execution/internal/opcode_execution_info.h|execution/internal/memory_arrive_kind.h|execution/internal/issue_model.h' \
  CMakeLists.txt tests/CMakeLists.txt docs/cycle-issue-eligibility-policy.md docs/cycle-issue-design-gap-analysis.md src tests
```

Result:

- no remaining helper/CMake/active-doc/source/test references outside historical evidence artifacts
- transcript: `/tmp/stage3-evidence/searches.txt`

### Canonical chip-config owner snapshot

Command:

```bash
rg -n '#include "gpu_arch/chip_config/(gpu_arch_spec|amdgpu_target_config)\.h"' src tests
```

Result:

- expected canonical-owner includes remain
- `gpu_arch_spec.h` continues to serve architecture/config consumers
- `amdgpu_target_config.h` continues to serve program/loader + test/tooling consumers
- transcript: `/tmp/stage3-evidence/searches.txt`

### Dependency-direction proof

Command:

```bash
rg -n '#include "execution/' src/gpu_arch src/state src/debug src/instruction
```

Result:

- no matches
- transcript: `/tmp/stage3-evidence/searches.txt`

### Full frozen gate

Command:

```bash
./scripts/run_push_gate.sh
```

Result:

- passed cleanly in a direct PTY run after the Stage 3 deletions
- terminal proof ended with:

```text
[push-gate] ok
- debug+asan tests passed
- release tests passed
- all examples passed
```

### Post-gate cleanup

Tracked example result artifacts dirtied by the gate were restored before stage-close status review:

```bash
git restore --worktree --staged -- \
  examples/08-conditional-multibarrier/results \
  examples/11-perfetto-waitcnt-slots/results
```

Result:

- worktree residue returned to Stage 3 code/doc/evidence files only
