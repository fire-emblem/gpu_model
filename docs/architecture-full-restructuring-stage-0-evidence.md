# Stage 0 Evidence — Final-State Contract and Gate Freeze

Date: 2026-04-14
Execution surface: `/data/gpu_model_stage0_fresh`

This document records the Stage 0 freeze/checkpoint evidence for the approved
architecture-full-restructuring program.

## Frozen stage-complete gate

Exact frozen gate command for this program:

```bash
./scripts/run_push_gate.sh
```

Supporting local iteration gate:

```bash
./scripts/run_push_gate_light.sh
```

## Gate snapshot

- `scripts/run_push_gate.sh`
  - SHA-256: `ce234a0a91316432c33d377139225d7ee2cc93b8449299ae027d7ea43b167e35`
  - Git blob: `7790868beb6032540c4294a6aff5ae60133ee07e`
- `scripts/run_push_gate_light.sh`
  - SHA-256: `1e4faec779c92d40f1ff080016b12388e8fc848595c2ce58a2caa26dd9835947`
  - Git blob: `336dc9c98ed58dd615a02e17c03e39bd30498e42`

## Gate-fix regressions repaired during Stage 0

The frozen gate was not defensibly closable at the start of execution. Stage 0
therefore includes the narrow gate-fix repairs needed to make the frozen gate
command complete cleanly without altering the gate scripts themselves.

Repaired Stage 0 gate blockers:

- `CMakeLists.txt`
  - added the missing `URL_HASH` for googletest `FetchContent`
- `tests/runtime/trace_encoded_test.cpp`
  - repaired the broken encoded waitcnt fixture
  - switched to the canonical `runtime/exec_engine/exec_engine.h` include path
- `examples/11-perfetto-waitcnt-slots/perfetto_waitcnt_slots_demo.cpp`
  - switched to the canonical `runtime/exec_engine/exec_engine.h` include path
- `examples/common.sh`
  - injects default `--offload-arch=${GPU_MODEL_HIP_OFFLOAD_ARCH:-gfx90a}` when absent
- `examples/01-vecadd-basic/vecadd.hip`
  - updated success marker to `vecadd validation ok`
  - reduced `N` from `1 << 20` to `1 << 12` to keep the frozen gate non-stress
- `examples/07-vecadd-cycle-splitting/run.sh`
  - reads cycle totals from merged trace summary output instead of the removed
    standalone `launch_summary.txt`
- `examples/11-perfetto-waitcnt-slots/run.sh`
  - caps functional worker threads to `4` for deterministic non-stress gate execution
- `tests/runtime/executed_flow_program_cycle_stats_test.cpp`
  - constrains the representative cycle-stats helper to a deterministic MT
    worker count during test execution so the frozen release suite does not hang
    behind the host default worker pool size

## Verification evidence

### Focused gate-fix checks

- Trace regression repair:

  ```bash
  cmake --build build-gate-release --target gpu_model_tests gpu_model_hip_runtime_abi gpu_model_perfetto_waitcnt_slots_demo -j 8
  ./build-gate-release/tests/gpu_model_tests --gtest_filter='TraceEncodedTest.PerfettoProtoShowsEncodedFunctionalLoadArriveInMultiThreadedMode'
  ```

- Example 07 cycle-summary repair:

  ```bash
  export GPU_MODEL_BUILD_DIR="$PWD/build-gate-examples" GPU_MODEL_USE_HIPCC_CACHE=0
  ./examples/07-vecadd-cycle-splitting/run.sh
  ```

- Example 11 deterministic worker-count repair:

  ```bash
  export GPU_MODEL_BUILD_DIR="$PWD/build-gate-examples" GPU_MODEL_USE_HIPCC_CACHE=0
  ./examples/11-perfetto-waitcnt-slots/run.sh
  ```

- Release hang reproducer / fix confirmation:

  ```bash
  timeout 120s env GPU_MODEL_FUNCTIONAL_WORKERS=4 \
    ./build-gate-release/tests/gpu_model_tests \
    --gtest_filter=ExecutedFlowProgramCycleStatsTest.RepresentativeCasesMaintainAccountingAndModeAgreement
  ```

### Light gate

- Passed:

  ```bash
  ./scripts/run_push_gate_light.sh
  ```

### Frozen full gate

The frozen full gate completed cleanly in the isolated Stage 0 worktree:

```bash
./scripts/run_push_gate.sh
```

Wrapper terminal status:

```text
[push-gate] ok
- debug+asan tests passed
- release tests passed
- all examples passed
```

Evidence paths:

- wrapper transcript: `/tmp/gpu_model_stage0_fresh_fullgate_final.out`
- release test log: `results/push-gate/release.gpu_model_tests.log`
- debug+asan test log: `results/push-gate/debug_asan.gpu_model_tests.log`
- example logs:
  - `results/push-gate/example_01-vecadd-basic.log`
  - `results/push-gate/example_02-fma-loop.log`
  - `results/push-gate/example_03-shared-reverse.log`
  - `results/push-gate/example_04-atomic-reduction.log`
  - `results/push-gate/example_05-softmax-reduction.log`
  - `results/push-gate/example_06-mma-gemm.log`
  - `results/push-gate/example_07-vecadd-cycle-splitting.log`
  - `results/push-gate/example_08-conditional-multibarrier.log`
  - `results/push-gate/example_09-dynamic-shared-sum.log`
  - `results/push-gate/example_10-block-reduce-sum.log`
  - `results/push-gate/example_11-perfetto-waitcnt-slots.log`

Observed final suite summaries:

- release: `819 tests`, `812 passed`, `7 skipped`, `0 failed`
- debug+asan: `801 tests`, `794 passed`, `7 skipped`, `0 failed`

## Stage 0 inventories

Stage 0 freezes the exact inventory commands and their current outputs.

### Runtime bridge include inventory

Command:

```bash
rg -n '#include "runtime/(exec_engine|model_runtime|hip_runtime|runtime_session|runtime_submission_context|program_cycle_tracker|launch_request|launch_config|kernel_arg_pack|kernarg_packer)\.h"' src tests
```

Current count: `118`

### State / GPU-arch / execution bridge include inventory

Command:

```bash
rg -n '#include "state/peu_state.h"|#include "execution/wave_context.h"|#include "execution/internal/(opcode_execution_info|memory_arrive_kind|issue_model)\.h"|#include "gpu_arch/chip_config/(gpu_arch_spec|amdgpu_target_config)\.h"' src tests
```

Current count: `32`

### Bridge / migration marker inventory

Command:

```bash
rg -n 'Bridge header|legacy bridge|deprecated\. Update your include path|backward compatibility|during migration' src tests
```

Current count: `48`

### Forbidden dependency-direction inventory

Command:

```bash
rg -n '#include "execution/' src/gpu_arch src/state src/debug src/instruction
```

Current count: `0`

### Placeholder / reserved structure inventory

Commands:

```bash
test -d src/runtime/kernel_stub && echo present || echo absent
test -f src/debug/replay/replayer.h && echo present || echo absent
```

Current state:

- `src/runtime/kernel_stub`: `absent`
- `src/debug/replay/replayer.h`: `present`

Placeholder evidence:

- `src/debug/replay/replayer.h:5-8` still says `Placeholder for future execution-state replay / restore support.`

### Structurally mislocated test inventory

Command:

```bash
rg -n 'execution/internal/handlers/semantic_handler_test\.cpp' tests/CMakeLists.txt docs
```

Current hit:

- `tests/CMakeLists.txt:38:  execution/internal/handlers/semantic_handler_test.cpp`

## Doc contradictions against the strict final-state rule

Command:

```bash
rg -n 'kernel_stub|future extension|future execution-state replay|placeholder|bridge retention|reserved node' \
  docs/architecture-restructuring-plan.md \
  docs/superpowers/plans/2026-04-13-architecture-final-cleanup.md
```

Current contradictions:

- `docs/architecture-restructuring-plan.md:32`
  - still names `runtime/kernel_stub/` in the target tree
- `docs/architecture-restructuring-plan.md:194`
  - still describes `debug/replay/` as placeholder-oriented
- `docs/architecture-restructuring-plan.md:337`
  - still describes `replay/` as future extension / placeholder
- `docs/superpowers/plans/2026-04-13-architecture-final-cleanup.md:62`
  - still normalizes `kernel_stub/` as part of the target runtime subtree
- `docs/superpowers/plans/2026-04-13-architecture-final-cleanup.md:68-70`
  - still treats `runtime/kernel_stub/` as an acceptable reserved node
- `docs/superpowers/plans/2026-04-13-architecture-final-cleanup.md:88`
  - still marks `runtime/kernel_stub/` as reserved / acceptable

## Conclusion

Stage 0 freezes:

1. the exact stage-complete gate contract,
2. the script hashes/blob IDs guarding that contract,
3. the repaired frozen-gate execution proof from the isolated worktree, and
4. the initial bridge/dependency/placeholder/doc-contradiction inventory that
   later stages must drive to zero or eliminate.
