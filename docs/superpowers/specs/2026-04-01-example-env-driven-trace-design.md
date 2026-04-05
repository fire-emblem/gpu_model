# Example Env-Driven Trace Design

**Goal:** Make every example run through the same compiled `.out` entrypoint while using environment variables to choose `st`, `mt`, or `cycle`, and default to writing trace artifacts into per-mode results directories.

**Scope:** `examples/01-07`, HIP interposer `.out` path, execution-mode env selection, default trace artifact dumping, and enough machine-readable launch summary for bash-side cycle validation in example scripts.

## Requirements

- Every example keeps the current host-side validation behavior.
- The same compiled `.out` must be executed for all modes.
- Mode selection must be env-driven, not tool-driven:
  - `functional + st`
  - `functional + mt`
  - `cycle`
- Example outputs must land under mode-specific directories:
  - `results/st/`
  - `results/mt/`
  - `results/cycle/`
- Each mode directory must default to containing:
  - `stdout.txt`
  - `trace.txt`
  - `trace.jsonl`
  - `timeline.perfetto.json`
  - `launch_summary.txt`
- `timeline.perfetto.json` must be produced in a format that Perfetto / Chrome trace viewers can open directly.
- `examples/07` must use the emitted launch summary to perform bash-side cycle comparison validation across its three kernels.

## Design

### Execution Mode

- Keep `GPU_MODEL_FUNCTIONAL_MODE=st|mt` as the existing functional scheduler selector.
- Add `GPU_MODEL_EXECUTION_MODE=functional|cycle` for the higher-level execution backend.
- The HIP interposer resolves the execution mode before forwarding the kernel launch into `HipRuntime::LaunchExecutableKernel(...)`.
- Functional launches continue to use the current runtime path.
- Cycle launches reuse the same `.out -> ProgramObject -> ExecEngine` path, only with `ExecutionMode::Cycle`.

### Trace Artifact Emission

- Add a small trace artifact sink/recorder in the debug layer.
- It multiplexes three outputs:
  - text trace stream
  - JSONL trace stream
  - in-memory event collection for timeline rendering
- After each model launch, it rewrites `timeline.perfetto.json` from the collected events using the existing Google Trace renderer.
- The interposer creates and owns this recorder when `GPU_MODEL_TRACE_DIR` is set.
- The trace recorder is process-local and is reused across all launches in one example process.

### Launch Summary

- The interposer writes a stable text summary per kernel launch to `launch_summary.txt`.
- Summary format is line-oriented `key=value` pairs so bash can parse it without `jq`.
- Required fields:
  - `kernel`
  - `execution_mode`
  - `functional_mode`
  - `ok`
  - `submit_cycle`
  - `begin_cycle`
  - `end_cycle`
  - `total_cycles`
  - `program_total_cycles` when available

### Example Script Layout

- Each example still compiles exactly one `.out` per source.
- Each example script runs the compiled `.out` three times:
  - `st`
  - `mt`
  - `cycle`
- Each run sets:
  - `GPU_MODEL_EXECUTION_MODE`
  - `GPU_MODEL_FUNCTIONAL_MODE` when functional
  - `GPU_MODEL_FUNCTIONAL_WORKERS` for `mt`
  - `GPU_MODEL_TRACE_DIR`
- `examples/common.sh` gains shared helpers for:
  - mode directory creation
  - ROCm library path detection
  - env selection
  - per-mode interposed execution

### Example 07 Validation

- `examples/07/run.sh` keeps the existing host validation checks for all three kernels.
- It additionally reads `results/cycle/launch_summary.txt` for:
  - `vecadd_direct`
  - `vecadd_grid_stride`
  - `vecadd_chunk2`
- It asserts:
  - cycle launches succeeded
  - cycle totals are positive
  - the three kernels produced distinct comparable cycle totals
- The script writes a small comparison summary into the case results directory.

## Testing

- Add focused runtime/interposer tests for:
  - execution-mode env selection
  - cycle launch through the `.out` interposer path
  - trace artifact recorder output files
- Re-run focused trace, interposer, and cycle tests.
- Re-run updated example scripts in at least one representative ring.

## Non-Goals

- No attempt to make the current cycle model hardware-accurate.
- No new standalone runner binary.
- No global automatic trace dumping when `GPU_MODEL_TRACE_DIR` is unset.
