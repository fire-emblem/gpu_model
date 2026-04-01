# Remove Lowering Mainline Design

**Goal:** Remove lowering from the runtime mainline so `.out` / raw-code-object execution uses one encoded-driven path for both functional and cycle backends, then fix tests and examples against that unified path.

## Decision

- `lowering` is no longer allowed in the runtime mainline.
- Raw HIP / AMDGPU artifacts must not be bridged into modeled asm before execution.
- `ExecutionRoute::LoweredModeled` should be removed from the runtime-facing artifact path.
- `ProgramLoweringRegistry`, `GcnLoweringRuleRegistry`, and the loader-side lowering bridge are treated as legacy code to be removed from the mainline.
- `ExecutableKernel` may remain temporarily for internal hand-written tests only. It is not part of the raw artifact mainline anymore.

## Why

Current code is split across two incompatible execution inputs:

- encoded/raw artifacts can execute directly in the encoded functional path
- cycle currently consumes `ExecutableKernel` / modeled instructions

That split forces `.out -> cycle` to detour through loader-side lowering, which is why coverage gaps keep appearing as more real HIP programs are added. The result is codebase divergence: new real kernels require more lowering rules instead of improving the real execution path.

## Target Architecture

### One Artifact Mainline

For HIP `.out`, AMDGPU object, or encoded program object:

1. load artifact
2. decode to encoded instruction representation
3. initialize ABI / memory / launch metadata
4. execute through a backend that consumes encoded instructions

This applies to both:

- functional execution
- cycle execution

### Route Simplification

`ExecutionRoute` should converge to raw/encoded-oriented choices only.

- keep encoded/raw route as the default artifact route
- remove lowered/modeled route from the runtime artifact path
- stop auto-rewriting raw programs into modeled asm

### Backend Alignment

- `EncodedExecEngine` already executes encoded instructions in functional mode and should remain the functional raw-artifact mainline.
- cycle execution needs a corresponding encoded-driven path instead of requiring `ExecutableKernel`.
- the long-term shape should be one shared decoded/semantic substrate with different scheduling/timing policies layered on top.

## Removal Scope

### Remove From Mainline

- `ProgramLoweringRegistry`
- `program_lowering.cpp`
- `gcn_lowering_rules.cpp`
- `gcn_lowering_rule.h`
- runtime code that auto-selects `LoweredModeled` for raw artifacts
- tests whose only purpose is validating loader-side lowering

### Keep Temporarily

- `ExecutableKernel`
- `InstructionBuilder`
- `AsmParser`

These remain only to avoid breaking internal hand-written tests while the encoded path becomes the sole real execution path.

## Example Framework Impact

The env-driven example framework remains valid:

- same compiled `.out`
- env picks `st` / `mt` / `cycle`
- mode-specific outputs land under `results/<mode>/`

But `cycle` must now run through the encoded mainline, not through lowering. Any example failures after the cut are treated as encoded-path capability gaps to fix directly.

## Expected Breakage After The Cut

The first wave of failures is expected in:

- cycle execution for real HIP artifacts
- tests assuming lowered/modeled raw-artifact execution
- examples that currently only work because lowering rewrites unsupported raw patterns

Those failures are acceptable and should be fixed by expanding the encoded mainline, not by restoring lowering.

## Validation Strategy

1. remove lowering from the raw artifact runtime path
2. keep focused coverage for encoded functional and encoded/cycle artifact execution
3. run examples and tests
4. fix only encoded-path gaps revealed by those runs

## Non-Goals

- not deleting `ExecutableKernel` in this pass
- not rewriting every internal unit test to encoded input immediately
- not preserving compatibility for runtime paths that depend on loader-side lowering
