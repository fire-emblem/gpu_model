# Usage Bundles

This directory groups runnable workflows by usage instead of by source module.

- `functional-vecadd/`
  Functional execution of the built-in `vecadd` example.
- `cycle-fma-trace/`
  Cycle-mode execution of the FMA loop example with text trace, JSON trace, and ASCII timeline output.
- `scaling-regression/`
  Parameterized regression coverage for requested functional and cycle launch scales.
- `hip-fatbin-launch/`
  Compile a minimal HIP artifact, extract the embedded `.hip_fatbin` device bundle, and run the loader path.
- `hip-command-line-interposer/`
  Run a real HIP `.out` executable with `LD_PRELOAD` so host code executes normally while kernel launches are redirected into the model.
- `hip-fma-loop-interposer/`
  Run a real HIP `.out` executable with a looped floating-point kernel to validate raw GCN loop, branch, and FMA execution.
- `raw-code-object-decode/`
  Decode raw AMDGPU instructions from an AMDGPU ELF or HIP `.out` using the project-side format/encoding scaffold.

Each usage directory contains:

- `README.md`
  What the workflow is for.
- `PLAN.md`
  What the script does step by step.
- `DOC.md`
  Technical notes and expected artifacts.
- `run.sh`
  The executable script.
- `results/`
  Captured outputs produced by running the script.
