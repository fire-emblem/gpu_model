# Usage Bundles

This directory groups runnable workflows by usage instead of by source module.

- `functional-vecadd/`
  Functional execution of the built-in `vecadd` example.
- `cycle-fma-trace/`
  Cycle-mode execution of the FMA loop example with text trace, JSON trace, and ASCII timeline output.
- `scaling-regression/`
  Parameterized regression coverage for requested functional and cycle launch scales.

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
