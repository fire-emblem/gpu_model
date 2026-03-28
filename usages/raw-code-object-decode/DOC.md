# Doc

This usage bundle demonstrates the current binary-side scaffold:

- code object extraction from HIP `.out`
- `text bytes -> raw instruction array -> decoded instruction array`
- instruction object factory instantiation during decode
- raw instruction words
- GCN format classification
- canonical opcode-name fallback
- project formatter output including:
  - `fmt`
  - `op_type`
  - instantiated object `class`

Current limitation:

- decoded operands only cover a small bootstrap subset
- many instantiated instruction objects still delegate execution to legacy semantic handlers
- descriptor/image/export/interp families are still placeholder objects in the compute-first model

Expected artifacts:

- `results/gtest_stdout.txt`
- `results/hip_vecadd.cpp`
- `results/hip_vecadd.out`
- `results/stdout.txt`
- `results/validation.txt`
