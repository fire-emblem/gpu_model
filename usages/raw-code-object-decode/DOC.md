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

- compute-focused真实 HIP kernel 的 decoded operands 已覆盖较广，但仍未达到“全部 GCN ISA”
- 很多 instantiated instruction object 仍会委托到旧 semantic handler
- descriptor/image/export/interp families 仍主要是占位，当前主覆盖方向仍是 compute path

Expected artifacts:

- `results/gtest_stdout.txt`
- `results/hip_vecadd.cpp`
- `results/hip_vecadd.out`
- `results/stdout.txt`
- `results/validation.txt`
