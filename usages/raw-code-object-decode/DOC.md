# Doc

This usage bundle demonstrates the current binary-side scaffold:

- code object extraction from HIP `.out`
- raw instruction words
- GCN format classification
- minimal encoding-id resolution
- project formatter output

Current limitation:

- decoded operands only cover a small bootstrap subset
- project formatting is still partial
- this is a scaffold for the eventual full binary decode / disassemble path

Expected artifacts:

- `results/hip_vecadd.cpp`
- `results/hip_vecadd.out`
- `results/stdout.txt`
