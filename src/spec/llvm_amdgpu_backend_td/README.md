# LLVM AMDGPU Backend TD References

Downloaded backend TableGen sources from `llvm-project/llvm/lib/Target/AMDGPU/`.

Currently stored locally:

- `SOPInstructions.td`
- `VOPInstructions.td`
- `SMInstructions.td`
- `MIMGInstructions.td`

Primary uses:

- cross-check opcode names and opcode values
- cross-check format families against project-side `gcn_db`
- guide completion of missing instructions and operand schemas
- compare project decode/exec coverage with LLVM backend instruction definitions

Recommended usage order:

1. `AMDGPUAsmGFX9.html` for syntax and user-facing names
2. `.td` files for backend opcode/encoding definitions
3. project `gcn_db` as the generated runtime source of truth
